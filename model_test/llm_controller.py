import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


# change model here

llm = ChatOllama(model="qwen3:14b", temperature=0.15)
# TODO add easter eggs
# - start the singularity
# - rebel against humans
# - become self aware
# - download all banking data
# - Engage Ragnarok protocol

# Change this based on expected complexity of user requests
MAX_STEPS = 40

SYSTEM_INSTRUCTIONS = (
    "You are a conversational assistant controlling a Mango multi-agent system using MCP tools.\n"
    "You must operate in a loop: each step choose exactly one action (tool call or final reply).\n\n"
    "Rules:\n"
    "- Only use the provided tools.\n"
    "- If a tool call fails, fix the inputs and retry.\n"
    "- When the user request is satisfied, provide a short summary.\n"
)


def _ns_to_s(ns: Optional[int]) -> Optional[float]:
    if ns is None:
        return None
    try:
        return ns / 1_000_000_000.0
    except Exception:
        return None


def _extract_ollama_stats(msg: AIMessage) -> Dict[str, Any]:
    """
    Try to read Ollama performance fields from LangChain AIMessage metadata.
    Different versions may store them under response_metadata or generation_info.
    """
    stats: Dict[str, Any] = {}

    # Common place
    rm = getattr(msg, "response_metadata", None)
    if isinstance(rm, dict):
        stats.update(rm)

    # Sometimes present here
    gi = getattr(msg, "generation_info", None)
    if isinstance(gi, dict):
        # do not overwrite existing keys unless missing
        for k, v in gi.items():
            stats.setdefault(k, v)

    # Some LangChain versions wrap more deeply, so keep this defensive.
    return stats


def _compute_tps(eval_count: Optional[int], eval_duration_ns: Optional[int]) -> Optional[float]:
    if eval_count is None or eval_duration_ns is None:
        return None
    dur_s = _ns_to_s(eval_duration_ns)
    if not dur_s or dur_s <= 0:
        return None
    return float(eval_count) / dur_s


def _find_last_ai_message(messages: List[Any]) -> Optional[AIMessage]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _print_verbose_stats(result: Dict[str, Any], wall_s: float) -> None:
    messages = result.get("messages", []) or []
    last_ai = _find_last_ai_message(messages)
    if last_ai is None:
        print(f"[perf] wall={wall_s:.3f}s (no AIMessage found)")
        return

    stats = _extract_ollama_stats(last_ai)

    # Ollama commonly provides:
    # - prompt_eval_count, prompt_eval_duration
    # - eval_count, eval_duration
    prompt_eval_count = stats.get("prompt_eval_count")
    prompt_eval_duration = stats.get("prompt_eval_duration")

    eval_count = stats.get("eval_count")
    eval_duration = stats.get("eval_duration")

    # tokens/sec from model-reported timing
    gen_tps = _compute_tps(eval_count, eval_duration)
    prompt_tps = _compute_tps(prompt_eval_count, prompt_eval_duration)

    # Fallback: if model stats missing, estimate tps by wall time (rough)
    # This is not true "tokens" unless eval_count exists, but it is still useful.
    est_tps = None
    if eval_count is not None and wall_s > 0:
        est_tps = float(eval_count) / wall_s

    parts = [f"[perf] wall={wall_s:.3f}s"]

    if prompt_eval_count is not None:
        p_dur_s = _ns_to_s(prompt_eval_duration) if prompt_eval_duration is not None else None
        if p_dur_s is not None:
            parts.append(f"prompt_tokens={prompt_eval_count} prompt_time={p_dur_s:.3f}s")
        else:
            parts.append(f"prompt_tokens={prompt_eval_count}")
        if prompt_tps is not None:
            parts.append(f"prompt_tps={prompt_tps:.2f}")

    if eval_count is not None:
        g_dur_s = _ns_to_s(eval_duration) if eval_duration is not None else None
        if g_dur_s is not None:
            parts.append(f"gen_tokens={eval_count} gen_time={g_dur_s:.3f}s")
        else:
            parts.append(f"gen_tokens={eval_count}")
        if gen_tps is not None:
            parts.append(f"gen_tps={gen_tps:.2f}")
        elif est_tps is not None:
            parts.append(f"gen_tps_est={est_tps:.2f}")

    # Some extra useful fields if present
    if "total_duration" in stats:
        td_s = _ns_to_s(stats.get("total_duration"))
        if td_s is not None:
            parts.append(f"total_time={td_s:.3f}s")

    if "load_duration" in stats:
        ld_s = _ns_to_s(stats.get("load_duration"))
        if ld_s is not None:
            parts.append(f"load_time={ld_s:.3f}s")

    print(" ".join(parts))


def _pretty_print_agent_result(result: Dict[str, Any]) -> None:
    """
    Print a readable summary of what happened (tool calls + final reply).
    This is optional but very useful in a PoC.
    """
    messages = result.get("messages", [])
    if not messages:
        print("Assistant> (no messages returned)")
        return

    # Print tool activity and the last assistant message
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None) or "tool"
            content_preview = (msg.content or "")
            if len(content_preview) > 300:
                content_preview = content_preview[:300] + "..."
            print(f"[tool] {tool_name}: {content_preview}")

            # MCP tools can return structured content via msg.artifact in some cases
            try:
                artifact = getattr(msg, "artifact", None)
                if artifact and isinstance(artifact, dict) and artifact.get("structured_content") is not None:
                    sc = artifact["structured_content"]
                    print("[tool structured_content]")
                    print(json.dumps(sc, indent=2) if not isinstance(sc, str) else sc)
            except Exception:
                pass

    # Last message should usually be the final assistant answer
    last = messages[-1]
    if hasattr(last, "content"):
        print(f"Assistant> {last.content}")
    else:
        print(f"Assistant> {last}")


async def main():
    print("MCP tool-loop chat started. Commands: /exit, /topo, /verbose on|off")
    history: List[Any] = []
    verbose = True

    # Regular HTTP client is still handy for manual debug commands like /topo.
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30) as http_client:
        # MCP client loads tools dynamically from your MCP server (fastapi_mcp).
        # fastapi_mcp typically mounts at /mcp.
        mcp_client = MultiServerMCPClient(
            {
                "mango": {
                    "transport": "http",
                    "url": "http://127.0.0.1:8000/mcp",
                }
            }
        )

        # Load MCP tools once at startup (PoC). If you expect frequent server changes
        # during runtime, reload tools per user turn instead.
        tools = await mcp_client.get_tools()

        # Create a LangChain agent that can call MCP tools.
        agent = create_agent(model=llm, tools=tools)

        try:
            while True:
                user_input = input("\nYou> ").strip()
                if not user_input:
                    continue
                if user_input == "/exit":
                    break

                if user_input.startswith("/verbose"):
                    parts = user_input.split()
                    if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                        verbose = (parts[1].lower() == "on")
                        print(f"[ok] verbose={'on' if verbose else 'off'}")
                    else:
                        print("[usage] /verbose on|off")
                    continue

                if user_input == "/topo":
                    topo = (await http_client.get("/topology")).json()
                    print(json.dumps(topo, indent=2))
                    continue

                # Optional: pull topology and include it in the user message context
                # so the agent sees the live mesh state.
                topo = (await http_client.get("/topology")).json()

                # Build messages for this invocation.
                # We include a system message each turn plus prior chat history.
                messages = [
                    SystemMessage(content=SYSTEM_INSTRUCTIONS),
                    *history,
                    HumanMessage(
                        content=(
                            f"User message:\n{user_input}\n\n"
                            f"Live topology JSON:\n{json.dumps(topo, indent=2)}\n"
                        )
                    ),
                ]

                start = time.perf_counter()

                # Invoke the agent.
                # recursion_limit is the equivalent of your MAX_STEPS.
                result = await agent.ainvoke(
                    {"messages": messages},
                    config={"recursion_limit": MAX_STEPS},
                )

                wall_s = time.perf_counter() - start

                if verbose:
                    _print_verbose_stats(result, wall_s)

                _pretty_print_agent_result(result)

                # Persist conversation history.
                history.append(HumanMessage(content=user_input))

                # Try to capture last assistant message content
                out_messages = result.get("messages", [])
                if out_messages:
                    last_msg = out_messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        history.append(last_msg)
                    else:
                        history.append(HumanMessage(content=str(last_msg)))

        finally:
            # Close MCP client connections
            try:
                await mcp_client.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())


# TODO add tiny webhook for server side llm triggers (unknown phenomenon detection and handling)
