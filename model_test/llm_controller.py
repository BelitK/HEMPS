import asyncio
import json
from typing import Any, Dict, List

import httpx

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

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
MAX_STEPS = 20

SYSTEM_INSTRUCTIONS = (
    "You are a conversational assistant controlling a Mango multi-agent system using MCP tools.\n"
    "You must operate in a loop: each step choose exactly one action (tool call or final reply).\n\n"
    "Rules:\n"
    "- Only use the provided tools.\n"
    "- If a tool call fails, fix the inputs and retry.\n"
    "- When the user request is satisfied, provide a short summary.\n"
)


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
    print("MCP tool-loop chat started. Commands: /exit, /topo")
    history: List[Any] = []

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

                # Invoke the agent.
                # recursion_limit is the equivalent of your MAX_STEPS.
                result = await agent.ainvoke(
                    {"messages": messages},
                    config={"recursion_limit": MAX_STEPS},
                )

                _pretty_print_agent_result(result)

                # Persist conversation history.
                # Agent returns a full list of messages; we can store only the delta,
                # but for PoC simplicity store the last user and last assistant reply.
                history.append(HumanMessage(content=user_input))

                # Try to capture last assistant message content
                out_messages = result.get("messages", [])
                if out_messages:
                    last_msg = out_messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        history.append(last_msg)
                    else:
                        # Fallback: store a simple text summary
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
