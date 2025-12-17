import asyncio
import json
from typing import Optional, Literal, Dict, Any, List

import httpx
from pydantic import BaseModel, Field, ConfigDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


class ToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["call_tool", "reply"]

    # call_tool
    tool_name: Optional[str] = Field(default=None, description="Must match one of tools[].name from /tools")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call")

    # reply
    text: Optional[str] = None

    # private notes (NOT executed, NOT sent to server)
    notes: Optional[str] = Field(
        default=None,
        max_length=800,
        description="Private reasoning/notes for this step. Never include tool arguments here."
    )


llm = ChatOllama(model="gemma3:12b", temperature=0.15)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversational assistant controlling a Mango multi-agent system using server-exposed HTTP tools.\n"
     "You must operate in a loop: each step choose exactly one action.\n\n"
     "You will receive a JSON list of tools from the server at /tools.\n"
     "To do any change, select one tool by name and provide args.\n\n"
     "Important:\n"
     "- You may include private notes in the 'notes' field for debugging.\n"
     "- Notes are never executed and never sent to the server.\n"
     "- Do not put tool parameters in notes. Put them in args.\n\n"
     "Rules:\n"
     "- Always choose tool_name from the tools list.\n"
     "- Provide args that match the tool args_schema.\n"
     "- If the last tool call returned an error, fix it in the next step.\n"
     "- When the user request is satisfied, respond with action=reply and a short summary.\n"
     "Return only a ToolDecision object."
    ),
    MessagesPlaceholder("history"),
    ("human",
     "User message:\n{user_input}\n\n"
     "Tools JSON:\n{tools_json}\n\n"
     "Live topology JSON:\n{topology_json}\n\n"
     "This-turn execution log (most recent last):\n{exec_log}\n")
])

decision_chain = prompt | llm.with_structured_output(ToolDecision)

MAX_STEPS = 20


def index_tools(tools_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tools = tools_payload.get("tools", [])
    out: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        name = t.get("name")
        if isinstance(name, str):
            out[name] = t
    return out


async def call_server_tool(client: httpx.AsyncClient, tool_def: Dict[str, Any], args: Dict[str, Any]) -> httpx.Response:
    method = (tool_def.get("method") or "GET").upper()
    path = tool_def.get("path") or "/"

    if method == "GET":
        return await client.get(path, params=args or None)
    if method == "POST":
        return await client.post(path, json=args)
    if method == "PUT":
        return await client.put(path, json=args)
    if method == "PATCH":
        return await client.patch(path, json=args)
    if method == "DELETE":
        return await client.delete(path, params=args or None)

    raise RuntimeError(f"Unsupported HTTP method in tool definition: {method}")


async def main():
    print("Tool-loop chat started. Commands: /exit, /topo")
    history: List[Any] = []

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30) as client:
        while True:
            user_input = input("\nYou> ").strip()
            if not user_input:
                continue
            if user_input == "/exit":
                break

            if user_input == "/topo":
                topo = (await client.get("/topology")).json()
                print(json.dumps(topo, indent=2))
                continue

            exec_log: List[Dict[str, Any]] = []

            for _ in range(MAX_STEPS):
                topo = (await client.get("/topology")).json()
                tools_payload = (await client.get("/tools")).json()
                tools_by_name = index_tools(tools_payload)

                decision: ToolDecision = await decision_chain.ainvoke({
                    "history": history,
                    "user_input": user_input,
                    "tools_json": tools_payload,
                    "topology_json": topo,
                    "exec_log": exec_log,
                })

                # Print notes locally (optional)
                if decision.notes:
                    print(f"[notes] {decision.notes}")

                if decision.action == "reply":
                    text = decision.text or "Done."
                    print(f"Assistant> {text}")
                    history.append(HumanMessage(content=user_input))
                    history.append(AIMessage(content=text))
                    break

                tool_name = (decision.tool_name or "").strip()
                if tool_name not in tools_by_name:
                    exec_log.append({
                        "action": "call_tool",
                        "tool_name": tool_name,
                        "args": decision.args,
                        "error": f"Unknown tool_name. Must be one of: {sorted(list(tools_by_name.keys()))}",
                    })
                    continue

                tool_def = tools_by_name[tool_name]
                args = decision.args or {}

                try:
                    r = await call_server_tool(client, tool_def, args)

                    if r.status_code >= 400:
                        exec_log.append({
                            "action": "call_tool",
                            "tool_name": tool_name,
                            "args": args,
                            "error": {"status": r.status_code, "detail": r.text},
                        })
                    else:
                        try:
                            result = r.json()
                        except Exception:
                            result = {"raw": r.text}

                        exec_log.append({
                            "action": "call_tool",
                            "tool_name": tool_name,
                            "args": args,
                            "result": result,
                        })

                    print(f"{tool_name} -> {r.status_code}")
                except Exception as e:
                    exec_log.append({
                        "action": "call_tool",
                        "tool_name": tool_name,
                        "args": args,
                        "error": str(e),
                    })
                    continue

            else:
                text = "I hit the step limit for this request. Try splitting it into smaller parts."
                print(f"Assistant> {text}")
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=text))


if __name__ == "__main__":
    asyncio.run(main())


# TODO add tiny webhook for server side llm triggers (unknown phenomenon detection and handling)
