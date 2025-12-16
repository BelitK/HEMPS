import asyncio
import json
from typing import Optional, List, Literal, Dict, Any

import httpx
from pydantic import BaseModel, Field, ConfigDict, constr

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


AgentName = constr(pattern=r"^[a-z][a-z0-9_]{0,31}$")


class OneAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["create_agent", "add_edge", "reply"]

    # create_agent
    name: Optional[AgentName] = None
    persona: Optional[str] = Field(default=None, min_length=10, max_length=240)
    connect_to: Optional[List[AgentName]] = None

    # add_edge
    src: Optional[AgentName] = None
    dst: Optional[AgentName] = None
    bidirectional: bool = False

    # reply
    text: Optional[str] = None


llm = ChatOllama(model="gemma3:12b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversational assistant that manages a Mango multi-agent system via HTTP tools.\n"
     "You must act in a loop: choose exactly one action each time.\n"
     "Available tools are provided as JSON.\n\n"
     "Rules:\n"
     "- Never output procedural names like create_agent_1 or connect_agent_to_router.\n"
     "- Agent names must be domain nouns like house_battery, pv_panels, ev_charger.\n"
     "- For add_edge, src and dst must be existing agent names from the live topology.\n"
     "- If a server error occurs, fix it in the next action.\n"
     "- When all requested changes are done, return action=reply with a short summary.\n"
     "Return only a OneAction object."
    ),
    MessagesPlaceholder("history"),
    ("human",
     "User message:\n{user_input}\n\n"
     "Tools JSON:\n{tools_json}\n\n"
     "Live topology JSON:\n{topology_json}\n\n"
     "This-turn execution log (most recent last):\n{exec_log}\n")
])

action_chain = prompt | llm.with_structured_output(OneAction)


MAX_STEPS = 20


async def main():
    print("Tool-loop chat started. Commands: /exit, /topo")
    history = []

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30) as client:
        tools = (await client.get("/tools")).json()

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

            # loop until reply
            for _ in range(MAX_STEPS):
                topo = (await client.get("/topology")).json()

                act: OneAction = await action_chain.ainvoke({
                    "history": history,
                    "user_input": user_input,
                    "tools_json": tools,
                    "topology_json": topo,
                    "exec_log": exec_log,
                })

                if act.action == "reply":
                    text = act.text or "Done."
                    print(f"Assistant> {text}")
                    history.append(HumanMessage(content=user_input))
                    history.append(AIMessage(content=text))
                    break

                if act.action == "create_agent":
                    payload = {
                        "name": act.name,
                        "persona": act.persona,
                        "connect_to": act.connect_to or [],
                    }
                    try:
                        r = await client.post("/agents", json=payload)
                        if r.status_code >= 400:
                            exec_log.append({
                                "action": "create_agent",
                                "payload": payload,
                                "error": {"status": r.status_code, "detail": r.text},
                            })
                        else:
                            exec_log.append({
                                "action": "create_agent",
                                "payload": payload,
                                "result": r.json(),
                            })
                    except Exception as e:
                        exec_log.append({"action": "create_agent", "payload": payload, "error": str(e)})
                    continue

                if act.action == "add_edge":
                    payload = {
                        "src": act.src,
                        "dst": act.dst,
                        "bidirectional": bool(act.bidirectional),
                    }
                    try:
                        r = await client.post("/edges", json=payload)
                        if r.status_code >= 400:
                            exec_log.append({
                                "action": "add_edge",
                                "payload": payload,
                                "error": {"status": r.status_code, "detail": r.text},
                            })
                        else:
                            exec_log.append({
                                "action": "add_edge",
                                "payload": payload,
                                "result": r.json(),
                            })
                        print(r.status_code, r.text)
                    except Exception as e:
                        exec_log.append({"action": "add_edge", "payload": payload, "error": str(e)})
                    continue
                print(f"{act.action}")
            else:
                # max steps exceeded
                text = "I hit the step limit for this request. Try splitting it into smaller parts."
                print(f"Assistant> {text}")
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=text))

                


if __name__ == "__main__":
    asyncio.run(main())
