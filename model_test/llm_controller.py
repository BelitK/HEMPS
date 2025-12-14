import asyncio
import json
from typing import Optional, List, Literal

import httpx
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# -------------------------
# LLM "decision" schema
# -------------------------
class Decision(BaseModel):
    action: Literal["create_agent", "reply", "do_nothing"]
    reply: Optional[str] = None
    name: Optional[str] = None
    persona: Optional[str] = None
    connect_to: Optional[List[str]] = None


# -------------------------
# LLM setup
# -------------------------
llm = ChatOllama(model="mistral:7b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversational assistant that also manages a Mango multi-agent system.\n"
     "You can chat normally.\n"
     "If the user asks to add something as an agent, choose action=create_agent.\n"
     "If the user is just talking or asking questions, choose action=reply.\n"
     "Always keep replies concise.\n"
     "When creating an agent:\n"
     "- name should be short and snake_case\n"
     "- persona should be 1-2 sentences\n"
     "- connect_to must use existing agent names from the topology\n"),
    MessagesPlaceholder("history"),
    ("human", "User message: {user_input}\n\nCurrent topology JSON: {topology_json}\n")
])

chain = prompt | llm.with_structured_output(Decision)


# -------------------------
# Conversational loop
# -------------------------
async def main():
    print("Chat started. Commands: /exit, /topo")
    history = []

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30) as client:
        while True:
            user_input = input("\nYou> ").strip()
            if not user_input:
                continue
            if user_input == "/exit":
                break

            # Always fetch topology so model has live view
            topo = (await client.get("/topology")).json()

            if user_input == "/topo":
                print(json.dumps(topo, indent=2))
                continue

            # Ask LLM what to do, with chat history
            decision: Decision = await chain.ainvoke({
                "history": history,
                "user_input": user_input,
                "topology_json": topo,
            })

            if decision.action == "create_agent":
                payload = {
                    "name": decision.name or "new_agent",
                    "persona": decision.persona or "A helpful agent.",
                    "connect_to": decision.connect_to or (["router"] if any(n["name"] == "router" for n in topo["nodes"]) else []),
                }

                created = (await client.post("/agents", json=payload)).json()

                msg = f"Created agent **{created['name']}** and connected to {created.get('connected_to', [])}."
                print(f"Assistant> {msg}")

                # Update history (so user can refer to it)
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=msg))
                continue

            if decision.action == "reply":
                reply_text = decision.reply or "Okay."
                print(f"Assistant> {reply_text}")

                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=reply_text))
                continue

            # do_nothing fallback
            print("Assistant> (No action taken.)")
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content="(No action taken.)"))


if __name__ == "__main__":
    asyncio.run(main())
