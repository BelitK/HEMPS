import asyncio
import json
from typing import Optional, List, Literal, Union, Set

import httpx
from pydantic import BaseModel, Field, ConfigDict, constr

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# =========================================================
# 1) Strict identifier type (prevents planning labels)
# =========================================================
AgentName = constr(pattern=r"^[a-z][a-z0-9_]{0,31}$")  # router, battery_agent, ev_charger


# =========================================================
# 2) Step schemas (extra forbidden prevents schema drift)
# =========================================================
class CreateAgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["create_agent"] = "create_agent"
    name: AgentName = Field(..., description="New agent name (snake_case)")
    persona: str = Field(..., min_length=10, max_length=240, description="1-2 sentence role")
    connect_to: Optional[List[AgentName]] = Field(default=None, description="Existing agent names to connect to")


class AddEdgeStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["add_edge"] = "add_edge"
    src: AgentName = Field(..., description="Existing agent name (source)")
    dst: AgentName = Field(..., description="Existing agent name (destination)")
    bidirectional: bool = Field(default=False, description="Add edges both ways if true")


class ReplyStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["reply"] = "reply"
    text: str = Field(..., description="Assistant reply to user")


Step = Union[CreateAgentStep, AddEdgeStep, ReplyStep]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: List[Step] = Field(default_factory=list, description="Ordered list of actions to execute")
    final_reply: Optional[str] = Field(default=None, description="Optional final message summarizing what happened")


# =========================================================
# 3) LLM setup
# =========================================================
llm = ChatOllama(model="mistral:7b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversational assistant that also manages a Mango multi-agent system.\n"
     "For each user message, output a multi-step Plan.\n\n"
     "CRITICAL RULES:\n"
     "- Agent names must be real topology agent names OR a new snake_case name like battery_agent.\n"
     "- NEVER use placeholder step labels as agent names (examples of forbidden names: create_5_new_agents, "
     "set_agent_properties, repeat_step_1).\n"
     "- connect_to/src/dst must reference existing agent names from the topology.\n"
     "- If an agent must be created before wiring, put create_agent steps first.\n"
     "- If the user is just chatting, return one reply step.\n"
     "Return only a Plan object."
    ),
    MessagesPlaceholder("history"),
    ("human", "User message: {user_input}\n\nCurrent topology JSON: {topology_json}\n")
])

plan_chain = prompt | llm.with_structured_output(Plan)


# =========================================================
# 4) Helpers
# =========================================================
def topo_names(topo: dict) -> Set[str]:
    return {n.get("name") for n in topo.get("nodes", []) if isinstance(n.get("name"), str)}


# =========================================================
# 5) Conversation loop with multi-step execution
# =========================================================
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

            topo = (await client.get("/topology")).json()

            if user_input == "/topo":
                print(json.dumps(topo, indent=2))
                continue

            plan: Plan = await plan_chain.ainvoke({
                "history": history,
                "user_input": user_input,
                "topology_json": topo,
            })

            executed_msgs: List[str] = []

            for step in plan.steps:
                # Always refresh topology so later steps see newly created agents
                topo = (await client.get("/topology")).json()
                names = topo_names(topo)

                if isinstance(step, ReplyStep):
                    executed_msgs.append(step.text)
                    continue

                if isinstance(step, CreateAgentStep):
                    # Ensure connect_to exists; default to router if available
                    connect_to = list(step.connect_to or [])
                    if not connect_to and "router" in names:
                        connect_to = ["router"]

                    # Filter to existing names only (extra safety)
                    connect_to = [x for x in connect_to if x in names]

                    payload = {
                        "name": step.name,
                        "persona": step.persona,
                        "connect_to": connect_to,
                    }
                    res = (await client.post("/agents", json=payload)).json()
                    executed_msgs.append(
                        f"Created agent {res.get('name', payload['name'])} connected to {res.get('connected_to', [])}."
                    )
                    continue

                if isinstance(step, AddEdgeStep):
                    # Hard gate: only add edges between existing agents
                    if step.src not in names or step.dst not in names:
                        executed_msgs.append(
                            f"Skipped edge {step.src}->{step.dst} (unknown agent name)."
                        )
                        continue

                    payload = {
                        "src": step.src,
                        "dst": step.dst,
                        "bidirectional": bool(step.bidirectional),
                    }
                    res = (await client.post("/edges", json=payload)).json()
                    executed_msgs.append(f"Added edges: {res.get('edges', [])}")
                    continue

            # Choose a final assistant message
            if plan.final_reply:
                assistant_text = plan.final_reply
            else:
                assistant_text = "\n".join(executed_msgs) if executed_msgs else "Okay."

            print(f"Assistant> {assistant_text}")

            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=assistant_text))


if __name__ == "__main__":
    asyncio.run(main())
