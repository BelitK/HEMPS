import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, constr

from mango import Agent, create_topology, activate, create_tcp_container


# -------------------------
# Guards
# -------------------------
AgentName = constr(pattern=r"^[a-z][a-z0-9_]{0,31}$")

FORBIDDEN_NAME_PARTS = [
    "create_agent",
    "connect_agent",
    "set_agent",
    "repeat_step",
    "step_",
    "plan",
    "task",
    "do_",
]


def reject_bad_name(name: str) -> None:
    lowered = name.lower()
    for bad in FORBIDDEN_NAME_PARTS:
        if bad in lowered:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid agent name '{name}'. Use a domain noun like house_battery, pv_panels, ev_charger."
                ),
            )


# -------------------------
# Mango Agent
# -------------------------
class DynamicAgent(Agent):
    def __init__(self, name: str, persona: str):
        super().__init__()
        self.name = name
        self.persona = persona

    def handle_message(self, content, meta):
        print(f"[{self.name}] {content}")


# -------------------------
# Registry to export topology
# -------------------------
class TopologyRegistry:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, str]] = []

    def export(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges)}

    def add_node(self, name: str, node_id: int, persona: str):
        self.nodes[name] = {"id": node_id, "name": name, "persona": persona}

    def add_edge(self, src: str, dst: str):
        self.edges.append({"from": src, "to": dst})


def unique_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# -------------------------
# API Schemas
# -------------------------
class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: AgentName = Field(..., description="Unique agent name (snake_case)")
    persona: str = Field(..., min_length=10, max_length=240, description="1-2 sentence role description")
    connect_to: Optional[List[AgentName]] = Field(default=None, description="Existing agent names to connect to")


class CreateAgentResponse(BaseModel):
    created: bool
    name: str
    node_id: int
    connected_to: List[str]


class AddEdgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    src: AgentName = Field(..., description="Existing agent name (source)")
    dst: AgentName = Field(..., description="Existing agent name (destination)")
    bidirectional: bool = Field(default=False, description="If true, add reverse edge too")


class AddEdgeResponse(BaseModel):
    added: bool
    edges: List[Dict[str, str]]


# -------------------------
# FastAPI app + runtime state
# -------------------------
app = FastAPI(title="Mango Runtime Server")

registry = TopologyRegistry()
agents_by_name: Dict[str, DynamicAgent] = {}

container = None
topology_ctx = None
topology = None
activation_manager = None


@app.on_event("startup")
async def startup():
    global container, topology_ctx, topology, activation_manager

    container = create_tcp_container(("127.0.0.1", 0))

    topology_ctx = create_topology()
    topology = topology_ctx.__enter__()

    router = DynamicAgent("router", "Routes messages and acts as the central hub.")
    router_id = topology.add_node(router)
    registry.add_node("router", router_id, router.persona)
    agents_by_name["router"] = router

    container.register(router)

    activation_manager = activate(container)
    await activation_manager.__aenter__()


@app.on_event("shutdown")
async def shutdown():
    global activation_manager, topology_ctx

    if activation_manager is not None:
        await activation_manager.__aexit__(None, None, None)

    if topology_ctx is not None:
        topology_ctx.__exit__(None, None, None)


@app.get("/tools")
async def get_tools():
    # Simple, stable "tool catalog" for the model
    return {
        "tools": [
            {
                "name": "get_topology",
                "method": "GET",
                "path": "/topology",
                "description": "Get current agents and edges.",
                "args_schema": {},
            },
            {
                "name": "create_agent",
                "method": "POST",
                "path": "/agents",
                "description": "Create a new agent and optionally connect it to existing agents.",
                "args_schema": {
                    "name": "snake_case string, required",
                    "persona": "string (10-240 chars), required",
                    "connect_to": "list of existing agent names, optional",
                },
                "name_rules": [
                    "Use a domain noun like house_battery, pv_panels, ev_charger",
                    "Do not use procedural names like create_agent_1, connect_agent_to_router",
                ],
            },
            {
                "name": "add_edge",
                "method": "POST",
                "path": "/edges",
                "description": "Add an edge between two existing agents. Optionally add reverse edge.",
                "args_schema": {
                    "src": "existing agent name, required",
                    "dst": "existing agent name, required",
                    "bidirectional": "bool, optional (default false)",
                },
            },
        ]
    }


@app.get("/topology")
async def get_topology():
    return registry.export()


@app.post("/agents", response_model=CreateAgentResponse)
async def create_agent(req: CreateAgentRequest):
    name = req.name
    reject_bad_name(name)

    persona = req.persona.strip()
    connect_to = req.connect_to or []

    if name in agents_by_name:
        # allow idempotent-ish behavior by uniquifying instead of hard failing
        name = unique_name(name, set(agents_by_name.keys()))
        reject_bad_name(name)

    missing = [t for t in connect_to if t not in agents_by_name]
    if missing:
        raise HTTPException(status_code=400, detail=f"unknown connect_to targets: {missing}")

    agent = DynamicAgent(name, persona)

    node_id = topology.add_node(agent)
    registry.add_node(name, node_id, persona)
    agents_by_name[name] = agent

    for target in connect_to:
        topology.add_edge(registry.nodes[name]["id"], registry.nodes[target]["id"])
        registry.add_edge(name, target)

    if hasattr(topology, "inject"):
        topology.inject()

    container.register(agent)

    return CreateAgentResponse(created=True, name=name, node_id=node_id, connected_to=connect_to)


@app.post("/edges", response_model=AddEdgeResponse)
async def add_edge(req: AddEdgeRequest):
    src = req.src
    dst = req.dst

    if src not in agents_by_name:
        raise HTTPException(status_code=400, detail=f"unknown src agent: {src}")
    if dst not in agents_by_name:
        raise HTTPException(status_code=400, detail=f"unknown dst agent: {dst}")

    src_id = registry.nodes[src]["id"]
    dst_id = registry.nodes[dst]["id"]

    added_edges: List[Dict[str, str]] = []

    topology.add_edge(src_id, dst_id)
    registry.add_edge(src, dst)
    added_edges.append({"from": src, "to": dst})

    if req.bidirectional and src != dst:
        topology.add_edge(dst_id, src_id)
        registry.add_edge(dst, src)
        added_edges.append({"from": dst, "to": src})

    if hasattr(topology, "inject"):
        topology.inject()

    return AddEdgeResponse(added=True, edges=added_edges)
