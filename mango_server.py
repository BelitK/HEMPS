import re
from typing import Any, Dict, List, Optional, Literal, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, constr

from mango import Agent, create_topology, activate, create_tcp_container
from agents.dynamic_agent import DynamicAgent

# Try importing Mango State enum for link activation
try:
    from mango.agent.core import State  # development branch path
except Exception:
    State = None  # Fallback: we will still track state in registry


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
# Registry to export topology
# -------------------------
class TopologyRegistry:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # edges store state: NORMAL | INACTIVE | BROKEN
        self.edges: List[Dict[str, Any]] = []
        self._edge_index: Dict[Tuple[str, str], int] = {}

    def export(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges)}

    def add_node(self, name: str, node_id: int, persona: str):
        self.nodes[name] = {"id": node_id, "name": name, "persona": persona}

    def upsert_edge(self, src: str, dst: str, state: str = "NORMAL") -> bool:
        """
        Insert edge if missing, otherwise update its state.
        Returns True if state changed or edge inserted.
        """
        key = (src, dst)
        if key in self._edge_index:
            idx = self._edge_index[key]
            old_state = self.edges[idx].get("state", "NORMAL")
            if old_state != state:
                self.edges[idx]["state"] = state
                return True
            return False

        self._edge_index[key] = len(self.edges)
        self.edges.append({"from": src, "to": dst, "state": state})
        return True

    def get_edge_state(self, src: str, dst: str) -> Optional[str]:
        key = (src, dst)
        if key not in self._edge_index:
            return None
        return self.edges[self._edge_index[key]].get("state", "NORMAL")


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
    edges: List[Dict[str, Any]]  # includes state


class EdgeStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    src: AgentName = Field(..., description="Existing agent name (source)")
    dst: AgentName = Field(..., description="Existing agent name (destination)")
    bidirectional: bool = Field(default=False, description="If true, apply to both directions")
    state: Literal["NORMAL", "INACTIVE", "BROKEN"] = Field(..., description="Edge state")


class EdgeStateResponse(BaseModel):
    ok: bool
    edges: List[Dict[str, Any]]


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


def _require_agent(name: str):
    if name not in agents_by_name:
        raise HTTPException(status_code=400, detail=f"unknown agent: {name}")


def _set_mango_edge_state(src_id: int, dst_id: int, state_str: str) -> None:
    """
    Best-effort: set edge state in Mango topology if supported.
    Supports both signatures:
      - set_edge_state(src_id, dst_id, state)
      - set_edge_state((src_id, dst_id), state)
    """
    if topology is None or not hasattr(topology, "set_edge_state"):
        return
    if State is None:
        return

    if state_str == "NORMAL":
        state = State.NORMAL
    elif state_str == "INACTIVE":
        state = State.INACTIVE
    else:
        state = State.BROKEN

    # Try signature: set_edge_state(src_id, dst_id, state)
    try:
        topology.set_edge_state(src_id, dst_id, state)
        return
    except TypeError:
        pass

    # Fallback signature: set_edge_state((src_id, dst_id), state)
    try:
        topology.set_edge_state((src_id, dst_id), state)
        return
    except TypeError:
        # If Mango topology doesn't support edge states, we still track it in registry
        return



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
    return {
        "tools": [
            {
                "name": "get_topology",
                "method": "GET",
                "path": "/topology",
                "description": "Get current agents and edges. Edges include a state: NORMAL, INACTIVE, BROKEN.",
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
                "description": "Add an edge between two existing agents. Optionally add reverse edge. Adds with state NORMAL.",
                "args_schema": {
                    "src": "existing agent name, required",
                    "dst": "existing agent name, required",
                    "bidirectional": "bool, optional (default false)",
                },
            },
            {
                "name": "deactivate_edge",
                "method": "POST",
                "path": "/edges/deactivate",
                "description": "Delete an edge by deactivating it (set state to INACTIVE). Optionally bidirectional.",
                "args_schema": {
                    "src": "existing agent name, required",
                    "dst": "existing agent name, required",
                    "bidirectional": "bool, optional (default false)",
                },
            },
            {
                "name": "activate_edge",
                "method": "POST",
                "path": "/edges/activate",
                "description": "Activate an existing edge (set state to NORMAL). Optionally bidirectional.",
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
        src_id = registry.nodes[name]["id"]
        dst_id = registry.nodes[target]["id"]
        topology.add_edge(src_id, dst_id)
        registry.upsert_edge(name, target, state="NORMAL")

    if hasattr(topology, "inject"):
        topology.inject()

    container.register(agent)

    return CreateAgentResponse(created=True, name=name, node_id=node_id, connected_to=connect_to)


@app.post("/edges", response_model=AddEdgeResponse)
async def add_edge(req: AddEdgeRequest):
    src = req.src
    dst = req.dst

    _require_agent(src)
    _require_agent(dst)

    src_id = registry.nodes[src]["id"]
    dst_id = registry.nodes[dst]["id"]

    changed_edges: List[Dict[str, Any]] = []

    # src -> dst
    topology.add_edge(src_id, dst_id)
    changed = registry.upsert_edge(src, dst, state="NORMAL")
    changed_edges.append({"from": src, "to": dst, "state": registry.get_edge_state(src, dst)})

    # reverse if needed
    if req.bidirectional and src != dst:
        topology.add_edge(dst_id, src_id)
        registry.upsert_edge(dst, src, state="NORMAL")
        changed_edges.append({"from": dst, "to": src, "state": registry.get_edge_state(dst, src)})

    if hasattr(topology, "inject"):
        topology.inject()

    return AddEdgeResponse(added=True, edges=changed_edges)


@app.post("/edges/deactivate", response_model=EdgeStateResponse)
async def deactivate_edge(req: AddEdgeRequest):
    src = req.src
    dst = req.dst

    _require_agent(src)
    _require_agent(dst)

    src_id = registry.nodes[src]["id"]
    dst_id = registry.nodes[dst]["id"]

    updated: List[Dict[str, Any]] = []

    # src -> dst inactive
    _set_mango_edge_state(src_id, dst_id, "INACTIVE")
    registry.upsert_edge(src, dst, state="INACTIVE")
    updated.append({"from": src, "to": dst, "state": "INACTIVE"})

    # optional reverse
    if req.bidirectional and src != dst:
        _set_mango_edge_state(dst_id, src_id, "INACTIVE")
        registry.upsert_edge(dst, src, state="INACTIVE")
        updated.append({"from": dst, "to": src, "state": "INACTIVE"})

    if hasattr(topology, "inject"):
        topology.inject()

    return EdgeStateResponse(ok=True, edges=updated)


@app.post("/edges/activate", response_model=EdgeStateResponse)
async def activate_edge(req: AddEdgeRequest):
    src = req.src
    dst = req.dst

    _require_agent(src)
    _require_agent(dst)

    src_id = registry.nodes[src]["id"]
    dst_id = registry.nodes[dst]["id"]

    updated: List[Dict[str, Any]] = []

    # src -> dst normal
    _set_mango_edge_state(src_id, dst_id, "NORMAL")
    registry.upsert_edge(src, dst, state="NORMAL")
    updated.append({"from": src, "to": dst, "state": "NORMAL"})

    # optional reverse
    if req.bidirectional and src != dst:
        _set_mango_edge_state(dst_id, src_id, "NORMAL")
        registry.upsert_edge(dst, src, state="NORMAL")
        updated.append({"from": dst, "to": src, "state": "NORMAL"})

    if hasattr(topology, "inject"):
        topology.inject()

    return EdgeStateResponse(ok=True, edges=updated)


# TODO add llm trigger for unknown phenomenon, after trigger llm will gather data from mesh network and devise a plan then execute to compansate for the situation
# TODO separation of functions for better modularity and testing also classes for better state management
# TODO 