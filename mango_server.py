import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mango import Agent, create_topology, activate, create_tcp_container


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
# Registry to export topology for the LLM
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


def normalize_name(raw: str) -> str:
    raw = raw.strip().lower()
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        raw = "agent"
    if not raw[0].isalpha():
        raw = f"a_{raw}"
    return raw


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
    name: str = Field(..., description="Unique agent name")
    persona: str = Field(..., description="1-2 sentence role description")
    connect_to: Optional[List[str]] = Field(default=None, description="Existing agent names to connect to")


class CreateAgentResponse(BaseModel):
    created: bool
    name: str
    node_id: int
    connected_to: List[str]


class AddEdgeRequest(BaseModel):
    src: str = Field(..., description="Existing agent name (source)")
    dst: str = Field(..., description="Existing agent name (destination)")
    bidirectional: bool = Field(default=False, description="If true, add edges both ways")


class AddEdgeResponse(BaseModel):
    added: bool
    edges: List[Dict[str, str]]


# -------------------------
# FastAPI App + Mango Runtime State
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

    # Avoid bind conflicts by letting OS pick a free port for Mango TCP container
    container = create_tcp_container(("127.0.0.1", 0))

    topology_ctx = create_topology()
    topology = topology_ctx.__enter__()

    # Seed agent(s)
    router = DynamicAgent("router", "Routes messages and acts as the central hub.")
    router_id = topology.add_node(router)
    registry.add_node("router", router_id, router.persona)
    agents_by_name["router"] = router

    # Register before activation
    container.register(router)

    # Activate once for server lifetime (no container.start() call)
    activation_manager = activate(container)
    await activation_manager.__aenter__()


@app.on_event("shutdown")
async def shutdown():
    global activation_manager, topology_ctx

    if activation_manager is not None:
        await activation_manager.__aexit__(None, None, None)

    if topology_ctx is not None:
        topology_ctx.__exit__(None, None, None)


@app.get("/topology")
async def get_topology():
    return registry.export()


@app.post("/agents", response_model=CreateAgentResponse)
async def create_agent(req: CreateAgentRequest):
    base = normalize_name(req.name)
    name = unique_name(base, set(agents_by_name.keys()))

    persona = (req.persona or "").strip()
    if len(persona) < 10:
        raise HTTPException(status_code=400, detail="persona too short (min 10 chars)")

    connect_to = req.connect_to or []
    connect_to = [normalize_name(x) for x in connect_to]

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

    # Container is active; registering is enough
    container.register(agent)

    return CreateAgentResponse(created=True, name=name, node_id=node_id, connected_to=connect_to)


@app.post("/edges", response_model=AddEdgeResponse)
async def add_edge(req: AddEdgeRequest):
    src = normalize_name(req.src)
    dst = normalize_name(req.dst)

    if src not in agents_by_name:
        raise HTTPException(status_code=400, detail=f"unknown src agent: {src}")
    if dst not in agents_by_name:
        raise HTTPException(status_code=400, detail=f"unknown dst agent: {dst}")

    src_id = registry.nodes[src]["id"]
    dst_id = registry.nodes[dst]["id"]

    added_edges: List[Dict[str, str]] = []

    # Add src -> dst
    topology.add_edge(src_id, dst_id)
    registry.add_edge(src, dst)
    added_edges.append({"from": src, "to": dst})

    # Optional reverse edge
    if req.bidirectional and src != dst:
        topology.add_edge(dst_id, src_id)
        registry.add_edge(dst, src)
        added_edges.append({"from": dst, "to": src})

    if hasattr(topology, "inject"):
        topology.inject()

    return AddEdgeResponse(added=True, edges=added_edges)
