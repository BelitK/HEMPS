import re
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, constr
from fastapi_mcp import FastApiMCP

from mango import Agent, create_topology, activate, create_tcp_container

from agents.CriticalMonitorAgent import CriticalMonitorAgent
from agents.dynamic_agent import DynamicAgent, IOAgent
from agents.agent_catalog import generate_agent_catalog

from tools.check_tools import CheckTools
from tools.TopoRegistry import TopologyRegistry


# Try importing Mango State enum for link activation
try:
    from mango.agent.core import State
except Exception:
    State = None


# -------------------------
# Guards
# -------------------------
AgentName = constr(pattern=r"^[a-z][a-z0-9_]{0,31}$")


# -------------------------
# Agent type map (auto)
# -------------------------
def _iter_all_subclasses(cls):
    out = []
    stack = list(cls.__subclasses__())
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(c.__subclasses__())
    return out


def _build_agent_class_map() -> Dict[str, type]:
    m: Dict[str, type] = {}
    for cls in _iter_all_subclasses(DynamicAgent):
        t = getattr(cls, "TYPE", None)
        if not t:
            continue
        t = str(t).strip().lower()
        if not t or t == "dynamic":
            continue
        m[t] = cls
    return m


AGENT_CLASS_MAP = _build_agent_class_map()


# -------------------------
# API Schemas
# -------------------------
class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: AgentName
    agent_type: str
    state: Literal["NORMAL", "INACTIVE", "BROKEN"]

    persona: Optional[str] = Field(default=None, max_length=240)
    usage: Optional[str] = Field(default=None, max_length=240)

    connect_to: Optional[List[AgentName]] = None


class CreateAgentResponse(BaseModel):
    created: bool
    name: str
    node_id: int
    state: str
    agent_type: str
    connected_to: List[str]


class AddEdgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    src: AgentName
    dst: AgentName
    bidirectional: bool = False


class AddEdgeResponse(BaseModel):
    added: bool
    edges: List[Dict[str, Any]]


class EdgeStateResponse(BaseModel):
    ok: bool
    edges: List[Dict[str, Any]]


# -------------------------
# FastAPI app + runtime state
# -------------------------
app = FastAPI(title="Mango Runtime Server")

mcp = FastApiMCP(
    app,
    name="Mango Runtime MCP",
    describe_full_response_schema=True,
    describe_all_responses=True,
)

registry = TopologyRegistry()

# NOTE: not all agents are DynamicAgent
agents_by_name: Dict[str, Agent] = {}

container = None
topology_ctx = None
topology = None
activation_manager = None


# -------------------------
# Helpers
# -------------------------
def _require_agent(name: str):
    if name not in agents_by_name:
        raise HTTPException(status_code=400, detail=f"unknown agent: {name}")


def _set_mango_edge_state(src_id: int, dst_id: int, state_str: str) -> None:
    if topology is None or not hasattr(topology, "set_edge_state") or State is None:
        return

    state = (
        State.NORMAL if state_str == "NORMAL"
        else State.INACTIVE if state_str == "INACTIVE"
        else State.BROKEN
    )

    try:
        topology.set_edge_state(src_id, dst_id, state)
    except TypeError:
        try:
            topology.set_edge_state((src_id, dst_id), state)
        except TypeError:
            pass


# -------------------------
# Lifecycle
# -------------------------
@app.on_event("startup")
async def startup():
    global container, topology_ctx, topology, activation_manager

    container = create_tcp_container(("127.0.0.1", 0))
    topology_ctx = create_topology()
    topology = topology_ctx.__enter__()

    # Router
    router = DynamicAgent(
        name="router",
        persona="Routes messages and acts as the central hub.",
        usage="network router",
    )
    router_id = topology.add_node(router)
    registry.add_node("router", router_id, router)
    agents_by_name["router"] = router
    container.register(router)

    # Critical monitor
    monitor = CriticalMonitorAgent(llm_trigger_url="http://127.0.0.1:9001/llm/trigger")
    monitor_id = topology.add_node(monitor)
    registry.add_node("critical_monitor", monitor_id, monitor)
    agents_by_name["critical_monitor"] = monitor
    container.register(monitor)

    # Connect router -> monitor
    topology.add_edge(registry.nodes["router"]["id"], monitor_id)
    registry.upsert_edge("router", "critical_monitor", state="NORMAL")

    # Test IO agent
    test_agent = IOAgent(
        name="test_agent",
        persona="A test IO agent for development purposes.",
    )
    test_agent_id = topology.add_node(test_agent)
    registry.add_node("test_agent", test_agent_id, test_agent)
    agents_by_name["test_agent"] = test_agent
    container.register(test_agent)

    if hasattr(topology, "inject"):
        topology.inject()

    activation_manager = activate(container)
    await activation_manager.__aenter__()


@app.on_event("shutdown")
async def shutdown():
    if activation_manager:
        await activation_manager.__aexit__(None, None, None)
    if topology_ctx:
        topology_ctx.__exit__(None, None, None)


# -------------------------
# Routes
# -------------------------
@app.get("/agent_catalog")
async def agent_catalog():
    return generate_agent_catalog()


@app.get("/topology")
async def get_topology():
    return registry.export()


@app.get("/agents")
async def get_agents():
    return list(agents_by_name.keys())


@app.post("/agents", response_model=CreateAgentResponse)
async def create_agent(req: CreateAgentRequest):
    name = req.name
    CheckTools.reject_bad_name(name)

    agent_type = req.agent_type.strip().lower()
    agent_cls = AGENT_CLASS_MAP.get(agent_type)
    if not agent_cls:
        raise HTTPException(status_code=400, detail=f"unknown agent_type: {agent_type}")

    if name in agents_by_name:
        name = CheckTools.unique_name(name, set(agents_by_name.keys()))
        CheckTools.reject_bad_name(name)

    connect_to = req.connect_to or []
    missing = [n for n in connect_to if n not in agents_by_name]
    if missing:
        raise HTTPException(status_code=400, detail=f"unknown connect_to targets: {missing}")

    persona = req.persona.strip() if req.persona else None
    usage = req.usage.strip() if req.usage else None

    agent = agent_cls(name=name, persona=persona, usage=usage)

    node_id = topology.add_node(agent)
    registry.add_node(name, node_id, agent)
    agents_by_name[name] = agent

    for target in connect_to:
        topology.add_edge(registry.nodes[name]["id"], registry.nodes[target]["id"])
        registry.upsert_edge(name, target, state="NORMAL")

    if hasattr(topology, "inject"):
        topology.inject()

    container.register(agent)

    return CreateAgentResponse(
        created=True,
        name=name,
        node_id=node_id,
        state=req.state,
        agent_type=agent_type,
        connected_to=connect_to,
    )


@app.post("/edges", response_model=AddEdgeResponse)
async def add_edge(req: AddEdgeRequest):
    _require_agent(req.src)
    _require_agent(req.dst)

    src_id = registry.nodes[req.src]["id"]
    dst_id = registry.nodes[req.dst]["id"]

    edges = []

    topology.add_edge(src_id, dst_id)
    registry.upsert_edge(req.src, req.dst, "NORMAL")
    edges.append({"from": req.src, "to": req.dst, "state": "NORMAL"})

    if req.bidirectional and req.src != req.dst:
        topology.add_edge(dst_id, src_id)
        registry.upsert_edge(req.dst, req.src, "NORMAL")
        edges.append({"from": req.dst, "to": req.src, "state": "NORMAL"})

    if hasattr(topology, "inject"):
        topology.inject()

    return AddEdgeResponse(added=True, edges=edges)


@app.post("/edges/deactivate", response_model=EdgeStateResponse)
async def deactivate_edge(req: AddEdgeRequest):
    _require_agent(req.src)
    _require_agent(req.dst)

    src_id = registry.nodes[req.src]["id"]
    dst_id = registry.nodes[req.dst]["id"]

    _set_mango_edge_state(src_id, dst_id, "INACTIVE")
    registry.upsert_edge(req.src, req.dst, "INACTIVE")

    return EdgeStateResponse(ok=True, edges=[{"from": req.src, "to": req.dst, "state": "INACTIVE"}])


@app.post("/edges/activate", response_model=EdgeStateResponse)
async def activate_edge(req: AddEdgeRequest):
    _require_agent(req.src)
    _require_agent(req.dst)

    src_id = registry.nodes[req.src]["id"]
    dst_id = registry.nodes[req.dst]["id"]

    _set_mango_edge_state(src_id, dst_id, "NORMAL")
    registry.upsert_edge(req.src, req.dst, "NORMAL")

    return EdgeStateResponse(ok=True, edges=[{"from": req.src, "to": req.dst, "state": "NORMAL"}])


mcp.setup_server()
mcp.mount_http()
