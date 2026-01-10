import re
from typing import Any, Dict, List, Optional, Literal, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict, constr
from fastapi_mcp import FastApiMCP

from mango import Agent, create_topology, activate, create_tcp_container
from agents.CriticalMonitorAgent import CriticalMonitorAgent
from agents.dynamic_agent import DynamicAgent, IOAgent, BatteryAgent
from tools.check_tools import CheckTools
from tools.TopoRegistry import TopologyRegistry

# Try importing Mango State enum for link activation
try:
    from mango.agent.core import State  # development branch path
except Exception:
    State = None  # Fallback: we will still track state in registry


# TODO add forecasting for other agent types and add to startup
# TODO add agent removal API
# TODO change the usage from explanation to custom functionality for different agent types : right now in implementation phase


# -------------------------
# Guards
# -------------------------
AgentName = constr(pattern=r"^[a-z][a-z0-9_]{0,31}$")


# -------------------------
# API Schemas
# -------------------------
class CreateAgentRequest(BaseModel):
    # add agent type for different types
    model_config = ConfigDict(extra="forbid")
    name: AgentName = Field(..., description="Unique agent name (snake_case)")
    state: Literal["NORMAL", "INACTIVE", "BROKEN"] = Field(..., description="Agent state")
    usage: str = Field(..., min_length=10, max_length=240, description="1-2 sentence usage description")
    persona: str = Field(..., min_length=10, max_length=240, description="1-2 sentence role description")
    connect_to: Optional[List[AgentName]] = Field(default=None, description="Existing agent names to connect to")


class CreateAgentResponse(BaseModel):
    # add agent type to response model
    created: bool
    name: str
    state: str
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

class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dst: AgentName = Field(..., description="Existing agent name (destination)")
    content: str = Field(..., min_length=1, max_length=1000, description="Message content")


# -------------------------
# FastAPI app + runtime state
# -------------------------
app = FastAPI(title="Mango Runtime Server")

mcp = FastApiMCP(app, name="Mango Runtime MCP",
    describe_full_response_schema=True,  # Describe the full response JSON-schema instead of just a response example
    describe_all_responses=True,  # Describe all the possible responses instead of just the success (2XX) response)
)

# Mount the MCP server directly to your FastAPI app


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
    # creating topology registry
    router = DynamicAgent("router", "Routes messages and acts as the central hub.", "network router")
    router_id = topology.add_node(router)
    registry.add_node("router", router_id, router.persona)
    agents_by_name["router"] = router

    container.register(router)

    monitor = CriticalMonitorAgent(llm_trigger_url="http://127.0.0.1:9001/llm/trigger")
    monitor_id = topology.add_node(monitor)
    registry.add_node("critical_monitor", monitor_id, monitor.persona)
    agents_by_name["critical_monitor"] = monitor
    container.register(monitor)

    # optionally connect router -> monitor so it receives traffic
    topology.add_edge(registry.nodes["router"]["id"], monitor_id)
    registry.upsert_edge("router", "critical_monitor", state="NORMAL")

    if hasattr(topology, "inject"):
        topology.inject()
    # first test agent for development, gonna remove before presentation
    test_agent = IOAgent(
        name="test_agent",
        persona="A test IO agent for development purposes.",
    )
    test_agent_id = topology.add_node(test_agent)
    registry.add_node("test_agent", test_agent_id, test_agent.persona)
    agents_by_name["test_agent"] = test_agent

    container.register(test_agent)

    activation_manager = activate(container)
    print()
    await activation_manager.__aenter__()


@app.on_event("shutdown")
async def shutdown():
    global activation_manager, topology_ctx

    if activation_manager is not None:
        await activation_manager.__aexit__(None, None, None)

    if topology_ctx is not None:
        topology_ctx.__exit__(None, None, None)

@app.get("/agent_list")
async def get_agent_list():
    return list(Agent.__subclasses__())

@app.get("/topology")
async def get_topology():
    return registry.export()

@app.get("/agents")
async def get_agents():
    return list(agents_by_name.keys())


@app.post("/agents", response_model=CreateAgentResponse)
async def create_agent(req: CreateAgentRequest):
    # add agent type for different types
    name = req.name
    CheckTools.reject_bad_name(name)
    print(req)
    persona = req.persona.strip()
    state = req.state or "NORMAL"
    usage = req.usage or "Not specified."
    connect_to = req.connect_to or []

    if name in agents_by_name:
        name = CheckTools.unique_name(name, set(agents_by_name.keys()))
        CheckTools.reject_bad_name(name)

    missing = [t for t in connect_to if t not in agents_by_name]
    if missing:
        raise HTTPException(status_code=400, detail=f"unknown connect_to targets: {missing}")
    # make this a choice later
    agent = DynamicAgent(name, persona, usage)

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

    return CreateAgentResponse(created=True, name=name, node_id=node_id, connected_to=connect_to, state=state)


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


mcp.setup_server()
mcp.mount_http()
# keep this in 
# TODO add llm trigger for unknown phenomenon, after trigger llm will gather data from mesh network and devise a plan then execute to compansate for the situation
# TODO separation of functions for better modularity and testing also classes for better state management