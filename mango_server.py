from encodings.cp932 import codec
import re
from typing import Any, Dict, List, Optional, Literal, Annotated


import time
from typing import Callable

import tools.scheduler as scheduler
from tools.scheduler import InMemoryScheduler, ScheduleItem  # adjust path/module name



from contextlib import asynccontextmanager
from fastapi_mcp import FastApiMCP
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, constr, StringConstraints

from mango import Agent, create_topology, activate, create_tcp_container

from agents.CriticalMonitorAgent import CriticalMonitorAgent
from agents.dynamic_agent import DynamicAgent
from agents.test_agent import Router_Agent
from agents.io_agent import IOAgent
from agents.agent_catalog import generate_agent_catalog
from agents.message import Message, MessageLevel

from tools.check_tools import CheckTools
from tools.TopoRegistry import TopologyRegistry


# Try importing Mango State enum for link activation
try:
    from mango.agent.core import State
    from mango import JSON
except Exception:
    State = None


# -------------------------
# Scheduler
# -------------------------


scheduler = InMemoryScheduler()

async def run_agent_action(agent_name: str, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Default action runner:
      1) find agent by name
      2) call agent.<action>(**payload) if exists
      3) else call agent.run_action(action, payload) if exists
    """
    _require_agent(agent_name)
    agent = agents_by_name[agent_name]

    # 1) direct method call: agent.<action>(**payload)
    fn = getattr(agent, action, None)
    if callable(fn):
        try:
            res = fn(**(payload or {}))
            # may be coroutine or normal value
            if hasattr(res, "__await__"):
                res = await res
            return res if isinstance(res, dict) else {"result": res}
        except TypeError as e:
            # common: wrong kwargs
            raise RuntimeError(f"action '{action}' called with invalid payload: {e}") from e

    # 2) generic handler: agent.run_action(action, payload)
    fn2 = getattr(agent, "run_action", None)
    if callable(fn2):
        res = fn2(action, payload or {})
        if hasattr(res, "__await__"):
            res = await res
        return res if isinstance(res, dict) else {"result": res}

    raise RuntimeError(f"agent '{agent_name}' has no action '{action}' and no run_action handler")



# -------------------------
# Guards
# -------------------------

AgentName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9_]{0,31}$",
        min_length=1,
        max_length=32,
        strip_whitespace=True,
    ),
]

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
AGENTS = sorted(list(AGENT_CLASS_MAP.keys()))

# -------------------------
# API Schemas
# -------------------------
class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: AgentName
    agent_type: Literal[*AGENTS] = Field(default="stock", description="Type of agent to create. Agent type must be selected from the catalog-derived set (fixed at startup).\n"
                            "Choose exactly one. Do not invent new values choose from the agent catalog.\n"
                            "Don't use 'dynamic' as agent_type.\n"
                            "There are no types such as dynamic or base or generic so don't make up new types.")

    state: Literal["NORMAL", "INACTIVE", "BROKEN"] = Field(default="NORMAL",
                                                               description=(
        "Operational state of the agent. "
        "Must be exactly one of: NORMAL, INACTIVE, BROKEN. "
        "NORMAL = agent is active and functioning. "
        "INACTIVE = agent exists but should not run. "
        "BROKEN = agent is faulty and should not be used. "
        "No other values are allowed."
    ))

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

class ScheduleCreateRequest(BaseModel):
    created_by: str = Field(..., min_length=1)
    run_at_epoch: float = Field(..., description="Unix epoch seconds")
    agent_name: AgentName
    action: str = Field(..., min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)

class ScheduleResponse(BaseModel):
    id: str
    created_at: float
    created_by: str
    status: str
    run_at: float
    agent_name: str
    action: str
    payload: Dict[str, Any]
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_item(x: ScheduleItem) -> "ScheduleResponse":
        return ScheduleResponse(
            id=x.id,
            created_at=x.created_at,
            created_by=x.created_by,
            status=x.status,
            run_at=x.run_at,
            agent_name=x.agent_name,
            action=x.action,
            payload=x.payload,
            error=x.error,
            result=x.result,
        )



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
    global container, topology_ctx, topology, activation_managerv
    codec = JSON()
    codec.add_serializer(*Message.__serializer__())

    container = create_tcp_container(("127.0.0.1", 0),codec=codec)
    topology_ctx = create_topology()
    topology = topology_ctx.__enter__()

    # Router
    router = Router_Agent(
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
        name="io_agent",
        persona="A test IO agent for development purposes.",
    )
    test_agent_id = topology.add_node(test_agent)
    registry.add_node("io_agent", test_agent_id, test_agent)
    agents_by_name["io_agent"] = test_agent
    container.register(test_agent)
    registry.upsert_edge("io_agent", "router",  state="NORMAL")
    registry.upsert_edge("router", "io_agent",  state="NORMAL")

    if hasattr(topology, "inject"):
        topology.inject()

    activation_manager = activate(container)
    await activation_manager.__aenter__()

    scheduler.set_action_runner(run_agent_action)
    await scheduler.start(poll_s=0.5)


@app.on_event("shutdown")
async def shutdown():
    await scheduler.stop()
    if activation_manager:
        await activation_manager.__aexit__(None, None, None)
    if topology_ctx:
        topology_ctx.__exit__(None, None, None)



# -------------------------
# FastAPI app + runtime state
# -------------------------

# -------------------------
# Routes
# -------------------------
@app.get("/agent_catalog")
async def agent_catalog():
    # Return the generated catalog but only include types that the server supports
    catalog = generate_agent_catalog()
    allowed = set(AGENT_CLASS_MAP.keys())
    types = [a for a in catalog.get("agent_types", []) if a.get("type") in allowed]
    catalog["agent_types"] = types
    return catalog


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


@app.post("/schedules", response_model=ScheduleResponse)
async def create_schedule(req: ScheduleCreateRequest):
    _require_agent(req.agent_name)
    ## time grace for when model calls this tool
    GRACE_T= 600000

    # optional: reject scheduling too far in the past
    if req.run_at_epoch < time.time() - GRACE_T:
        raise HTTPException(status_code=400, detail="run_at_epoch is in the past")

    item = await scheduler.create(
        created_by=req.created_by,
        run_at_epoch=req.run_at_epoch,
        agent_name=req.agent_name,
        action=req.action,
        payload=req.payload,
    )
    return ScheduleResponse.from_item(item)


@app.get("/schedules", response_model=List[ScheduleResponse])
async def list_schedules(status: Optional[str] = None):
    items = await scheduler.list(status=status)
    return [ScheduleResponse.from_item(x) for x in items]


@app.get("/schedules/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(schedule_id: str):
    item = await scheduler.get(schedule_id)
    if not item:
        raise HTTPException(status_code=404, detail="schedule not found")
    return ScheduleResponse.from_item(item)


@app.post("/schedules/{schedule_id}/cancel")
async def cancel_schedule(schedule_id: str):
    ok = await scheduler.cancel(schedule_id)
    if not ok:
        raise HTTPException(status_code=400, detail="cannot cancel (not found or already finished)")
    return {"cancelled": True, "id": schedule_id}

@app.get("/io/status")
async def io_status():
    io_agent = agents_by_name.get("io_agent")
    if not io_agent or not isinstance(io_agent, IOAgent):
        raise HTTPException(status_code=500, detail="IOAgent 'io_agent' not found")

    info = io_agent.get_aggregated_info()
    return info

@app.post("/io/send_message")
async def send_message_to_io_agent(req: dict):
    io_agent = agents_by_name.get("io_agent")
    if not io_agent or not isinstance(io_agent, IOAgent):
        raise HTTPException(status_code=500, detail="IOAgent 'io_agent' not found")

    # Forward the message to the IO agent
    await container.send_message(content=req["content"], meta=req["meta"], receiver_addr=io_agent.addr)
    return {"status": "message sent to IO agent"}

class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_name: AgentName
    content: str
    meta: str


@app.post("/send_message")
async def send_message_to_agent(req: SendMessageRequest):
    agent_name = req.agent_name
    content = req.content
    meta = req.meta

    agent = agents_by_name.get(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    await agents_by_name.get("io_agent").send_message(content=content, meta=meta, receiver_addr=agent.addr)
    return {"status": f"message sent to agent '{agent_name}'"}

@app.get('/trigger_emergency')
async def trigger_emergency():
    critical_agent = agents_by_name.get("critical_monitor")
    if not critical_agent or not isinstance(critical_agent, CriticalMonitorAgent):
        raise HTTPException(status_code=500, detail="CriticalMonitorAgent 'critical_monitor' not found")

    # Simulate an emergency condition
    emergency_message = {
        "type": "emergency",
        "details": "Simulated critical condition triggered."
    }
    critical_agent.handle_message(content=emergency_message, meta={"source": "test"})

    return {"status": "emergency condition triggered in Critical agent"}

mcp.setup_server()
mcp.mount_http()


# @app.get("/io/forecast")
# async def io_forecast():
#     io_agent = agents_by_name.get("io_agent")
#     if not io_agent or not isinstance(io_agent, IOAgent):
#         raise HTTPException(status_code=500, detail="io_agent 'test_agent' not found")

#     info = io_agent.to_llm_format()
#     return info




## add function for agent message sending, querying, etc.
## update topology function to include agent states and more details
## add function to interact with io agent functions
## add io agent to startup sequence

