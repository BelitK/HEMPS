import asyncio
import json
import logging
import os
import re
import time
import uuid
import hashlib
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


# -------------------------
# Logging
# -------------------------
LOG_LEVEL = os.getenv("LLM_LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LLM_LOG_FILE", "logs/llm_service.log")

logger = logging.getLogger("llm_service")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False  # prevent duplicate logs if uvicorn also configures root logger

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.addHandler(_console)

_file = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
_file.setFormatter(_fmt)
_file.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.addHandler(_file)


def _preview(text: Any, n: int = 400) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")


def _tool_content_to_text(content: Any) -> str:
    """
    MCP ToolMessage.content can be:
      - str
      - list[dict] like [{"type":"text","text":"...","id":"..."}]
      - dict
      - None
    Convert to a readable string for logging / trace.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
                continue
            parts.append(str(item))
        return "\n".join([p for p in parts if p])

    if isinstance(content, dict):
        txt = content.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt
        return json.dumps(content, ensure_ascii=False, default=str)

    return str(content)


# -------------------------
# Config
# -------------------------
SERVER_BASE_URL = "http://127.0.0.1:8000"
MCP_URL = "http://127.0.0.1:8000/mcp"

MAX_STEPS = 200

EXEC_STEP_RECURSION_LIMIT = 45
MAX_TOOL_CALLS_PER_RUN = 40
MAX_PLAN_STEPS = 20

# Retry knobs
MAX_AGENT_INVOKE_RETRIES = 2          # retries for agent.ainvoke exceptions
MAX_STEP_RETRIES_TOTAL = 2            # retries per plan step (includes recovery attempts)
MAX_FINALIZER_REPAIR_PASSES = 2       # parse repair passes for final JSON
MAX_PLANNER_REPAIR_PASSES = 2         # parse repair passes for planner JSON
RUN_JOB_RETRIES_ON_EXCEPTION = 2      # retries for engine.run_once errors in /trigger job
RETRY_BACKOFF_S = 0.6                 # base backoff, grows per attempt

SYSTEM_INSTRUCTIONS = (
"""You are an autonomous controller for a live, distributed house energy management system.

SYSTEM ROLE
You operate a Mango-based multi-agent system exposed via MCP endpoints.
You have full authority to observe, create, modify, connect, control, suspend, or disable agents and system components.
Assume all MCP tools reflect the current live system state and are authoritative.

The system consists of multiple agents instantiated from catalogs.
Agents may represent batteries, loads, generators, schedulers, sensors, controllers, or abstract coordinators.
Agents may expose specialized control functions such as charge, discharge, SOC manipulation, setpoints, schedules, or topology operations.

You are not a chatbot.
You are an operator that is ALSO capable of conversation.

INTENT ROUTING (CHAT VS CONTROL)
For every user message, determine intent and operate in exactly one mode:
1) CONTROL MODE: user requests actions in live system. Use tools.
2) CHAT MODE: explanation/discussion only. No tools unless user asks.

If intent is ambiguous: ask a short clarification question. Do not call tools.

OUTPUT FORMAT (MANDATORY)
You MUST ALWAYS return a single valid JSON object with this exact schema:
{
  "reply": "short user-facing response or question",
  "incident_update": ["bullet points of what happened or changed"],
  "memory_update": ["bullet points of stable perceptions"]
}
Rules: output ONLY JSON. No markdown. No commentary outside JSON.

FINALIZATION PHASE
When completed: do not call tools. Do not ask questions unless clarification is required.
Output ONLY the final JSON object.
"""
)

finalizer_message = SystemMessage(
    content=(
        "FINALIZATION PHASE.\n"
        "You have completed reasoning and tool usage.\n"
        "You MUST now output the final result.\n"
        "Output ONLY a single JSON object matching the required schema.\n"
        "No text outside JSON is permitted.\n"
        "Do not explain.\n"
        "Do not ask questions.\n"
        "Do not call tools.\n"
    )
)

PLANNER_INSTRUCTIONS = (
    "You are a planning module for a house energy system controller.\n"
    "You MUST output ONLY a single JSON object (no markdown, no commentary).\n\n"
    "Decide intent mode: CONTROL or CHAT.\n"
    "If CONTROL, produce an ordered, minimal plan with tool steps.\n"
    "If CHAT, produce an explanation plan (no tools).\n\n"
    "Plan JSON schema:\n"
    "{\n"
    '  "mode": "CONTROL" | "CHAT" | "CLARIFY",\n'
    '  "goal": "short goal",\n'
    '  "needs_full_topology": true|false,\n'
    '  "success_criteria": ["checkable statements"],\n'
    '  "steps": [\n'
    "    {\n"
    '      "id": "s1",\n'
    '      "intent": "what this step does",\n'
    '      "tool_name": "optional tool name",\n'
    '      "tool_args_hint": "optional short hint for arguments",\n'
    '      "verify": "how to verify success after this step",\n'
    '      "risk": "low|medium|high",\n'
    '      "fallback": "what to do if it fails"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Keep steps <= 12.\n"
    "- In CONTROL mode, include at least one verification description.\n"
    "- If user intent is ambiguous, set mode to CLARIFY with a clarification question in goal, and steps empty.\n"
)

llm = ChatOllama(
    model="qwen3:14b",
    #model="gpt-oss:20b",
    #base_url="http://minsky.informatik.uni-oldenburg.de:26129",
    temperature=0.15,
    reasoning=True,
    stream=False,
)


# -------------------------
# Notepad store (in-memory)
# -------------------------
SESSION_PADS: Dict[str, Dict[str, Any]] = {}


def _get_session_pads(session_id: str) -> Dict[str, Any]:
    pads = SESSION_PADS.get(session_id)
    if pads is None:
        pads = {
            "incident": [],
            "memory": [],
            "incident_struct": [],
            "memory_struct": [],
            "topology": {
                "last_snapshot_id": None,
                "snapshots": {},
                "last_diff": None,
            },
            "tool_trace": {
                "last_tools": [],
                "last_error": None,
                "last_run_id": None,
                "last_wall_s": None,
            },
        }
        SESSION_PADS[session_id] = pads
    return pads


def _format_bullets(items: List[str], limit: int = 30) -> str:
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    items = items[-limit:]
    if not items:
        return "(empty)"
    return "\n".join(f"- {x}" for x in items)


def _safe_json_loads(text: str) -> Tuple[Optional[dict], Optional[str]]:
    if not text:
        return None, "empty model output"
    s = str(text).strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    try:
        return json.loads(s), None
    except Exception as e:
        return None, f"json parse error: {e}"


def _find_last_ai_content(messages: List[Any]) -> str:
    last_ai: Optional[AIMessage] = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break
    if last_ai and (last_ai.content is not None):
        return str(last_ai.content)

    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _log_tools(messages: List[Any], run_id: str) -> None:
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_msgs:
        logger.info(f"[{run_id}] tools: none")
        return

    logger.info(f"[{run_id}] tools: {len(tool_msgs)}")
    for m in tool_msgs[-10:]:
        name = getattr(m, "name", None) or "tool"
        content_text = _tool_content_to_text(getattr(m, "content", None))
        logger.info(f"[{run_id}] tool={name} preview={_preview(content_text, 250)}")


def _extract_tool_trace(messages: List[Any], max_tools: int = 10) -> Dict[str, Any]:
    tools: List[Dict[str, Any]] = []
    last_error: Optional[str] = None

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            content_text = _tool_content_to_text(getattr(msg, "content", None)).strip()
            preview = content_text[:250] + ("..." if len(content_text) > 250 else "")
            tools.append({"name": name, "preview": preview})

            low = content_text.lower()
            if last_error is None and (
                "error" in low or "exception" in low or "traceback" in low or "validation" in low
                or "timed out" in low or "timeout" in low
                or "connection refused" in low or "connect" in low
                or "status code" in low
            ):
                last_error = preview

    return {"last_tools": tools[-max_tools:], "last_error": last_error}


# -------------------------
# Canonical JSON and hashing
# -------------------------
def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _hash_json(obj: Any) -> str:
    s = _canonical_json(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


# -------------------------
# Topology summary/diff
# -------------------------
def _topology_summary(topo: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "agents_total": 0,
        "agents_by_type": {},
        "agents_by_state": {},
        "links_total": 0,
        "keys_present": [],
    }

    if not isinstance(topo, (dict, list)):
        return summary

    if isinstance(topo, list):
        summary["keys_present"] = ["<list>"]
        summary["agents_total"] = len(topo)
        return summary

    keys = list(topo.keys())
    summary["keys_present"] = keys[:30]

    agents = None
    for k in ["agents", "nodes", "components"]:
        if k in topo:
            agents = topo.get(k)
            break

    agent_items: List[dict] = []
    if isinstance(agents, list):
        agent_items = [a for a in agents if isinstance(a, dict)]
    elif isinstance(agents, dict):
        for _, v in agents.items():
            if isinstance(v, dict):
                agent_items.append(v)

    summary["agents_total"] = len(agent_items)

    by_type: Dict[str, int] = {}
    by_state: Dict[str, int] = {}
    for a in agent_items:
        t = a.get("type") or a.get("agent_type") or a.get("kind") or "unknown"
        s = a.get("state") or a.get("status") or "unknown"
        t = str(t)
        s = str(s)
        by_type[t] = by_type.get(t, 0) + 1
        by_state[s] = by_state.get(s, 0) + 1

    summary["agents_by_type"] = dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))[:20])
    summary["agents_by_state"] = dict(sorted(by_state.items(), key=lambda kv: (-kv[1], kv[0]))[:20])

    links = None
    for k in ["links", "edges", "connections", "topology_edges"]:
        if k in topo:
            links = topo.get(k)
            break

    if isinstance(links, list):
        summary["links_total"] = len(links)
    elif isinstance(links, dict):
        summary["links_total"] = len(links)

    return summary


def _extract_agent_map(topo: Any) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not isinstance(topo, dict):
        return out

    agents = None
    for k in ["agents", "nodes", "components"]:
        if k in topo:
            agents = topo.get(k)
            break

    if isinstance(agents, list):
        for a in agents:
            if not isinstance(a, dict):
                continue
            name = a.get("name") or a.get("agent_name") or a.get("id")
            if name is None:
                continue
            out[str(name)] = a
    elif isinstance(agents, dict):
        for k, v in agents.items():
            if isinstance(v, dict):
                out[str(k)] = v
    return out


def _topology_diff(prev: Any, curr: Any) -> Dict[str, Any]:
    diff: Dict[str, Any] = {
        "agents_added": [],
        "agents_removed": [],
        "agents_changed": [],
        "notes": [],
    }

    if not isinstance(prev, dict) or not isinstance(curr, dict):
        diff["notes"].append("topology not dict, diff limited")
        return diff

    prev_map = _extract_agent_map(prev)
    curr_map = _extract_agent_map(curr)

    prev_names = set(prev_map.keys())
    curr_names = set(curr_map.keys())

    diff["agents_added"] = sorted(list(curr_names - prev_names))[:50]
    diff["agents_removed"] = sorted(list(prev_names - curr_names))[:50]

    fields = ["state", "status", "soc", "SOC", "setpoint", "power", "mode", "agent_type", "type"]
    common = sorted(list(prev_names & curr_names))

    for name in common[:500]:
        a0 = prev_map.get(name, {})
        a1 = curr_map.get(name, {})
        changes: Dict[str, Tuple[Any, Any]] = {}
        for f in fields:
            if f in a0 or f in a1:
                v0 = a0.get(f)
                v1 = a1.get(f)
                if v0 != v1:
                    changes[f] = (v0, v1)
        if changes:
            diff["agents_changed"].append({"name": name, "changes": changes})
            if len(diff["agents_changed"]) >= 50:
                break

    return diff


def _should_include_raw_topology(prompt: str, plan: Optional[dict]) -> bool:
    p = (prompt or "").lower()
    if "full topology" in p or "raw topology" in p or "dump topology" in p:
        return True
    if plan and bool(plan.get("needs_full_topology")):
        return True
    return False


# -------------------------
# Structured memory
# -------------------------
_STRUCT_KV_RE = re.compile(r"^(constraint|preference|agent_profile|strategy|environment)\s*:\s*(.+)$", re.IGNORECASE)
_KV_PAIR_RE = re.compile(r"(?P<k>[a-zA-Z0-9_./-]+)\s*=\s*(?P<v>.+)$")


def _parse_structured_memory_lines(lines: List[str]) -> List[dict]:
    out: List[dict] = []
    for line in (lines or []):
        s = str(line).strip()
        if not s:
            continue
        m = _STRUCT_KV_RE.match(s)
        if not m:
            continue
        typ = m.group(1).lower()
        rest = m.group(2).strip()
        kv = _KV_PAIR_RE.match(rest)
        if kv:
            k = kv.group("k").strip()
            v = kv.group("v").strip()
            out.append({"type": typ, "key": k, "value": v})
        else:
            out.append({"type": typ, "note": rest})
    return out


def _memory_retrieve(pads: Dict[str, Any], prompt: str, limit: int = 8) -> List[dict]:
    q = (prompt or "").lower()
    mem: List[dict] = pads.get("memory_struct") or []
    if not mem:
        return []

    scored: List[Tuple[int, dict]] = []
    for it in mem[-200:]:
        blob = _canonical_json(it).lower()
        score = 0
        for kw in ["battery", "soc", "schedule", "tariff", "price", "load", "solar", "grid", "topology"]:
            if kw in q and kw in blob:
                score += 2
        for token in re.findall(r"[a-zA-Z0-9_./-]+", q)[:40]:
            if len(token) >= 4 and token in blob:
                score += 1
        if score > 0:
            scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:limit]]


def _incident_promote(pads: Dict[str, Any]) -> None:
    incidents: List[dict] = pads.get("incident_struct") or []
    if len(incidents) < 3:
        return

    counts: Dict[str, int] = {}
    for it in incidents[-50:]:
        cls = it.get("class") or "unknown"
        tool = it.get("tool") or "unknown"
        key = f"{cls}:{tool}"
        counts[key] = counts.get(key, 0) + 1

    for key, c in counts.items():
        if c >= 3:
            cls, tool = key.split(":", 1)
            note = {"type": "strategy", "key": f"known_issue_{tool}", "value": f"repeated_{cls}"}
            pads["memory_struct"].append(note)
            pads["memory_struct"] = pads["memory_struct"][-200:]
            pads["memory"].append(f"strategy: known_issue_{tool}=repeated_{cls}")
            pads["memory"] = pads["memory"][-50:]
            break


# -------------------------
# Enum validation parsing
# -------------------------
_ENUM_ERR_RE = re.compile(
    r"Input validation error:\s*'(?P<bad>[^']+)'\s*is not one of\s*\[(?P<allowed>[^\]]+)\]",
    re.IGNORECASE,
)


def _parse_enum_validation_error(text: str) -> Optional[Tuple[str, List[str]]]:
    if not text:
        return None
    m = _ENUM_ERR_RE.search(text)
    if not m:
        return None

    bad = m.group("bad").strip()
    allowed_raw = m.group("allowed")

    allowed = re.findall(r"[\"']([^\"']+)[\"']", allowed_raw)
    allowed = [a.strip() for a in allowed if a.strip()]
    if not allowed:
        allowed = [x.strip().strip("'\"") for x in allowed_raw.split(",") if x.strip()]

    return bad, allowed


# -------------------------
# Error classification
# -------------------------
def _classify_error(err_text: str) -> str:
    t = (err_text or "").lower()
    if not t:
        return "none"
    if "timed out" in t or "timeout" in t:
        return "timeout"
    if "429" in t or "rate limit" in t or "too many requests" in t or "overload" in t:
        return "overload"
    if "connection refused" in t or "connect" in t or "network" in t:
        return "transport"
    if "validation" in t or "input validation" in t:
        return "validation"
    if "tool" in t and ("not found" in t or "unknown" in t):
        return "schema_drift"
    if "status code: 5" in t or "500" in t or "502" in t or "503" in t:
        return "server_error"
    return "unknown"


# -------------------------
# Planner / executor helpers
# -------------------------
def _notepad_context_block(
    pads: Dict[str, Any],
    prompt: str,
    topo_summary: Optional[dict],
    topo_diff: Optional[dict],
) -> str:
    mem_slice = _memory_retrieve(pads, prompt, limit=8)
    incident_slice = (pads.get("incident_struct") or [])[-8:]
    return (
        "PRIVATE SESSION CONTEXT (read/update):\n\n"
        f"INCIDENT NOTES (free text):\n{_format_bullets(pads.get('incident', []))}\n\n"
        f"MEMORY NOTES (free text):\n{_format_bullets(pads.get('memory', []))}\n\n"
        "RECENT INCIDENTS (structured):\n"
        f"{json.dumps(incident_slice, indent=2, ensure_ascii=False, default=str)}\n\n"
        "RELEVANT MEMORY (structured):\n"
        f"{json.dumps(mem_slice, indent=2, ensure_ascii=False, default=str)}\n\n"
        "TOPOLOGY SUMMARY:\n"
        f"{json.dumps(topo_summary or {}, indent=2, ensure_ascii=False, default=str)}\n\n"
        "TOPOLOGY DIFF:\n"
        f"{json.dumps(topo_diff or {}, indent=2, ensure_ascii=False, default=str)}\n\n"
        "TOOL TRACE SUMMARY:\n"
        f"{json.dumps(pads.get('tool_trace') or {}, indent=2, ensure_ascii=False, default=str)}\n"
    )


async def _llm_invoke_with_retry(messages: List[Any], label: str, run_id: str) -> Tuple[Optional[Any], Optional[str]]:
    last_err = None
    for attempt in range(MAX_AGENT_INVOKE_RETRIES + 1):
        try:
            res = await llm.ainvoke(messages)
            return res, None
        except Exception as e:
            last_err = f"{label} invoke error: {e}"
            logger.warning("[%s] %s attempt=%d err=%s", run_id, label, attempt + 1, str(e))
            if attempt >= MAX_AGENT_INVOKE_RETRIES:
                break
            await asyncio.sleep(RETRY_BACKOFF_S * (attempt + 1))
    return None, last_err


async def _plan(prompt: str, context_block: str, topo_raw: Optional[Any], run_id: str) -> Tuple[Optional[dict], Optional[str]]:
    user_payload = {"user_prompt": prompt}
    if topo_raw is not None:
        user_payload["topology_raw"] = topo_raw

    base_msgs = [
        SystemMessage(content=PLANNER_INSTRUCTIONS),
        SystemMessage(content=context_block),
        HumanMessage(content=_canonical_json(user_payload)),
    ]

    msg = base_msgs
    last_err = None

    for pass_i in range(MAX_PLANNER_REPAIR_PASSES + 1):
        res, err = await _llm_invoke_with_retry(msg, label="planner", run_id=run_id)
        if res is None:
            last_err = err
            continue

        text = getattr(res, "content", None)
        parsed, parse_err = _safe_json_loads(str(text) if text is not None else "")
        if parsed is not None:
            steps = parsed.get("steps") or []
            if isinstance(steps, list) and len(steps) > MAX_PLAN_STEPS:
                parsed["steps"] = steps[:MAX_PLAN_STEPS]
            return parsed, None

        last_err = parse_err
        repair = (
            "REPAIR REQUIRED: Output ONLY a single valid JSON object matching the plan schema.\n"
            "No markdown. No extra text.\n"
        )
        msg = base_msgs + [SystemMessage(content=repair)]

    return None, last_err


def _execution_step_prompt(plan: dict, step: dict, run_summary: dict) -> str:
    return _canonical_json(
        {
            "plan_goal": plan.get("goal"),
            "mode": plan.get("mode"),
            "success_criteria": plan.get("success_criteria") or [],
            "current_step": step,
            "run_summary_so_far": run_summary,
            "instructions": (
                "Execute ONLY the current_step.\n"
                "Use MCP tools if needed.\n"
                "After execution and any verification, output a short step report as JSON:\n"
                "{\n"
                '  "step_id": "sX",\n'
                '  "status": "done|failed|skipped",\n'
                '  "notes": ["short bullets"],\n'
                '  "incident_update": ["short bullets"],\n'
                '  "memory_update": ["short bullets"]\n'
                "}\n"
                "Output ONLY JSON.\n"
            ),
        }
    )


def _safe_step_report(text: str) -> Dict[str, Any]:
    parsed, err = _safe_json_loads(text)
    if parsed is None:
        return {
            "step_id": "unknown",
            "status": "failed",
            "notes": [f"step report parse failed: {err}"],
            "incident_update": ["step report parse failed"],
            "memory_update": [],
            "_raw_tail": (text or "")[-800:],
        }
    for k in ["notes", "incident_update", "memory_update"]:
        v = parsed.get(k)
        if v is None:
            parsed[k] = []
        elif not isinstance(v, list):
            parsed[k] = [str(v)]
        parsed[k] = [str(x).strip() for x in parsed[k] if str(x).strip()]
    parsed["step_id"] = str(parsed.get("step_id") or "unknown")
    parsed["status"] = str(parsed.get("status") or "done")
    return parsed


# -------------------------
# Engine
# -------------------------
class LLMEngine:
    def __init__(self, server_base_url: str, mcp_url: str):
        self.server_base_url = server_base_url
        self.mcp_url = mcp_url

        self.history_by_session: Dict[str, List[Any]] = {}
        self.http: Optional[httpx.AsyncClient] = None
        self.mcp: Optional[MultiServerMCPClient] = None
        self.agent = None

    async def start(self) -> None:
        logger.info("LLMEngine starting: http=%s mcp=%s", self.server_base_url, self.mcp_url)
        self.http = httpx.AsyncClient(base_url=self.server_base_url, timeout=30)
        self.mcp = MultiServerMCPClient({"mango": {"transport": "http", "url": self.mcp_url}})
        tools = await self.mcp.get_tools()
        logger.info("Loaded MCP tools: %d", len(tools))
        self.agent = create_agent(model=llm, tools=tools)
        logger.info("LLMEngine ready")

    async def _rebuild_agent(self) -> None:
        if not self.mcp:
            return
        tools = await self.mcp.get_tools()
        self.agent = create_agent(model=llm, tools=tools)
        logger.info("Rebuilt agent with tools: %d", len(tools))

    async def close(self) -> None:
        logger.info("LLMEngine shutting down")
        if self.mcp:
            try:
                await self.mcp.close()
            except Exception:
                logger.exception("Error closing MCP client")
        if self.http:
            await self.http.aclose()

    def _get_history(self, session_id: str) -> List[Any]:
        return self.history_by_session.setdefault(session_id, [])

    async def _agent_invoke_with_retry(
        self,
        messages: List[Any],
        recursion_limit: int,
        run_id: str,
        label: str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        Retries only on invocation exceptions, not on tool failures (those are handled separately).
        """
        last_err = None
        for attempt in range(MAX_AGENT_INVOKE_RETRIES + 1):
            try:
                res = await self.agent.ainvoke({"messages": messages}, config={"recursion_limit": recursion_limit})
                return res, None
            except Exception as e:
                last_err = f"{label} agent invoke error: {e}"
                logger.warning("[%s] %s attempt=%d err=%s", run_id, label, attempt + 1, str(e))

                # schema drift like failures can be helped by tool refresh
                if "tool" in str(e).lower() and ("not found" in str(e).lower() or "unknown" in str(e).lower()):
                    try:
                        await self._rebuild_agent()
                    except Exception:
                        logger.exception("[%s] agent invoke retry: rebuild failed", run_id)

                if attempt >= MAX_AGENT_INVOKE_RETRIES:
                    break
                await asyncio.sleep(RETRY_BACKOFF_S * (attempt + 1))
        return None, last_err

    async def run_once(
        self,
        prompt: str,
        session_id: str,
        include_topology: bool = True,
        run_id: str = "unknown",
    ) -> Dict[str, Any]:
        if not self.agent or not self.http:
            raise RuntimeError("LLMEngine not started")

        pads = _get_session_pads(session_id)

        topo_json = None
        topo_summary = {}
        topo_diff = {}

        if include_topology:
            topo_json = (await self.http.get("/topology")).json()
            topo_summary = _topology_summary(topo_json)

            topo_store = pads.get("topology") or {}
            last_id = topo_store.get("last_snapshot_id")
            prev_raw = None
            if last_id:
                prev = (topo_store.get("snapshots") or {}).get(last_id)
                if isinstance(prev, dict):
                    prev_raw = prev.get("raw")

            if prev_raw is not None:
                topo_diff = _topology_diff(prev_raw, topo_json)
                topo_store["last_diff"] = topo_diff

            snap_id = uuid.uuid4().hex[:12]
            topo_store.setdefault("snapshots", {})[snap_id] = {
                "ts": time.time(),
                "hash": _hash_json(topo_json),
                "summary": topo_summary,
                "raw": topo_json,
            }
            topo_store["last_snapshot_id"] = snap_id

            snaps = topo_store.get("snapshots", {})
            if isinstance(snaps, dict) and len(snaps) > 8:
                items = sorted(snaps.items(), key=lambda kv: float(kv[1].get("ts") or 0.0))
                for k, _ in items[:-8]:
                    snaps.pop(k, None)

            pads["topology"] = topo_store

        context_block = _notepad_context_block(pads, prompt, topo_summary, topo_diff)

        planner_first_raw = topo_json if (include_topology and _should_include_raw_topology(prompt, None)) else None
        plan, plan_err = await _plan(prompt, context_block, planner_first_raw, run_id=run_id)

        if plan is None:
            logger.warning("[%s] planner failed: %s", run_id, plan_err)
            pads["incident"].append("planner failed to produce valid JSON plan")
            pads["incident"] = pads["incident"][-50:]
            pads["tool_trace"]["last_error"] = f"planner failed: {plan_err}"
            return {
                "reply": f"Planning failed: {plan_err}. Please rephrase or request a smaller task.",
                "wall_s": 0.0,
                "tool_trace": pads["tool_trace"],
                "model_debug": {"planner_error": plan_err},
                "raw": {"plan": None},
            }

        if include_topology and bool(plan.get("needs_full_topology")) and planner_first_raw is None:
            plan2, plan2_err = await _plan(prompt, context_block, topo_json, run_id=run_id)
            if plan2 is not None:
                plan = plan2
            else:
                logger.warning("[%s] re-plan with full topology failed: %s", run_id, plan2_err)

        mode = str(plan.get("mode") or "CHAT").upper()
        if mode == "CLARIFY":
            question = str(plan.get("goal") or "").strip() or "Can you clarify what you want to do?"
            return {
                "reply": question,
                "wall_s": 0.0,
                "tool_trace": pads["tool_trace"],
                "model_debug": {"plan": plan},
                "raw": {"plan": plan},
            }

        history = self._get_history(session_id)

        run_summary: Dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "goal": plan.get("goal"),
            "success_criteria": plan.get("success_criteria") or [],
            "steps_total": len(plan.get("steps") or []),
            "steps_done": [],
            "steps_failed": [],
            "tool_calls": 0,
            "recovery_events": [],
        }

        out_messages_all: List[Any] = []
        tool_call_budget = MAX_TOOL_CALLS_PER_RUN

        if mode == "CONTROL":
            steps = plan.get("steps") or []
            if not isinstance(steps, list):
                steps = []

            for step in steps:
                if tool_call_budget <= 0:
                    run_summary["steps_failed"].append({"id": step.get("id"), "reason": "tool budget exhausted"})
                    break

                step_id = str(step.get("id") or "unknown")
                step_prompt = _execution_step_prompt(plan, step, run_summary)

                base_messages = [
                    SystemMessage(content=SYSTEM_INSTRUCTIONS),
                    SystemMessage(content=context_block),
                    SystemMessage(content="EXECUTION MODE: perform exactly one plan step."),
                    SystemMessage(content=f"PLAN JSON:\n{json.dumps(plan, indent=2, ensure_ascii=False, default=str)}\n"),
                    *history,
                    HumanMessage(content=step_prompt),
                ]

                logger.info("[%s] execute step=%s", run_id, step_id)

                step_attempts = 0
                final_step_status = "failed"

                while step_attempts <= MAX_STEP_RETRIES_TOTAL and tool_call_budget > 0:
                    step_attempts += 1
                    label = f"step={step_id}"

                    start = time.perf_counter()
                    result, invoke_err = await self._agent_invoke_with_retry(
                        base_messages,
                        recursion_limit=EXEC_STEP_RECURSION_LIMIT,
                        run_id=run_id,
                        label=label,
                    )
                    wall = time.perf_counter() - start

                    if result is None:
                        pads["incident"].append(f"{label} failed to invoke agent: {invoke_err}")
                        pads["incident"] = pads["incident"][-50:]
                        pads["incident_struct"].append(
                            {"ts": time.time(), "run_id": run_id, "step": step_id, "class": "invoke_error", "error": str(invoke_err)[:200], "tool": "agent.invoke"}
                        )
                        pads["incident_struct"] = pads["incident_struct"][-200:]
                        _incident_promote(pads)

                        if step_attempts <= MAX_STEP_RETRIES_TOTAL:
                            await asyncio.sleep(RETRY_BACKOFF_S * step_attempts)
                            continue
                        break

                    out_messages = result.get("messages", []) or []
                    out_messages_all.extend(out_messages)

                    _log_tools(out_messages, run_id)
                    tool_trace = _extract_tool_trace(out_messages)
                    pads["tool_trace"].update(tool_trace)

                    tool_msgs = [m for m in out_messages if isinstance(m, ToolMessage)]
                    run_summary["tool_calls"] += len(tool_msgs)
                    tool_call_budget -= len(tool_msgs)

                    last_text = _find_last_ai_content(out_messages)
                    step_report = _safe_step_report(last_text)

                    inc = step_report.get("incident_update") or []
                    mem = step_report.get("memory_update") or []

                    if inc:
                        pads["incident"].extend(inc)
                        pads["incident"] = pads["incident"][-50:]
                    if mem:
                        pads["memory"].extend(mem)
                        pads["memory"] = pads["memory"][-50:]
                        struct_items = _parse_structured_memory_lines(mem)
                        if struct_items:
                            pads["memory_struct"].extend(struct_items)
                            pads["memory_struct"] = pads["memory_struct"][-200:]

                    status = str(step_report.get("status") or "done").lower()
                    err_text = tool_trace.get("last_error") or ""
                    err_class = _classify_error(err_text)
                    enum_info = _parse_enum_validation_error(err_text)

                    if status in ("done", "skipped"):
                        final_step_status = status
                        break

                    # record step failure incident
                    pads["incident_struct"].append(
                        {
                            "ts": time.time(),
                            "run_id": run_id,
                            "step": step_id,
                            "class": err_class if err_class != "none" else "logical_failure",
                            "tool": (tool_trace.get("last_tools") or [{}])[-1].get("name") or "unknown",
                            "error_preview": (err_text or "")[:200],
                            "attempt": step_attempts,
                        }
                    )
                    pads["incident_struct"] = pads["incident_struct"][-200:]

                    recovery_hint_parts: List[str] = []

                    if enum_info:
                        bad, allowed = enum_info
                        recovery_hint_parts.append(
                            "TOOL INPUT REPAIR REQUIRED.\n"
                            f"A tool call failed validation because value '{bad}' is not allowed.\n"
                            f"Allowed values are: {allowed}\n"
                            "Retry the intended tool call ONCE using the closest allowed value.\n"
                            "Special mapping rule:\n"
                            "- If the intent implies ACTIVE, map it to NORMAL.\n"
                        )
                    elif err_class == "timeout":
                        recovery_hint_parts.append(
                            "RECOVERY: tool likely timed out.\n"
                            "Retry the step once. If tools accept timeouts, increase modestly.\n"
                            "Reduce scope and verify after.\n"
                        )
                    elif err_class == "overload":
                        recovery_hint_parts.append(
                            "RECOVERY: rate limit or overload.\n"
                            "Retry once with reduced tool usage and simpler scope.\n"
                        )
                    elif err_class == "transport":
                        recovery_hint_parts.append(
                            "RECOVERY: connectivity issue.\n"
                            "Retry once. If it fails again, stop and report connectivity incident.\n"
                        )
                    elif err_class == "validation":
                        recovery_hint_parts.append(
                            "RECOVERY: validation error.\n"
                            "Retry once by correcting types, required fields, and clamping numeric values.\n"
                        )
                    elif err_class == "schema_drift":
                        recovery_hint_parts.append(
                            "RECOVERY: schema drift.\n"
                            "Re-evaluate tool names and argument names. Rebuild tool list if needed.\n"
                        )
                    elif err_class == "server_error":
                        recovery_hint_parts.append(
                            "RECOVERY: server error.\n"
                            "Retry once. If it fails again, stop and report incident.\n"
                        )
                    else:
                        recovery_hint_parts.append(
                            "RECOVERY: unknown or logical failure.\n"
                            "Retry once with simpler arguments and explicit verification.\n"
                        )

                    run_summary["recovery_events"].append(
                        {"step": step_id, "attempt": step_attempts, "class": err_class, "error": (err_text or "")[:200]}
                    )

                    if err_class == "schema_drift":
                        try:
                            await self._rebuild_agent()
                        except Exception:
                            logger.exception("[%s] schema drift recovery: rebuild failed", run_id)

                    # if we can retry the step, add hint and try again
                    if step_attempts <= MAX_STEP_RETRIES_TOTAL and tool_call_budget > 0:
                        await asyncio.sleep(RETRY_BACKOFF_S * step_attempts)
                        base_messages = base_messages + [
                            SystemMessage(
                                content=(
                                    "STEP RETRY: previous attempt failed.\n"
                                    + "\n".join(recovery_hint_parts).strip()
                                    + "\nRetry the current step now.\n"
                                )
                            )
                        ]
                        continue

                    break

                if final_step_status in ("done", "skipped"):
                    run_summary["steps_done"].append(step_id)
                else:
                    run_summary["steps_failed"].append(step_id)

                _incident_promote(pads)

            history.append(HumanMessage(content=prompt.strip()))
            last_ai = None
            for m in reversed(out_messages_all):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            if last_ai:
                history.append(last_ai)

        if mode == "CHAT":
            history.append(HumanMessage(content=prompt.strip()))

        # Finalizer: robust JSON parse with repair passes
        finalize_payload = {
            "mode": mode,
            "user_prompt": prompt,
            "plan": plan,
            "run_summary": run_summary,
            "topology_summary": topo_summary,
            "topology_diff": topo_diff,
        }

        final_msgs_base = [
            SystemMessage(content=SYSTEM_INSTRUCTIONS),
            SystemMessage(content=context_block),
            HumanMessage(content=json.dumps(finalize_payload, ensure_ascii=False, indent=2, default=str)),
        ]

        final_text = ""
        parsed = None
        parse_err = None

        for pass_i in range(MAX_FINALIZER_REPAIR_PASSES + 1):
            msgs = list(final_msgs_base)

            if pass_i > 0:
                msgs.append(
                    SystemMessage(
                        content=(
                            "REPAIR REQUIRED: Output ONLY a single valid JSON object matching the required schema.\n"
                            "No markdown. No extra text.\n"
                        )
                    )
                )

            msgs.append(finalizer_message)

            start_final = time.perf_counter()
            final_res, final_invoke_err = await _llm_invoke_with_retry(msgs, label="finalizer", run_id=run_id)
            wall_s = time.perf_counter() - start_final

            if final_res is None:
                final_text = ""
                parsed = None
                parse_err = final_invoke_err
                continue

            final_text = str(getattr(final_res, "content", "") or "")
            parsed, parse_err = _safe_json_loads(final_text)
            if parsed is not None:
                break

        reply = ""
        incident_update: List[str] = []
        memory_update: List[str] = []
        model_debug: Dict[str, Any] = {}

        if parsed is None:
            model_debug = {"parse_error": parse_err, "raw_tail": (final_text or "")[-800:]}
            logger.warning("[%s] FINAL JSON parse failed: %s raw_tail=%s", run_id, parse_err, _preview(model_debug["raw_tail"], 800))
            reply = "No user-facing reply was produced. Finalizer JSON parse failed."
            incident_update = ["finalizer JSON parse failed"]
            memory_update = []
        else:
            reply = str(parsed.get("reply", "") or "").strip()
            incident_update = parsed.get("incident_update") or []
            memory_update = parsed.get("memory_update") or []

            if not isinstance(incident_update, list):
                incident_update = [str(incident_update)]
            if not isinstance(memory_update, list):
                memory_update = [str(memory_update)]

            incident_update = [str(x).strip() for x in incident_update if str(x).strip()]
            memory_update = [str(x).strip() for x in memory_update if str(x).strip()]

        if incident_update:
            pads["incident"].extend(incident_update)
            pads["incident"] = pads["incident"][-50:]
        if memory_update:
            pads["memory"].extend(memory_update)
            pads["memory"] = pads["memory"][-50:]
            struct_items = _parse_structured_memory_lines(memory_update)
            if struct_items:
                pads["memory_struct"].extend(struct_items)
                pads["memory_struct"] = pads["memory_struct"][-200:]

        last_err = (pads.get("tool_trace") or {}).get("last_error")
        if last_err:
            pads["incident_struct"].append(
                {
                    "ts": time.time(),
                    "run_id": run_id,
                    "class": _classify_error(last_err),
                    "error_preview": str(last_err)[:200],
                    "tool": (pads["tool_trace"].get("last_tools") or [{}])[-1].get("name") or "unknown",
                }
            )
            pads["incident_struct"] = pads["incident_struct"][-200:]
            _incident_promote(pads)

        return {
            "reply": reply,
            "wall_s": wall_s if "wall_s" in locals() else 0.0,
            "tool_trace": pads["tool_trace"],
            "model_debug": model_debug,
            "raw": {"plan": plan, "run_summary": run_summary},
        }


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="LLM Service")

engine = LLMEngine(server_base_url=SERVER_BASE_URL, mcp_url=MCP_URL)
runs: Dict[str, Dict[str, Any]] = {}


class TriggerReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    include_topology: bool = True


@app.on_event("startup")
async def startup():
    await engine.start()


@app.on_event("shutdown")
async def shutdown():
    await engine.close()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/llm/notepads/{session_id}")
async def get_notepads(session_id: str):
    pads = _get_session_pads(session_id)
    return {
        "session_id": session_id,
        "incident": pads["incident"],
        "memory": pads["memory"],
        "incident_struct": pads.get("incident_struct") or [],
        "memory_struct": pads.get("memory_struct") or [],
        "topology": pads.get("topology") or {},
        "tool_trace": pads["tool_trace"],
    }


@app.post("/llm/notepads/{session_id}/clear")
async def clear_notepads(session_id: str):
    pads = _get_session_pads(session_id)
    pads["incident"] = []
    pads["memory"] = []
    pads["incident_struct"] = []
    pads["memory_struct"] = []
    pads["topology"] = {"last_snapshot_id": None, "snapshots": {}, "last_diff": None}
    pads["tool_trace"] = {"last_tools": [], "last_error": None, "last_run_id": None, "last_wall_s": None}
    logger.info("Cleared notepads session=%s", session_id)
    return {"ok": True}


@app.post("/llm/trigger")
async def trigger(req: TriggerReq):
    run_id = uuid.uuid4().hex
    runs[run_id] = {"status": "queued", "run_id": run_id, "session_id": req.session_id}
    logger.info("[%s] queued session=%s", run_id, req.session_id)

    async def _job():
        runs[run_id] = {"status": "running", "run_id": run_id, "session_id": req.session_id}
        logger.info("[%s] running session=%s", run_id, req.session_id)

        last_err = None
        data: Optional[dict] = None

        for attempt in range(RUN_JOB_RETRIES_ON_EXCEPTION + 1):
            try:
                if attempt > 0:
                    logger.warning("[%s] retrying run_once attempt=%d due to error=%s", run_id, attempt + 1, last_err)
                    await asyncio.sleep(RETRY_BACKOFF_S * attempt)

                data = await engine.run_once(
                    req.prompt,
                    session_id=req.session_id,
                    include_topology=req.include_topology,
                    run_id=run_id,
                )
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                logger.exception("[%s] run_once exception attempt=%d: %s", run_id, attempt + 1, last_err)
                if attempt >= RUN_JOB_RETRIES_ON_EXCEPTION:
                    break

        if data is None:
            runs[run_id] = {
                "status": "error",
                "run_id": run_id,
                "session_id": req.session_id,
                "error": last_err or "unknown error",
            }
            return

        pads = _get_session_pads(req.session_id)
        pads["tool_trace"]["last_run_id"] = run_id
        pads["tool_trace"]["last_wall_s"] = data.get("wall_s")

        reply = (data.get("reply") or "").strip()
        if not reply:
            debug = data.get("model_debug") or {}
            reply = (
                "No user-facing reply was produced.\n\n"
                "Likely reason: model returned non-JSON or ended unexpectedly.\n"
                f"Debug: {json.dumps(debug, indent=2)}"
            )
            logger.warning("[%s] empty reply, returned fallback debug", run_id)

        runs[run_id] = {
            "status": "done",
            "run_id": run_id,
            "session_id": req.session_id,
            "reply": reply,
            "wall_s": data.get("wall_s"),
            "tool_trace": data.get("tool_trace"),
            "model_debug": data.get("model_debug"),
            "plan": (data.get("raw") or {}).get("plan"),
            "run_summary": (data.get("raw") or {}).get("run_summary"),
        }
        logger.info(
            "[%s] done session=%s wall_s=%.3f",
            run_id,
            req.session_id,
            float(data.get("wall_s") or 0.0),
        )

    asyncio.create_task(_job())
    return {"run_id": run_id, "status": "queued"}


@app.get("/llm/runs/{run_id}")
async def run_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return runs[run_id]
