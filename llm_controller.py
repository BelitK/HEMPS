import asyncio
import json
import logging
import os
import time
import uuid
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
LOG_FILE = os.getenv("LLM_LOG_FILE", "llm_service.log")

logger = logging.getLogger("llm_service")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False  # prevent duplicate logs if uvicorn also configures root logger

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# console
_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.addHandler(_console)

# rotating file (5MB x 5)
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

MAX_STEPS = 60

SYSTEM_INSTRUCTIONS = (
    "You are a controller for a Mango multi-agent system using MCP tools.\n"
    "You may call tools. If a tool call fails, fix inputs and retry.\n"
    "When finished, you MUST output ONLY valid JSON with this schema:\n"
    "{\n"
    '  "reply": "short user-facing summary",\n'
    '  "incident_update": ["1-5 bullets, system state, hypotheses, next actions"],\n'
    '  "memory_update": ["0-5 bullets, stable preferences/constraints/goals per session"]\n'
    "}\n"
    "Rules:\n"
    "- Output ONLY JSON in the final response (no markdown, no extra text).\n"
    "- Keep reply concise.\n"
    "- incident_update should reflect what changed or what you learned.\n"
    "- memory_update should include only stable facts worth remembering.\n"
    "Important:\n"
    "- If the user asks to create agents or edges, you MUST use tools to do it (do not just describe it).\n"
    "- After tool calls, return the final JSON.\n"
)

llm = ChatOllama(model="qwen3:14b", temperature=0.15)


# -------------------------
# Notepad store (PoC in-memory)
# -------------------------
SESSION_PADS: Dict[str, Dict[str, Any]] = {}


def _get_session_pads(session_id: str) -> Dict[str, Any]:
    pads = SESSION_PADS.get(session_id)
    if pads is None:
        pads = {
            "incident": [],
            "memory": [],
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
            if last_error is None and ("error" in low or "exception" in low or "traceback" in low):
                last_error = preview

    return {"last_tools": tools[-max_tools:], "last_error": last_error}


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
        if include_topology:
            topo_json = (await self.http.get("/topology")).json()

        notepad_message = (
            "PRIVATE SESSION NOTEPADS (read/update):\n\n"
            f"INCIDENT NOTES:\n{_format_bullets(pads['incident'])}\n\n"
            f"MEMORY NOTES:\n{_format_bullets(pads['memory'])}\n\n"
            "TOOL TRACE SUMMARY (auto, for context):\n"
            f"{json.dumps(pads['tool_trace'], indent=2)}\n"
        )

        history = self._get_history(session_id)

        user_block = f"{prompt.strip()}\n"
        if include_topology:
            user_block += "\nLive topology JSON:\n" + json.dumps(topo_json, indent=2) + "\n"

        messages = [
            SystemMessage(content=SYSTEM_INSTRUCTIONS),
            SystemMessage(content=notepad_message),
            *history,
            HumanMessage(content=user_block),
        ]

        logger.info(
            "[%s] run_once start session=%s include_topology=%s prompt=%s",
            run_id,
            session_id,
            include_topology,
            _preview(prompt, 400),
        )

        start = time.perf_counter()
        result = await self.agent.ainvoke({"messages": messages}, config={"recursion_limit": MAX_STEPS})
        wall_s = time.perf_counter() - start

        out_messages = result.get("messages", []) or []

        _log_tools(out_messages, run_id)
        logger.info("[%s] model finished wall_s=%.3f", run_id, wall_s)

        history.append(HumanMessage(content=prompt.strip()))
        if out_messages:
            history.append(out_messages[-1])

        tool_trace = _extract_tool_trace(out_messages)

        last_text = _find_last_ai_content(out_messages)
        parsed, parse_err = _safe_json_loads(last_text)

        reply = ""
        incident_update: List[str] = []
        memory_update: List[str] = []
        model_debug: Dict[str, Any] = {}

        if parsed is None:
            model_debug = {
                "parse_error": parse_err,
                "raw_tail": (last_text or "")[-800:],
                "last_message_type": out_messages[-1].__class__.__name__ if out_messages else None,
            }
            logger.warning(
                "[%s] JSON parse failed: %s raw_tail=%s",
                run_id,
                parse_err,
                _preview(model_debug["raw_tail"], 800),
            )
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

            logger.info(
                "[%s] parsed ok reply_len=%d incident_items=%d memory_items=%d",
                run_id,
                len(reply),
                len(incident_update),
                len(memory_update),
            )

        if incident_update:
            pads["incident"].extend(incident_update)
            pads["incident"] = pads["incident"][-50:]

        if memory_update:
            pads["memory"].extend(memory_update)
            pads["memory"] = pads["memory"][-50:]

        pads["tool_trace"].update(tool_trace)

        return {
            "reply": reply,
            "wall_s": wall_s,
            "tool_trace": pads["tool_trace"],
            "model_debug": model_debug,
            "raw": result,
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
        "tool_trace": pads["tool_trace"],
    }


@app.post("/llm/notepads/{session_id}/clear")
async def clear_notepads(session_id: str):
    pads = _get_session_pads(session_id)
    pads["incident"] = []
    pads["memory"] = []
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

        try:
            data = await engine.run_once(
                req.prompt,
                session_id=req.session_id,
                include_topology=req.include_topology,
                run_id=run_id,
            )

            pads = _get_session_pads(req.session_id)
            pads["tool_trace"]["last_run_id"] = run_id
            pads["tool_trace"]["last_wall_s"] = data.get("wall_s")

            reply = (data.get("reply") or "").strip()
            if not reply:
                debug = data.get("model_debug") or {}
                reply = (
                    "No user-facing reply was produced.\n\n"
                    "Likely reason: model returned non-JSON or ended on tool output.\n"
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
            }
            logger.info(
                "[%s] done session=%s wall_s=%.3f",
                run_id,
                req.session_id,
                float(data.get("wall_s") or 0.0),
            )

        except Exception as e:
            logger.exception("[%s] error session=%s: %s", run_id, req.session_id, str(e))
            runs[run_id] = {"status": "error", "run_id": run_id, "session_id": req.session_id, "error": str(e)}

    asyncio.create_task(_job())
    return {"run_id": run_id, "status": "queued"}


@app.get("/llm/runs/{run_id}")
async def run_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return runs[run_id]
