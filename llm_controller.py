import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


# -------------------------
# Config
# -------------------------
SERVER_BASE_URL = "http://127.0.0.1:8000"
MCP_URL = "http://127.0.0.1:8000/mcp"

MAX_STEPS = 40
SYSTEM_INSTRUCTIONS = (
    "You are a conversational assistant controlling a Mango multi-agent system using MCP tools.\n"
    "You must operate in a loop: each step choose exactly one action (tool call or final reply).\n\n"
    "Rules:\n"
    "- Only use the provided tools.\n"
    "- If a tool call fails, fix the inputs and retry.\n"
    "- When the user request is satisfied, provide a short summary.\n"
)

llm = ChatOllama(model="qwen3:14b", temperature=0.15)


# -------------------------
# Engine
# -------------------------
class LLMEngine:
    def __init__(self, server_base_url: str, mcp_url: str):
        self.server_base_url = server_base_url
        self.mcp_url = mcp_url

        self.history: List[Any] = []
        self.http: Optional[httpx.AsyncClient] = None
        self.mcp: Optional[MultiServerMCPClient] = None
        self.agent = None

    async def start(self) -> None:
        self.http = httpx.AsyncClient(base_url=self.server_base_url, timeout=30)
        self.mcp = MultiServerMCPClient({"mango": {"transport": "http", "url": self.mcp_url}})
        tools = await self.mcp.get_tools()
        self.agent = create_agent(model=llm, tools=tools)

    async def close(self) -> None:
        if self.mcp:
            try:
                await self.mcp.close()
            except Exception:
                pass
        if self.http:
            await self.http.aclose()

    async def run_once(self, prompt: str, include_topology: bool = True) -> Dict[str, Any]:
        if not self.agent or not self.http:
            raise RuntimeError("LLMEngine not started")

        topo = None
        if include_topology:
            topo = (await self.http.get("/topology")).json()

        messages = [
            SystemMessage(content=SYSTEM_INSTRUCTIONS),
            *self.history,
            HumanMessage(
                content=(
                    f"{prompt}\n\n"
                    f"Live topology JSON:\n{json.dumps(topo, indent=2)}\n"
                )
            ),
        ]

        start = time.perf_counter()
        result = await self.agent.ainvoke({"messages": messages}, config={"recursion_limit": MAX_STEPS})
        wall_s = time.perf_counter() - start

        # Save minimal conversational continuity
        self.history.append(HumanMessage(content=prompt))
        out = result.get("messages", []) or []
        if out:
            self.history.append(out[-1])

        # Extract final assistant reply
        last = out[-1].content if out and hasattr(out[-1], "content") else ""
        return {"reply": last, "wall_s": wall_s, "raw": result}


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="LLM Service")

engine = LLMEngine(server_base_url=SERVER_BASE_URL, mcp_url=MCP_URL)

# in-memory run store (PoC)
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


@app.post("/llm/trigger")
async def trigger(req: TriggerReq):
    """
    Starts an LLM run asynchronously and returns a run_id for UI polling.
    """
    run_id = uuid.uuid4().hex
    runs[run_id] = {"status": "queued"}

    async def _job():
        runs[run_id] = {"status": "running"}
        try:
            data = await engine.run_once(req.prompt, include_topology=req.include_topology)
            runs[run_id] = {
                "status": "done",
                "reply": data["reply"],
                "wall_s": data["wall_s"],
            }
        except Exception as e:
            runs[run_id] = {"status": "error", "error": str(e)}

    asyncio.create_task(_job())
    return {"run_id": run_id, "status": "queued"}


@app.get("/llm/runs/{run_id}")
async def run_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return runs[run_id]
