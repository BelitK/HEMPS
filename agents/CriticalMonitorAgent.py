import asyncio
import time
import httpx

from agents.dynamic_agent import DynamicAgent


class CriticalMonitorAgent(DynamicAgent):
    def __init__(self, llm_trigger_url: str, cooldown_s: float = 10.0):
        super().__init__(
            name="critical_monitor",
            persona="Watches for critical events and triggers the LLM incident response.",
            usage="Triggers LLM when critical messages arrive.",
        )
        self.llm_trigger_url = llm_trigger_url
        self.cooldown_s = cooldown_s
        self._last_trigger_ts = 0.0

    def _is_critical(self, content: str) -> bool:
        text = (content or "").lower()
        # adjust this to your real signal
        return ("critical" in text) or ("panic" in text) or ("incident" in text)

    async def _trigger_llm(self, content: str, meta: dict):
        payload = {
            "prompt": (
                "CRITICAL EVENT DETECTED.\n"
                f"Message: {content}\n"
                f"Meta: {meta}\n"
                "Action: inspect topology, identify affected agents/edges, mitigate, and report summary."
            ),
            "include_topology": True,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self.llm_trigger_url, json=payload)
            r.raise_for_status()

    def handle_message(self, content, meta):
        # 1) fast filter
        if not self._is_critical(content):
            return

        # 2) cooldown (prevents trigger storms)
        now = time.time()
        if now - self._last_trigger_ts < self.cooldown_s:
            print(f"[{self.name}] critical ignored (cooldown)")
            return
        self._last_trigger_ts = now

        print(f"[{self.name}] critical detected, triggering LLM...")

        # 3) schedule async trigger without blocking Mango
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._trigger_llm(content, meta or {}))
        except RuntimeError:
            # if no running loop in this thread, fallback
            asyncio.run(self._trigger_llm(content, meta or {}))
