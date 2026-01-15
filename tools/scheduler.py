from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScheduleItem:
    id: str
    created_at: float
    created_by: str
    status: str  # scheduled | running | done | failed | cancelled
    run_at: float  # epoch seconds
    agent_name: str
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class InMemoryScheduler:
    def __init__(self):
        self._items: Dict[str, ScheduleItem] = {}
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        # injected from server:
        self._action_runner = None  # callable(agent_name, action, payload) -> dict

    def set_action_runner(self, fn):
        self._action_runner = fn

    async def start(self, poll_s: float = 0.5) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(poll_s=poll_s))

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                pass

    async def create(
        self,
        created_by: str,
        run_at_epoch: float,
        agent_name: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> ScheduleItem:
        item = ScheduleItem(
            id=uuid.uuid4().hex,
            created_at=time.time(),
            created_by=created_by,
            status="scheduled",
            run_at=float(run_at_epoch),
            agent_name=agent_name,
            action=action,
            payload=payload or {},
        )
        async with self._lock:
            self._items[item.id] = item
        return item

    async def cancel(self, schedule_id: str) -> bool:
        async with self._lock:
            item = self._items.get(schedule_id)
            if not item:
                return False
            if item.status in ("done", "failed", "cancelled"):
                return False
            item.status = "cancelled"
            return True

    async def list(self, status: Optional[str] = None) -> List[ScheduleItem]:
        async with self._lock:
            items = list(self._items.values())
        if status:
            items = [x for x in items if x.status == status]
        items.sort(key=lambda x: x.run_at)
        return items

    async def get(self, schedule_id: str) -> Optional[ScheduleItem]:
        async with self._lock:
            return self._items.get(schedule_id)

    async def _loop(self, poll_s: float) -> None:
        while not self._stop.is_set():
            await self._tick()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=poll_s)
            except asyncio.TimeoutError:
                continue

    async def _tick(self) -> None:
        if self._action_runner is None:
            return

        now = time.time()

        due: List[ScheduleItem] = []
        async with self._lock:
            for item in self._items.values():
                if item.status == "scheduled" and item.run_at <= now:
                    item.status = "running"
                    due.append(item)

        # run outside lock
        for item in due:
            try:
                res = await self._maybe_await(self._action_runner(item.agent_name, item.action, item.payload))
                item.result = res if isinstance(res, dict) else {"result": res}
                item.status = "done"
            except Exception as e:
                item.error = str(e)
                item.status = "failed"

    async def _maybe_await(self, x):
        if asyncio.iscoroutine(x):
            return await x
        return x
