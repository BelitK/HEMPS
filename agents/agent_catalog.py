from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Type

from .dynamic_agent import DynamicAgent


@dataclass(frozen=True)
class AgentTypeSpec:
    type: str
    label: str
    default_persona: str
    default_usage: str
    capabilities: List[str]
    required_fields: List[str]
    optional_fields: List[str]


def _iter_all_subclasses(cls: Type) -> List[Type]:
    out: List[Type] = []
    stack = list(cls.__subclasses__())
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(c.__subclasses__())
    return out


def generate_agent_catalog() -> Dict:
    specs: List[AgentTypeSpec] = []

    for cls in _iter_all_subclasses(DynamicAgent):
        agent_type = getattr(cls, "TYPE", None)

        # skip base/generic or misconfigured classes
        if not agent_type or str(agent_type).strip() in ("dynamic", ""):
            continue

        specs.append(
            AgentTypeSpec(
                type=str(agent_type),
                label=str(getattr(cls, "LABEL", cls.__name__)),
                default_persona=str(getattr(cls, "DEFAULT_PERSONA", "")),
                default_usage=str(getattr(cls, "DEFAULT_USAGE", "")),
                capabilities=list(getattr(cls, "CAPABILITIES", [])) or [],
                required_fields=["name"],
                optional_fields=["persona", "usage"],
            )
        )

    specs.sort(key=lambda s: s.type)

    return {
        "version": 1,
        "agent_types": [asdict(s) for s in specs],
    }
