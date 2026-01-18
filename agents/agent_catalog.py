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

def create_agent_spec(cls: Type[DynamicAgent]) -> AgentTypeSpec:
    # Inspect __init__ to find required and optional fields
    required_fields = []
    optional_fields = []
    init_params = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
    defaults = cls.__init__.__defaults__ or ()
    num_required = len(init_params) - len(defaults)

    for i, param in enumerate(init_params):
        if i < num_required:
            required_fields.append(param)
        else:
            optional_fields.append(param)

    return AgentTypeSpec(
        type=cls.TYPE,
        label=cls.LABEL,
        default_persona=cls.DEFAULT_PERSONA,
        default_usage=cls.DEFAULT_USAGE,
        capabilities=cls.CAPABILITIES,
        required_fields=required_fields,
        optional_fields=optional_fields,
    )


def generate_agent_catalog() -> Dict:
    specs: List[AgentTypeSpec] = []

    for cls in _iter_all_subclasses(DynamicAgent):
        agent_type = getattr(cls, "TYPE", None)

        # skip base/generic or misconfigured classes
        if not agent_type or str(agent_type).strip() in ('dynamic',"","io"):
            continue

        specs.append(create_agent_spec(cls))

    specs.sort(key=lambda s: s.type)

    return {
        "version": 1,
        "agent_types": [asdict(s) for s in specs],
    }


