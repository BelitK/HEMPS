from mango import Agent


# -------------------------
# Base Mango Agent
# -------------------------
class DynamicAgent(Agent):
    """
    Base class for all HEMS agents.
    
    Provides:
    - Catalog metadata (TYPE, LABEL, CAPABILITIES)
    - Basic info() and handle_message() methods
    - State management
    
    Subclass this and define your own CAPABILITIES and methods.
    See io_agent.py, pv_agent.py, battery_agent.py, grid_agent.py for examples.
    """
    
    # Catalog metadata (defaults - override in subclasses)
    TYPE = "dynamic"
    LABEL = "Dynamic Agent"
    DEFAULT_PERSONA = "Generic dynamic agent."
    DEFAULT_USAGE = "Base agent class."
    CAPABILITIES: list[str] = []

    def __init__(self, name: str, persona: str | None = None, usage: str | None = None):
        super().__init__()
        self.name = name
        self.persona = persona or self.DEFAULT_PERSONA
        self.usage = usage or self.DEFAULT_USAGE
        self.state = "active"

    def handle_message(self, content, meta):
        """Default message handler - override in subclasses."""
        print(f"[{self.name}] {content}")

        # Normalize to case-insensitive
        if "critical" in str(content).lower():
            self.state = "critical"
            print(f"[{self.name}] Agent {self.name} is now in CRITICAL state!")

    def info(self):
        """Return agent info for LLM/catalog."""
        return {
            "name": self.name,
            "persona": self.persona,
            "usage": self.usage,
            "state": self.state,
        }


# -------------------------
# Agent implementations are in separate files:
# - io_agent.py      -> IOAgent
# - pv_agent.py      -> PVAgent
# - battery_agent.py -> BatteryAgent
# - grid_agent.py    -> GridAgent
# -------------------------

