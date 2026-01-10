from mango import Agent

from .forecasters import sinusoidal_prices


# -------------------------
# Mango Agent
# -------------------------
class DynamicAgent(Agent):
    # Catalog metadata (defaults)
    TYPE = "dynamic"
    LABEL = "Dynamic Agent"
    DEFAULT_PERSONA = "Generic dynamic agent."
    DEFAULT_USAGE = "None"
    CAPABILITIES: list[str] = []

    def __init__(self, name: str, persona: str | None = None, usage: str | None = None):
        super().__init__()
        self.name = name
        self.persona = persona or self.DEFAULT_PERSONA
        self.usage = usage or self.DEFAULT_USAGE
        self.state = "active"

    def handle_message(self, content, meta):
        print(f"[{self.name}] {content}")

        # Normalize to case-insensitive
        if "critical" in str(content).lower():
            self.state = "critical"
            print(f"[{self.name}] Agent {self.name} is now in CRITICAL state!")

    def info(self):
        return {
            "name": self.name,
            "persona": self.persona,
            "usage": self.usage,
            "state": self.state,
        }


class IOAgent(DynamicAgent):
    TYPE = "io"
    LABEL = "IO Agent"
    DEFAULT_PERSONA = "Handles input and output operations."
    DEFAULT_USAGE = "Manages data flow, acknowledgements, and IO-related messages."
    CAPABILITIES = ["input", "output", "acknowledge"]

    def handle_message(self, content, meta):
        c = str(content).lower()
        if "input" in c:
            print(f"[{self.name}] Input received and processed.")
        elif "output" in c:
            print(f"[{self.name}] Output generated successfully.")
        else:
            super().handle_message(content, meta)


class BatteryAgent(DynamicAgent):
    TYPE = "battery"
    LABEL = "Battery Agent"
    DEFAULT_PERSONA = "Monitors and reports battery status."
    DEFAULT_USAGE = "Responds to battery queries and reports energy levels."
    CAPABILITIES = ["battery", "energy", "status"]

    def handle_message(self, content, meta):
        if "battery" in str(content).lower():
            print(f"[{self.name}] The current battery level is 85%.")
        else:
            super().handle_message(content, meta)


class GridAgent(DynamicAgent):
    TYPE = "grid"
    LABEL = "Grid Agent"
    DEFAULT_PERSONA = "Manages grid interactions and pricing."
    DEFAULT_USAGE = "Provides grid price forecasts and responds to pricing requests."
    CAPABILITIES = ["grid", "pricing", "forecast"]

    def get_price_forecast(self, hours: int) -> list:
        time_values = list(range(hours))
        return sinusoidal_prices(
            time_values,
            base_price=50.0,
            amplitude=15.0,
            period=24.0,
        )

    def handle_message(self, content, meta):
        if "price forecast" in str(content).lower():
            forecast = self.get_price_forecast(24)
            print(f"[{self.name}] Next 24-hour price forecast: {forecast}")
        else:
            super().handle_message(content, meta)
