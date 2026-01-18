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
        self.state = "NORMAL"

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

class StockAgent(DynamicAgent):
    TYPE = "stock"
    LABEL = "Stock Agent"
    DEFAULT_PERSONA = "Manages stock portfolio and trading."
    DEFAULT_USAGE = "Provides stock price forecasts and trading recommendations."
    CAPABILITIES = ["stocks", "trading", "forecast"]

    def get_price_forecast(self, hours: int) -> list:
        time_values = list(range(hours))
        return sinusoidal_prices(
            time_values,
            base_price=100.0,
            amplitude=20.0,
            period=24.0,
        )

    def handle_message(self, content, meta):
        if "price forecast" in str(content).lower():
            forecast = self.get_price_forecast(24)
            print(f"[{self.name}] Next 24-hour stock price forecast: {forecast}")
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

if __name__ == "__main__":
    # Simple test
    agents = [
        GridAgent(name="grid_agent"),
    ]

    test_messages = [
        ("Input data received", {}),
        ("What is the battery status?", {}),
        ("Provide price forecast", {}),
        ("This is a critical alert!", {}),
    ]

    for agent in agents:
        for msg, meta in test_messages:
            agent.handle_message(msg, meta)