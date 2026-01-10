import asyncio
import time
import httpx
from mango import Agent

from .forecasters import sinusoidal_prices

# -------------------------
# Mango Agent
# -------------------------
class DynamicAgent(Agent):
    def __init__(self, name: str, persona: str, usage: str = "None"):
        super().__init__()
        self.name = name
        self.persona = persona
        self.usage = usage
        self.state = "active"

    def handle_message(self, content, meta):
        print(f"[{self.name}] {content}")
        if "Critical" in content:
            self.state = "critical"
            response = f"Agent {self.name} is now in CRITICAL state!"

    def info(self):
        return {
            "name": self.name,
            "persona": self.persona,
            "usage": self.usage,
        }
    


class IOAgent(DynamicAgent):
    def __init__(self, name: str, persona: str, usage: str = "None"):
        super().__init__(
            name=name,
            persona=persona,
            usage=usage
        )

    def handle_message(self, content, meta):
        if "input" in content.lower():
            response = "Input received and processed."
            print(f"[{self.name}] {response}")
        elif "output" in content.lower():
            response = "Output generated successfully."
            print(f"[{self.name}] {response}")
        else:
            super().handle_message(content, meta)


class BatteryAgent(DynamicAgent):
    def __init__(self, name: str, persona: str, usage: str = "None"):
        super().__init__(
            name=name,
            persona=persona,
            usage=usage
        )

    def handle_message(self, content, meta):
        if "battery" in content.lower():
            response = "The current battery level is 85%."
            print(f"[{self.name}] {response}")
        else:
            super().handle_message(content, meta)


class GridAgent(DynamicAgent):
    def __init__(self, name: str, persona: str, usage: str = "None"):
        super().__init__(
            name=name,
            persona=persona,
            usage=usage
        )

    def get_price_forecast(self, hours: int) -> list:
        time_values = list(range(hours))
        prices = sinusoidal_prices(
            time_values,
            base_price=50.0,
            amplitude=15.0,
            period=24.0
        )
        return prices

    def handle_message(self, content, meta):
        if "price forecast" in content.lower():
            forecast = self.get_price_forecast(24)
            response = f"Next 24-hour price forecast: {forecast}"
            print(f"[{self.name}] {response}")
        else:
            super().handle_message(content, meta)
