from mango import Agent

# -------------------------
# Mango Agent
# -------------------------
class DynamicAgent(Agent):
    def __init__(self, name: str, persona: str, usage: str):
        super().__init__()
        self.name = name
        self.persona = persona
        self.state = "active"
        self.usage= "Not specified."

    def handle_message(self, content, meta):
        print(f"[{self.name}] {content}")

    def info(self):
        return {
            "name": self.name,
            "persona": self.persona,
            "usage": self.usage,
        }

class IOAgent(DynamicAgent):
    def __init__(self):
        super().__init__("io_agent", "Handles input and output operations for the llm.", "Manages data flow.")

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
    def __init__(self):
        super().__init__("battery_agent", "Monitors and reports battery status.")

    def handle_message(self, content, meta):
        if "battery" in content.lower():
            response = "The current battery level is 85%."
            print(f"[{self.name}] {response}")
        else:
            super().handle_message(content, meta)