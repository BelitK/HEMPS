from mango import Agent

# -------------------------
# Mango Agent
# -------------------------
class DynamicAgent(Agent):
    def __init__(self, name: str, persona: str):
        super().__init__()
        self.name = name
        self.persona = persona
        self.state = "active"

    def handle_message(self, content, meta):
        print(f"[{self.name}] {content}")

    def info(self):
        return {
            "name": self.name,
            "persona": self.persona,
            "usage": "This agent dynamically responds to messages based on its persona.",
        }
