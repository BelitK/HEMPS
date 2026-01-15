from .dynamic_agent import DynamicAgent

class test_Agent(DynamicAgent):
    TYPE = "test_agent"
    LABEL = "Test Agent"
    DEFAULT_PERSONA = "An agent for testing purposes."
    DEFAULT_USAGE = "Used to test agent functionalities."
    CAPABILITIES = ["test", "debug"]

    def handle_message(self, content, meta):
        print(f"[{self.name}] Test message received: {content}")
        # Implement test-specific logic here