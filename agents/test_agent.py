from curses import meta
from .dynamic_agent import DynamicAgent
import logging
from mango import agent


logger = logging.getLogger(__name__)

class Router_Agent(DynamicAgent):
    TYPE = "Router"
    LABEL = "Test Agent"
    DEFAULT_PERSONA = "An agent for testing purposes."
    DEFAULT_USAGE = "Used to test agent functionalities."
    CAPABILITIES = ["test", "debug"]

    def handle_message(self, content, meta):
        agent_name = getattr(self, "name", "unknown")
        print(f"[{self.name}] Test message received: {content}")

        logger.info("[%s] Test message received: %s | meta=%s", agent_name, content, meta)
        # retain simple stdout for quick debugging
        print(f"[{agent_name}] Test message received: {content}")

        