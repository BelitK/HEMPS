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

        if "critical" in content.lower():
            logger.critical("[%s] Critical issue detected in message: %s", agent_name, content)
            print(f"[{agent_name}] CRITICAL issue detected in message: {content}")
            neighbor_agents = self.neighbors()
            #
            for neighbor in neighbor_agents:
                if 'critical' in neighbor.name.lower():
                    alert_message = f"ALERT from {agent_name}: Critical issue detected."
                    self.send_message(neighbor, alert_message)
                    logger.info("[%s] Sent alert to neighbor %s", agent_name, neighbor)
                    print(f"[{agent_name}] Sent alert to neighbor {neighbor}")

        