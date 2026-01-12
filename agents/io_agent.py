from typing import Any, Callable, Dict, List, Optional
import json

from .dynamic_agent import DynamicAgent
from mango import sender_addr


class IOAgent(DynamicAgent):
    """
    IO Agent - The Bridge/Gateway between device agents and the Dispatcher.
    
    Responsibilities:
    1. Collect info() from all connected agents
    2. Aggregate and format data for the LLM/Dispatcher
    3. Handle incoming commands from Dispatcher and route to appropriate Agents
    """
    
    # Catalog metadata (required for agent factory & catalog)
    TYPE = "io"
    LABEL = "IO Agent"
    DEFAULT_PERSONA = "Central IO interface that aggregates agent information and routes commands."
    DEFAULT_USAGE = "Manages data flow between device agents and the dispatcher."
    CAPABILITIES = ["aggregate", "route", "register", "query"]
    
    def __init__(self, name: str = "io_agent", persona: str | None = None, usage: str | None = None):
        super().__init__(
            name=name,
            persona=persona or self.DEFAULT_PERSONA,
            usage=usage or self.DEFAULT_USAGE
        )
        # Registry of connected agents and their info callbacks
        self.connected_agents: Dict[str, Dict[str, Any]] = {}
        # Actions this agent can perform
        self.actions: Dict[str, Dict[str, Any]] = {}
        # State callback for dynamic state retrieval
        self._state_callback: Optional[Callable[[], Dict[str, Any]]] = None
        
        # Register default actions
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register the actions this IOAgent can perform."""
        self.register_action(
            action_name="get_all_agents_info",
            description="Retrieve information from all connected agents.",
            parameters={}
        )
        self.register_action(
            action_name="send_command",
            description="Send a command to a specific connected agent.",
            parameters={
                "target_agent": "Name of the agent to send command to",
                "command": "The command/action to execute",
                "args": "Arguments for the command (optional)"
            }
        )
        self.register_action(
            action_name="get_aggregated_state",
            description="Get the combined state of all connected agents.",
            parameters={}
        )
    
    def register_action(self, action_name: str, description: str, parameters: Dict[str, str]):
        """
        Register an action that this agent can perform.
        """
        self.actions[action_name] = {
            "description": description,
            "parameters": parameters
        }
    
    def register_agent(self, agent_name: str, agent_info: Dict[str, Any]):
        """
        Register a connected agent's info for aggregation.
        Called when an agent connects to the IOAgent.
        """
        self.connected_agents[agent_name] = agent_info
        print(f"[{self.name}] Registered agent: {agent_name}")
    
    def unregister_agent(self, agent_name: str):
        """Remove an agent from the registry."""
        if agent_name in self.connected_agents:
            del self.connected_agents[agent_name]
            print(f"[{self.name}] Unregistered agent: {agent_name}")
    
    def set_state_callback(self, callback: Callable[[], Dict[str, Any]]):
        """Set a callback to retrieve dynamic state."""
        self._state_callback = callback
    
    def get_aggregated_info(self) -> Dict[str, Any]:
        """
        Aggregate info from all connected agents.
        Returns the complete picture for the Dispatcher/LLM.
        """
        return {
            "io_agent": self.name,
            "connected_agents": self.connected_agents,
            "available_actions": self.actions,
            "agent_count": len(self.connected_agents)
        }
    
    def info(self) -> Dict[str, Any]:
        """
        Override parent info() to include IOAgent-specific data.
        This is what the LLM sees when querying this agent.
        """
        base_info = super().info()
        base_info.update({
            "type": "io_agent",
            "connected_agents": list(self.connected_agents.keys()),
            "available_actions": self.actions,
            "aggregated_data": self.get_aggregated_info()
        })
        return base_info
    
    def to_llm_format(self) -> str:
        """
        Convert the agent's info to a JSON string for LLM consumption.
        """
        return json.dumps(self.info(), indent=2, default=str)
    
    def handle_message(self, content, meta):
        """
        Handle incoming messages.
        - From Dispatcher: Route commands to appropriate agents
        - From Device Agents: Update their registered info
        """
        sender = sender_addr(meta)
        print(f"[{self.name}] Received from {sender}: {content}")
        
        # If content is a dict with 'type' field, handle accordingly
        if isinstance(content, dict):
            msg_type = content.get("type")
            
            if msg_type == "register":
                # Agent is registering itself
                agent_name = content.get("agent_name")
                agent_info = content.get("info", {})
                if agent_name:
                    self.register_agent(agent_name, agent_info)
            
            elif msg_type == "info_update":
                # Agent is updating its info
                agent_name = content.get("agent_name")
                agent_info = content.get("info", {})
                if agent_name:
                    self.connected_agents[agent_name] = agent_info
            
            elif msg_type == "command":
                # Command from Dispatcher to route
                target = content.get("target")
                command = content.get("command")
                args = content.get("args", {})
                print(f"[{self.name}] Routing command '{command}' to '{target}' with args: {args}")
                # TODO: Actually route the message to target agent via mango messaging
            
            elif msg_type == "query":
                # Request for aggregated info
                response = self.get_aggregated_info()
                self.schedule_instant_message(response, sender)
        
        else:
            # Default: just print (parent behavior)
            super().handle_message(content, meta)
