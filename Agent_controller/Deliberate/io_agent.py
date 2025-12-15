import sys
import os
# Allow imports from src and project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import mango
from typing import Any, Dict
from env import UserRequest, RequestType, UserResponse

class IOAgent(mango.Agent):
    """
    IO Agent responsible for handling user interaction.
    Acts as the interface layer for the multi-agent system.
    """
    def __init__(self, dispatch_agent_addr=None):
        super().__init__()
        self.dispatch_agent_addr = dispatch_agent_addr
        # Context to be updated by Dispatch Agent updates
        self.system_context = {
            "battery_soc": 0.5,
            "current_price": 0.30,
            "schedule": "No schedule yet"
        }

    def handle_message(self, content, meta):
        if isinstance(content, UserRequest):
            self.handle_user_request(content, meta)
        
        # Future: Listen for context updates from DispatchAgent
        # elif isinstance(content, ContextUpdate):
        #     self.system_context.update(content)

    def handle_user_request(self, request: UserRequest, meta):
        sender = mango.sender_addr(meta)
        print(f"[IOAgent] Received Request: {request.type} - {request.message}")

        if request.type == RequestType.INFORM:
            # Handle Information/Command
            if not self.validate_input(request.message):
                response = UserResponse("Request rejected: Input seems invalid or unsafe.")
                self.schedule_instant_message(response, sender)
                return

            # TODO: serialized_intent = parse_intent(request.message)
            # self.schedule_instant_message(serialized_intent, self.dispatch_agent_addr)

            # Ack to user
            self.schedule_instant_message(UserResponse(f"Acknowledged. Processing: '{request.message}'"), sender)

        elif request.type == RequestType.EXPLAIN:
            # I return the data context. 
            # The actual NL generation happens at the higher-level controller/binding.
            
            # Bundles the state that clarifies WHY decisions were made.
            explanation_data = f"Context: {self.system_context}"
            self.schedule_instant_message(UserResponse(explanation_data), sender)

    def validate_input(self, text: str) -> bool:
        """
        Basic Input Verification: Checks for unsafe or nonsensical keywords.
        """
        unsafe_keywords = ["destroy", "hack", "infinite", "power usage -1"]
        if any(word in text.lower() for word in unsafe_keywords):
            return False
        return True
