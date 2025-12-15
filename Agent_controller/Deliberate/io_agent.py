import sys
import os
# Allow imports from src and project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import mango
import asyncio
from typing import Any, Dict
from env import UserRequest, RequestType, UserResponse
from Agent_controller.Common.llm_interface import LLMInterface

class IOAgent(mango.Agent):
    """
    IO Agent responsible for handling user interaction and LLM communication.
    Acts as the interface layer for the multi-agent system.
    """
    def __init__(self, dispatch_agent_addr):
        super().__init__()
        self.dispatch_agent_addr = dispatch_agent_addr
        self.llm = LLMInterface()
        # Context to be updated by Dispatch Agent updates
        self.system_context = {
            "battery_soc": 0.5,
            "current_price": 0.30,
            "schedule": "No schedule yet"
        }

    def handle_message(self, content, meta):
        if isinstance(content, UserRequest):
            self.handle_user_request(content, meta)
        # TODO: Handle updates from Dispatch Agent to update self.system_context

    def handle_user_request(self, request: UserRequest, meta):
        sender = mango.sender_addr(meta)
        print(f"[IOAgent] Received Request: {request.type} - {request.message}")

        if request.type == RequestType.INFORM:
            # Handle Information/Command
            if not self.guardrail_check_input(request.message):
                response = UserResponse("I'm sorry, I cannot process that request. It seems invalid or unsafe.")
                self.schedule_instant_message(response, sender)
                return

            # Ack to user
            self.schedule_instant_message(UserResponse(f"Understood. Accessing system to handle: '{request.message}'"), sender)

        elif request.type == RequestType.EXPLAIN:
            # Handle Explanation Request async
            asyncio.create_task(self.process_explanation(request, sender))

    async def process_explanation(self, request: UserRequest, sender):
        """Async handler for generating explanations"""
        prompt = self.llm.format_prompt(
            system_role="You are a helpful Home Energy Assistant. Explain decisions to the user.",
            context=self.system_context,
            user_input=request.message
        )
        
        # NOTE: mango agents are async, so we can await the network call
        explanation = await self.llm.generate_response(prompt)
        
        # Guardrail Output (Basic check)
        if "error" in explanation.lower() and "llm api" in explanation.lower():
             # Fallback if model is down
             explanation = "I encountered an issue connecting to the AI model. " + explanation

        self.schedule_instant_message(UserResponse(explanation), sender)

    def guardrail_check_input(self, text: str) -> bool:
        """
        Basic Input Guardrail: Checks for unsafe or nonsensical keywords.
        """
        unsafe_keywords = ["destroy", "hack", "infinite", "power usage -1"]
        if any(word in text.lower() for word in unsafe_keywords):
            return False
        return True
