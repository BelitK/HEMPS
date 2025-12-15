import json
import aiohttp
import asyncio
from typing import Any, Dict, Optional

class LLMInterface:
    """
    Interface for agents to communicate with LLM models.
    Handles serialization of state/context and parsing of LLM responses.
    Supports connecting to a hosted LLM service (e.g., local vLLM, Ollama, or OpenAI).
    """
    def __init__(self, model_name: str = "gpt-oss-20b", api_base: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key

    def serialize_state(self, state: Dict[str, Any]) -> str:
        """Serializes agent state to a structured string for LLM context."""
        try:
            return json.dumps(state, indent=2, default=str)
        except TypeError as e:
            return f"Error serializing state: {e}"

    def format_prompt(self, system_role: str, context: Dict[str, Any], user_input: str) -> str:
        """
        Formats the input into a prompt structure for the LLM.
        """
        prompt = f"""
### System Role
{system_role}

### Context / State
{self.serialize_state(context)}

### User Input
{user_input}

### Instructions
Provide your response in JSON format if possible, or clear natural language.
"""
        return prompt.strip()

    async def generate_response(self, prompt: str) -> str:
        """
        Sends the prompt to the hosted LLM API and returns the text response.
        Assumes OpenAI-compatible Chat Completion API by default.
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await resp.text()
                        return f"Error from LLM API: {resp.status} - {error_text}"
        except Exception as e:
            return f"Connection error to LLM Host: {e}"

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Attempts to parse the LLM response as JSON.
        Returns a dict. If parsing fails, returns a dict with the raw text.
        """
        try:
            # Attempt to find JSON block if wrapped in markdown code blocks
            clean_text = response_text
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                 clean_text = clean_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return {"text": response_text, "parsed": False}
