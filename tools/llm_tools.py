class LLMTool:
    def __init__(self, name, method, path, description, args_schema, name_rules=None):
        self.name = name
        self.method = method
        self.path = path
        self.description = description
        self.args_schema = args_schema
        self.name_rules = name_rules or []

    def LLM_tools():
        return {
            "tools": [
                {
                    "name": "get_tools",
                    "method": "GET",
                    "path": "/openapi.json",
                    "description": "Get documentation from server about available endpoints.",
                    "args_schema": {},
                }
            ]
        }