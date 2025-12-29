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
                    "name": "get_topology",
                    "method": "GET",
                    "path": "/topology",
                    "description": "Get current agents and edges. Edges include a state: NORMAL, INACTIVE, BROKEN.",
                    "args_schema": {},
                },
                {
                    "name": "get_agents",
                    "method": "GET",
                    "path": "/agents",
                    "description": "Get available agent templates for creation of new agents.",
                    "args_schema": {},
                },
                {
                    "name": "create_agent",
                    "method": "POST",
                    "path": "/agents",
                    "description": "Create a new agent and optionally connect it to existing agents.",
                    "args_schema": {
                        "name": "snake_case string, required",
                        "persona": "string (10-240 chars), required",
                        "state": "string, optional, one of NORMAL, INACTIVE, BROKEN",
                        "connect_to": "list of existing agent names, optional",
                    },
                    "name_rules": [
                        "Use a domain noun like house_battery, pv_panels, ev_charger",
                        "Do not use procedural names like create_agent_1, connect_agent_to_router",
                    ],
                },
                {
                    "name": "add_edge",
                    "method": "POST",
                    "path": "/edges",
                    "description": "Add an edge between two existing agents. Optionally add reverse edge. Adds with state NORMAL.",
                    "args_schema": {
                        "src": "existing agent name, required",
                        "dst": "existing agent name, required",
                        "bidirectional": "bool, optional (default false)",
                    },
                },
                {
                    "name": "deactivate_edge",
                    "method": "POST",
                    "path": "/edges/deactivate",
                    "description": "Delete an edge by deactivating it (set state to INACTIVE). Optionally bidirectional.",
                    "args_schema": {
                        "src": "existing agent name, required",
                        "dst": "existing agent name, required",
                        "bidirectional": "bool, optional (default false)",
                    },
                },
                {
                    "name": "activate_edge",
                    "method": "POST",
                    "path": "/edges/activate",
                    "description": "Activate an existing edge (set state to NORMAL). Optionally bidirectional.",
                    "args_schema": {
                        "src": "existing agent name, required",
                        "dst": "existing agent name, required",
                        "bidirectional": "bool, optional (default false)",
                    },
                },
            ]
        }