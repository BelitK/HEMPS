from typing import Any, Dict, List, Optional, Literal, Tuple
# -------------------------
# Registry to export topology
# -------------------------


class TopologyRegistry:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # edges store state: NORMAL | INACTIVE | BROKEN
        self.edges: List[Dict[str, Any]] = []
        self._edge_index: Dict[Tuple[str, str], int] = {}

    def export(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges)}

    def add_node(self, name: str, node_id: int, persona: str):
        self.nodes[name] = {"id": node_id, "name": name, "persona": persona}

    def upsert_edge(self, src: str, dst: str, state: str = "NORMAL") -> bool:
        """
        Insert edge if missing, otherwise update its state.
        Returns True if state changed or edge inserted.
        """
        key = (src, dst)
        if key in self._edge_index:
            idx = self._edge_index[key]
            old_state = self.edges[idx].get("state", "NORMAL")
            if old_state != state:
                self.edges[idx]["state"] = state
                return True
            return False

        self._edge_index[key] = len(self.edges)
        self.edges.append({"from": src, "to": dst, "state": state})
        return True

    def get_edge_state(self, src: str, dst: str) -> Optional[str]:
        key = (src, dst)
        if key not in self._edge_index:
            return None
        return self.edges[self._edge_index[key]].get("state", "NORMAL")

