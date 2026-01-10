import json

import requests
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
from streamlit_autorefresh import st_autorefresh

# -------------------------
# Config
# -------------------------
MANGO_BASE = "http://127.0.0.1:8000"
LLM_BASE = "http://127.0.0.1:9001"


# -------------------------
# Helpers
# -------------------------
def fetch_topology() -> dict:
    resp = requests.get(f"{MANGO_BASE}/topology", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_notepads(session_id: str) -> dict:
    resp = requests.get(f"{LLM_BASE}/llm/notepads/{session_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def clear_notepads(session_id: str) -> dict:
    resp = requests.post(f"{LLM_BASE}/llm/notepads/{session_id}/clear", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _safe(s) -> str:
    return "" if s is None else str(s)


def topo_to_agraph(topology: dict):
    nodes_raw = topology.get("nodes", []) or []
    edges_raw = topology.get("edges", []) or []

    node_by_name: dict[str, dict] = {}
    for n in nodes_raw:
        name = n.get("name")
        if name:
            node_by_name[name] = n

    nodes: list[Node] = []
    for n in nodes_raw:
        name = n.get("name")
        if not name:
            continue

        node_id = _safe(n.get("id"))
        persona = _safe(n.get("persona"))
        agent_class = _safe(n.get("agent_class"))
        agent_module = _safe(n.get("agent_module"))
        agent_type = _safe(n.get("agent_type"))

        title_lines = [
            f"Name: {name}",
            f"ID: {node_id}",
        ]
        if agent_class:
            title_lines.append(f"Class: {agent_class}")
        if agent_module:
            title_lines.append(f"Module: {agent_module}")
        if agent_type and agent_type.lower() != "none":
            title_lines.append(f"Type: {agent_type}")
        if persona:
            title_lines.append("")
            title_lines.append("Persona:")
            title_lines.append(persona)

        title = "\n".join(title_lines)
        label = name

        nodes.append(Node(id=name, label=label, title=title, group=agent_class or None))

    edges: list[Edge] = []
    for e in edges_raw:
        src = e.get("from")
        dst = e.get("to")
        if not src or not dst:
            continue
        state = e.get("state", "NORMAL")
        edges.append(Edge(source=src, target=dst, label=state))

    return nodes, edges, node_by_name


def llm_trigger(prompt: str, session_id: str) -> str:
    resp = requests.post(
        f"{LLM_BASE}/llm/trigger",
        json={
            "prompt": prompt,
            "session_id": session_id,
            # keep default include_topology on server side; UI doesn't send it
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["run_id"]


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Mango Control Room", layout="wide")
st.title("Mango Control Room")

with st.sidebar:
    st.subheader("Endpoints")

    mango_in = st.text_input("Mango base URL", value=MANGO_BASE)
    llm_in = st.text_input("LLM base URL", value=LLM_BASE)

    if mango_in != MANGO_BASE:
        MANGO_BASE = mango_in
    if llm_in != LLM_BASE:
        LLM_BASE = llm_in

    st.divider()
    st.subheader("LLM Session")
    session_id = st.text_input("session_id", value="default", help="Notepads are stored per session_id.")

tabs = st.tabs(["Topology", "LLM Console", "Notepads"])


# -------------------------
# Tab: Topology
# -------------------------
with tabs[0]:
    st_autorefresh(interval=3000, key="topo_refresh")  # ms

    col1, col2 = st.columns([2, 1], gap="large")

    try:
        topo = fetch_topology()
    except Exception as e:
        st.error(f"Failed to fetch /topology from {MANGO_BASE}: {e}")
        st.stop()

    nodes, edges, node_by_name = topo_to_agraph(topo)

    config = Config(
        width="100%",
        height=700,
        directed=True,
        physics=True,
        hierarchical=False,
    )

    with col1:
        selected = agraph(nodes=nodes, edges=edges, config=config)

        if selected:
            st.subheader("Selected")

            selected_id = None
            if isinstance(selected, dict):
                if "id" in selected:
                    selected_id = selected.get("id")
                elif "node" in selected and isinstance(selected["node"], dict):
                    selected_id = selected["node"].get("id")
                elif "nodes" in selected and isinstance(selected["nodes"], list) and selected["nodes"]:
                    if isinstance(selected["nodes"][0], dict):
                        selected_id = selected["nodes"][0].get("id")
                    else:
                        selected_id = selected["nodes"][0]

            if selected_id and selected_id in node_by_name:
                st.json(node_by_name[selected_id])
            else:
                st.json(selected)

    with col2:
        st.subheader("Topology summary")

        class_counts: dict[str, int] = {}
        for n in topo.get("nodes", []) or []:
            cls = n.get("agent_class") or "Unknown"
            class_counts[cls] = class_counts.get(cls, 0) + 1

        st.write("Nodes by class:")
        st.json(class_counts)

        st.divider()
        st.subheader("Raw /topology JSON")
        st.code(json.dumps(topo, indent=2), language="json")


# -------------------------
# Tab: LLM Console
# -------------------------
with tabs[1]:
    st.subheader("Prompt the LLM (uses MCP tools on Mango server)")

    prompt = st.text_area(
        "Prompt",
        height=180,
        placeholder=(
            "Example: A critical event occurred. "
            "Investigate the system and propose mitigation steps."
        ),
    )

    if "llm_run_id" not in st.session_state:
        st.session_state.llm_run_id = None
    if "llm_last_result" not in st.session_state:
        st.session_state.llm_last_result = None

    col_a, col_b = st.columns([1, 2], gap="large")
    with col_a:
        send = st.button("Send", use_container_width=True)
    with col_b:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.llm_run_id = None
        st.session_state.llm_last_result = None
        st.rerun()

    if send:
        if not prompt.strip():
            st.warning("Write a prompt first.")
        else:
            try:
                run_id = llm_trigger(prompt.strip(), session_id=session_id.strip() or "default")
                st.session_state.llm_run_id = run_id
                st.session_state.llm_last_result = {"status": "queued"}
                st.info(f"Run started: {run_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to trigger LLM at {LLM_BASE}: {e}")
                st.stop()

    run_id = st.session_state.llm_run_id
    if run_id:
        try:
            resp = requests.get(f"{LLM_BASE}/llm/runs/{run_id}", timeout=10)
            resp.raise_for_status()
            result = resp.json()
            st.session_state.llm_last_result = result
        except Exception as e:
            st.session_state.llm_last_result = {"status": "error", "error": str(e)}

        result = st.session_state.llm_last_result or {}
        status = result.get("status", "unknown")

        if status in ("queued", "running"):
            st_autorefresh(interval=1000, key="llm_poll_refresh")
            st.info(f"LLM status: {status} (polling...)")

        elif status == "done":
            st.success("Done")
            st.write(result.get("reply", ""))

            if "wall_s" in result and result["wall_s"] is not None:
                st.caption(f"Wall time: {result['wall_s']:.3f}s")

            # Optional extra visibility
            with st.expander("Tool trace (latest)", expanded=False):
                st.json(result.get("tool_trace", {}))

            with st.expander("Model debug", expanded=False):
                st.json(result.get("model_debug", {}))

        else:
            st.error(result.get("error", "Unknown error"))


# -------------------------
# Tab: Notepads
# -------------------------
with tabs[2]:
    st.subheader("Session Notepads")
    st.caption("These are maintained server-side per session_id and injected into the model context.")

    st_autorefresh(interval=2000, key="notepad_refresh")

    col1, col2 = st.columns([1, 1], gap="large")

    with col2:
        if st.button("Clear notepads for this session", use_container_width=True):
            try:
                clear_notepads(session_id=session_id.strip() or "default")
                st.success("Cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear notepads: {e}")
                st.stop()

    try:
        pads = fetch_notepads(session_id=session_id.strip() or "default")
    except Exception as e:
        st.error(f"Failed to fetch notepads from {LLM_BASE}: {e}")
        st.stop()

    incident = pads.get("incident", []) or []
    memory = pads.get("memory", []) or []
    tool_trace = pads.get("tool_trace", {}) or {}

    with col1:
        st.markdown("### Incident notepad")
        if incident:
            st.code("\n".join(f"- {x}" for x in incident), language="text")
        else:
            st.info("No incident notes yet.")

        st.markdown("### Memory notepad")
        if memory:
            st.code("\n".join(f"- {x}" for x in memory), language="text")
        else:
            st.info("No memory notes yet.")

    with col2:
        st.markdown("### Tool trace summary")
        st.json(tool_trace)
