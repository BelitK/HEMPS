import json
import time

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


def topo_to_agraph(topology: dict):
    nodes_raw = topology.get("nodes", []) or []
    edges_raw = topology.get("edges", []) or []

    # Use agent name as node id (matches your edge format)
    nodes: list[Node] = []
    for n in nodes_raw:
        name = n.get("name")
        if not name:
            continue
        title = f'ID: {n.get("id")}\n{n.get("persona", "")}'
        nodes.append(Node(id=name, label=name, title=title))

    edges: list[Edge] = []
    for e in edges_raw:
        src = e.get("from")
        dst = e.get("to")
        if not src or not dst:
            continue
        state = e.get("state", "NORMAL")
        edges.append(Edge(source=src, target=dst, label=state))

    return nodes, edges


def llm_trigger(prompt: str) -> str:
    resp = requests.post(
        f"{LLM_BASE}/llm/trigger",
        json={"prompt": prompt},
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
    st.subheader("Refresh")

tabs = st.tabs(["Topology", "LLM Console"])

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

    nodes, edges = topo_to_agraph(topo)

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
            st.json(selected)

    with col2:
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

    # Session state for polling
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
                run_id = llm_trigger(prompt.strip())
                st.session_state.llm_run_id = run_id
                st.session_state.llm_last_result = {"status": "queued"}
                st.info(f"Run started: {run_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to trigger LLM at {LLM_BASE}: {e}")
                st.stop()

    run_id = st.session_state.llm_run_id
    if run_id:
        # Poll once per rerun
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
            if "wall_s" in result:
                st.caption(f"Wall time: {result['wall_s']:.3f}s")

        else:
            st.error(result.get("error", "Unknown error"))
