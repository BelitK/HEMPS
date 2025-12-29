# pip install requests pyvis

import json
import requests
from pyvis.network import Network

TOPOLOGY_URL = "http://localhost:8000/topology"  # change to your endpoint
OUT_HTML = "mango_topology.html"

def load_topology():
    r = requests.get(TOPOLOGY_URL, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    topo = load_topology()

    # Build lookup by name, because edges reference names
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    name_set = set()
    for n in nodes:
        if "name" in n:
            name_set.add(str(n["name"]))

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # Add nodes
    for n in nodes:
        name = str(n.get("name"))
        title = json.dumps(n, indent=2, ensure_ascii=False)
        net.add_node(
            name,
            label=name,
            title=f"<pre>{title}</pre>",
            group="agent" if name != "router" else "router",
        )

    # Add edges
    for e in edges:
        src = str(e.get("from"))
        dst = str(e.get("to"))

        # If an edge references a node not in nodes[], add it so graph doesn't break
        if src not in name_set:
            net.add_node(src, label=src, title="<pre>{}</pre>", group="unknown")
            name_set.add(src)
        if dst not in name_set:
            net.add_node(dst, label=dst, title="<pre>{}</pre>", group="unknown")
            name_set.add(dst)

        label = str(e.get("state", ""))  # NORMAL, etc.
        net.add_edge(src, dst, label=label, arrows="to")

    # Optional: some UI controls (physics/layout tweaks) in the HTML
    net.show_buttons(filter_=["physics", "layout"])

    net.write_html(OUT_HTML, open_browser=False, notebook=False)
    print(f"Wrote {OUT_HTML} (open it in your browser)")

if __name__ == "__main__":
    main()
