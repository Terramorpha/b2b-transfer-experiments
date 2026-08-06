"""Export a building's b2b morphology as graphviz DOT (for the thesis).

Sober style: nodes colour-coded by node type (fill only, short label), one
edge per node pair with its edge type(s) as the label. Prints edge-kind
counts (diagnostic for missing adjacency).

    python scripts/export_morphology_dot.py --building-type OfficeMedium \
        --split train --index 0 --out figures/morphology_officemedium_4001.dot
"""
import argparse, os, sys
from collections import Counter, defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("vendor/morel", "vendor/Building2Building", "scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.environ.setdefault("WANDB_MODE", "disabled")
import _pin_dataset  # noqa: F401
import building2building as b2b

FILL = {"vav_supply": "#DAF0F4", "vav_zone": "#DCE9FA", "vav_zone_no_cooling": "#DCE9FA",
        "unitary_zone": "#DCE9FA", "heating_zone": "#DCE9FA",
        "uncontrolled_zone": "#E9E9E7"}

def short(nid):
    s = nid.split(":")[-1].strip()
    return s.replace("B2B vav supply temp setpoint schedule", "supply")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-type", default="OfficeMedium")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="figures/morphology_graph.dot")
    ap.add_argument("--task", default="task_occ_e0")
    args = ap.parse_args()
    env = b2b.make_env(args.building_type, split=args.split, index=args.index,
                       task=args.task, run_period="full_year")
    m = env.metadata["morphology"]

    lines = ["graph morphology {",
             "  layout=neato; overlap=false; splines=true;",
             '  node [shape=box, fontname="sans-serif", fontsize=9, '
             'style=filled, color="#9A9A95"];',
             '  edge [fontname="sans-serif", fontsize=7, color="#9A9A95", '
             'fontcolor="#52514E"];']
    typed = set()
    for n in m.nodes:
        t = n.node_type.name
        if t not in FILL:
            continue
        typed.add(n.node_id)
        lines.append(f'  "{n.node_id}" [label="{short(n.node_id)}", '
                     f'fillcolor="{FILL[t]}"];')
    kinds_per_pair = defaultdict(list)
    kind_count = Counter()
    for e in m.edges:
        kind = getattr(e.edge_type, "value", str(e.edge_type))
        kind_count[kind] += 1
        if e.source in typed and e.target in typed:
            pair = tuple(sorted((e.source, e.target)))
            if kind not in kinds_per_pair[pair]:
                kinds_per_pair[pair].append(kind)
    for (a, bnode), kinds in sorted(kinds_per_pair.items()):
        lines.append(f'  "{a}" -- "{bnode}" [label="{", ".join(kinds)}"];')
    lines.append("}")
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    open(out, "w").write("\n".join(lines))
    print(f"wrote {out}: {len(typed)} nodes, {len(kinds_per_pair)} pair-edges")
    print("edge kinds in metadata morphology:", dict(kind_count))
    env.close()

if __name__ == "__main__":
    main()
