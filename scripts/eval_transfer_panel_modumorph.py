"""Eval a ModuMorph checkpoint across all 4 building types on held-out test
buildings vs RBC -- the ModuMorph counterpart of `eval_transfer_panel.py`.

    python scripts/eval_transfer_panel_modumorph.py --checkpoint runs/transfer_modumorph/model_s0.eqx

WARNING: this is the 672-step CHUNK panel -- a SMOKE TEST, not a thesis result.
Per `eval_fullyear_panel.py`'s docstring, the chunk (2.33 days of January, ~0.6%
of a year) systematically FLATTERS policies: they beat RBC here yet lose over the
full year. The "0/20" fig-1 number is FULL-YEAR; for a comparable ModuMorph
number use `eval_fullyear_panel_modumorph.py`, NOT this. Identical chunk protocol
to the Amorpheus panel (same 5 test buildings/type, same `data/baseline_chunk.csv`).
The ONLY differences: the ModuMorph
`load_model` + `MODUMORPH_B2B_BRIDGE` (commons_to_node only -- ModuMorph reads
geometry through the attribute pathway, not the observation). `amorpheus_policy`
is model-agnostic (it just conditions the model on each morphology and calls it),
and ModuMorph's ConditionedPolicy threads its own node_attrs, so it is reused
verbatim.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from building2building.data.registry import get_registry
from morel_amorpheus.policy import AmorpheusStepFn  # model-agnostic conditioned step fn
from morel_b2b_modumorph import MODUMORPH_B2B_BRIDGE, load_model
from morel.morphology import trivial_morphology
from evaluate import _b2b_factory_impl

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
TASK, RP, CHUNK, N = "task_occ_e0", "full_year", 672, 5


def _step_fn(model, morph, zero_attrs):
    """Conditioned deterministic step fn. `zero_attrs` sets each node's static
    attribute vector to zeros -- an inference-time knockout of the hypernetwork's
    geometry signal, so the policy falls back to its type-shared base I/O layers.
    If a ModuMorph trained WITH attributes collapses under this, its decisions
    provably depend on the attribute pathway."""
    cp = model.condition(morph)
    if zero_attrs:
        cp = eqx.tree_at(
            lambda c: c.node_attrs, cp,
            tuple(jnp.zeros_like(a) for a in cp.node_attrs))
    return AmorpheusStepFn(policy=cp, morphology=morph, deterministic=True,
                           key=jax.random.PRNGKey(0))


def policy_return(ckpt, bt, idx, d_context, zero_attrs=False):
    env, source = _b2b_factory_impl(bt, idx, "test", TASK, RP)
    lens = MODUMORPH_B2B_BRIDGE.apply(trivial_morphology(source))
    model = load_model(ckpt, d_model=64, d_context=d_context, n_heads=4, n_layers=3)
    sp = _step_fn(model, MODUMORPH_B2B_BRIDGE.apply(source), zero_attrs)
    raw, _ = env.reset()
    ret = 0.0
    for _t in range(CHUNK):
        tca, tpa = sp(*lens.split_observation(source.split_observation(raw)))
        a = np.asarray(source.join_actions(lens.join_actions((tca, tpa))), dtype=np.float64)
        raw, r, term, trunc, _ = env.step(a)
        ret += float(r)
        if term or trunc:
            break
    env.close()
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--d-context", type=int, default=32)
    ap.add_argument("--zero-attrs", action="store_true",
                    help="inference-time knockout: feed zeros as node attributes "
                         "(hypernetwork geometry signal removed)")
    args = ap.parse_args()
    if args.zero_attrs:
        print("[probe] ATTRIBUTE KNOCKOUT: node_attrs zeroed at inference", flush=True)
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(REPO, args.checkpoint)

    base = pd.read_csv(os.path.join(REPO, "data", "baseline_chunk.csv"))
    rbc = {r.building_id: r.baseline_return for r in base.itertuples()}
    reg = get_registry()

    summary = {}
    for bt in TYPES:
        print(f"\n=== {bt} (test) ===", flush=True)
        rows, wins = [], 0
        for idx in range(N):
            bid = reg.get_building_by_index(bt, "test", idx).building_id
            p = policy_return(ckpt, bt, idx, args.d_context, zero_attrs=args.zero_attrs)
            b = rbc.get(bid, float("nan"))
            w = p > b
            wins += bool(w)
            rows.append((p, b))
            print(f"  {bid:26s} policy {p:9.2f}  RBC {b:9.2f}  {'POLICY' if w else 'rbc'}", flush=True)
        pm = float(np.mean([r[0] for r in rows])); bm = float(np.mean([r[1] for r in rows]))
        summary[bt] = (pm, bm, wins)
        print(f"  -> mean policy {pm:.2f} vs RBC {bm:.2f} | beats RBC on {wins}/{N}", flush=True)

    print("\n=== SUMMARY (mean over 5 test buildings) ===")
    tot = 0
    for bt, (pm, bm, w) in summary.items():
        tot += w
        print(f"  {bt:22s} policy {pm:9.2f}  RBC {bm:9.2f}  wins {w}/{N}")
    print(f"  TOTAL wins: {tot}/{len(TYPES)*N}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
