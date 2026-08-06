"""Prove the vav_supply symmetry collapse under Amorpheus (thesis claim).

The three OfficeMedium `vav_supply` nodes have empty observation spaces AND
trivial attribute spaces, so after encoding they are the SAME constant token.
A permutation-equivariant trunk attending over the same shared token set must
therefore emit IDENTICAL actions for all three loops -- at every step, for any
weights, regardless of history. This script verifies it on a trained checkpoint:
all three supply nodes' deterministic (supply_temp, oa_mass_flow) actions are
bit-identical.

    python scripts/check_supply_symmetry.py \
        [--checkpoint runs/transfer4_bcwarm/model_s0.eqx] [--index 0]
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("vendor/morel", "vendor/Building2Building", "scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.environ.setdefault("WANDB_MODE", "disabled")

import _pin_dataset  # noqa: F401
import numpy as np

from morel.morphology import trivial_morphology
from morel.check import type_check
from morel.encode import encode
from morel_amorpheus import amorpheus_policy
from morel_b2b_amorpheus import (AMORPHEUS_B2B_BRIDGE, AMORPHEUS_B2B_SPE_BRIDGE,
                                 AMORPHEUS_B2B_SPE_UNIVERSE, load_model, make_model)
from evaluate import _b2b_factory_impl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/transfer4_bcwarm/model_s0.eqx",
                    help="'none' => fresh random model (the claim is architectural)")
    ap.add_argument("--bridge", default="plain", choices=["plain", "spe"],
                    help="spe composes spectral_pe in front: symmetry should BREAK")
    ap.add_argument("--building-type", default="OfficeMedium")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=1,
                    help=">1: roll the env executing the policy and report the MAX "
                         "pairwise supply-action deviation over the whole rollout")
    args = ap.parse_args()
    ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
            else os.path.join(REPO, args.checkpoint))

    env, source = _b2b_factory_impl(
        args.building_type, args.index, args.split, "task_occ_e0", "full_year")
    bridge = AMORPHEUS_B2B_SPE_BRIDGE if args.bridge == "spe" else AMORPHEUS_B2B_BRIDGE
    universe = AMORPHEUS_B2B_SPE_UNIVERSE if args.bridge == "spe" else None
    m = bridge.apply(source)
    lens = bridge.apply(trivial_morphology(source))
    nodes = dict(m.nodes) if not hasattr(m.nodes, "items") else m.nodes
    spaces = {nid: nt.action_space for nid, nt in nodes.items()}
    if args.checkpoint.lower() == "none":
        model = make_model(d_model=64, n_heads=4, n_layers=3, seed=0, universe=universe)
    else:
        model = load_model(ckpt, d_model=64, n_heads=4, n_layers=3, universe=universe)
    sp = amorpheus_policy(model)(m)

    raw, _ = env.reset()
    max_dev = 0.0
    sup = {}
    for step in range(max(1, args.steps)):
        tca, tpa = sp(*lens.split_observation(source.split_observation(raw)))
        sup = {nid: np.asarray(encode(type_check(spaces[nid], v)), float)
               for nid, v in tpa.items() if "supply" in nid.lower()}
        vals = list(sup.values())
        dev = max((float(np.max(np.abs(vals[0] - v))) for v in vals[1:]), default=0.0)
        max_dev = max(max_dev, dev)
        if args.steps > 1:
            a = np.asarray(source.join_actions(lens.join_actions((tca, tpa))),
                           dtype=np.float32)
            raw, _r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    env.close()

    print(f"{len(sup)} vav_supply nodes, deterministic actions (last step):")
    for nid, a in sup.items():
        print(f"  {nid[:60]:60s} {np.round(a, 6)}")
    print(f"max pairwise |action difference| over {step + 1} step(s): {max_dev:.3e}")
    vals = list(sup.values())
    identical = max_dev < 1e-12
    print("IDENTICAL:", identical)
    if args.bridge == "plain":
        assert identical, "symmetry collapse NOT observed -- claim would be false!"
    else:
        assert not identical, "SPE bridge failed to break the supply symmetry!"


if __name__ == "__main__":
    main()
