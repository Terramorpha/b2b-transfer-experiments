"""Roll the policy; at every step also ask RBC what it would command in that state.

The env is driven by the POLICY, so this is a counterfactual comparison on the policy's
own state distribution: "given exactly this situation, what would the reactive baseline
have done?" Records supply-air-temp setpoint (policy vs RBC), fan flow, task setpoint,
and zone temperature. Writes data/policy_vs_rbc_rollout.npz for the figure.

Uses the correct reactive controller per type (AirLoopPolicy for OfficeMedium/VAV,
UnitaryHvacPolicy otherwise).
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
import numpy as np

from baselines.controllers import AirLoopPolicy, UnitaryHvacPolicy
from building2building.data.registry import get_registry
from morel_amorpheus import amorpheus_policy
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, load_model
from morel.morphology import trivial_morphology
from evaluate import _b2b_factory_impl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/transfer4_persistent_long/model_s0.eqx")
    ap.add_argument("--building-type", default="OfficeSmall")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2016)   # 7 days at 12 steps/h
    ap.add_argument("--out", default="data/policy_vs_rbc_rollout.npz")
    args = ap.parse_args()
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(REPO, args.checkpoint)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    bid = get_registry().get_building_by_index(args.building_type, "test", args.index).building_id
    env, source = _b2b_factory_impl(args.building_type, args.index, "test", "task_occ_e0", "full_year")
    lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
    step_policy = amorpheus_policy(
        load_model(ckpt, d_model=64, n_heads=4, n_layers=3)
    )(AMORPHEUS_B2B_BRIDGE.apply(source))
    rbc = AirLoopPolicy() if args.building_type == "OfficeMedium" else UnitaryHvacPolicy()
    rbc.bind_env(env)

    an = list(env.metadata["action_names"]); on = list(env.metadata["observation_names"])
    sat_i = [i for i, n in enumerate(an) if "temp setpoint schedule" in n.lower()]
    fan_i = [i for i, n in enumerate(an) if "mass flow rate" in n.lower()]
    tgt_i = [i for i, n in enumerate(on) if n.startswith("target_temperature")]
    tz_i = [i for i, n in enumerate(on) if n.upper().startswith("ZONE AIR TEMPERATURE")]

    raw, _ = env.reset(); rbc.reset()
    rec = {k: [] for k in ("p_sat", "r_sat", "p_fan", "r_fan", "tgt", "tz", "rew")}
    for _t in range(args.steps):
        tca, tpa = step_policy(*lens.split_observation(source.split_observation(raw)))
        a_pol = np.asarray(source.join_actions(lens.join_actions((tca, tpa))), dtype=np.float64)
        a_rbc = np.asarray(rbc.predict(raw)[0], dtype=np.float64)  # counterfactual
        o = np.asarray(raw, dtype=float)
        rec["p_sat"].append(a_pol[sat_i].mean()); rec["r_sat"].append(a_rbc[sat_i].mean())
        rec["p_fan"].append(a_pol[fan_i].mean()); rec["r_fan"].append(a_rbc[fan_i].mean())
        rec["tgt"].append(o[tgt_i].mean()); rec["tz"].append(o[tz_i].mean())
        raw, r, term, trunc, _ = env.step(a_pol)
        rec["rew"].append(float(r))
        if term or trunc:
            break
    env.close()

    d = {k: np.asarray(v) for k, v in rec.items()}
    np.savez(out, building_id=bid, checkpoint=os.path.basename(os.path.dirname(ckpt)), **d)
    ps, rs = d["p_sat"], d["r_sat"]
    print(f"{bid}: {len(ps)} steps | mean(policy-RBC) SAT = {(ps-rs).mean():+.2f} C | "
          f"policy fan span {d['p_fan'].max()-d['p_fan'].min():.2f} vs RBC "
          f"{d['r_fan'].max()-d['r_fan'].min():.2f}", flush=True)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
