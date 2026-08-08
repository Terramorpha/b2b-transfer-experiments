"""Branched-rollout comparison with per-term reward decomposition.

Two INDEPENDENT trajectories from the SAME initial state (fresh identical
envs, deterministic EnergyPlus from Jan 1): one driven by the learned policy,
one by the energy-tuned lookup RBC. Each controller acts on-policy in its own
branch -- no counterfactual queries, so neither controller is ever evaluated
on states it wouldn't itself visit.

Per step and per branch, records the emed reward SPLIT into its two terms
(recomputed from the flat observation using the env's own reward constants,
and verified against the env-reported reward at every step):

    comfort_term = temp_penalty / tau_T      (mean sq. deviation, reward zones)
    energy_term  = w_E * (elec + gas) / tau_E

plus zone temperatures, targets, outdoor temp, and the full action vector.

    python scripts/rollout_branch_compare.py --building-type OfficeMedium \
        --index 0 --steps 2016 \
        --checkpoint runs/emed_fromscratch_amorpheus_y2/model_s0.eqx

Writes data/branch_compare_<bt>_<idx>.npz for the figures.
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("vendor/morel", "vendor/Building2Building", "scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.environ.setdefault("WANDB_MODE", "disabled")

import _pin_dataset  # noqa: F401
import numpy as np

import building2building as b2b
from baselines.controllers import (
    UnitaryHvacPolicy, UnitaryHvacConfig, AirLoopPolicy, AirLoopConfig,
)
from building2building.data.registry import get_registry
from morel_amorpheus import amorpheus_policy
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, load_model
from morel.morphology import trivial_morphology
from morel_b2b import ScalarOnlyInfo
from evaluate import b2b_source_morphology

TUNED_DIR = os.path.join(REPO, "data", "rbc_emed_tuned_pertypecz")


def _lookup_params(bt, cz):
    hits = glob.glob(os.path.join(TUNED_DIR, f"{bt}_cz{cz}", "*.json"))
    assert len(hits) == 1, f"no unique tuned config for ({bt}, cz{cz}): {hits}"
    return json.load(open(hits[0]))["params"]


class TermRecorder:
    """Recompute the emed reward decomposition from the FLAT observation,
    using the constants of the env's own NormalizedDeadbandReward. Verified:
    comfort_term + energy_term must equal -reward at every step."""

    def __init__(self, env):
        rf = env.unwrapped.reward_fn
        names = list(env.metadata["observation_names"])
        self.tau_T, self.tau_E = float(rf.tau_T), float(rf.tau_E)
        self.w_E = float(rf.energy_weight)
        self.zones = list(rf.controlled_zones)
        self.task_config = rf.task_config
        low = [n.lower() for n in names]
        self.i_elec = [i for i, n in enumerate(low) if "electricity" in n]
        self.i_gas = [i for i, n in enumerate(low)
                      if "natural_gas" in n or "natural gas" in n or n == "energy_gas"]
        assert self.i_elec and self.i_gas, f"energy obs not found in {names}"
        self.i_temp, self.i_tgt = {}, {}
        for z in self.zones:
            zl = z.lower()
            t = [i for i, n in enumerate(low)
                 if n.startswith("zone air temperature") and n.endswith(zl)]
            g = [i for i, n in enumerate(low)
                 if n.startswith("target_temperature") and n.endswith(zl)]
            assert len(t) == 1, f"zone temp col for {z}: {t}"
            self.i_temp[z] = t[0]
            self.i_tgt[z] = g[0] if len(g) == 1 else None
        self.i_out = [i for i, n in enumerate(low)
                      if "outdoor" in n and "temperature" in n]

    def terms(self, obs):
        o = np.asarray(obs, dtype=float)
        power = float(o[self.i_elec].sum() + o[self.i_gas].sum())
        err = 0.0
        for z in self.zones:
            tgt = (float(o[self.i_tgt[z]]) if self.i_tgt[z] is not None
                   else self.task_config.target_for_zone(z).occupied_c)
            err += (float(o[self.i_temp[z]]) - tgt) ** 2
        err /= len(self.zones)
        return err / self.tau_T, self.w_E * power / self.tau_E

    def snapshot(self, obs):
        o = np.asarray(obs, dtype=float)
        temps = np.array([o[self.i_temp[z]] for z in self.zones])
        tgts = np.array([o[self.i_tgt[z]] if self.i_tgt[z] is not None
                         else self.task_config.target_for_zone(z).occupied_c
                         for z in self.zones])
        out = float(o[self.i_out].mean()) if self.i_out else np.nan
        elec = float(o[self.i_elec].sum())
        gas = float(o[self.i_gas].sum())
        return temps, tgts, out, elec, gas


RUN_PERIOD = "full_year"  # set from --run-period in main()


def _make_env(bt, idx, split, task):
    return b2b.make_env(bt, split=split, index=idx, task=task,
                        run_period=RUN_PERIOD)


def run_policy_branch(bt, idx, split, task, ckpt, steps, clamp_oa_min=None):
    raw_env = _make_env(bt, idx, split, task)
    env = ScalarOnlyInfo(raw_env)
    source = b2b_source_morphology(env)
    lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
    sp = amorpheus_policy(load_model(ckpt, d_model=64, n_heads=4, n_layers=3))(
        AMORPHEUS_B2B_BRIDGE.apply(source))
    rec = TermRecorder(raw_env)
    an = [str(n).lower() for n in raw_env.metadata["action_names"]]
    oa_i = np.array([i for i, n in enumerate(an)
                     if "outdoor air controller" in n and "mass flow" in n])

    def act(o):
        a = np.asarray(source.join_actions(lens.join_actions(
            sp(*lens.split_observation(source.split_observation(o))))),
            dtype=np.float64)
        if clamp_oa_min is not None and oa_i.size:
            a[oa_i] = np.maximum(a[oa_i], clamp_oa_min)
        return a

    obs, _ = env.reset()
    out = _drive(env, rec, steps, act, obs)
    env.close()
    return out


def run_rbc_branch(bt, idx, split, task, steps, params=None):
    env = _make_env(bt, idx, split, task)
    is_vav = bt == "OfficeMedium"
    if params is None:
        cfg = AirLoopConfig() if is_vav else UnitaryHvacConfig()
    else:
        cfg = AirLoopConfig(**params) if is_vav else UnitaryHvacConfig(**params)
    pol = AirLoopPolicy(cfg) if is_vav else UnitaryHvacPolicy(cfg)
    pol.bind_env(env)
    rec = TermRecorder(env)
    obs, _ = env.reset()
    pol.reset()
    out = _drive(env, rec, steps,
                 lambda o: np.asarray(pol.predict(o)[0], dtype=np.float64), obs)
    env.close()
    return out


def _drive(env, rec, steps, act_fn, obs):
    R = {k: [] for k in ("reward", "comfort", "energy", "temps", "tgts",
                         "outdoor", "elec", "gas", "actions")}
    max_mismatch = 0.0
    for _t in range(steps):
        a = act_fn(obs)
        cT, cE = rec.terms(obs)
        temps, tgts, outdoor, elec, gas = rec.snapshot(obs)
        obs, r, term, trunc, _ = env.step(a)
        # decomposition check: b2b computes r from the NEXT structured obs?
        # No -- reward is computed from the obs the action produced. Verify
        # against the terms of the NEW obs instead (post-step state).
        nT, nE = rec.terms(obs)
        max_mismatch = max(max_mismatch, abs((-r) - (nT + nE)))
        R["reward"].append(float(r))
        R["comfort"].append(nT)
        R["energy"].append(nE)
        R["temps"].append(temps)
        R["tgts"].append(tgts)
        R["outdoor"].append(outdoor)
        R["elec"].append(elec)
        R["gas"].append(gas)
        R["actions"].append(a)
        if term or trunc:
            break
    out = {k: np.asarray(v) for k, v in R.items()}
    out["max_mismatch"] = max_mismatch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-type", default="OfficeMedium")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default="task_occ_emed")
    ap.add_argument("--steps", type=int, default=2016, help="prefix length (1 week)")
    ap.add_argument("--run-period", default="full_year",
                    choices=["full_year", "winter", "summer"],
                    help="b2b run-period preset; the prefix starts at its first day")
    ap.add_argument("--clamp-oa-min", type=float, default=None,
                    help="safety-layer projection: floor the policy's Outdoor "
                         "Air Controller mass-flow commands at this value "
                         "(kg/s), e.g. 1.37 = the design ventilation intake")
    ap.add_argument("--checkpoint",
                    default="runs/emed_fromscratch_amorpheus_y2/model_s0.eqx")
    ap.add_argument("--with-default-rbc", action="store_true",
                    help="also roll a third branch with the default-config RBC")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
            else os.path.join(REPO, args.checkpoint))
    global RUN_PERIOD
    RUN_PERIOD = args.run_period

    bid = get_registry().get_building_by_index(
        args.building_type, args.split, args.index).building_id
    cz = (None if args.building_type in b2b.TYPES_WITHOUT_CLIMATE_ZONE
          else b2b.get_climate_zone(args.building_type, bid))
    tuned_params = _lookup_params(args.building_type, cz)
    print(f"[branch-compare] {bid} cz{cz} task={args.task} "
          f"prefix={args.steps} steps", flush=True)

    branches = {"policy": run_policy_branch(
        args.building_type, args.index, args.split, args.task, ckpt, args.steps,
        clamp_oa_min=args.clamp_oa_min)}
    print("  policy branch done"
          + (f" (OA floored at {args.clamp_oa_min})" if args.clamp_oa_min else ""),
          flush=True)
    branches["tuned_rbc"] = run_rbc_branch(
        args.building_type, args.index, args.split, args.task, args.steps,
        params=tuned_params)
    print("  tuned-RBC branch done", flush=True)
    if args.with_default_rbc:
        branches["default_rbc"] = run_rbc_branch(
            args.building_type, args.index, args.split, args.task, args.steps)
        print("  default-RBC branch done", flush=True)

    # one env probe for names (any branch env would do; rebuild cheaply)
    env = _make_env(args.building_type, args.index, args.split, args.task)
    rec = TermRecorder(env)
    meta = dict(building_id=bid, climate_zone=cz, task=args.task,
                steps=args.steps, checkpoint=args.checkpoint,
                zones=np.asarray(rec.zones),
                action_names=np.asarray(list(env.metadata["action_names"])),
                tau_T=rec.tau_T, tau_E=rec.tau_E, w_E=rec.w_E)
    env.close()

    suffix = "" if args.run_period == "full_year" else f"_{args.run_period}"
    out = args.out or os.path.join(
        REPO, "data",
        f"branch_compare_{args.building_type}_{args.split}{args.index}{suffix}.npz")
    flat = dict(meta)
    for bname, R in branches.items():
        for k, v in R.items():
            flat[f"{bname}/{k}"] = v
    np.savez(out, **flat)

    print(f"\n=== prefix totals ({args.steps} steps) ===")
    for bname, R in branches.items():
        print(f"  {bname:12s} return {R['reward'].sum():10.2f} | "
              f"comfort {R['comfort'].sum():10.2f} | "
              f"energy {R['energy'].sum():10.2f} | "
              f"elec {R['elec'].sum():12.0f} | gas {R['gas'].sum():12.0f} | "
              f"decomp check max|Δ| {R['max_mismatch']:.2e}")
    print(f"[done] {out}\nBRANCH_COMPARE_DONE")


if __name__ == "__main__":
    main()
