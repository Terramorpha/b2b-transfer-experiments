"""Thin CLI: evaluate a saved URMA policy on B2B buildings, scored
against the published reactive-controller baseline.

Wires the b2b/URMA stack into `morel.evaluation.run_evaluation`. The
heavy lifting (parallel dispatch, self-describing artifact store, skip-
existing) lives in the generic eval module; this script just supplies
factories, a morphism, and a policy factory. The lens pieces come from
`run_training` so evaluation uses byte-identical encoding/decoding to
the training pipeline.

Baseline comparison: B2B publishes per-building reactive-controller
returns (`building2building/scores/baseline_returns.csv`, the same
table the b2b benchmarks score against). For every completed episode we
report `normalized_score = episode_return / baseline_return` via
`b2b.scoring.compute_normalized_score`. Returns are negative costs, so
**lower is better**: score < 1 beats the baseline, > 1 loses to it.
Scores are only attached when the episode ran to its natural end —
a partial-episode return is not comparable to a full-period baseline —
and only for the **test split**, which is what the baseline table
covers (100 buildings per type, all tasks, all three run periods).

Usage:
    python evaluate.py --load runs/<id>/models/model_latest_jax --split test
    python evaluate.py --load <ckpt> --indices 800 801 802 803
    python evaluate.py --load <ckpt> --types OfficeSmall Warehouse --steps 5000
    # full-episode winter eval, baseline-scored:
    python evaluate.py --load <ckpt> --run-period winter --steps 0
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

os.environ.setdefault("WANDB_MODE", "offline")

import building2building as b2b
import numpy as np
from building2building.data.registry import get_registry
from building2building.scoring import compute_normalized_score

from morel.evaluation import Run, run_evaluation
from morel_b2b import ScalarOnlyInfo
from one_policy_to_run_them_all.morel import (
    FOOT_NODE_TYPE,
    JOINT_NODE_TYPE,
    count_nodes,
)
from run_training import RBC_TRACE_DIR, b2b_source_morphology, urma_bridge
from workbench import UrmaPolicyEvaluator, make_policy_and_params


# ---------------------------------------------------------------------------
# Top-level functions — picklable across multiprocessing workers when bound
# via `functools.partial` (nested closures would silently break under
# Pool's spawn-context pickling).
# ---------------------------------------------------------------------------


def _b2b_factory_impl(
    building_type: str, index: int, split: str, task: str, run_period: str,
):
    """Returns `(env, source_morphology)` for `drive_episode`. The
    `source_morphology` here is the post-normalization b2b morphology
    (run_training's lens, env-side half); the URMA-side morphism is
    applied inside `drive_episode`."""
    raw = b2b.make_env(
        building_type, split=split, index=index, task=task,
        run_period=run_period,
    )
    env = ScalarOnlyInfo(raw)
    return env, b2b_source_morphology(env)


def _urma_step_for_morph(target, *, policy_module, params):
    """Constructed inside the worker once `_urma_policy_factory_impl`
    has loaded the orbax checkpoint. Bound via `partial` so the worker
    can call `policy(target)` to get a step-callable for that morphology."""
    return UrmaPolicyEvaluator(
        target, policy_module, params, deterministic=True,
    )


def _urma_policy_factory_impl(ckpt_path: str, n_joints: int, n_feet: int):
    """Loads orbax inside the worker (called once per task in the
    pool); returns a `Policy: Morphology → StepFn` partial."""
    policy_module, params = make_policy_and_params(
        n_joints, n_feet, ckpt_path,
    )
    return functools.partial(
        _urma_step_for_morph,
        policy_module=policy_module, params=params,
    )


def _ref_node_counts(building_type: str, split: str, task: str, identity_one_hot: bool):
    """Probe one env to size the policy module's init dummy. Any single
    env of the same morphology shape gives the same answer."""
    raw = b2b.make_env(
        building_type, split=split, index=0, task=task,
    )
    env = ScalarOnlyInfo(raw)
    source = b2b_source_morphology(env)
    target = urma_bridge(identity_one_hot=identity_one_hot).apply(source)
    n_joints = count_nodes(target, JOINT_NODE_TYPE)
    n_feet = count_nodes(target, FOOT_NODE_TYPE)
    raw.close()
    return n_joints, n_feet


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def _local_rbc_return(row) -> float:
    """Baseline return from our own cached RBC trace (run_training's
    `rbc_reward_trace` cache) — the fallback for buildings the published
    table doesn't cover (it only has the test split). Same controller,
    locally replayed. NaN when no trace is cached."""
    path = os.path.join(
        RBC_TRACE_DIR, f"{row.building_id}_{row.task}_{row.run_period}.npy"
    )
    if not os.path.exists(path):
        return float("nan")
    return float(np.load(path).sum())


def attach_baseline_scores(df, *, max_steps: int | None):
    """Add `baseline_return` / `normalized_score` / `baseline_source`
    columns.

    The baseline return comes from B2B's published reactive-controller
    table (`baseline_source = "published"`, test-split buildings), or
    falls back to our locally-replayed RBC trace cache
    (`baseline_source = "local_rbc"`, e.g. training buildings).

    A row gets a score only when its episode is complete: either no step
    cap was set, or the env signaled done before hitting the cap. A
    step-capped partial return is not comparable to the full-period
    baseline, so those rows get NaN rather than a misleading number.
    """
    baseline_returns = []
    scores = []
    sources = []
    for row in df.itertuples():
        complete = max_steps is None or row.n_steps < max_steps
        baseline = float("nan")
        source = ""
        if complete:
            try:
                score = compute_normalized_score(
                    row.episode_return,
                    row.building_type,
                    row.task,
                    row.run_period,
                    row.building_id,
                )
                baseline = row.episode_return / score if score else float("nan")
                source = "published"
            except KeyError:
                baseline = _local_rbc_return(row)
                source = "local_rbc" if baseline == baseline else ""
        baseline_returns.append(baseline)
        scores.append(
            row.episode_return / baseline if baseline else float("nan")
        )
        sources.append(source)
    return df.assign(
        baseline_return=baseline_returns,
        normalized_score=scores,
        baseline_source=sources,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True, help="orbax checkpoint dir")
    parser.add_argument(
        "--types", nargs="+", default=["OfficeSmall"],
        help="building types to evaluate",
    )
    parser.add_argument(
        "--indices", nargs="+", type=int,
        default=[800, 801, 802, 803, 804, 805, 806, 807, 808, 809],
        help="building indices within --split",
    )
    parser.add_argument("--split", default="train", help="b2b split (train / test)")
    parser.add_argument("--task", default="task_const_e0")
    parser.add_argument(
        "--run-period", default="full_year",
        choices=["full_year", "winter", "summer"],
        help="simulation run period (baselines exist for all three)",
    )
    parser.add_argument(
        "--steps", type=int, default=10000,
        help="per-env step cap; 0 = full episode (required for "
             "baseline-normalized scores)",
    )
    parser.add_argument(
        "--store-dir", default="runs/eval_store",
        help="directory for self-describing pickles",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="parallel processes; 1 = in-process (debuggable)",
    )
    parser.add_argument(
        "--identity-one-hot", action="store_true",
        help="must match the loaded checkpoint's training config",
    )
    args = parser.parse_args()
    max_steps = args.steps if args.steps > 0 else None

    ckpt = os.path.abspath(args.load)
    policy_id = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))  # e.g. "s8zcewc4"

    # Probe a representative env (first type, first index) for joint/foot counts.
    n_joints, n_feet = _ref_node_counts(
        args.types[0], args.split, args.task, args.identity_one_hot,
    )
    print(f"[setup] policy_id={policy_id}, ref n_joints={n_joints}, n_feet={n_feet}", flush=True)

    morphism = urma_bridge(identity_one_hot=args.identity_one_hot)
    policy_factory = functools.partial(
        _urma_policy_factory_impl, ckpt, n_joints, n_feet,
    )

    # Resolve building ids up front — they key the baseline lookup and
    # make the stored artifacts self-describing.
    registry = get_registry()
    runs = [
        Run(
            metadata={
                "policy_id": policy_id,
                "ckpt": ckpt,
                "building_type": bt,
                "split": args.split,
                "index": idx,
                "building_id": registry.get_building_by_index(
                    bt, args.split, idx
                ).building_id,
                "task": args.task,
                "run_period": args.run_period,
                "steps": max_steps,
                "identity_one_hot": args.identity_one_hot,
            },
            factory=functools.partial(
                _b2b_factory_impl, bt, idx, args.split, args.task,
                args.run_period,
            ),
        )
        for bt in args.types
        for idx in args.indices
    ]

    print(
        f"[eval] {len(runs)} runs (types={args.types}, "
        f"split={args.split}, indices={args.indices}, "
        f"run_period={args.run_period}, steps={max_steps}); "
        f"n_workers={args.n_workers}, store={args.store_dir}",
        flush=True,
    )
    df = run_evaluation(
        runs,
        morphism=morphism,
        policy_factory=policy_factory,
        max_steps=max_steps,
        store_dir=args.store_dir,
        n_workers=args.n_workers,
    )

    df = attach_baseline_scores(df, max_steps=max_steps)

    # Per-(type, split) aggregates.
    print()
    print("=" * 100)
    cols = ["building_type", "split", "index", "building_id", "n_steps",
            "episode_return", "baseline_return", "normalized_score",
            "baseline_source"]
    print(df[cols].sort_values(["building_type", "split", "index"]).to_string(index=False))
    print("=" * 100)
    print()
    print("normalized_score = episode_return / RBC baseline return "
          "(returns are negative costs: < 1 beats the baseline)")
    if df["normalized_score"].isna().all():
        if max_steps is not None:
            print("note: scores omitted — episodes were step-capped; "
                  "rerun with --steps 0 for baseline-comparable returns")
        elif args.split == "train":
            print("note: scores omitted — the published table covers the "
                  "test split only, and no local RBC traces are cached "
                  "for these buildings (run_training.rbc_reward_trace "
                  "builds them)")
    print()
    print("aggregates (mean ± std across rows):")
    for (bt, sp), sub in df.groupby(["building_type", "split"]):
        ret = sub["episode_return"]
        score = sub["normalized_score"].dropna()
        score_part = (
            f"score={score.mean():.3f} ± {score.std():.3f} (n={len(score)})"
            if len(score) else "score=n/a"
        )
        print(
            f"  {bt:<22} split={sp:<6} n={len(sub):>3} "
            f"return={ret.mean():+.1f} ± {ret.std():.1f} {score_part}"
        )


if __name__ == "__main__":
    sys.exit(main())
