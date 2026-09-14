"""Reproduce every experiment behind a figure in the mémoire, as a dependency DAG.

Each task is one command from the reproduction manifest (RBC baselines and
tuning campaigns, BC seed, the 11 training runs, the 10 full-year evals, the
4 diagnostic rollouts, then the figure driver). Tasks run as subprocesses
from the repo root, each with its own log under logs/, scheduled by pools
so the box is neither idle nor thrashed:

    train  3 concurrent 20-env training runs (each ~4-6 cores)
    eval   1 at a time (each uses 20 worker processes)
    rbc    Optuna tuning: 24 slots (per-building tuners weigh 1, the emed
           campaign weighs 11 = its internal workers)
    diag   2 short diagnostic rollouts
    light  cheap glue (JSON joins, BC fit, figures)

A task whose outputs all exist is skipped, so the driver is resumable. A
failed task marks its dependents as blocked; everything else continues.

Run from the checkout whose data/ and runs/ should receive the results
(the reproduction worktree), with the guix profile exported:

    PY=/path/to/.venv/bin/python python scripts/reproduce_thesis.py [--dry-run] [--only T3 E4]

Old-box reference totals: training 39.7 h, evals ~83 h, RBC campaigns
9-20 h. Expected here: ~4-5 h wall.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("PY") or sys.executable
TYPES4 = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
EVAL_ENV = {"OMP_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}

POOL_CAP = {"train": 3, "eval": 1, "rbc": 16, "diag": 2, "light": 4}


@dataclass
class Task:
    name: str
    cmd: list[str] | str
    outputs: list[str]
    deps: list[str] = field(default_factory=list)
    pool: str = "light"
    weight: int = 1
    env: dict = field(default_factory=dict)
    note: str = ""


def _py(script, *args):
    return [PY, os.path.join("scripts", script), *map(str, args)]


def _train(script, *args):
    return _py(script, "--building-types", *TYPES4, "--n-envs", 5, *args)


def _eval(ckpt, tag, *extra):
    return _py("eval_fullyear_panel.py", "--checkpoint", ckpt, "--tag", tag, "--n-workers", 20, *extra)


def build_tasks(seeds: list[int] | None = None) -> list[Task]:
    """The base campaign (seed 0, as in the thesis) plus, for each extra seed in
    `seeds`, a seed-suffixed copy of every LEARNED chain (BC fit + warm-start,
    from-scratch runs, DAgger/SPE/critic/fine-tune) with its evals. RBC
    baselines, tuning campaigns, demo collection, diagnostics and figures are
    shared (seed-independent or single-seed by design)."""
    T: list[Task] = []
    add = T.append

    # ---- RBC baselines & tuning ------------------------------------------
    add(Task("R0_rbc_default_e0",
             _py("compute_rbc_emed_baseline.py", "--task", "task_occ_e0", "--n-workers", 20,
                 "--out", "data/rbc_fullyear_ourharness.json"),
             ["data/rbc_fullyear_ourharness.json"], pool="eval",
             note="default RBC, comfort, 20 test buildings (the comfort baseline JSON)"))
    add(Task("R1_pertype_bo_eval",
             _py("eval_fullyear_pertype_bo.py", "--n-workers", 20),
             ["data/pertype_bo_fullyear_eval.csv"], deps=["R0_rbc_default_e0"], pool="eval",
             note="paper's BO-tuned per-(type,CZ) RBC on the comfort test panel"))
    for bt in TYPES4:
        for i in range(5):
            add(Task(f"R2_tune_{bt}_{i}",
                     _py("tune_rbc_comfort.py", "--building-type", bt, "--index", i,
                         "--split", "train", "--n-trials", 25),
                     [f"data/rbc_tuned_configs/{bt}_train_{i}.json"], pool="rbc",
                     note="per-building comfort oracle (DAgger teacher)"))
    add(Task("R2_join_tuned_train_baseline",
             [PY, "-c",
              "import json,glob; d={}\n"
              "for p in sorted(glob.glob('data/rbc_tuned_configs/*_train_*.json')):\n"
              "    c=json.load(open(p)); d[c['building_id']]=c['tuned_return']\n"
              "json.dump(d, open('data/tuned_train_baseline.json','w'), indent=1); print(len(d))"],
             ["data/tuned_train_baseline.json"],
             deps=[f"R2_tune_{bt}_{i}" for bt in TYPES4 for i in range(5)]))
    add(Task("R3_rbc_default_emed",
             _py("compute_rbc_emed_baseline.py", "--n-workers", 20),
             ["data/rbc_emed_fullyear.json"], pool="eval"))
    add(Task("R4_tune_emed_pertypecz",
             _py("tune_rbc_emed_campaign.py", "--n-workers", 11),
             ["data/rbc_emed_tuned_pertypecz"], pool="rbc", weight=11,
             note="energy-tuned per-(type,CZ) RBC, 11 pairs x 25 trials"))
    add(Task("B1a_bc_collect", _py("bc_collect.py", "--n-workers", 20),
             ["data/bc_demos"], pool="eval"))

    # ---- Learned chains, one copy per seed ---------------------------------
    for seed in [0] + list(seeds or []):
        suf = "" if seed == 0 else f"_seed{seed}"       # task names, out dirs, eval tags
        S = str(seed)
        out = lambda d: f"runs/{d}{suf}"                 # noqa: E731
        ckpt = lambda d: f"{out(d)}/model_s{S}.eqx"      # noqa: E731

        add(Task(f"B1b_bc_fit{suf}",
                 _py("bc_fit.py", "--policy-steps", 20000, "--out", ckpt("transfer4_bc_long"),
                     *([] if seed == 0 else ["--seed", S])),
                 [ckpt("transfer4_bc_long")], deps=["B1a_bc_collect"], pool="light",
                 note="thesis: 20k Adam steps, lr 3e-4 (fit budget of the original _long run not logged)"))
        add(Task(f"T1_transfer4_persistent_long{suf}",
                 _train("train_allactive.py", "--task", "task_occ_e0", "--total-steps", 5376000,
                        "--n-steps", 672, "--lr", "5e-5", "--ent-coef", 0.01, "--gamma", 0.98,
                        "--seed", S, "--out", out("transfer4_persistent_long")),
                 [ckpt("transfer4_persistent_long")], pool="train",
                 note="from-scratch comfort (400 updates at the era's n_steps=672)"))
        add(Task(f"T2_transfer4_bcwarm{suf}",
                 _train("train_allactive.py", "--task", "task_occ_e0", "--years", 1, "--lr", "2e-5",
                        "--ent-coef", 0.0, "--gamma", 0.99, "--seed", S,
                        "--init-checkpoint", ckpt("transfer4_bc_long"),
                        "--out", out("transfer4_bcwarm")),
                 [ckpt("transfer4_bcwarm")], deps=[f"B1b_bc_fit{suf}"], pool="train"))
        add(Task(f"T3_dagger_y1{suf}",
                 _train("dagger_stream_amorpheus.py", "--task", "task_occ_e0", "--total-steps", 2000000,
                        "--n-steps", 512, "--updates-per-cycle", 32, "--minibatch", 256,
                        "--buffer-cap", 100000, "--beta-half-life", 200000,
                        "--checkpoint-every", 10, "--snapshot-every", 50, "--seed", S,
                        "--out", out("dagger_stream_e0_amorpheus")),
                 [ckpt("dagger_stream_e0_amorpheus")],
                 deps=["R2_join_tuned_train_baseline"], pool="train"))
        add(Task(f"T4_dagger_y2{suf}",
                 _train("dagger_stream_amorpheus.py", "--task", "task_occ_e0", "--total-steps", 2000000,
                        "--n-steps", 512, "--updates-per-cycle", 64, "--minibatch", 256,
                        "--buffer-cap", 500000, "--beta-half-life", 1000, "--seed", S,
                        "--init-checkpoint", ckpt("dagger_stream_e0_amorpheus"),
                        "--checkpoint-every", 10, "--snapshot-every", 50,
                        "--out", out("dagger_stream_e0_amorpheus_y2")),
                 [ckpt("dagger_stream_e0_amorpheus_y2")], deps=[f"T3_dagger_y1{suf}"], pool="train"))
        add(Task(f"T5_dagger_ppoft{suf}",
                 _train("train_allactive.py", "--task", "task_occ_e0", "--years", 1, "--ent-coef", 0,
                        "--lr", "2e-5", "--seed", S,
                        "--init-checkpoint", ckpt("dagger_stream_e0_amorpheus_y2"),
                        "--out", out("dagger_stream_e0_ppoft")),
                 [ckpt("dagger_stream_e0_ppoft")], deps=[f"T4_dagger_y2{suf}"], pool="train"))
        add(Task(f"T6_spe_y1{suf}",
                 _py("dagger_stream_amorpheus.py", "--bridge", "spe", "--seed", S,
                     "--out", out("dagger_stream_spe_amorpheus")),
                 [ckpt("dagger_stream_spe_amorpheus")],
                 deps=["R2_join_tuned_train_baseline"], pool="train",
                 note="all other flags at defaults = y1 protocol"))
        add(Task(f"T7_spe_y2{suf}",
                 _py("dagger_stream_amorpheus.py", "--bridge", "spe", "--seed", S,
                     "--init-checkpoint", ckpt("dagger_stream_spe_amorpheus"),
                     "--beta-half-life", 1000, "--out", out("dagger_stream_spe_amorpheus_y2")),
                 [ckpt("dagger_stream_spe_amorpheus_y2")], deps=[f"T6_spe_y1{suf}"], pool="train",
                 note="kept defaults updates-per-cycle 32 / buffer 100k (as the original run did)"))
        add(Task(f"T8_critic_year_emed{suf}",
                 _train("dagger_stream_amorpheus.py", "--critic-only", "--task", "task_occ_emed",
                        "--total-steps", 2000000, "--n-steps", 512, "--updates-per-cycle", 64,
                        "--minibatch", 256, "--buffer-cap", 500000, "--beta-half-life", 1000, "--seed", S,
                        "--init-checkpoint", ckpt("dagger_stream_e0_amorpheus_y2"),
                        "--checkpoint-every", 10, "--snapshot-every", 50,
                        "--out", out("dagger_stream_emed_criticyear")),
                 [ckpt("dagger_stream_emed_criticyear")], deps=[f"T4_dagger_y2{suf}"], pool="train"))
        add(Task(f"T9_emed_ppoft{suf}",
                 _train("train_allactive.py", "--task", "task_occ_emed", "--years", 1, "--ent-coef", 0,
                        "--lr", "2e-5", "--seed", S,
                        "--init-checkpoint", ckpt("dagger_stream_emed_criticyear"),
                        "--out", out("dagger_emed_ppoft")),
                 [ckpt("dagger_emed_ppoft")], deps=[f"T8_critic_year_emed{suf}"], pool="train"))
        add(Task(f"T10_emed_fromscratch{suf}",
                 _train("train_allactive.py", "--task", "task_occ_emed", "--total-steps", 5376000,
                        "--lr", "5e-5", "--ent-coef", 0.01, "--seed", S,
                        "--out", out("emed_fromscratch_amorpheus")),
                 [ckpt("emed_fromscratch_amorpheus")], pool="train"))
        add(Task(f"T11_emed_fromscratch_y2{suf}",
                 _train("train_allactive.py", "--task", "task_occ_emed", "--lr", "5e-05",
                        "--ent-coef", 0.01, "--total-steps", 5376000, "--seed", S,
                        "--init-checkpoint", ckpt("emed_fromscratch_amorpheus"),
                        "--out", out("emed_fromscratch_amorpheus_y2")),
                 [ckpt("emed_fromscratch_amorpheus_y2")],
                 deps=[f"T10_emed_fromscratch{suf}"], pool="train"))

        emed = ["--task", "task_occ_emed", "--baselines", "data/rbc_emed_fullyear.json"]
        evals = [
            ("E1_transfer4_final", "transfer4_persistent_long", "transfer4_final", [], [f"T1_transfer4_persistent_long{suf}", "R0_rbc_default_e0"]),
            ("E2_transfer4_bcwarm", "transfer4_bcwarm", "transfer4_bcwarm", [], [f"T2_transfer4_bcwarm{suf}", "R0_rbc_default_e0"]),
            ("E3_bcwarm_train", "transfer4_bcwarm", "bcwarm_train",
             ["--split", "train", "--baselines", "data/tuned_train_baseline.json"],
             [f"T2_transfer4_bcwarm{suf}", "R2_join_tuned_train_baseline"]),
            ("E4_dagger_y1_test", "dagger_stream_e0_amorpheus", "dagger_stream_y1_test", [], [f"T3_dagger_y1{suf}", "R0_rbc_default_e0"]),
            ("E5_dagger_y2_test", "dagger_stream_e0_amorpheus_y2", "dagger_stream_y2_test", [], [f"T4_dagger_y2{suf}", "R0_rbc_default_e0"]),
            ("E6_dagger_ppoft_test", "dagger_stream_e0_ppoft", "dagger_ppoft_test", [], [f"T5_dagger_ppoft{suf}", "R0_rbc_default_e0"]),
            ("E7_spe_y2_test", "dagger_stream_spe_amorpheus_y2", "spe_y2_test", ["--bridge", "spe"], [f"T7_spe_y2{suf}", "R0_rbc_default_e0"]),
            ("E8_emed_ppoft_test", "dagger_emed_ppoft", "emed_ppoft_test", emed, [f"T9_emed_ppoft{suf}", "R3_rbc_default_emed"]),
            ("E9_emed_fromscratch_test", "emed_fromscratch_amorpheus", "emed_fromscratch_test", emed, [f"T10_emed_fromscratch{suf}", "R3_rbc_default_emed"]),
        ]
        for name, rundir, tag, extra, deps in evals:
            add(Task(f"{name}{suf}", _eval(ckpt(rundir), f"{tag}{suf}", *extra),
                     [f"data/{tag}{suf}_fullyear_eval.csv"], deps=deps, pool="eval", env=EVAL_ENV))
        add(Task(f"E10_emed_fromscratch_5y_test{suf}",
                 " ".join(_eval(ckpt("emed_fromscratch_amorpheus_y2"), f"emed_fromscratch_y2_test{suf}", *emed))
                 + f" && mv data/emed_fromscratch_y2_test{suf}_fullyear_eval.csv data/emed_fromscratch_5y_test{suf}_fullyear_eval.csv",
                 [f"data/emed_fromscratch_5y_test{suf}_fullyear_eval.csv"],
                 deps=[f"T11_emed_fromscratch_y2{suf}", "R3_rbc_default_emed"], pool="eval", env=EVAL_ENV))

    # ---- Single-seed: energy-tuned panel (reads E8 seed 0), diagnostics, figures
    add(Task("R5_emedtuned_rbc_eval",
             _py("eval_fullyear_emedtuned_rbc.py", "--n-workers", 20),
             ["data/emedtuned_rbc_test_fullyear_eval.csv"],
             deps=["R4_tune_emed_pertypecz", "R3_rbc_default_emed", "E8_emed_ppoft_test"], pool="eval",
             note="also prints the re-judgment, which reads E8's CSV"))
    add(Task("D1_policy_vs_rbc", _py("rollout_policy_vs_rbc.py"),
             ["data/policy_vs_rbc_rollout.npz"], deps=["T1_transfer4_persistent_long"], pool="diag"))
    add(Task("D2_supply_symmetry",
             _py("check_supply_symmetry.py", "--steps", 1000,
                 "--checkpoint", "runs/dagger_stream_e0_amorpheus_y2/model_s0.eqx",
                 "--dump", "data/supply_symmetry_rollout.npz"),
             ["data/supply_symmetry_rollout.npz"], deps=["T4_dagger_y2"], pool="diag"))
    add(Task("D3_supply_symmetry_spe",
             _py("check_supply_symmetry.py", "--bridge", "spe", "--steps", 1000,
                 "--checkpoint", "runs/dagger_stream_spe_amorpheus_y2/model_s0.eqx",
                 "--dump", "data/supply_symmetry_rollout_spe.npz"),
             ["data/supply_symmetry_rollout_spe.npz"], deps=["T7_spe_y2"], pool="diag"))
    add(Task("D4_branch_compare",
             _py("rollout_branch_compare.py", "--building-type", "OfficeMedium", "--index", 0,
                 "--steps", 2016, "--with-default-rbc"),
             ["data/branch_compare_OfficeMedium_test0.npz"],
             deps=["T11_emed_fromscratch_y2", "R4_tune_emed_pertypecz"], pool="diag"))
    add(Task("FIG_make_thesis_figures", f"PY={PY} bash figures/make_thesis_figures.sh",
             ["figures_out/comfort_summary.pdf"],
             deps=[t.name for t in T if t.name[:1] in "ERD" and not t.name.endswith(tuple(f"_seed{s}" for s in (seeds or [])))],
             pool="light"))
    return T


def _outputs_exist(t: Task) -> bool:
    for o in t.outputs:
        p = os.path.join(REPO, o)
        if os.path.isdir(p):
            if not os.listdir(p):
                return False
        elif not os.path.isfile(p):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="task name prefixes to run (deps implied)")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="extra learner seeds: add a seed-suffixed copy of every learned chain")
    ap.add_argument("--pool-train", type=int, default=None, help="override the training pool size")
    ap.add_argument("--fresh", action="store_true",
                    help="delete every task's outputs first (a checkout carries the banked "
                         "results as tracked files; a reproduction must regenerate them)")
    args = ap.parse_args()

    if args.pool_train:
        POOL_CAP["train"] = args.pool_train
    tasks = {t.name: t for t in build_tasks(args.seeds)}
    if args.only:
        keep: set[str] = set()
        stack = [n for n in tasks if any(n.startswith(p) for p in args.only)]
        while stack:
            n = stack.pop()
            if n in keep:
                continue
            keep.add(n)
            stack.extend(tasks[n].deps)
        tasks = {n: t for n, t in tasks.items() if n in keep}

    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    master = open(os.path.join(REPO, "logs", "reproduce_thesis.log"), "a")
    if args.fresh and not args.dry_run:
        import shutil
        for t in tasks.values():
            for o in t.outputs:
                p = os.path.join(REPO, o)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)
        master.write(f"[{time.strftime('%F %T')}] FRESH: removed outputs of {len(tasks)} tasks\n")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        master.write(line + "\n"); master.flush()

    state = {n: ("done" if _outputs_exist(t) else "pending") for n, t in tasks.items()}
    if args.dry_run:
        for n, t in tasks.items():
            cmd = t.cmd if isinstance(t.cmd, str) else " ".join(t.cmd)
            print(f"{state[n]:8s} [{t.pool}:{t.weight}] {n}\n    deps={t.deps}\n    {cmd}")
        return

    log(f"REPRODUCE START: {len(tasks)} tasks, {sum(s == 'done' for s in state.values())} already done, PY={PY}")
    running: dict[str, tuple[subprocess.Popen, float]] = {}
    load = {p: 0 for p in POOL_CAP}
    base_env = {**os.environ, "PYTHONUNBUFFERED": "1", "WANDB_MODE": os.environ.get("WANDB_MODE", "offline")}

    while True:
        # reap
        for n in list(running):
            proc, t0 = running[n]
            rc = proc.poll()
            if rc is None:
                continue
            del running[n]
            t = tasks[n]; load[t.pool] -= t.weight
            ok = rc == 0 and _outputs_exist(t)
            state[n] = "done" if ok else "failed"
            log(f"{'DONE' if ok else 'FAILED'} {n} rc={rc} in {(time.time() - t0) / 60:.1f} min"
                + ("" if ok else f" (outputs present: {_outputs_exist(t)}) -> logs/{n}.log"))
        # block dependents of failures
        for n, t in tasks.items():
            if state[n] == "pending" and any(state[d] in ("failed", "blocked") for d in t.deps):
                state[n] = "blocked"; log(f"BLOCKED {n} (failed dependency)")
        # launch
        for n, t in tasks.items():
            if state[n] != "pending" or n in running:
                continue
            if not all(state[d] == "done" for d in t.deps):
                continue
            if load[t.pool] + t.weight > POOL_CAP[t.pool]:
                continue
            logf = open(os.path.join(REPO, "logs", f"{n}.log"), "a")
            cmd = t.cmd
            env = {**base_env, **t.env}
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                    shell=isinstance(cmd, str))
            running[n] = (proc, time.time()); load[t.pool] += t.weight; state[n] = "running"
            log(f"START {n} [{t.pool}] pid={proc.pid}")
        if not running and all(s != "pending" or True for s in state.values()):
            pending = [n for n, s in state.items() if s == "pending"]
            if not pending:
                break
            # pending but never launchable (deps blocked/failed handled above) -> nothing running & nothing launchable
            if not any(all(state[d] == "done" for d in tasks[n].deps) for n in pending):
                log(f"STUCK: {pending}"); break
        time.sleep(args.poll)

    summary = {s: [n for n, st in state.items() if st == s] for s in ("done", "failed", "blocked", "pending")}
    log("REPRODUCE END " + json.dumps({k: len(v) for k, v in summary.items()}))
    for k in ("failed", "blocked"):
        if summary[k]:
            log(f"{k}: {summary[k]}")
    json.dump(state, open(os.path.join(REPO, "logs", "reproduce_thesis_state.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
