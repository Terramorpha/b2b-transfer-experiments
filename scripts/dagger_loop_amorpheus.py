"""AMORPHEUS DAgger outer loop: iterate shadow-mode collect + aggregated refit,
cloning the per-building-tuned RBCs into a single Amorpheus policy.

The Amorpheus counterpart of dagger_loop.py. beta_i = 0.5**i (beta_0 = 1.0 =
pure-expert = plain BC = it0). Each iteration ADDS iteration-tagged npz files to
the aggregated demo dir (never deletes), then refits a FRESH Amorpheus net on
ALL accumulated demos -> model_it{i}.eqx, which becomes the next student.

Collect + fit run as SUBPROCESSES (EnergyPlus / JAX process hygiene).

    python scripts/dagger_loop_amorpheus.py --iterations 3 \
        --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall \
        --n-buildings 5 --task task_occ_e0 --out runs/dagger_e0_amorpheus
"""
import argparse
import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")


def run(cmd, tail=None):
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={**os.environ})
    captured = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if tail is not None and any(mark in line for mark in tail):
            captured.append(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--building-types", nargs="+",
                    default=["RetailStandalone", "RestaurantFastFood",
                             "OfficeMedium", "OfficeSmall"])
    ap.add_argument("--n-buildings", type=int, default=5)
    ap.add_argument("--task", default="task_occ_e0",
                    help="b2b task preset threaded to collect+fit")
    ap.add_argument("--out", default="runs/dagger_e0_amorpheus",
                    help="output dir for model_it{i}.eqx and aggregated demos")
    ap.add_argument("--steps", type=int, default=105_120)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy-steps", type=int, default=4000)
    ap.add_argument("--value-steps", type=int, default=2000)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    demo_dir = os.path.join(out_dir, "demos")
    os.makedirs(demo_dir, exist_ok=True)

    py = sys.executable
    student = "none"  # no student at i=0

    print(f"[dagger-loop-amorpheus] {args.iterations} iterations | task={args.task} | "
          f"types={args.building_types} x {args.n_buildings} | "
          f"steps={args.steps} stride={args.stride} | out={out_dir}", flush=True)

    for i in range(args.iterations):
        beta = 0.5 ** i
        print(f"\n{'='*70}\n[iteration {i}] beta={beta} student={student}\n{'='*70}",
              flush=True)

        collect = [
            py, os.path.join(SCRIPTS, "dagger_collect_amorpheus.py"),
            "--student-checkpoint", student,
            "--beta", str(beta),
            "--task", args.task,
            "--building-types", *args.building_types,
            "--n-buildings", str(args.n_buildings),
            "--steps", str(args.steps),
            "--stride", str(args.stride),
            "--n-workers", str(args.n_workers),
            "--out-dir", demo_dir,
            "--iteration", str(i),
        ]
        run(collect, tail=["recon_err", "DAGGER_COLLECT_DONE"])

        n_files = len(glob.glob(os.path.join(demo_dir, "*.npz")))
        print(f"[iteration {i}] aggregated demo files: {n_files}", flush=True)

        model_out = os.path.join(out_dir, f"model_it{i}.eqx")
        fit = [
            py, os.path.join(SCRIPTS, "dagger_fit_amorpheus.py"),
            "--demos", demo_dir,
            "--out", model_out,
            "--task", args.task,
            "--gamma", str(args.gamma),
            "--policy-steps", str(args.policy_steps),
            "--value-steps", str(args.value_steps),
            "--minibatch", str(args.minibatch),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
        ]
        losses = run(fit, tail=["action MSE", "value MSE", "DAGGER_FIT_DONE"])

        print(f"\n[iteration {i}] SUMMARY: beta={beta}  demo_files={n_files}  "
              f"model={model_out}", flush=True)
        for ln in losses[-4:]:
            print(f"    {ln.strip()}", flush=True)

        student = model_out

    print(f"\n[dagger-loop-amorpheus] DONE. final model: "
          f"{os.path.join(out_dir, f'model_it{args.iterations-1}.eqx')}", flush=True)
    print("DAGGER_LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
