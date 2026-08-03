"""DAgger outer loop: iterate collect (shadow-mode) + refit for a ModuMorph
student learning from the tuned RBC expert.

beta schedule: beta_i = 0.5**i (i from 0), so beta_0 = 1.0 = pure-expert =
plain BC, and later iterations mix in the student more and more.

Each iteration:
  1. COLLECT with the CURRENT student checkpoint (none at i=0) and beta_i, ADDING
     iteration-tagged npz files into the aggregated demo dir (never deleting
     prior iterations' files -- the dataset AGGREGATES).
  2. FIT a fresh ModuMorph net on ALL accumulated demos -> model_it{i}.eqx.
  3. That checkpoint becomes the next iteration's student.

Collect and fit run as SUBPROCESSES (clean EnergyPlus / JAX process hygiene).

    python scripts/dagger_loop.py --iterations 4 \
        --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall \
        --n-buildings 5 --variant hn --out runs/dagger
"""
import argparse
import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")


def run(cmd, tail=None):
    """Run a subprocess, streaming its output; return captured lines matching
    tail-of-interest markers. Raises on non-zero exit."""
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
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--building-types", nargs="+",
                    default=["RetailStandalone", "RestaurantFastFood",
                             "OfficeMedium", "OfficeSmall"])
    ap.add_argument("--n-buildings", type=int, default=5)
    ap.add_argument("--variant", default="hn", choices=["hn", "faithful", "blind"])
    ap.add_argument("--task", default="task_occ_e0",
                    help="b2b task preset threaded to collect+fit (comfort task_occ_e0, energy task_occ_emed)")
    ap.add_argument("--out", default="runs/dagger",
                    help="output dir for model_it{i}.eqx and aggregated demos")
    # collect pass-throughs
    ap.add_argument("--steps", type=int, default=105_120)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n-workers", type=int, default=3)
    # fit pass-throughs
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

    print(f"[dagger-loop] {args.iterations} iterations | variant={args.variant} | "
          f"types={args.building_types} x {args.n_buildings} | "
          f"steps={args.steps} stride={args.stride} | out={out_dir}", flush=True)

    for i in range(args.iterations):
        beta = 0.5 ** i
        print(f"\n{'='*70}\n[iteration {i}] beta={beta} student={student}\n{'='*70}",
              flush=True)

        # 1. COLLECT (aggregates: iteration-tagged files added to demo_dir)
        collect = [
            py, os.path.join(SCRIPTS, "dagger_collect.py"),
            "--student-checkpoint", student,
            "--beta", str(beta),
            "--variant", args.variant,
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

        # 2. FIT on ALL accumulated demos -> model_it{i}.eqx
        model_out = os.path.join(out_dir, f"model_it{i}.eqx")
        fit = [
            py, os.path.join(SCRIPTS, "dagger_fit.py"),
            "--demos", demo_dir,
            "--out", model_out,
            "--variant", args.variant,
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

        # 3. next student = this iteration's model
        student = model_out

    print(f"\n[dagger-loop] DONE. final model: "
          f"{os.path.join(out_dir, f'model_it{args.iterations-1}.eqx')}", flush=True)
    print("DAGGER_LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
