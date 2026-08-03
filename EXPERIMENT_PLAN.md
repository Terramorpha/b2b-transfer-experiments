# Experiment Plan — DAgger + energy ladder (LIVING DOC)

**This is the source of truth for the current experimental campaign. Read it after any
context compression before acting. Update it whenever state changes.**

Last updated: 2026-07-31

## The plan (supersedes the stale grid in the thesis §1815)
1. Get the DAgger setup working. — **DONE** (built + smoke-tested)
2. DAgger on comfort only (`task_occ_e0`). — **RUNNING** (scoped validation run)
3. Full method ladder — **from-scratch → BC → BC-warm → DAgger(+PPO)** — on the
   **energy+comfort task `task_occ_emed`** (w_E=1.0). — **NOT STARTED** (launch after step 2 validates)

Architecture for the ladder: **ModuMorph** (variant `hn`) is the flagship student. Amorpheus = prior baseline.

## Locked decisions
- Task pair: `{task_occ_e0` (comfort), `task_occ_emed` (energy+comfort, w_E=1.0)}`. `ehigh` not planned.
- Energy teacher for BC/DAgger: **reuse the comfort-tuned per-building RBC configs as-is**
  (`data/rbc_tuned_configs/`) — NO re-tune pass. Rationale: imitation clones the pinned-OA teacher
  regardless; the RL fine-tune is what chases the energy win, so teacher energy-quality matters less.
- **DAgger on energy = DAgger + PPO fine-tune** (DAgger-warm), so it can deviate from the teacher.
- β schedule: β_i = 0.5^i, i from 0 (β_0 = 1.0 = pure expert = plain BC = it0).
- Shadow-mode expert: RBC steps every timestep on the student-induced obs; per-step Bernoulli(β)
  picks who executes; RBC action is ALWAYS the label. (Non-Markov expert → lose the no-regret
  theorem; β-schedule + observation-driven state is the justification.)

## Currently running (verify with /proc before trusting)
- **Comfort DAgger (step 2)**: CRASHED at it2 fit (OOM). STATUS (2026-08-01):
  it0 (=BC) DONE → `model_it0.eqx`; it1 (β=0.5) DONE → `model_it1.eqx`; it2 DATA collected
  (all 20 buildings in `demos/`), but **it2 fit OOM'd** (`LLVM compilation error: Cannot allocate
  memory`, exit 127) — the aggregated fit loaded 60 buildings on a 16GB box (it0 fit=20 ok, it1 fit=40 ok, it2 fit=60 FAIL).
  RESULTS (2026-08-02, comfort full-year, confirmed from CSVs):
  - it0 (BC): 7/20 vs DEFAULT RBC (Restaurant 4/5, OfficeSmall 3/5, Retail 0/5, OffMed 0/5); **0/20 vs BO**.
  - it1 (DAgger-1): 1/20 vs default; **0/20 vs BO**. Worse than BC — but CONFOUNDED (see below).
  - it2: not re-fit yet (dagger_fit now fixed; deprioritized given findings).
  THREE FINDINGS:
  1. Neither beats the honest BO bar (0/20). Expected — comfort is RBC-dominated.
  2. DAgger-1<BC is CONFOUNDED: loop uses fixed 4000 fit steps regardless of dataset → it0=200 steps/bldg,
     it1=100 steps/bldg (undertrained). MUST scale --policy-steps with #buffers before trusting DAgger-vs-BC.
  3. **OfficeMedium/VAV CATASTROPHICALLY BROKEN** (policy −68k to −240k vs RBC −10k, 7-16×). Same as old
     Amorpheus VAV BC (−155k) → persistent across architectures. CRITICAL: OfficeMedium is the ONLY type
     with the OA lever, so the energy-win thesis depends on VAV working. Must diagnose before energy ladder.
  CSVs: `data/dagger_e0_it{0,1}_fullyear_eval.csv`. **Checkpoints are NOT auto-evaluated.**
- **Per-type-BO baseline eval**: **DONE** → `data/pertype_bo_fullyear_eval.csv`. See Data points.

## Energy ladder — exact commands (launch AFTER comfort validates + after wiring `--task`)
All `--task task_occ_emed`:
- from-scratch: `train_allactive_modumorph.py --task task_occ_emed --out runs/emed_fromscratch --variant hn`
- BC + DAgger: `dagger_loop.py --task task_occ_emed --iterations 3 --n-buildings 5 --variant hn --out runs/dagger_emed`  (it0 = BC, it2 = DAgger)
- BC-warm: `train_allactive_modumorph.py --task task_occ_emed --init-checkpoint runs/dagger_emed/model_it0.eqx --ent-coef 0 --out runs/emed_bcwarm`
- DAgger-warm: `train_allactive_modumorph.py --task task_occ_emed --init-checkpoint runs/dagger_emed/model_it2.eqx --ent-coef 0 --out runs/emed_daggerwarm`

## Wiring for step 3 (comfort loop is DONE — safe to edit now)
- Add `--task` flag to `dagger_collect.py`, `dagger_fit.py`, `dagger_loop.py`, AND
  `eval_fullyear_panel_modumorph.py` (all hardcode `task_occ_e0`). Default `task_occ_e0`.
- `train_allactive_modumorph.py` ALREADY has `--task` + `--init-checkpoint` — no edit.
- **DECISION (2026-08-02): "same protocol as comfort" — do NOT change fit budget** (keep 4000/2000);
  DAgger-vs-BC stays confounded consistently across comfort+energy (accepted for comparability).
- **Energy needs its own RBC baseline**: run default RBC full-year on `task_occ_emed` ->
  `data/rbc_emed_fullyear.json` (comfort's `rbc_fullyear_ourharness.json` is e0 only). BO-on-emed = later.
- STATUS (2026-08-02): FULL 4-cell energy ladder LIVE. `--task` wired into dagger_collect/fit/loop +
  eval_fullyear_panel_modumorph (+ `--baselines` arg). OfficeMedium expected broken. Compat OK (d_context=32).
  - Driver: `scripts/run_energy_ladder.sh`, PID **125505**, master log `runs/energy_ladder.log`.
    Order: dagger_loop(emed) -> from-scratch -> bc-warm(init it0,ent0) -> dagger-warm(init it2,ent0) -> eval all 5.
  - Sub-logs: runs/dagger_emed_loop.log, runs/train_emed_{fromscratch,bcwarm,daggerwarm}.log, runs/eval_<tag>.log.
  - Energy RBC baseline: PID 125430, log data/rbc_emed_baseline.log -> `data/rbc_emed_fullyear.json`
    (default RBC on emed; emed returns ~3-15x more negative than e0). Eval tags: emed_{it0_BC,it2_DAgger,fromscratch,bcwarm,daggerwarm}.
  - Runs for DAYS. Monitor runs/energy_ladder.log. New scripts: run_energy_ladder.sh, compute_rbc_emed_baseline.py.
  - PROGRESS (2026-08-03): baseline DONE; STEP1 dagger_loop DONE (it0/it1/it2 all written — **memory fix
    VALIDATED, it2 fit completed no OOM**); STEP2 from-scratch DONE (400 updates); STEP3 bc-warm RUNNING
    (~50/400); STEP4 dagger-warm + STEP5 evals pending. No eval CSVs yet → no performance numbers yet.
  - INCREMENTAL EVAL (2026-08-03, user asked for bc-warm scores before the end): eval ready checkpoints
    as they land (n-workers 2 to coexist with training), sequential, vs data/rbc_emed_fullyear.json.
    Order: it0_BC, it2_DAgger, fromscratch (ready now) → bc-warm (when STEP3 done) → dagger-warm.
    Reports per-result. Tags emed_it0_BC/emed_it2_DAgger/emed_fromscratch/emed_bcwarm/emed_daggerwarm.
- COMFORT eval final (2026-08-02): it0 BC 7/20 vs default RBC (0/20 vs BO); it1 DAgger-1 1/20 (0/20 vs BO);
  OfficeMedium catastrophic both. See `data/dagger_e0_it{0,1}_fullyear_eval.csv`.

## Compat checks before the `-warm` cells
1. `d_context` match: `dagger_fit` must save ModuMorph with `d_model=64, d_context=32, n_heads=4, n_layers=3` or `--init-checkpoint` load fails.
2. `--ent-coef 0` on warm cells (trainer defaults 0.01 = from-scratch value; warm needs entropy off).

## MAJOR LEAD (2026-08-02): VAV breaks because the morphology prior is too weak
- CONFIRMED at code level: b2b builds `supply->zone` edges (`morphology.py:589-590` "hvac_system"/"controls")
  + zone-zone "thermal_adjacency" (`:721`); morel_b2b forwards them (`morphology.py:281`). BUT Amorpheus
  (`morel_amorpheus/model.py:20` "morel carries no edge topology yet") AND ModuMorph-hn BOTH ignore edges —
  pure all-to-all attention, zero graph prior.
- MECHANISM: `vav_supply` has EMPTY obs; it must set SAT/OA purely from attention-routed zone info.
  OfficeMedium = 3 loops x 5 zones. With no graph prior the supply node can't tell which zones are on its
  loop -> conflates all 15 -> wrong loop-wide SAT/OA -> catastrophe. Unitary zones are independent -> weak
  prior is fine. Explains "unitary ok, VAV explodes" and predicts multi-loop = worst.
- WHY IT MATTERS: this is the experiment that MOTIVATES the morphology prior AND justifies the spectral_pe
  adapter (currently "off-by-default", `model.py:20`). Fix = one-morphism change (compose `spectral_pe`, or
  edge-masked attention). Also a PREREQUISITE for the OfficeMedium energy win (VAV must work first).
- STATUS: HYPOTHESIS, not yet run. Confirm by: no-prior (have it, VAV broken) vs +spectral_pe vs +edge-attn
  on OfficeMedium. Co-factor: only 5 VAV train buildings (scarcity). Need to wire spectral_pe into modumorph.

## Key facts / insights (don't re-derive)
- **OA lever is VAV-only** (`vav_supply` node, `oa_mass_flow`). Only **OfficeMedium** among the 4 test
  types has it. 3 unitary types (Retail, Restaurant, OfficeSmall) have no OA actuator.
- **The RBC pins OA at 1.37 kg/s by design** (`air_loop.py:358`); the b2b authors intend the agent to
  learn OA modulation. → energy is where a learner can beat the RBC; comfort is where RBC is near-optimal.
- **BC/DAgger clone the pinned-OA RBC** → pure imitation inherits energy-blindness → the energy win
  requires the RL fine-tune phase (from-scratch / -warm), not imitation alone.
- **The b2b published-scores baseline = the BO-tuned RBC**, which is **3–5× stronger** than the default RBC.
  (RestaurantFastFood-3997: BO −2261.9 = published table, vs default −7701.4.) The old transfer results
  (incl. thesis §1718 "beats RBC 3/4") were vs the DEFAULT (weak) RBC — must name the baseline.
- Reward: `r = −(p_T/τ_T + w_E·p_E/τ_E)`; `p_T` = mean zone (T−T*)²  (pure quadratic, NO deadband —
  the `dT` param is vestigial/unused); `p_E = E_elec + E_gas`; τ per (type, CZ).
- Full-year = 105120 steps = the real eval; chunk (672) FLATTERS — never report chunk.
- **MEMORY (2026-08-01, CONFIRMED + FIXED): `dagger_fit` OOM was COMPILATION, not data.** It built a fresh
  `eqx.filter_jit` PER demo FILE (lines 168-171); the aggregated dir grows each iteration (it0=20 files,
  it1=40, it2=60), so compiled-executable count grew 80→160→240. Standalone re-fit confirmed: data load +
  full policy phase ran clean at ~8.3GB free; OOM hit at the CRITIC phase (2nd wave of 60 value-grad
  compilations stacking on the resident 60 policy compilations). BC never aggregates (fixed 20 files);
  PPO reuses one batched step (constant jit footprint) — that's why only DAgger hit it.
  **FIX APPLIED**: memoize jit fns by `id(m)` (files share one cached `m` per (bt,idx)) → caps compilations
  at #unique buildings, constant across iterations = the proven-good it0 level. Keep energy ladder at
  `--n-buildings 5` (20 buildings); if scaling to 10, may also need schema-dedup or separate-process critic.
- **RESOLVED (2026-08-01): the `unassigned observation slots: [...target_temperature...]` warning is
  COSMETIC, not a bug.** The network DOES observe the dynamic per-zone setpoint. b2b's native split
  leaves the `target_temperature <zone>` flat slot unassigned (→ the warning), but the morel adapter
  claims it by name: `_target_source_by_node` (morel_b2b/morphology.py:193) + `SplitObservation.__call__`
  (line 157-160) append the live scalar so each controlled zone decodes to `(zone_temp, target_temperature)`.
  Tested by `test_controlled_zone_observes_its_target_temperature` (+ a dynamic-target test). Do NOT re-chase.

## Files built this campaign
- `scripts/dagger_collect.py` — shadow-mode DAgger collector (ModuMorph student, β-Bernoulli, shadow-RBC labels via A_pinv).
- `scripts/dagger_fit.py` — ModuMorph two-phase refit over aggregated demos.
- `scripts/dagger_loop.py` — outer loop, β_i=0.5^i.
- `scripts/eval_fullyear_pertype_bo.py` — per-(type,CZ) BO-tuned RBC baseline eval.
- `bc_fit.py` LEFT UNTOUCHED (Amorpheus, paper reproduction).

## Data points so far
- **BO-tuned RBC vs default RBC (full-year, 5 test/type)**: BO wins 20/20. Means (BO / default):
  Restaurant −1572.5/−5935.6 (0.25), OfficeSmall −563.4/−1851.0 (0.32), Retail −3920.6/−7104.0 (0.56),
  OfficeMedium −8484.2/−10732.0 (0.80). BO is the honest bar; OfficeMedium is the closest margin
  (consistent with pinned-OA headroom). `data/pertype_bo_fullyear_eval.csv`.
- Amorpheus from-scratch (e0): 0/20 vs default RBC.
- ModuMorph from-scratch (e0): 0/20 full-year (confirmed correct).
- Amorpheus BC-warm (e0, generic teacher): 3/4 TYPES beat DEFAULT RBC (full 15-bldg tally never computed).
- Warm-start from per-building-tuned teacher (e0): 2/15 vs default (weak).
- Per-building Optuna tuning (comfort): large, building-dependent gains.

## Environment note (how to launch)
Runs need the guix profile + venv: `vendor/morel/.envrc` does `guix-load energyplus python python-numpy
python-pandas python-pytorch ty uv` + `export ENERGYPLUS_PATH=$GUIX_LOAD_PROFILE` + `source .venv/bin/activate`.
`guix-load` is NOT available in the bare non-interactive shell — launch via the profile-sourcing wrapper
(the pattern the DAgger subagent used). Use `nohup … & disown` (setsid unavailable). n-workers ≤ 3 (4 cores).
