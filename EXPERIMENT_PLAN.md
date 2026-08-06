# Experiment Plan — DAgger + energy ladder (LIVING DOC)

## 📌 USER'S PLAN (2026-08-06 late — the governing sequence):
1. BO energy&comfort-tuning of the ORACLES (per-building).  ⚠️ DISCREPANCY FLAGGED: what is ACTUALLY
   running is step 2's per-(type,CZ) campaign (PID 9716, 11 pairs on TRAIN instances → test lookup).
   A per-BUILDING emed oracle campaign is NOT running and NOT yet launched — awaiting user confirmation
   whether it's wanted (would be 20 train bldgs × 25 trials ≈ 15-20h; use = emed-DAgger teachers and/or
   train-side oracle bar). If confirmed: queue AFTER the current campaign.
2. BO energy&comfort per-(type,CZ) TEST baselines ← THIS is the running campaign; on completion build the
   lookup-apply eval → the energy-tuned RBC bar → re-judge the 13/20.
3. Meanwhile user WRITES "why does OfficeMedium behave so weirdly?" — assets ready for them:
   check_supply_symmetry.py (0.0 deviation over 500 steps, trained ckpt; --bridge spe flips it),
   figures/morphology_officemedium_4001.dot (fixed graph, 64 pair-edges), the teacher-noise/conditional-mean
   mechanism (+ Jensen argument), y2 VAV numbers (0.23x comfort / 0.80x emed), SPE inversion hypothesis.
4. Then user writes the e+c (emed) RESULTS section — needs: energy-tuned bar (step 2), decomposition eval
   (energy-only vs comfort-only terms — APPROVED, queued), from-scratch eval, final emed figure.
REMEMBER: never draft prose; tuned/best-of-two is the only bar; waiters armed on campaign+from-scratch;
SPE-DAgger 2yr comfort ablation queued for free box; minergym adjacency fix UNCOMMITTED (push round pending:
minergym + morel submodule + b2b-transfer scripts/figures); MLP rerun likely cut.

## 🔧 DEPENDENCY MODEL MIGRATION (user directive 2026-08-06): kill submodules + hand-patching
POLICY (effective now): **no hand-patching installed copies, ever.** Fixes land in the SOURCE repo
(~/code_sync/maitrise/{morel,Building2Building,minergym}), get pushed, the commit pin is bumped, env rebuilt.
DONE (phase 1, non-destructive, uv-based per user): **pyproject.toml + uv.lock** — first-party deps
declared by FULL commit sha via [tool.uv.sources] (ssh URLs — repos are private); two
[tool.uv] override-dependencies entries force our minergym@7520086 + b2b@698ca69 over the OLDER pins
inside b2b's and morel-b2b's own pyprojects (the transitive-pin chain). `uv lock` VALIDATED (67 pkgs).
NOTE: had to push the b2b submodule state upstream (vtaboga/Building2Building branch
`transfer-experiments` @ 698ca69) — pins must be fetchable. bootstrap_env.sh = `uv sync` (+ --dev
editable-overrides from siblings). scripts/evaluate.py moved from vendor/morel root. NO requirements.txt.
PHASE 2 (ONLY WHEN BOX IDLE — running campaign uses the OLD venv at vendor/morel/.venv):
  1. bash scripts/bootstrap_env.sh --dev  → new .venv at repo root
  2. smoke: check_supply_symmetry --steps 5 (+ verify adjacency: 49 pairs on OfficeMedium-4001)
  3. update launch wrappers/drivers: vendor/morel/.venv/bin/python → .venv/bin/python
  4. git rm the vendor/ submodules + .gitmodules, commit, push
  GOTCHAS: scripts sys.path-insert vendor/* (harmless once dirs gone); drivers hardcode the old venv path
  (run_energy_ladder.sh, run_mlp_rerun.sh, PROF wrappers); wandb creds/netrc unaffected; guix profile
  sourcing unchanged. The old venv's minergym hand-patch becomes irrelevant (pin 7520086 has the fix).

## ⚡ CURRENT SNAPSHOT 2026-08-06 — READ THIS FIRST after compression
**FINAL RESULTS (all zero-shot on TEST, full-year, deterministic eval):**
- **COMFORT (e0)** vs BO-tuned RBC (the bar; user directive: never default-RBC as headline):
  from-scratch 0/20 · BC-warm 1/20 · DAgger-y1 1/20 · **DAgger-y2 5/20 (OfficeMed 5/5 @ 0.23x)** ·
  y2+PPO 5/20 (OfficeMed 5/5 @ 0.70x; fixed Retail/OffSmall, hurt Restaurant 2.70x).
  CONCLUSION: tuned RBC unbeaten on single-zone comfort across 6 methods (1.1-1.7x);
  VAV surpassed 4x by pure imitation. (vs default RBC for reference: y1 16/20, y2 17/20.)
- **ENERGY (emed)** vs BEST-of-two RBC per type (no energy-tuned RBC exists; comfort-BO RBC is WORSE
  than default on 2/4 types under energy pricing): frozen-y2 2/20 · **y2+PPO(1yr) 13/20 — THE HEADLINE**
  (OfficeMed 5/5 @0.80x, Restaurant 5/5 @0.93x, OffSmall 3/5 @0.93x, Retail 0/5 @1.66x).
  Mechanism story: imitation can't touch energy levers (teacher pinned-OA) → RL from imitation+calibrated
  critic captures them. ⚠️ RETRACTED: "PPO degrades on emed" was a sampled-vs-deterministic rollout artifact
  — NEVER compare training rollouts across different action-selection modes; compare EVALS only.
- Train-side (motivation, banked): BC-warm 4/20 vs per-building oracles; y2 8/20 (OfficeMed 5/5, +46-61%).
- Symmetry proof: OfficeMedium's 3 vav_supply nodes get BIT-IDENTICAL actions under Amorpheus
  (scripts/check_supply_symmetry.py, asserts it) — architectural, motivates ModuMorph/spectral-PE.
**RUNNING NOW (2026-08-06 late):**
- ~~5-YEAR run~~ **KILLED by user** ("won't be conclusive") after 50/730 updates.
- from-scratch emed Amorpheus (PID 9087, runs/emed_fromscratch_amorpheus, 373 updates, ~75+ done).
- **EMED-BO TUNING CAMPAIGN (PID 9716, waiter armed)**: per-(type,CZ) Optuna on TRAIN instances,
  EXACT comfort protocol (TPE seed42, 25 trials, full-year) but --task task_occ_emed. 11 pairs cover the
  test set, all matched to train instances, 0 missing. → data/rbc_emed_tuned_pertypecz/<type>_cz<z>/...json
  (+ per-pair logs). ~275 rollouts, 3-wide, ETA ~15h. AFTER: write+run the lookup-apply eval on emed test
  → the ENERGY-TUNED RBC bar → re-judge the 13/20 headline against it (RISK: may shrink it; honest bar).
  tune_rbc_comfort.py now takes --task/--out-dir (defaults unchanged).
- USER-CONFIRMED: headline baselines are per-(type,CZ) LOOKUP (realistic); per-building tunes were
  teachers/oracle-figure only.
**SPECTRAL-PE THREAD (2026-08-06, user-approved incl. the INVERSION HYPOTHESIS):**
- BUILT + VALIDATED: `AMORPHEUS_B2B_SPE_BRIDGE` (= spectral_pe(k=4) ∘ inline ∘ commons_to_node) in
  morel_b2b_amorpheus/bridge.py; make_model/load_model take `universe=`; dagger_stream has `--bridge spe`;
  check_supply_symmetry has `--bridge spe --checkpoint none` → **ZERO-TRAINING DEMO PASSED: the 3 supply
  nodes emit DISTINCT actions under SPE** (symmetry broken by construction; plain bridge asserts identical).
- USER HYPOTHESIS (the interesting one): y2's VAV win may be CAUSED by the forced symmetry (identical
  supply tokens + different labels → regression to the LOOP-MEAN SAT = accidental ensemble). SPE lets the
  student imitate per-loop teacher behavior faithfully = possibly a WORSE strategy better imitated →
  **SPE might LOSE the OfficeMedium gains.** Either outcome is thesis-grade (prior-strength twist vs
  reinjection validated).
- RUN (queue when box frees, after emed-BO campaign): SPE-DAgger 2 years comfort (same protocol as y1+y2:
  2M steps beta-decay year then 2M beta≈0 year, --bridge spe) → eval on test vs y2. THEN decide on the
  SPE critic-year+PPO energy leg. Checkpoints NOT compatible with plain-bridge ones.
**ALSO QUEUED:** energy-decomposition eval (approved); from-scratch eval when it finishes;
emed-tuned-bar lookup eval after campaign; specialists-distillation = future work.
**BUG FOUND + FIXED (2026-08-06): thermal_adjacency edges were missing** — root cause was in MINERGYM, not
b2b: `zone_adjacency()`'s SPARQL only understood the outside_boundary_condition="Surface" (mirror-surface)
idiom; the ASHRAE prototypes use the "Zone" idiom exclusively → empty adjacency, b2b's edge-builder
(b435015) silently received nothing. FIX: added the Zone-idiom query to `zone_adjacency()` in BOTH
`~/code_sync/maitrise/minergym/minergym/ontology.py` (source checkout — UNCOMMITTED, include in next push
round) and the venv copy (vendor/morel/.venv/.../minergym/ontology.py — NOTE: venv copy is NOT under git;
document in thesis repro notes). VERIFIED: OfficeMedium-4001 now 49 adjacencies; fresh morphologies carry
{hvac_system:15, controls:15, thermal_adjacency:98}; graph now CONNECTED → SPE Laplacian sees the full
thermal graph (the queued SPE-DAgger run will use the RICH graph — richer ablation, note in write-up);
SPE symmetry-break re-verified on the fixed graph (max |Δ| 5.5e-4 ≠ 0). Exporter regenerated the thesis dot
(64 pair-edges). Running jobs (campaign/from-scratch) unaffected — they don't consume edges.
**ON THEIR WAKE:** eval finished run(s) on emed test (--task task_occ_emed! now supported in
eval_fullyear_panel.py) vs data/rbc_emed_fullyear.json + join best-of-two; 5y: eval best snapshots too.
**TODO:** Phase-3 energy figure (house style, like comfort_summary but emed); from-scratch eval;
optional: Retail diagnostic (0/5 on emed; blowups on 2 train bldgs), u25 comfort snapshot eval;
MLP rerun still queued (run_mlp_rerun.sh — likely CUT for deadline); GITHUB PUSH pending (many new
scripts + morel submodule edits: critic-only flag, --task flags, beta-cap docstring fix).
**FIGURES:** all comfort finals DONE + reproducible via `bash figures/make_thesis_figures.sh`
(incl. comfort_summary: normalized by UNTUNED rbc, tuned RBC = orange hatched bar, validated palette).
**STANDING RULES:** never write thesis prose; tuned-RBC/best-of-two is the only reporting bar; no
train-split evals; morel edits in vendor/morel submodule; pub_style for all figures; arm waiters, act on wake.
Deadline ~2026-08-11. Remaining work is mostly the user WRITING; compute: only the 2 runs above + their evals.

**This is the source of truth for the current experimental campaign. Read it after any
context compression before acting. Update it whenever state changes.**

Last updated: 2026-07-31

## The plan (supersedes the stale grid in the thesis §1815)
1. Get the DAgger setup working. — **DONE** (built + smoke-tested)
2. DAgger on comfort only (`task_occ_e0`). — **RUNNING** (scoped validation run)
3. Full method ladder — **from-scratch → BC → BC-warm → DAgger(+PPO)** — on the
   **energy+comfort task `task_occ_emed`** (w_E=1.0). — **NOT STARTED** (launch after step 2 validates)

Architecture for the ladder: **ModuMorph** (variant `hn`) is the flagship student. Amorpheus = prior baseline.

## ✅ COMFORT CAMPAIGN CLOSED (2026-08-05). PHASE 3 STAGE 1 LIVE.
PPOFT FINAL vs TUNED: 5/20 — OfficeMed 5/5 (0.70x, eroded from y2's 0.23x), OffSmall 1.25x, Retail 1.64x,
Restaurant 2.70x (hurt). Verdict: y2 = flagship comfort policy; RL redistributes, doesn't lift the count.
Six-method sweep complete: single-zone comfort belongs to tuned RBC; VAV belongs to the learned policy.
csv: data/dagger_ppoft_test_fullyear_eval.csv (ppoft figure: render via driver if wanted).
PHASE 3 STAGE 1 DONE (2026-08-06): BO-emed baselines + critic-year both complete.
**EMED RBC LANDSCAPE (test, means)**: default / BO-tuned: Restaurant −30.5k/−43.7k, Retail −121.4k/−98.1k,
OfficeMed −55.0k/−52.6k, OffSmall −104.2k/−121.2k. KEY: comfort-tuned RBC is WORSE than default on 2/4 types
under energy pricing (comfort aggressiveness burns energy) → NO clean tuned bar on emed; compare vs BOTH
(best-of-two per type), state "no energy-tuned RBC exists". Headroom huge (5-70x comfort-scale penalties).
csv: data/pertype_bo_emed_fullyear_eval.csv. Critic-year ckpt: runs/dagger_stream_emed_criticyear/model_s0.eqx.
## 🏆 PHASE 3 VERDICT (2026-08-06): ENERGY FINE-TUNE **WORKS** — hypothesis CONFIRMED
**y2+PPO(1yr) on emed test: 13/20 vs BEST-of-two RBC** (OfficeMed 5/5 0.80x, Restaurant 5/5 0.93x,
OffSmall 3/5 0.93x, Retail 0/5 1.66x). Frozen y2: 2/20 (1.10-1.37x). The learned policy SURPASSES the best
reactive alternative under energy pricing on 13/20 held-out buildings. csvs: data/{y2_emed_test,emed_ppoft_test}_fullyear_eval.csv.
⚠️ **RETRACTION**: the earlier "fine-tune degraded ~19%" wandb analysis was an ARTIFACT — it compared PPO
TRAINING rollouts (SAMPLED actions, exploration noise) against critic-year rollouts (DETERMINISTIC mean).
Never compare rollout rewards across runs with different action-selection modes. At deterministic eval the
fine-tune is a large win. (Comfort ppoft's mixed verdict came from EVALS — that one stands.)
⚠️ USER-QUEUED (2026-08-06) ON WAKE OF bs99onuw7, AFTER the evals: launch the 5-YEAR emed PPO run
(undertraining hypothesis test): train_allactive.py --task task_occ_emed --years 5 --ent-coef 0 --lr 2e-5
--init-checkpoint runs/dagger_stream_emed_criticyear/model_s0.eqx --out runs/dagger_emed_ppoft_5y
(≈730 updates, ~12h). Watch wandb for an upward YEAR-OVER-YEAR same-season trend (seasons repeat in-run →
trend unambiguous). Snapshots every 25 updates → can eval best-year ckpt later. Arm waiter + periodic
year-over-year check via the wandb API analysis used 2026-08-06.
STAGE 3 LIVE: chained emed TEST evals — y2_emed_test (frozen y2 = likely energy headline) then
emed_ppoft_test, both --task task_occ_emed (flag added to eval_fullyear_panel.py; earlier mislaunch on the
e0 task was caught + killed, no CSV corruption). Waiter bs99onuw7. THEN: join vs BOTH RBC bars (default +
comfort-BO on emed; best-of-two per type), figure, OPTIONAL u25-snapshot eval (early-vs-final fine-tune),
and the campaign closes. If evals confirm: energy story = "imitation policy competitive with best RBC;
RL fine-tune harmful on both tasks" — honest negative for the hypothesis, consistent with field experience.

## (executing) PHASE 3 (ENERGY) — USER-APPROVED SEQUENCE:
REPORTING BAR (user directive 2026-08-05): compare vs TUNED RBC only; default RBC no longer a meaningful bar.
1. emed baselines (test): comfort-BO-tuned RBC rolled on task_occ_emed (~20 RBC rollouts; script similar to
   eval_fullyear_pertype_bo.py but --task task_occ_emed). Default-emed baseline exists (data/rbc_emed_fullyear.json).
   (No energy-TUNED RBC exists — would need new Optuna campaign; state honestly in thesis.)
2. ON-POLICY CRITIC YEAR (user simplified: no DAgger needed, just policy evaluation):
   dagger_stream_amorpheus.py **--critic-only** --task task_occ_emed
   --init-checkpoint runs/dagger_stream_e0_amorpheus_y2/model_s0.eqx --buffer-cap 500000
   --updates-per-cycle 64 → runs/dagger_stream_emed_criticyear (~2.5h). Actor FROZEN (= y2 exactly);
   student always drives; no teacher queries; ONLY value heads trained on emed returns-to-go →
   V^π for the exact policy+reward PPO starts with, on a stationary distribution. Flag added to the
   script this session (skips policy SGD + teacher; labels zeroed; imit metrics guarded).
3. PPO fine-tune on emed FROM the y3 checkpoint (train_allactive.py --task task_occ_emed --ent-coef 0
   --lr 2e-5 --years 1 --init-checkpoint runs/dagger_stream_emed_y3/model_s0.eqx → runs/dagger_emed_ppoft).
4. Eval emed test split vs tuned bar (+ default for the record) + house figure.
HYPOTHESIS (precommitted): fine-tuned policy beats even tuned RBC on VAV by MORE than the comfort margin
(OA modulation = free energy the RBC forgoes); possibly closes unitary gap (deadband exploitation).
COMFORT CONCLUSION (once ppoft eval final): tuned RBC unbeaten on single-zone types across 6 methods
(1.1-1.7x); surpassed only on VAV (y2: 5/5 at 0.23x; ppoft eroded to ~0.75x while fixing Retail/OffSmall).

## WHERE WE ARE / WHERE GOING (snapshot 2026-08-03)
- Thesis empirical story = 3 phases: (1) comfort vs UNTUNED RBC [from-scratch fails → BC-warm beats it];
  (2) comfort vs BO-TUNED RBC [fails: BC-warm 14/20 vs default collapses to 1/20 vs BO]; (3) energy (emed)
  vs untuned RBC [ladder running]. Plus a DAgger-motivation subplot (train-split eval, below).
- Figures (house-style via `figures/pub_style.py`, see figure_style.md): `tuned_vs_amorpheus` (test, comfort)
  DONE. Phase-1/2 win tables computed & horizon-verified (CAVEAT: transfer4_bcwarm has no n_steps → re-verify).
- IN FLIGHT: Amorpheus TRAIN-split evals (bcwarm_train, bctuned_train) vs per-building-tuned RBC on TRAIN, to
  motivate DAgger; running at 4 workers. Energy ladder SIGSTOP-frozen to free cores (see MUST-DO resume above).
- QUEUED (NOT launched — days of compute, box busy): MLP-ceiling rerun after the norm bug fix →
  `scripts/run_mlp_rerun.sh`.
- WRITING next (unblocked): problem formulation + energy setup (OA motivation), DAgger methodology
  (shadow expert, β-schedule, non-Markov caveat), honest baseline naming (default vs BO-tuned).
- Standing constraints: never write thesis prose for the user; use pub_style for figures; edit morel in the
  SUBMODULE (vendor/morel), not the standalone ~/code_sync/maitrise/morel.

## ADVISOR-FOUND BUGS (code review, 2026-08-03)
1. **Beta concentration cap (`conc_cap`) — REAL bug, ZERO impact.** Scales (α−1,β−1) → preserves the MODE, not
   the mean; the docstring "preserves the mean α/(α+β)" is WRONG (α=10,β=2,cap=4 → mean 0.833→0.7). BUT it's
   off by default, used only in `runs/probe_conccap_OfficeSmall_0` (unreported), AND never applied at eval
   (static, NON-serialized field; `load_model` defaults None; eval scripts don't pass it) — so no reported eval
   action was shifted. Code: `morel_amorpheus/model.py:379-385`, `morel_modumorph/model.py:257-261`.
   TODO (cosmetic): fix the docstring or delete the feature. No rerun needed.
2. **MLP obs-normalizer — REAL bug, CONTAINED impact, FIXED.** `train_mlp_specialist.py` fed NORMALIZED obs to
   `norm.update()` → running stats drift toward (0,1)=identity over a long run → MLP progressively fed raw,
   badly-scaled obs → handicapped (worst at 1.5M steps). CONFINED to the MLP pipeline: the main `ppo.py` has NO
   such running normalizer, so transfer/BC/DAgger/energy results are UNAFFECTED. `bc_fit_mlp.py` verified CLEAN
   (updates on RAW demo obs; line 78) → existing `runs/mlp_bc_*` clones + obs_norm.npz are REUSABLE.
   FIXED (this session): store `O_raw` and `norm.update(O_raw)`; added `--freeze-norm` (use for warm-start).
   AFFECTED FIGURE: §specialist-comparison MLP bars (incl. the 1.5M "ceiling") → the "Amorpheus specialist
   dominates the MLP specialist" claim is CONFOUNDED (MLP was crippled, gap overstated). **Do NOT use the MLP
   ceiling figure until rerun.** RERUN queued: `scripts/run_mlp_rerun.sh` → `*_fix_*` dirs, reuses mlp_bc clones.

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

## ENERGY LADDER KILLED (2026-08-04) — freed 5.8GB; batch energy ON HOLD
The frozen energy-ladder group (pgid 125503: 126784/126820/125505) was SIGKILLed to free ~5.8GB RAM (it was
hogging memory on the 16GB box). Batch energy ladder is fully STOPPED — NOT resumable; needs a full re-run
(`run_energy_ladder.sh`) if energy results are wanted later. On hold given the streaming pivot. MemAvail ~7.5GB.
Obsolete waiter: byo4gp8js watched pgid 125503 → its "driver gone" wake is now spurious, IGNORE it.

## 🚀 STREAMING DAgger: FULL RUN LAUNCHED (2026-08-04, post-session-crash relaunch)
`scripts/dagger_stream_amorpheus.py` — SINGLE-PROCESS (envs are threads via minergym; model is a local var,
no subprocess weight transfer; tradeoff: sequential env stepping ~1 core). Jit memoized by id(m).
SMOKE PASSED (runs/_smoke_stream2.log): imit_mse 0.625→0.050 over 16 cycles; beta 1→0.07; tail shows gap
re-widening as student drives (the DAgger effect); ~107 steps/s → 2M steps ≈ 5-6h.
FULL RUN: PID **184**, log `runs/dagger_stream_e0_amorpheus/run.log`, wandb online.
Params: 4 types x 5 train bldgs (20 envs), task_occ_e0, tuned-RBC teachers, total 2M steps, n-steps 512,
32 SGD/cycle, mb 256, buffer 100k, beta-half-life 200k, ckpt every 10 cycles (snapshots every 50).
Completion waiter b6gyq366l armed (DAGGER_STREAM_DONE / Traceback / death, 10-min poll).
MID-RUN EYEBALL (cycle ~134, Aug window, beta 0.009): student −0.0339/step vs tuned-teacher same-window
−0.0297 → 1.14x. Naive BC-warm full-year: −0.0615 (2.25x). ~10x smaller imitation gap IF it holds in eval.
KNOWN LIMITS of run 1 (explained to user): single pass through the year (each season seen once), buffer
cap 100k = ~17 sim-days/bldg sliding window (NOT the full DAgger aggregate → seasonal forgetting),
only ~6.2k policy SGD steps. Hence wiggly imit_mse — flat "converged" curves impossible in run 1.

## ✅ YEAR 2 DONE (2026-08-04, ~3h): stabilization CONFIRMED
Same Nov-Dec cycles y1→y2: imit_mse 0.075-0.099(swinging) → **0.048-0.059(flat)**; reward −0.086..−0.183 →
**−0.057..−0.104**. Winter dip was DISTRIBUTION SHIFT, not season difficulty (teacher's Dec window −0.0237 is
BETTER than its year mean −0.0273). Student/teacher ratio: Aug 1.14x, Dec ~2.4x (down from ~5x in y1) —
winter still relatively weakest. Ckpt: runs/dagger_stream_e0_amorpheus_y2/model_s0.eqx.
## 🏆 Y2-TEST FINAL (2026-08-05): **17/20 vs default; 5/20 vs BO-TUNED — OfficeMedium 5/5 vs BO!**
The y2 streaming-DAgger policy (pure imitation, zero-shot) BEATS THE PAPER'S BO-TUNED BASELINE on ALL FIVE
held-out VAV buildings (OfficeMedium norm 0.204 vs default). Thesis story #13 achieved on the hardest class.
Per type vs default: Restaurant 5/5 (.40), OfficeMedium 5/5 (.20), OfficeSmall 5/5 (.49), Retail 2/5 (1.07).
Figure: figures_out/dagger_y2_* (delivered). csv: data/dagger_stream_y2_test_fullyear_eval.csv.
PPOFT LAUNCHED: PID 1968, runs/dagger_stream_e0_ppoft/run.log (train_allactive, init=y2, e0, ent0, 1yr,
lr 2e-5), waiter armed → then TEST eval only (vs default + BO). Goal: extend the surpass beyond VAV.

## (executed) ON WAKE (b2y01zaw8 = y2-test eval done) — USER-APPROVED NEXT TRAINING RUN:
Launch PPO FINE-TUNE YEAR on COMFORT (goal: SURPASS the tuned RBC, per user):
  scripts/train_allactive.py --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall \
    --n-envs 5 --task task_occ_e0 --years 1 --ent-coef 0 --lr 2e-5 --seed 0 \
    --init-checkpoint runs/dagger_stream_e0_amorpheus_y2/model_s0.eqx --out runs/dagger_stream_e0_ppoft
  (guix PROF env, nohup, log runs/dagger_stream_e0_ppoft/run.log, mkdir first, arm waiter.)
  Rationale: y2 already BEATS the VAV oracles by imitation; PPO now starts from competent policy + trained
  critic. After it: eval on TEST (vs default RBC + join vs BO-tuned).
  **USER DIRECTIVE (2026-08-05, corrected — user initially said the reverse by mistake): NO MORE TRAIN-SPLIT
  EVALS from now on.** The train-vs-tuned-oracle comparisons were the rhetorical setup motivating DAgger and
  are banked (bcwarm 4/20 → y2 8/20, OfficeMedium 5/5). All future checkpoints are judged on TEST — the real
  thesis question is zero-shot transfer, and ppoft's success bar is the BO-tuned RBC on test (currently 1/20).
Y2-TRAIN RESULT (FINAL): **8/20 vs tuned oracles** (2x BC-warm's 4/20): OfficeMedium **5/5 norm 0.56**
(beats every VAV oracle 46-61%!), Retail 2/5 norm 3.2 (TWO train-specific blowups 2002/2003 — diagnostic
pending, test shows no such collapse), OfficeSmall 1/5, Restaurant 0/5.
Y2-TEST partial (16/20 rollouts): 13/16 wins vs default; OfficeMedium 4/4 with 8-9x margins (norm ~0.2-0.3).

**Y1 TEST RESULT (2026-08-05, FINAL): 16/20 vs DEFAULT RBC** (Restaurant 5/5 norm .43, OfficeSmall 5/5 .47,
**OfficeMedium 4/5 norm .97 — VAV FIXED & TRANSFERS**, Retail 2/5 1.16); 1/20 vs BO-tuned (expected).
PURE IMITATION, no RL — beats BC-warm's 14/20-with-PPO. csv: data/dagger_stream_y1_test_fullyear_eval.csv.
IN FLIGHT: y2 TRAIN eval vs tuned teachers (PID 1029, partial 2/4 incl. OfficeMedium-4001 BEATING its oracle
2.5x — sanity-check when done) + y2 TEST eval (PID 1413, launched, waiter armed).

## ✅ EXECUTED (2026-08-04): run 1 DONE → year 2 + y1 test eval LAUNCHED
Run 1 finished (195 cycles, ~2h). Final: imit_mse ~0.075-0.099 @ beta 0.001. Late cycles = SELF-DRIVEN WINTER
(Nov-Dec): reward dipped to −0.18/step then recovered to −0.086 by cycle 194 — live seasonal-forgetting evidence,
motivates y2. Y1 ckpt: runs/dagger_stream_e0_amorpheus/model_s0.eqx (+ snapshots _c50/_c100/_c150).
- YEAR 2: PID **522**, log runs/dagger_stream_e0_amorpheus_y2/run.log, waiter b0w5r4qpm. (2M steps, beta≈0
  from cycle 1, buffer 500k, 64 SGD/cycle, init=y1 ckpt.)
- Y1 TEST EVAL: PID **523**, log runs/eval_dagger_stream_y1_test.log, waiter b6es3a5cd →
  data/dagger_stream_y1_test_fullyear_eval.csv (vs default RBC; join BO from pertype_bo csv).
## (executed) ON-WAKE spec was:
1. LAUNCH YEAR-2 CONTINUATION (user said yes; launch manually on wake, no script-chaining):
   nohup env (same guix PROF vars) vendor/morel/.venv/bin/python -u scripts/dagger_stream_amorpheus.py
     --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall --n-envs 5
     --task task_occ_e0 --total-steps 2000000 --n-steps 512
     --updates-per-cycle 64 --minibatch 256
     --buffer-cap 500000 --beta-half-life 1000
     --init-checkpoint runs/dagger_stream_e0_amorpheus/model_s0.eqx
     --checkpoint-every 10 --snapshot-every 50
     --out runs/dagger_stream_e0_amorpheus_y2   > runs/dagger_stream_e0_amorpheus_y2/run.log
   (beta-half-life 1000 => beta≈0 from cycle 1 on = fully student-driven year 2; buffer 500k = 25k/bldg
   ≈ 3 sim-months retention; 2x SGD. Envs restart at Jan 1 => student now self-drives WINTER. mkdir out dir first.)
   Then arm a new completion waiter on the y2 log (same pattern: DAGGER_STREAM_DONE/Traceback/death).
2. LAUNCH YEAR-1 TEST EVAL alongside (2 workers, coexists with y2 training):
   eval_fullyear_panel.py --checkpoint runs/dagger_stream_e0_amorpheus/model_s0.eqx --split test
   --tag dagger_stream_y1_test --n-workers 2   (default baselines file = default RBC; join BO from
   data/pertype_bo_fullyear_eval.csv as before). Compare vs BC-warm 14/20-vs-default / 1/20-vs-BO.
(bctuned_train KILLED per user "don't care"; batch amorpheus DAgger scripts kept but superseded.)

## (superseded) PIVOT rationale + design detail
Why: EnergyPlus full-year episodes (~105k steps) make BATCH DAgger too slow (whole year rolling a frozen
policy before any update; ~2-3 iters/day). Streaming = update WITHIN episodes.
- DESIGN (my rec): adapt the persistent-env PPO driver (train_allactive). Keep 20 persistent envs + round-robin
  + update every n_steps; at each step ALSO query the tuned-RBC (shadow label) + execute a beta-mixture action;
  REPLACE the PPO loss with supervised imitation MSE to the RBC labels (+ optional critic). Bounded replay buffer;
  beta decays continuously over steps. Reuses the forward_fn seam. Supervised targets => replay valid off-policy
  (no RL instability). Learns within episodes; can use short interleaved segments (never wait a full year).
- SCOPE (proposed = same as the held batch): Amorpheus student, all 4 types, comfort task_occ_e0,
  per-building-tuned-RBC teacher, then PPO-warm.
- AWAITING USER: (1) straight to streaming (drop batch) vs keep batch as fallback? (2) confirm scope. I lean straight-to-streaming.

## BATCH Amorpheus DAgger — BUILT + smoke-tested, HELD (not launched)
Scripts ready (subagent, kept intact): `scripts/dagger_{collect,fit,loop}_amorpheus.py` + `run_amorpheus_dagger_ladder.sh`.
- collect: Amorpheus shadow-mode; wandb ONLINE; logs per-building rollout_return/mean_reward + per-step
  imitation_mse (student vs RBC in normalized action space via A_pinv) + downsampled curves; tagged per iter+beta.
- fit: aggregated, KEEPS memoize-jit-by-id(m) OOM fix; tagged per iteration.
- loop: beta_i=0.5^i, 3 iters. driver: DAgger loop -> PPO warm from it2 (--ent-coef 0 --years 2, matches warmstart_unitary_tuned).
- Smoke passed (recon_err 0; imitation_mse~0.133 vs fit action_mse~0.125). Accidental 2/20 it0 launch was killed + partial demos removed (clean slate).

## STILL RUNNING / WAITERS (2026-08-04)
- `bctuned_train` (Amorpheus BC cloned from TUNED RBC, on TRAIN, vs tuned RBC): RUNNING (PID ~130546). Waiter bmh77l54z (3h cap) will wake me.
  When done: tally 4/20-style + remake the `bcwarm_train_vs_tuned` figure with THIS (tuned-clone) — the current figure used the DEFAULT-clone (transfer4_bcwarm), a teacher mismatch.
- Bugs (advisor): beta-cap docstring FIXED (comment-only); MLP obs-norm FIXED + rerun QUEUED (`run_mlp_rerun.sh`, not launched, ~2 days).

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
- RESULT (2026-08-03, bcwarm_train 18/20): Amorpheus BC-warm vs per-building-tuned RBC ON TRAIN = **4/20**
  (OfficeSmall 4/4 win; Restaurant 0/5; Retail 0/5; OfficeMedium 0/4 catastrophic VAV). Motivates DAgger:
  imitation loses even on trained buildings. Caveats: tuned RBC = per-building ORACLE; OfficeSmall is the
  exception. `data/bcwarm_train_fullyear_eval.csv`. bctuned_train (clone-of-tuned) still running.
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

## TRAIN-split eval (motivate DAgger, 2026-08-03)
- Goal: Amorpheus on TRAIN buildings vs per-building-tuned RBC on TRAIN (oracle bar) → shows imitation is
  lossy even on trained buildings → motivates DAgger. Tuned-RBC-on-train = `tuned_return` in
  `data/rbc_tuned_configs/*_train_*.json` (20). Amorpheus-on-train = was MISSING (all evals were split=test).
- `eval_fullyear_panel.py` now has `--split` (default test) + `--baselines` (default the test default-RBC json).
- Running (subagent): built `data/tuned_train_baseline.json` ({train building_id: tuned_return}); eval
  `transfer4_bcwarm` (clone of DEFAULT rbc) and `transfer4_bc_tuned` (clone of TUNED rbc) on train,
  `--split train --baselines data/tuned_train_baseline.json` → data/{bcwarm_train,bctuned_train}_fullyear_eval.csv.
- Box freed: killed the redundant incremental `eval_fullyear_panel_modumorph` evals (energy ladder's step 5
  re-evals anyway); KEPT the energy-ladder driver (PID 125505).

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

## TWO morel checkouts — edit the SUBMODULE for experiments (2026-08-03)
- Experiments import the SUBMODULE: `b2b-transfer-experiments/vendor/morel` (branch
  `officesmall-warmstart-hooks`, pushed @ 4508c2d) — editable-installed, so this is the code that RUNS.
  **Make all morel edits for experiments HERE.**
- There is a SEPARATE standalone clone `~/code_sync/maitrise/morel` (branch `feature/setpoint-native-obs`
  @ 08998ef) — NOT used by experiments. Both share remote Terramorpha/morel.git.
- Divergence (merge-base 139c79c, 2026-07-16): standalone has ONE commit the experiments lack —
  "morel_b2b: drop the target_temperature re-attach hack" (a native-setpoint REFACTOR, likely needs a
  paired Building2Building change). Submodule has ALL 10 experiment commits (ModuMorph, spectral PE, PPO
  forward_fn, warm-start hooks, snapshots, per-building logging, etc.) that the standalone lacks.
- The experiments are NOT missing functional morel work: the dropped hack is the WORKING setpoint mechanism
  verified earlier; the standalone just re-implements it natively. DON'T reconcile before submission (risk:
  breaks setpoint obs without the matching b2b bump). Merge native-setpoint refactor post-deadline, both repos together.

## Environment note (how to launch)
Runs need the guix profile + venv: `vendor/morel/.envrc` does `guix-load energyplus python python-numpy
python-pandas python-pytorch ty uv` + `export ENERGYPLUS_PATH=$GUIX_LOAD_PROFILE` + `source .venv/bin/activate`.
`guix-load` is NOT available in the bare non-interactive shell — launch via the profile-sourcing wrapper
(the pattern the DAgger subagent used). Use `nohup … & disown` (setsid unavailable). n-workers ≤ 3 (4 cores).
