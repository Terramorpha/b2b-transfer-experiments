# Beating RBC: OfficeSmall investigation → 4-type transfer — running log

**Living document.** Hypotheses, the experiments run against them, and results.
Part I (H1–H16) is the OfficeSmall investigation. Part II is the 4-type transfer
work it led to, plus **protocol corrections that change how Part I's numbers read**.

> ## ⚠️ READ FIRST — evaluation-protocol corrections (2026-07-20)
>
> `timesteps_per_hour = 12` (not 4, which I assumed for a long time). Therefore:
>
> | quantity | correct value |
> |---|---|
> | the 672-step eval "chunk" | **2.33 days** (56 h), starting Jan 1 — *not* 7 days |
> | a full year | **105,120 steps** |
> | updates per simulated year (672-step segments) | **~156** |
> | 400-update run | **2.56 passes over the SAME year** |
>
> **Every win-rate in this document — including the OfficeSmall 10/10 — is measured
> on that 672-step window, i.e. ~0.6% of a year, in January.** Those comparisons are
> internally fair (policy and RBC use the identical window, `baseline_chunk.csv`), but
> they are **not** annual performance. Full-year evaluation is a separate, harder test
> (see Part II §Full-year evaluation), and early evidence is that results do **not**
> automatically carry over.
>
> **EnergyPlus is deterministic**: replaying the year adds *no* new environmental data.
> Coverage saturates at ~156 updates; beyond that, extra updates are **epochs over fixed
> data** (with the attendant overfitting risk), not added diversity. Genuine temporal
> diversity would need multiple weather years, which the dataset does not provide.

## The problem
The transfer policy (`task_occ_e0` = occupancy/dynamic setpoint, comfort-only
`energy_weight=0`; Amorpheus, cold-start + resampling) **beats RBC on
Retail/OfficeMedium but loses on every OfficeSmall test building (0/5)**.
Representative: OfficeSmall-5996 policy `-42.3` vs RBC `-9.4`.

## Established facts (foundation, all verified)
- **The policy loses on the TRAIN buildings too — not a test-set/generalization
  artifact.** Eval of the OfficeSmall specialist on the 5 buildings it *trained
  on* vs the 5 held-out: **1/5 train, 0/5 test** — and the single train "win" is a
  degenerate exact tie (−173.31 vs −173.31 on 5005). So the failure is a genuine
  control/optimization failure, not "the held-out split is just hard." The policy
  is essentially **building-invariant** (under-heats to ~−20…−45 everywhere) while
  RBC *adapts* — RBC is actually *better* on test (−7…−10) than train (−15…−23), so
  the policy's gap to RBC is widest exactly where RBC does best (the easy test
  buildings). Not test-hardness; a fixed-behavior policy vs an adaptive baseline.
  Inter-building variance is huge: RBC return spans **−7 to −173** across 10
  buildings. (`scratchpad/eval_traintest.py`.)
- **The policy observes the setpoint at every layer.** b2b's *native* morphology
  drops `target_temperature` (the "10 unassigned observation slots" warning), but
  the morel bridge re-attaches it per zone via `FlatSlot`. Probes confirmed it in
  the source obs, in the Amorpheus network input, and that the action *responds*
  to setpoint perturbations (`Δaction` grows with setpoint). → not observability.
- **The action-conversion path is faithful and building-invariant.** Custom
  constant-command probe: a normalized level of `-1 / 0 / +1` maps to physical
  supply-temp `5 / 32.5 / 60 °C` and fan `0 / 7.5 / 15` on **both** OfficeSmall
  and Retail (fixed affine, no per-building rescale). The policy *could* command
  60 °C on OfficeSmall; it chooses low. → not an action-rescaling bug.
- **The failure mechanism is under-actuation of heating.** The policy's commanded
  supply-temp setpoint collapses to a near-constant low band on OfficeSmall while
  RBC ramps to 26 °C:
  | | policy supply-temp cmd | RBC |
  |---|---|---|
  | Retail-2998 (policy wins) | [9.4, 21.3] — ramps with setpoint | [12.0, 17.3] |
  | OfficeSmall-5996 (loses) | [9.0, 13.8] — stuck low | [12.0, 26.2] |
  The policy *can* heat (Retail), it just doesn't on OfficeSmall.

## Hypotheses

| # | Hypothesis | Status | Verdict |
|---|---|---|---|
| H1 | Policy is setpoint-blind (doesn't observe the target) | **ruled out** | Observed at all layers; action responds to it |
| H2 | Action conversion rescales badly per-building → can't command enough heat | **ruled out** | Fixed affine [5,60], identical on both buildings |
| H3 | Multi-type transfer averaging neglects OfficeSmall | **partial** | Specialist beats transfer (−21 vs −42) but still 0/5 — contributes, not the whole story |
| H4 | Undertraining — needs more iterations | **ruled out (reversed)** | More training → **worse** (see below) |
| H5 | Per-building reward normalization (`τ_T`) starves OfficeSmall's comfort gradient | **ruled out** | PPO normalizes advantages per-schema → reward scale cancels |
| H6 | Warm-start bias / big building coasts warm (hollow win) | **premise refuted** | OfficeMedium's policy *actively* tracks the setpoint, so winners control, not coast |
| H7 | Exploration / entropy collapse into a low-heat basin | **partial** | `ent_coef=0.05` helps (supply 13 vs 10.5, ret −27 vs −37) but still **0/5**, still ~3× worse than RBC |
| H8 | Cautious-heating from the two-sided deadband (avoid too-hot penalty → under-heat leaky OfficeSmall) | **doubtful** | Overshoot is *hard* on leaky OfficeSmall (even RBC only reaches ~20, not 21) → the too-hot penalty rarely fires |
| H9 | Discount horizon too short for anticipatory heating (`γ=0.98` ⇒ ~12 h; OfficeSmall needs to pre-heat vs a payoff ~1–2 days out) | **partial** | `γ=0.99+ent=0.05` improves test a few points, entropy holds — but still **0/5**, still ~2.5× worse than RBC |
| H10 | The 5 test buildings are just a hard/unlucky held-out sample (high inter-building variance) | **ruled out** | Policy loses on TRAIN buildings too (1/5, that 1 a degenerate tie); failure is fundamental, not a split artifact |
| H11 | Resampling non-stationarity (building drift every 10 updates) destroys plasticity / prevents learning | **ruled out** | Single fixed building (pool=1, zero drift) converges to the *identical* −17.17 on 5001, still loses to RBC. Drift removal changes nothing → not the cause. But the *eval-time* generalization gap (competitive on train, collapses on easy held-out) survives as the real shape of the failure |
| H12 | Born-Again self-distillation (fresh-init net matching teacher's action dist) improves held-out generalization | **ruled out** | Student ≈ teacher within rollout noise (<1 pt) on all 10 buildings, still 0/5. BAN's dark-knowledge regularization is a *classification* effect (softmax over classes); a continuous Beta policy has no class structure to exploit, so distillation just re-creates the controller |
| H13 | The net can't *represent* RBC-level control (representational limit) | **REFUTED — capacity confirmed** | Behavior-cloning RBC into the same net matches RBC on train (2/5 beat it, all within ~1 pt) → the good policy is representable + gradient-reachable; PPO just can't find it. Definitively localizes the failure to PPO optimization. Clone also generalizes ~2–3× closer to RBC on held-out test than PPO |
| H14 | Warm-starting PPO from the RBC clone (+ pretrained critic) lets it beat RBC | **ruled out — RL overfits (as configured)** | With a perfect warm-start, a pretrained critic, and ent=0 (no divergence), *resampled* PPO holds RBC-level on *train* but **overfits** — held-out test collapses. Superseded by H15: the overfitting was the **resampling batch structure**, not the RL objective |
| H15 | Training all buildings **jointly** per update (vs resampled 1-at-a-time) fixes the overfitting | **✅ persistent all-active BEATS RBC 5/5+5/5 — but the *cause* was misattributed (see H16)** | Persistent all-active, warm-started, ent=0: beats RBC on every building (5996 −2.7 vs −9.4). Changed *two* things vs H14: resampled→all-active AND cold-start→persistent. |
| H16 | Confound: is it the all-active *batching* or the persistent *full-year* rollout? | **persistence is the driver, NOT all-active** | Cold-start all-active (all-active held, cold-start restored) **FAILS: 1/5 train, 0/5 test** (5996 −29, 6000 −36). Cold-start only ever trains on the first 7-day chunk (resets each update); persistent rolls continuously through the whole year (~3× over 150 updates → all seasons). **Temporal data diversity (full year), not joint batching, is what generalizes.** |

**Framing:** a good policy provably *exists* — **RBC beats the policy 0/5** (RBC −9.4 vs policy −27..−37 on 5996). So this is an **RL optimization failure**, not a controllability/observability/scaling limit. PPO cannot find RBC-level heating for OfficeSmall's control problem *even as a specialist*. The open question is *why the optimizer stalls* (exploration H7 = partial; horizon/credit H9 = leading).

**CAPSTONE (H13) — the framing is now *proven*, not inferred.** Behavior-cloning RBC
into the *same* Amorpheus net reaches **RBC-level control on the training buildings**
(2/5 beat RBC, all within ~1 pt: 5001 −16.95 vs −16.96, 5002 −14.89 vs −15.51). Same
architecture, same observations, same buildings — trained by **imitation** instead of
**RL**. So the good policy is **representable and gradient-reachable**; PPO simply cannot
*find* it by exploration. This is a positive existence proof that the OfficeSmall failure
is **PPO optimization/exploration**, full stop — not architecture, observability, or
representation. Bonus: the clone **generalizes far better than PPO** on held-out test
(5996 −15.9 vs PPO −22.7, 5999 −13.7 vs PPO −35.2; ~2–3× closer to RBC), so supervised
imitation transfers across the building distribution much better than the RL policy —
the eval-time generalization edge H11/H12 were reaching for.

**H14 — RL *overfits* when trained one-building-at-a-time.** Warm-starting PPO from the RBC
clone (+ pretrained critic, ent=0) holds RBC-level on train but **overfits and destroys
held-out generalization** — under the *resampling* loop (one building per update). This
looked like "the RL objective can't generalize"... until H15.

**✅ SOLVED — but the mechanism is TEMPORAL DATA DIVERSITY, not joint batching (H15→H16).**
The persistent all-active trainer (warm-start + critic + ent=0) **beats RBC on all 5 train
AND all 5 held-out test buildings** (5996 −2.7 vs RBC −9.4). *Initial* read (H15): the fix was
all-active joint batching vs resampling. **Confound test (H16) corrected this:** cold-start
all-active — same joint batching but resetting to the first 7-day chunk each update — **FAILS
(1/5 train, 0/5 test)**. The difference is *what data the policy sees*: cold-start trains only
on **week 1** (reset every update); persistent rolls **continuously through the full year**
(~3× over 150 updates → all seasons). So generalization comes from **training on the whole
year**, not from the joint gradient. All-active on week-1-only data overfits like everything
else. **Best controller: `runs/officesmall_allactive` (persistent, beats RBC 10/10).** Recipe:
RBC behavior-clone → value-head pretrain → **persistent** all-active PPO fine-tune (ent=0,
γ=0.99, low LR). NB: the parallel *cold-start* path we built does not reproduce the win — to
parallelize the winner needs persistent-parallel workers, or adding random-chunk-start
(temporal diversity) to the cold-start trainer.

## Details

### H3 — transfer averaging (partial)
- **Exp:** OfficeSmall-only specialist, 150 iters, resampling across 10 buildings.
- **Result:** 5996 = `-21.4` (vs transfer `-42`), supply-temp [13.8, 14.9].
  Better than the shared policy, but still 0/5 and still collapsed low. So the
  transfer setup hurts OfficeSmall, but a specialist doesn't fix it.

### H4 — undertraining (ruled out, *reversed*)
- **Exp:** 400-iter specialist vs 150-iter.
- **Result:** *worse* with more training:
  | | supply-temp cmd (5996) | return (5996) |
  |---|---|---|
  | transfer | [9.0, 13.8] | −42 |
  | 150-iter specialist | [13.8, 14.9] | **−21** |
  | 400-iter specialist | **[10.1, 10.9]** | **−37** |
  The policy *converges to less heating*. This is a collapse, not a lack of
  training. Runs: wandb `edo2s649` (400-iter resampling).

### H5 — reward normalization `τ_T` (ruled out)
- **Motivation:** `r = -temp_penalty/τ_T`; measured mean `τ_T` (full_year):
  OfficeMedium **38** < Retail **97** < OfficeSmall **117** ≈ Restaurant **121**.
  Ordering tracks performance (low `τ_T` = wins), so it *looked* causal.
- **Exp / analysis:** PPO normalizes advantages to unit scale (`ppo.py:163`,
  `adv=(adv-adv.mean())/(adv.std()+1e-8)`), **per morphology schema** (per building
  type, in `_update_from_rollouts`). So the reward scale cancels for the policy
  gradient — for the specialist *and* the transfer. `τ_T` only leaks via the
  **un-normalized value targets** (`returns`), and the critic just adapts.
- **Result:** ruled out as the cause. Launched a `τ_T=1` (un-normalized) specialist
  then **killed it as moot** once the per-schema advantage normalization was
  confirmed. (`τ_T` values remain a fair *benchmark observation*, just not the
  mechanism.)

### H6 — warm-start bias / coasting (premise REFUTED)
- **Idea:** episodes reset ~20.5 °C (above the 18 °C unoccupied setpoint), so "don't
  heat" is locally correct early; big buildings coast warm all episode (hollow win),
  OfficeSmall cools fast and needs anticipatory heating the policy never learns.
- **Exp:** captured the OfficeMedium winner (4997) trajectory (transfer policy + VAV
  RBC). Stats (mean conditioned-zone temp): reset/min/%-below-setpoint —
  OfficeMedium `21.0 / 17.3 / 21%`, OfficeSmall `20.5 / 15.2 / 97%`. Plot:
  `scratchpad/traj/h6_coast.png`.
- **Result: refuted.** OfficeMedium's policy does **not** coast — it *actively
  tracks*: heats to 21–22 °C when occupied and **deliberately drops to ~18 °C when
  unoccupied**, following the setpoint down. It beats RBC *because* RBC stays ~21 °C
  during unoccupied (over-heating → penalized by the **two-sided** deadband) while
  the policy correctly cools. So winners genuinely control; OfficeSmall (below
  setpoint 97% of the time) is genuinely under-controlled. → **longer episodes
  deprioritized** (the same warm start doesn't stop OfficeMedium from learning).
- **New angle (H8):** the deadband penalizes *too hot* as well as too cold, so the
  learned policy is **cautious about heating** to avoid overshoot. That caution
  keeps retentive buildings (OfficeMedium, Retail) in-band but under-heats the
  leaky OfficeSmall (fast heat loss). Consistent with the collapse to a low,
  low-variance supply command. Worth testing.

### H7 — exploration / entropy collapse (SUPPORTED; testing)
- **Idea:** PPO settles into the low-heat basin and entropy decays, never sampling
  the large "heat hard" action change needed to escape. Scale-invariant; consistent
  with worse-with-training.
- **Exp:** pulled wandb curves for the 400-iter specialist (`edo2s649`):
  | metric | start | end |
  |---|---|---|
  | `loss/entropy` | 6.16 | 1.27 (dips to −0.34) |
  | `rollout/step_reward` | −0.65 | −0.31 (plateaus) |
  | `loss/value` | 66.4 | 1.6 (critic converged) |
- **Result: supported.** Entropy collapses hard, critic converges, reward improves
  early then plateaus at a poor level — PPO sharpening into a bad local optimum.
  Eval getting *worse* 150→400 iters = the collapse entrenching. The loss has an
  entropy bonus but `ent_coef=0.01` (`ppo.py:53`) is too weak to hold exploration.
- **Confirming exp — result: partial.** Retrained OfficeSmall specialist with
  `ent_coef=0.05` (5×; wandb `qazbhccl`). Entropy held higher through the midpoint
  (3.56 @193 vs 1.27), and the policy improved — but it still collapsed by the end
  (1.15) and the **held-out eval is still 0/5**: supply-temp cmd on 5996
  `[12.3, 14.5]` (vs collapsed `[10.1, 10.9]`, RBC 26), return −27 (vs −37), ~3×
  worse than RBC (policy ≈ −0.04/step vs RBC −0.014). NB: **train `step_reward` is
  a noisy single-building signal — trust the eval.** So exploration is a real
  contributing factor but not the root cause; the policy still under-heats.
- **Next:** larger `ent_coef` (0.1) is unlikely to close a 3× gap → pivot to H8.

### H10 — test set is just hard / unlucky held-out split (ruled out)
- **Motivation:** inter-building variance looked high; "0/5 test" rests on a small,
  fixed split (10 train / 5 test buildings, one 7-day chunk each).
- **Exp:** eval the specialist on the 5 buildings it *trained on* and the 5
  held-out, policy vs RBC, 672-step chunk (`scratchpad/eval_traintest.py`).
- **Result: ruled out.** **1/5 train, 0/5 test**, and the lone train "win" is a
  degenerate exact tie (−173.31 vs −173.31 on 5005). Losing on the *training*
  buildings means it's not a generalization/split artifact — it's fundamental.
  Numbers (policy vs RBC): train 5001 −17.2/−17.0, 5002 −21.3/−15.5, 5003
  −26.5/−22.9, 5004 −21.1/−17.9; test 5996 −26.7/−9.4, 5997 −25.0/−7.1, 5999
  −37.8/−9.0, 6000 −34.9/−10.7, 5014 −44.9/−44.3.
- **Why test *gaps* look bigger:** the policy is **building-invariant**
  (under-heats to ~−20…−45 everywhere); RBC *adapts* and is actually better on
  test (−7…−10) than train (−15…−23), so the policy's shortfall is widest exactly
  where RBC does best. RBC return itself spans −7…−173 across the 10 → variance is
  real, but it's not what causes the loss.

### H9 — discount horizon (partial)
- **Exp:** OfficeSmall specialist, `γ=0.99` + `ent_coef=0.05` (H9 horizon + H7
  exploration combined), 400 iters (wandb `6x7bzrqb`; `runs/officesmall_g99_e05`).
- **Result: partial.** Entropy held far higher (6.16→**3.34**, no collapse), and
  test returns improved a few points over ent0.05 alone — but **still 0/5 train and
  0/5 test**, still ~2.5× worse than RBC. Ladder on test-5996 (RBC −9.4):
  ent0.01 **−37** (4×) → ent0.05 **−26.7** (2.8×) → γ0.99+ent0.05 **−22.7** (2.4×).
  The two leading levers **monotonically improve** the policy but **plateau well
  short** of a trivial baseline. So horizon + exploration are real *contributing*
  factors, neither is *the* fix. Same shape as H7.
- **Caveat / lesson:** first g99 eval accidentally reloaded the ent05 checkpoint
  (a `sed` on `os.path.join` string components silently no-op'd) and printed
  numbers *identical to the decimal* — caught by the smell test. Always diff the
  checkpoint (md5) when comparing runs.

### H11 — resampling drift / plasticity loss (ruled out; but reframes the failure)
- **Motivation:** the resample loop swaps the active building every 10 updates —
  a second, large non-stationarity on top of RL's intrinsic one (inter-building
  variance is huge: RBC return spans −7…−173). Continual-learning theory says such
  drift causes **loss of plasticity** (Dohare/Sutton Nature 2024, Nikishin primacy
  bias 2022, Sokar dormant neurons 2023) — degrading the *ability to learn*, not
  just test generalization. The building-invariant under-heating collapse (worse
  with training, entropy collapse) fits that signature.
- **Exp:** train on a **single fixed building** (`--n-buildings-per-type 1`,
  OfficeSmall idx 0), zero resampling drift, γ=0.99+ent=0.05, 400 iters
  (wandb `ppw0w3f3`; `runs/officesmall_single`). Eval on 5001 vs RBC −16.96.
- **Result: ruled out.** Lands at **−17.17 on 5001 — identical to the
  drift-trained policies** (checkpoints verified distinct by md5; 5002–5005 differ,
  proving the eval loads different weights — 5001 is just policy-invariant like
  5005). Entropy collapsed *less* (1.82 vs 1.27) yet reached the same basin. So the
  resampling non-stationarity is **not** the cause — PPO under-heats identically
  with a perfectly stationary target.
- **What survives — the failure is eval-time generalization, not training drift.**
  The policy is ~competitive with RBC on buildings near its training set (5001 gap
  0.2) but **collapses on held-out buildings, worst on the *easy* ones** (RBC −7…−10
  → policy −25…−37, 2–4×). Single-building training can't fix that (generalizes
  worse). This is BAN's *actual* use case (distill → fresh plastic net → better
  generalization), so the distribution-shift instinct relocates from training to
  deployment.

### H12 — Born-Again self-distillation (ruled out)
- **Idea (user):** fresh-init an identical net (full plasticity) and distill the
  teacher's outputs into it; BANs (Furlanello et al. 2018) beat their teachers on
  generalization. Motivated by the loss-of-plasticity framing of H11.
- **Exp:** teacher = best specialist (g99_e05). Rolled it out on the 10 train
  OfficeSmall buildings, recording per-state observations + the teacher's full
  action distribution (ScaledBeta α,β — the continuous analog of dark knowledge).
  Fresh-init net (seed 999), fit α,β via MSE, 150 epochs (`scratchpad/ban_distill.py`;
  `runs/officesmall_ban`). Distill loss 5.53 → **0.0004** (near-perfect fit). Eval
  on train+test (`scratchpad/eval_ban.py`).
- **Result: ruled out.** Student ≈ teacher within rollout noise (<1 pt) on **every**
  building (test 5996 −22.2 vs −22.7, 5997 −23.6 vs −23.1, 6000 −29.0 vs −29.9),
  still **0/5 train, 0/5 test**.
- **Why it doesn't fire here (a real finding):** BAN's gain comes from *dark
  knowledge in a softmax over classes* — relative probabilities on wrong classes
  encode inter-class similarity that regularizes the student. A continuous Beta
  policy has no class structure; matching (α,β) is plain function-fitting, so the
  student re-creates the controller. And the fresh-init plasticity benefit is
  **undone** by training to convergence on the teacher's outputs (re-collapse).
  Self-distillation can't manufacture competence the teacher lacked.
- **Untried synthesis:** the version with a real mechanism is distilling a
  *competent* teacher — imitate **RBC** into the net, then RL-fine-tune — which
  injects the heating behavior PPO won't discover. That's imitation warm-start, not
  BAN.

### H14 — warm-started PPO fine-tune (RL overfits; imitation strictly better)
- **Idea:** load the RBC-clone (H13) as PPO's init and fine-tune so PPO starts in the
  good basin and only refines — the AlphaGo-style SL-init → RL path to *beat* RBC.
  (Added `init_model` hook to `train_transfer`; `--init-checkpoint` to the port script.)
- **Ablation ladder** (all warm-started from the RBC clone; OfficeSmall pool 10, 200 it):
  | run | ent_coef | critic | outcome |
  |---|---|---|---|
  | 1 (`kji95yzt`) | 0.005 | random | **destroyed** — entropy bonus inflated −1.2→+5.1, step_reward −0.026→−0.53 |
  | 2 (`80afqmwc`) | 0 | random | **destroyed** — random-critic advantages knocked it out; test −32…−40, 5005 −590 |
  | 3 (`2irn8wz0`) | 0 | **pretrained** | **stable on train, overfits test** (below) |
- **The critic-init flaw** (user-spotted): BC (`ban_rbc.py`) fits only the *policy*; the
  value head stays at random init, so PPO's first advantages are garbage and knock the
  good policy out before the critic catches up (run 2 dipped to −0.256 as value_loss
  spiked to 8.5). Fix = `value_pretrain.py`: re-roll RBC, compute discounted
  returns-to-go (γ=0.99), fit **only the value head** (policy frozen via grad-masking);
  value MSE 170→30. → `runs/officesmall_rbc_bcv`.
- **Result (run 3, the properly-controlled one).** The critic fix restored *stability*:
  train held RBC-competitive (5002 −15.7 vs RBC −15.5, 5005 −173 not −590), no entropy
  blow-up (−1.2→−4.9, sharpening not diverging), value_loss ~1.5. **BUT PPO still
  overfits:** held-out test collapsed back to from-scratch-PPO levels (5996 −32.5,
  5999 −36.7, 6000 −40.4 vs BC clone's −15.9/−13.7/−20.0), and it didn't even beat pure
  BC on train. Still **0/5 train, 0/5 test.**
- **Conclusion — the capstone.** The **best OfficeSmall controller is the pure RBC
  imitation; PPO fine-tuning strictly harms it.** Given a perfect policy *and* critic
  *and* no entropy pathology, PPO's objective itself drives the policy to **sharpen onto
  the training buildings and lose generalization** — supervised imitation generalizes
  across the building distribution, the RL objective does not. The OfficeSmall failure is,
  at root, that **PPO overfits the training buildings**; for this transfer problem
  imitation is strictly better than RL. (Best policy: `runs/officesmall_rbc_bc`.)

### H13 — RBC imitation warm-start (capacity confirmed; the key result)
- **Idea:** distill the *competent* controller (RBC), not the failed policy. Behavior-
  clone RBC into a fresh net; if it matches RBC, the net can represent the control law
  and the failure is purely PPO's; the clone also warm-starts an RL fine-tune.
- **Method:** RBC acts in physical action space; the policy emits a *target* (normalized
  [−1,1]) action mapped to physical by a fixed affine. Probed that affine per building
  (join at f=0 and each basis vector), pseudo-inverted it to turn RBC's actions into
  target actions (`recon_err = 0.000` on all 10 → map is exactly affine, D=10 = 5 zones ×
  {supply_temp, fan}). Fit a fresh net's Beta *mean* to those targets, 200 epochs, MSE
  0.48 → 0.0076 (`scratchpad/ban_rbc.py`; `runs/officesmall_rbc_bc`).
- **Result — capacity CONFIRMED.** On the imitated train buildings the clone **matches
  RBC**: 5001 −16.95/−16.96, 5002 −14.89/−15.51 (both beat RBC), 5003 −23.87/−22.89,
  5004 −18.07/−17.86 (within ~1 pt) → **2/5 beat RBC**. Same net + obs + buildings as the
  PPO runs; only the training signal differs (imitation vs RL). So the good policy is
  representable and gradient-reachable — **PPO's failure is optimization/exploration,
  proven by construction, not architecture/observability/representation.**
- **Generalization bonus:** on held-out test the clone is ~2–3× closer to RBC than PPO —
  5996 −15.9 (PPO −22.7), 5997 −14.1 (−23.1), 5999 −13.7 (−35.2), 6000 −20.0 (−29.9) —
  though still 0/5 (an approximation with ~2 °C action slack, never trained on these).
  Imitation transfers across buildings far better than the RL policy.
- **Next (the money experiment):** RL-fine-tune PPO **from this warm-start** instead of
  random init. PPO would start already heating (in the good basin) rather than having to
  discover heating by exploration — the concrete path to finally *beat* RBC on OfficeSmall.

## Sampling pipeline & the "~250-step cliff" (mechanism note)
- **Split:** `splits.json` fixes `type→split→[ids]`; `get_building_by_index` is
  `ids[index]` in stored order (no runtime sort/shuffle). Train pool =
  `(bt, idx) for idx in range(n_buildings_per_type)` = first N train IDs; test =
  first 5 test IDs (disjoint). (`registry.py:176,185`; `training.py:208`.)
- **Resample:** `sample_batch` (`training.py:224`) draws exactly one building per
  type, uniform-with-replacement from the pool, every `resample_interval=10`
  updates, seeded by `rng_seed=seed`.
- **The cliff is largely a plotting artifact.** `rollout/step_reward` is measured
  on whichever single building is currently active. That building switches every 10
  updates on a **seed-deterministic** schedule, so all seed-0 runs cliff at the
  *same* update (the reward's biggest jump sits at update 120 = a resample
  boundary). High inter-building variance makes those switches look like cliffs.
  It is **not** primarily a learning-collapse signal — the entropy curve (which is
  policy-level) has its own separate knee near update 200. Wandb x-axis is
  env-steps (`_step` = update × n_steps = 672…268800).

## Infra notes (not hypotheses, but bit us during the investigation)
- **JAX OOM on long runs.** Cold-start rebuilds the env/morphology **every update**;
  because the morphology's split/join callables are `eq=False`, JAX **retraces the
  JIT'd kernels each update** and the compiled XLA sections accumulate until the CPU
  JIT OOMs (~180 iters). The aggressive persistent compile cache
  (`min_compile_time_secs=0.0`) aggravates it (and manifests earlier as
  `Failed to materialize symbols`). **Workaround:** `jax.clear_caches()` every 10
  updates in `train_ppo_resampling` (added to the vendored `ppo.py` working tree —
  a real fix worth committing to morel). Do **not** disable the persistent cache —
  that forces more recompiles and OOMs sooner.

---

# Part II — 4-type transfer under the full-year regime

**Motivation.** H16 showed the failing regime was *cold-start*, which resets to step 0
every update and therefore trains only on the **first 672 steps (2.33 days) of January**.
The **original transfer experiment used exactly that regime**, so the published transfer
result may be limited by a training artifact rather than anything fundamental. Part II
re-runs it with persistent full-year rollouts.

## Runs

All: 4 types × 5 buildings = 20 persistent envs, `task_occ_e0`, from scratch, and the
**ORIGINAL hyperparameters** (lr 5e-5 / ent 0.01 / γ 0.98) so that only the *regime* changes.

| run | updates | chunk-eval vs RBC | notes |
|---|---|---|---|
| `transfer4_persistent` (wandb `a4qh14t1`) | 74 | **8/20** | under-trained |
| `transfer4_persistent_long` (wandb `t3x54yyn`) mid snapshot ≈100 | ~100 | **19/20** | best chunk result |
| `transfer4_persistent_long` final | 400 | **14/20** | OfficeMedium collapses |

**Budget was the dominant factor** (8/20 → 19/20 from more updates alone), confirming the
first "mixed" verdict was an artifact of under-training.

### Chunk-eval detail (672-step window; the caveat at the top applies)
| type | orig transfer | mid (~100) | final (400) | RBC |
|---|---|---|---|---|
| RetailStandalone | −68.5 | −93.4 (4/5) | **−73.4 (5/5)** | −79.3 |
| RestaurantFastFood | −14.1 | −10.1 (5/5) | −10.6 (4/5) | −11.9 |
| OfficeMedium | −61.0 | **−37.1 (5/5)** | **−186.1 (0/5)** | −55.8 |
| OfficeSmall | −31.0 | −14.3 (5/5) | −13.7 (5/5) | −16.1 |

### OfficeMedium: overfits with extended training
Between ~100 and 400 updates its **training** return kept improving (−263 → −65) while its
**held-out** return collapsed (−37 → −186). Training is the *stochastic* policy
(pessimistic) and eval the *deterministic* one (optimistic), so the gap runs the **wrong
way** — that is overfitting, not noise. OfficeMedium is the most complex morphology
(15 zones, ~30 action dims, the only VAV type) trained on only 5 buildings, and post-156
updates the extra passes are pure epochs on fixed data. **Fix: more OfficeMedium buildings**,
not fewer updates; early stopping is only a workaround.

### Full-year evaluation (the honest test) — IN PROGRESS
- **First attempt was INVALID and its numbers are void**: the eval capped at 40,000 steps
  (38% of a year) while the RBC table covers the full year, so the policy accrued less
  penalty purely by stopping early. It reported "16/20"; ignore it.
- Corrected eval runs to termination (105,120 steps) against
  `building2building/scores/baseline_returns.csv` (`task_occ_e0`, `full_year`).
- **Early valid data point**: mid checkpoint on Retail-2997 = **−9087.5 vs RBC −3479.0**
  (2.6× worse), and Retail-2998 −6085.7 vs −3025.4 — i.e. the chunk result **reverses**
  over a full year.
- **Leading explanation — a coverage hole.** The mid checkpoint ran 100 × 672 = 67,200 of
  105,120 steps = **64% of a year: it never saw Sept–Dec**. Degradation is *super-linear*
  (linear scaling predicts −4457, actual −9087), consistent with failing on unseen
  conditions. The final checkpoint (2.56 passes) has full coverage → prediction: it should
  do markedly better on the full year, reversing the chunk ranking. **Eval running.**

### Seasonality — a correction
Initial claim ("winter is harder, confirmed") was **an artifact**: I imposed the wrong
period (52 vs the true ~156) and reported *phases* rather than *indices*, so a monotonic
training trend masqueraded as seasonal clustering (the worst updates are simply indices
1–26, i.e. before the policy learned). Detrended autocorrelation is ≈0 at both lags.
**However**, the year-wrap *is* visible and large: at updates ≈156 and ≈312 the return
jumps from ~−5 to ~−40 (**8× degradation**) as the env resets to January. So a winter/reset
penalty is real; autocorrelation was simply the wrong instrument for a sharp, localised
spike. (The wrap conflates January weather with a cold thermal reset; this data can't
separate them.)

## Infrastructure added in Part II
- **Periodic checkpointing** (`snapshot_every=25`) — previously the rolling checkpoint was
  overwritten every update, so a run's best model was silently lost. The 19/20 weights
  survived only because they were copied by hand for an eval.
- **Per-building return logging** (`return/<type>_<idx>`) in both trainers.
- **RBC baselines per train pool** (`compute_rbc_baselines.py`, correct controller per type:
  `AirLoopPolicy` for OfficeMedium/VAV, `UnitaryHvacPolicy` otherwise) + `--baseline-json`
  for constant wandb reference lines.
- **Parallel full-year eval** (`eval_fullyear_panel.py --n-workers`): spawn process per
  building, idx-major ordering so a first cross-type read arrives early.
- **Figure renderer parameterized** (`--eval-csv/--baseline-csv/--outdir/--prefix`);
  defaults reproduce the original camera-ready figure unchanged (verified).
- **Live progress trick**: each rollout writes `/tmp/b2b_eplus_*/eplusout.eso`, whose
  `2,<day-of-year>,<month>,<day>` records give the simulation date — poll it for progress
  without touching the run.

## Open questions
1. **Full-year result** for the final checkpoint (running) — does full coverage reverse
   the chunk ranking?
2. **n=1.** Every Part II number is a single seed.
3. **OfficeMedium pool size** — 5 buildings for the most complex morphology.
4. **Which protocol should the mémoire headline?** Chunk (2.33 days) is what the original
   experiment used; full-year is the honest annual claim. They disagree.

## Key artifacts
- Part II checkpoints: `runs/transfer4_persistent_mid/` (≈100 upd, chunk 19/20),
  `runs/transfer4_persistent_long/` (400 upd, chunk 14/20).
- Part II scripts: `scripts/train_allactive.py`, `eval_transfer_panel.py`,
  `eval_fullyear_panel.py`, `compute_rbc_baselines.py`.
- Checkpoints: `runs/officesmall_only/` (150-iter), `runs/officesmall_resample_400/`
  (400-iter). Transfer: `checkpoints/model_s0..2.eqx`.
- Scratchpad scripts (capture/plot/probe): `capture_officesmall.py`,
  `capture_retail.py`, `plot_officesmall.py`, `plot_compare.py`,
  `custom_action_probe.py`, `probe_amorpheus_setpoint.py`, `tau_lookup.py`,
  `eval_spec400.py`, `train_officesmall_rawtau.py`.
- wandb project: `justin-veilleux-mila/morel-b2b-transfer-port`.
