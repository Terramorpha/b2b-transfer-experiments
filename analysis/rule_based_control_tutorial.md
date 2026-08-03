# Rule-Based HVAC Control — A Complete Tutorial

*How the Building2Building reactive baselines actually work, and what you need to
know to write real-world HVAC control policies.*

This document dissects the two hand-written controllers B2B uses as baselines —
`UnitaryHvacPolicy` (`baselines/controllers/unitary_hvac.py`) and `AirLoopPolicy`
(`baselines/controllers/air_loop.py`) — and uses them to teach the general craft
of reactive supervisory control. Read it end-to-end and you will understand not
just *these* controllers but the primitives, failure modes, and design decisions
behind essentially every "sequence of operations" running in a real building
automation system (BAS).

Everything here is **reactive**: the controller sees the current observation and
emits an action. No model of the building, no forecast, no optimization over a
horizon. That is exactly what makes it (a) the thing a learned policy must beat,
and (b) a surprisingly hard bar, because decades of building-automation practice
are distilled into these rules.

---

## 0. The mental model in one paragraph

A thermal zone is a leaky tank of air. Weather, sun, people, and equipment pour
heat in and out. HVAC fights back by blowing conditioned air into the zone. You
have two fundamental knobs: **how much air** (mass flow, kg/s) and **how
conditioned** (supply-air temperature, °C). The heat you deliver is roughly

```
Q ≈ ṁ · cp · (T_supply − T_zone)      [Watts]
```

— a **product** of the two knobs. Everything below is machinery for turning "the
zone is 1.4 °C too warm right now" into good values of `ṁ` and `T_supply`, for
one zone or for many zones sharing one air handler, without oscillating,
without wasting energy, and without ever fully shutting off the ventilation
people need to breathe.

---

## 1. The physical plant: what you are actually commanding

### 1.1 Two system archetypes

| | **PSZ unitary** (`UnitaryHvacPolicy`) | **Central VAV** (`AirLoopPolicy`) |
|---|---|---|
| Used for | RetailStandalone, RestaurantFastFood, OfficeSmall | OfficeMedium |
| Topology | one packaged unit **per zone** | one air handler → many **zones** via ducts |
| Per-zone control | its own fan + coil | a **damper** taps a shared air stream + local **reheat** |
| Hard part | tune one loop, replicate per zone | **one supply temperature must serve conflicting zones** |

The PSZ case is "N independent thermostats." The VAV case introduces the central
coordination problem that dominates real commercial HVAC: a single air handler
produces *one* stream of cold air, and every zone must be satisfied from it.

### 1.2 The actuators and their physical meaning

You never command a temperature *directly*. You command **setpoints and flows**;
the EnergyPlus plant model then tries to achieve them. Getting this distinction
right is the first thing that separates people who can write HVAC control from
people who can't.

- **Fan / supply-air mass flow** `[kg/s]` — *how much* air. The capacity knob.
  In the PSZ this is a direct mass-flow actuator; in VAV it is a per-zone
  **damper flow fraction** `[0..1]` that splits the shared stream.
- **Supply-air-temperature (SAT) setpoint** `[°C]` — *how conditioned* the air
  is. You set the target the coil regulates to; the coil (and its capacity
  limits) decide whether it's reached.
- **Reheat setpoint** `[°C]` (VAV only) — a small terminal coil that re-warms
  over-cooled air for a specific zone. This is how a central "cold air" plant
  keeps a *cold* zone comfortable without warming everyone.
- **Outdoor-air (OA) mass flow** `[kg/s]` — ventilation / economizer air.
- **Availability** `[0/1]` — is the system allowed to run at all this step.

Key consequence: **"tracking the setpoint" is not the same as "delivering the
capacity."** A policy can command exactly the right SAT and still fail, because
it never opened the fan/damper enough to move the air. This is not hypothetical —
it is precisely why the learned policies in this project scored 0/20 over a full
year (they froze the fan near mid-range while the RBC ramped it to ~70%; see §7).

---

## 2. Control primitives (the theory you must internalize)

### 2.1 Error and the deadband

The **error** is how far the zone is from where you want it. But "where you want
it" is usually not a point — it's a **band**: heat if below `heat_sp`, cool if
above `cool_sp`, do nothing in between. Two reasons:

1. **Physics/economics:** driving a free-floating zone to an exact degree wastes
   energy fighting noise; a band lets it drift cheaply.
2. **Actuator wear:** a point setpoint makes the plant hunt (constant small
   corrections). A band gives it rest.

Both controllers compute error relative to a band. The unitary loop uses the
*nearest edge* (`unitary_hvac.py:353-359`); the VAV loop uses a signed error to
a single target with an optional `deadband` cushion (`air_loop.py:115-126`).

> **Subtlety — the deadband can *hurt* you.** Under a comfort-only reward
> (`energy_weight = 0`), sitting anywhere inside the band earns the same reward,
> so a controller that "relaxes" inside the band merely lets the zone drift
> *toward the penalty edge* for no benefit. Measured cost on OfficeMedium
> full-year: **−35%**, enough to wipe out the +24.5% that correct setpoint
> tracking earned (`air_loop.py:36-42`). That is why `AirLoopConfig.deadband`
> defaults to `0.0`. A deadband is a *feature only when energy is priced.*

### 2.2 P, then PI

**Proportional (P):** `u = kp · e`. Simple, but has a fatal flaw — **steady-state
offset.** To hold any nonzero output you need a nonzero error, so a P-only loop
settles *near* the setpoint but never *on* it (heavier load → bigger residual
error).

**Proportional-Integral (PI):** `u = kp·e + ki·∫e`. The integral accumulates
past error and keeps pushing until the error is actually zero. This is the
workhorse of building control. See `_pi_step` (`unitary_hvac.py:95-99`):

```python
def _pi_step(error, state, kp, ki, i_max):
    state.integral = min(state.integral + error, i_max)   # accumulate (clamped)
    return kp * error + ki * state.integral
```

### 2.3 Integral windup — and three ways this code fights it

The integral term's strength is also its danger. If the plant is **saturated**
(fan already at max, still can't cool the zone), the error stays positive and
`∫e` grows without bound — "**windup**." When the load finally drops, the huge
accumulated integral keeps the actuator pinned for a long time, causing a big
overshoot. Anti-windup is not optional; it is the difference between a controller
that works and one that oscillates for hours. This codebase uses **three**
distinct techniques — learn to recognize all of them:

1. **Clamp the integral** to `±integral_max` (`unitary_hvac.py:98`,
   `air_loop.py:378-380`). Crude but effective ceiling.
2. **Conditional integration** — *don't* accumulate when you're saturated in the
   direction that would make windup worse (`air_loop.py:370-377`). This is the
   textbook-correct method: the integral only grows while it can still do
   something.
3. **Bleed / decay** — shrink the integral when there's no demand
   (`z.air_pi.integral *= 0.8` inside the band, `unitary_hvac.py:350,358`) or
   every step (`z.integral *= cfg.integral_decay` = ×0.99, `air_loop.py:369`).
   A leaky integrator forgets stale error.

### 2.4 Rate limiting (slew limits)

Real dampers, valves, and fans cannot slam from 0 to 100% in one step, and
doing so thermally shocks the space and the equipment. Every VAV command is
**rate-limited**: SAT ±0.3 °C/step (`air_loop.py:349-353`), flow ±0.04/step
(`air_loop.py:387-393`), reheat ±0.2 °C/step (`air_loop.py:412-418`). The pattern
is always: compute a `*_target`, then move `prev` toward it by at most
`rate_limit`. Rate limits trade responsiveness for smoothness and stability.

### 2.5 Trim-and-Respond (T&R) — the reset strategy from real BAS

This is the crown jewel and the one most worth learning. **Trim-and-Respond** is
the ASHRAE Guideline 36 standard way to reset a shared setpoint (SAT, static
pressure, hot-water temperature…). The idea:

- When a zone **demands** more (too warm), **respond** aggressively: step the
  setpoint hard in the helpful direction.
- When everyone is satisfied, **trim** gently: drift the setpoint back toward a
  neutral/efficient value in small steps.

The **asymmetry is the point** — fast to help, slow to relax — so the setpoint
finds the least-aggressive value that still satisfies demand (energy optimal)
without being sluggish when demand appears. The unitary SAT loop is textbook T&R
(`unitary_hvac.py:365-377`):

```python
if   t_zone - cool_sp > deadband:  sat_sp -= sat_respond   # too warm → colder air (respond, big step 0.5)
elif heat_sp - t_zone > deadband:  sat_sp += sat_respond   # too cold → warmer air (respond)
elif sat_sp < sat_initial:         sat_sp += sat_trim      # satisfied → drift to neutral (trim, small step 0.2)
elif sat_sp > sat_initial:         sat_sp -= sat_trim
sat_sp = clip(sat_sp, sat_min, sat_max)
```

### 2.6 Error smoothing (EMA)

Zone temperatures are noisy (sensor + simulation). The VAV loop runs each zone's
error through an exponential moving average before using it
(`air_loop.py:327-336`, `smooth = α·raw + (1−α)·smooth`, α=0.3). Smoothing
buys stability at the cost of a little lag — a classic control trade-off. Note
the first-step special case: on `reset` the EMA is *seeded* with the raw error
rather than starting at zero (`air_loop.py:329-330`), so the loop doesn't lurch
on step one.

---

## 3. Controller I — Unitary PSZ, line by line

### 3.1 Binding: discovering the building from metadata

A single controller object must work on RetailStandalone (5 zones), a restaurant,
a small office — buildings it has never seen. It does this by **reading the
building's equipment list at bind time** rather than hard-coding anything
(`unitary_hvac.py:209-309`):

- Pull `observation_names` and `action_names` (flat string lists) from
  `env.metadata`.
- For every `unitarysystem` in `hvac_equipment`, find its fan and SAT actuator
  **by matching `component_type::control_type::component_name` strings**
  (`_match_actuator_index`, `:128-138`), and find its zone-temperature
  observation by name (`find_zone_air_temp_index`, `metadata.py:30`).
- Read `fan_max` from `env.action_space.high[fan_idx]` — **the actuator's
  physical range is per-building**, not a constant (`:261`). This is how the
  same code adapts to different equipment sizing.
- Keep an independent `_ZoneState` (its own PI integrator and SAT setpoint) per
  zone (`:263-272`). Also collect `availability` actuators and heating-only
  **baseboard** zones (`:274-296`).

This name-based, per-zone autodiscovery is what makes a rule-based controller
"morphology-agnostic" *without learning* — contrast the learned policies, which
get morphology structure through the graph/attributes.

### 3.2 Setpoint sourcing — a 3-level fallback

`_current_setpoints` (`unitary_hvac.py:317-340`) resolves the band, in priority
order:

1. **Observed per-zone target** (best): if the task exposes a
   `target_temperature` for the zone, center the band on it:
   `heat = target − gap/2`, `cool = target + gap/2` (`:322-325`). This makes the
   controller **follow the task's dynamic occupancy schedule** (e.g. 21 °C
   occupied / 18 °C setback) automatically.
2. **Internal schedule** (weekday/weekend/setback by time-of-day and
   day-of-week) if enabled (`:327-340`).
3. **Fixed** `heating_setpoint_c` / `cooling_setpoint_c` (20/22) otherwise.

> **Cautionary tale.** The VAV baseline originally had *no* level-1 fallback —
> it regulated to a hard-coded 21 °C and never bound the `target_temperature`
> observation. Under `task_occ_e0` (21 occupied / 18 setback) it tracked the
> *wrong* target: measured correlation between commanded SAT and the true
> setpoint was **−0.45** (anti-correlated!). Fixing it (binding the observation)
> flipped correlation to **+0.71** and gained **+24.5%** full-year on
> OfficeMedium. **Lesson: the single most important line in an HVAC controller
> is the one that reads the right setpoint. Bind the observation, don't hardcode
> the constant.**

### 3.3 The fan airflow PI loop

`_airflow_command` (`unitary_hvac.py:342-363`):

- **Error mode** (`fan_error_mode`):
  - `"nearest_setpoint"` (default): `err = t−cool_sp` when hot, `heat_sp−t` when
    cold, and **0 inside the band** — the fan sits at its floor and the
    integrator bleeds (×0.8). The loop is *inactive* inside the deadband.
  - `"center_of_band"`: `err = |t − band_center|`, so the loop is active for any
    deviation. Use this when you want tighter tracking (at an energy cost).
- **The minimum-flow floor.** Output scales from `min_fan_frac · fan_max` up to
  `fan_max` (`:345,362`). The fan **never goes to zero** while available —
  ventilation and air mixing require a minimum airflow. Forgetting this floor is
  a classic beginner bug (dead zones, stale air, sensor lag).
- **Magnitude is everything.** Bigger error → bigger `u` → more air, ramping to
  `fan_max` under peak load. This "open the fan proportionally to how far off you
  are" is the behavior learned policies most often fail to reproduce.

### 3.4 The warm-up guard

`if |t_zone − prev_temp| > 3 °C: reset the PI integrator` (`unitary_hvac.py:393-394`).
EnergyPlus warm-up and midnight thermal resets can teleport a zone temperature;
without this guard the integrator would wind up on a discontinuity that isn't a
real control error. **Every real controller needs logic to reject
non-physical transients** (sensor dropouts, restarts, mode changes).

### 3.5 Putting it together (`predict`, `:379-404`)

Each step: force availability ON; for each zone resolve the band, read temp,
guard warm-up, write `fan_idx ← airflow PI` and `sat_idx ← SAT T&R`; for each
baseboard write its heating setpoint. Deterministic, O(zones), no history beyond
the per-zone integrator and SAT state.

---

## 4. Controller II — Central VAV air loop, line by line

The VAV controller (`air_loop.py`) solves the same problem *plus* the central
coordination problem. Structure per loop each step (`predict`, `:294-433`):

### 4.1 The shared-SAT conflict and its resolution

One SAT must serve all zones on the loop. Whose demand wins? The controller
computes a **weighted blend** of the per-zone errors (`:339-347`):

```
weighted_err = w_cold·min(errors) + w_warm·max(errors) + w_mean·mean(errors)
             = 0·min + 0.6·max + 0.4·mean          # defaults
sat_target   = sat_neutral − sat_kp·weighted_err + outdoor_offset
```

With `w_warm = 0.6`, SAT is **biased toward the warmest zone** (`max` error) —
supply cold enough to satisfy the worst cooling case, and let terminal **reheat**
handle any zone that is now too cold. This "cool for the hottest, reheat the
rest" is the canonical VAV strategy. (`w_cold` exists for heating-dominated
loops; `outdoor_sat_gain` lets SAT track outdoor temperature — both off by
default.) SAT is then clipped and **rate-limited** (`:348-356`).

### 4.2 Directional flow with a SAT-aware sign flip

The same VAV damper can *cool* (send cold supply air to a hot zone) or *heat*
(send warm supply air to a cold zone) — **more flow helps in opposite directions
depending on whether supply is colder or warmer than the zone.** The controller
detects the regime and flips the sign (`air_loop.py:367,382-386`):

```python
in_cooling = sat_aware_flow and sat < t_zone     # supplying colder-than-zone air
flow_sign  = -1.0 if in_cooling else 1.0
flow_target = flow_base − flow_sign·(flow_kp·err + flow_ki·integral)
```

- Cooling regime (`sign=−1`): too warm (`err>0`) ⇒ **more** flow. ✓
- Heating regime (`sign=+1`): too cold (`err<0`) ⇒ `−err>0` ⇒ **more** flow. ✓

Getting this sign wrong makes the controller *drive the zone away* from setpoint
— a subtle, catastrophic bug. The **conditional integration** (`:370-377`) is
also regime-aware: it refuses to accumulate when the damper is already pinned at
`flow_max`/`flow_min` in the unhelpful direction.

### 4.3 Terminal reheat

Only when a zone is cold beyond the reheat deadband does the reheat setpoint rise
(P-control on the cold excess, `:399-411`); otherwise it sits at `reheat_sp_min`.
This is the mechanism that lets one cold zone stay comfortable while the shared
SAT stays cold for the hot zones — at the cost of "simultaneous heating and
cooling," VAV's well-known energy penalty.

### 4.4 OA pinned constant — a deliberate baseline choice

`action[oa_act_idx] = cfg.oa_mass_flow` (constant 1.37 kg/s, `:358-362`). The RBC
does *not* modulate outdoor air; it supplies a fixed minimum-ventilation floor.
This is intentional: OA/economizer modulation is exactly the degree of freedom
the *learned* agent is meant to discover, so the baseline holds it constant to
make that learning measurable. **A good baseline is deliberately simple where you
want the learner to add value.**

---

## 5. Cross-cutting subtleties (the expert field guide)

1. **Setpoint ≠ delivered value.** You command a SAT *setpoint*; the coil may not
   reach it (capacity limits, saturation). You command a *flow*; the fan curve
   and duct static decide the rest. Always reason about what the plant can
   physically deliver, not just what you asked for.

2. **Capacity is a product of two knobs.** `Q ≈ ṁ·(T_sup−T_zone)`. You can be
   perfect on one and useless overall. The 0/20 failure was *good SAT tracking,
   frozen fan* → near-zero delivered capacity. When you audit a controller,
   audit **actuator range utilization**, not just setpoint error.

3. **Physical units vs normalized actions.** The RBC works in physical units
   (kg/s, °C) and reads `action_space.high` to learn each building's ranges. A
   learned policy usually emits normalized actions in `[-1,1]`/`[0,1]`; the
   env rescales them. Mixing the two frames up (e.g. treating "0" as "off" when
   normalized-0 maps to mid-range) is a common and invisible bug — and is
   exactly what makes an untrained policy sit at ~mid-range airflow.

4. **Per-building actuator ranges.** `fan_max` differs per building
   (`unitary_hvac.py:261`). Hard-coding a flow constant that worked on one
   building silently misbehaves on another. Always scale to discovered ranges.

5. **Deadband is a liability under comfort-only reward** (§2.1). Match the
   controller's "relax zone" to the reward's "indifference zone," or you leak
   performance.

6. **Anti-windup is mandatory, and there are several kinds** (§2.3). If your
   loop overshoots for a long time after a hot afternoon, you have windup.

7. **Reject transients.** Warm-up jumps, restarts, and year-wraps produce
   non-physical error spikes; reset integrators across them
   (`unitary_hvac.py:393`). B2B's own analysis found an **8× return degradation
   spike at the January year-wrap** — reactive controllers with memory must be
   defended against discontinuities.

8. **Multi-zone starvation.** A single shared SAT biased to the worst zone can
   under-serve the others; reheat covers cold zones but nothing covers a zone
   that needs *more cooling* than the blend provides. Central systems always
   involve someone being slightly unhappy — the weights choose who.

9. **Reactive control is fundamentally lagging.** No anticipation means it can't
   pre-cool before a known occupancy spike or pre-heat before a cold morning. It
   reacts after the zone has already drifted. This is the ceiling that motivates
   MPC and RL — and the reason beating a *well-tuned* reactive controller is a
   real result, not a formality.

---

## 6. How the baseline is scored (and why fairness is subtle)

The number a policy is compared against (`data/rbc_fullyear_ourharness.json`) is
produced by running **this exact controller through the same EnergyPlus harness**
for a full year, per building. Two rules the project learned the hard way:

- **Only ever compare within one harness.** The published
  `building2building/scores/baseline_returns.csv` reports RBC scores 2.3–3.4×
  *smaller in magnitude* than ours on identical buildings — almost certainly a
  different `timesteps_per_hour`. Comparing a policy against a mismatched
  baseline made it look ~3× worse than reality. Regenerate baselines with your
  own harness.
- **Evaluate on the full year, not a chunk.** A 672-step (2.3-day January) window
  *flatters* policies — they can beat RBC there while losing over the year (the
  chunk scored a policy 19/20 that was truly 0/20). Report the annual number.

---

## 7. Why the RBC is hard to beat — and what a learner must add

The RBC's fan/damper PI loop **actuates across the full range in proportion to
demand** (RBC used ~70% of the OfficeSmall fan range at peak; the learned policy
used ~2.3% — a 30× gap). It also tracks the dynamic setpoint *better* than a
naive policy. So a reactive controller, well tuned, is already:

- correct in **direction** (right sign, right regime),
- correct in **magnitude** (ramps the actuator to meet load),
- correct in **setpoint** (follows the occupancy schedule),
- stable (rate limits, anti-windup, smoothing).

To beat it, a learned policy must add what rules *cannot* do: **anticipation**
(pre-conditioning before load arrives), **cross-zone coordination** beyond a
fixed weight blend, and **energy-aware trade-offs** the fixed gains don't
capture — while at minimum *matching* the RBC's actuation magnitude. A policy
that only learns *when* to act but not *how much* loses to the RBC every time.

---

## 8. Recipe — implement your own reactive controller

The interface both baselines implement (SB3-compatible):

```python
class MyPolicy:
    def __init__(self, cfg): ...
    def bind_env(self, env):        # discover indices from env.metadata; build per-zone state
        ...
    def reset(self):                # zero integrators / setpoint state (call at episode start)
        ...
    def predict(self, obs, deterministic=True) -> tuple[np.ndarray, None]:
        # read obs by index, compute actions in PHYSICAL units, return full action vector
        ...
```

Build it in this order (each step is testable before the next):

1. **Wire the plumbing.** From `env.metadata`, map every zone to its
   temperature observation, its `target_temperature` observation (if any), and
   its actuator indices. Read actuator ranges from `action_space.high`. Get this
   *provably* right (assert names match) before any control math.
2. **Get the setpoint right.** Bind the dynamic target; fall back to a schedule,
   then a constant. (This one line dominates the score — §3.2.)
3. **Structure before gains.** Decide error definition (band vs point),
   regime detection (heating vs cooling), and which knob does what. A structurally
   correct P-only loop beats a structurally wrong PI loop.
4. **Add integral + anti-windup together.** Never ship a pure integrator; pair
   `ki` with a clamp *and* conditional integration or decay.
5. **Rate-limit every actuator.** Then tune `rate_limit` for smoothness.
6. **Floor the ventilation.** Fan/flow never truly zero while available.
7. **Guard transients.** Reset integrators on warm-up/wrap discontinuities.
8. **Validate on the full year, own harness, actuator-range audit** — not a
   sunny-week chunk and not setpoint error alone.

Tuning gains: start with `kp` alone (raise until it responds briskly without
oscillating), then add a small `ki` (eliminate residual offset), then set
`rate_limit` and anti-windup to kill the resulting overshoot. Trim-and-Respond
loops: `respond` step ≈ 2–5× the `trim` step (fast help, slow relax).

---

### Appendix — default parameters (authoritative: the dataclasses / YAML)

**Unitary** (`UnitaryHvacConfig`, `unitary_hvac.py:52-75`): `heat/cool = 20/22`,
`kp=0.8`, `ki=0.02`, `integral_max=5.0`, `min_fan_fraction=0.3`,
`sat ∈ [12,40]`, `sat_initial=14`, `sat_trim=0.2`, `sat_respond=0.5`,
`demand_deadband=0.5`, `fan_error_mode="nearest_setpoint"`.

**VAV** (`AirLoopConfig`, `air_loop.py:31-79`): `target=21`, `deadband=0.0`
(off — §2.1), `sat_neutral=20.5`, `sat_kp=0.8`, `sat ∈ [10,60]`,
`sat_rate_limit=0.3`, `warm/cold/mean bias = 0.6/0/0.4`, `flow_base=0.4`,
`flow_kp=0.2`, `flow_ki=0.025`, `flow ∈ [0.1,1.0]`, `flow_rate_limit=0.04`,
`integral_max=15`, `integral_decay=0.99`, `reheat ∈ [10,25]`, `reheat_kp=3.0`,
`error_ema_alpha=0.3`, `oa_mass_flow=1.37`.

> Note: `docs/baselines/controllers.md` lists an older parameter set (e.g.
> `kp=0.25`, `sat_initial=21`); the dataclass defaults and
> `baselines/configs/policy/*.yaml` above are the ones actually used.
