### The default RBC baseline is comfort-slack; a deadband-free RBC recovers ~29%

The default reactive baseline (`UnitaryHvacPolicy`) carries a fixed ±1 °C setpoint
band, a ±0.5 °C `demand_deadband`, and an idle-fan interior — comfort slack that a
purely quadratic (deadband-free) reward penalizes but the controller ignores.
Collapsing that slack (band → ±0.1 °C, `demand_deadband` → 0.05,
`fan_error_mode="center_of_band"`; all other gains unchanged) and re-scoring the 15
unitary test buildings on the full year recovers **≈29 % of the RBC's comfort
penalty** (mean episode return −4964 → −3519).

This materially changes the comparison against the BC-warm Amorpheus policy. Against
the *default* baseline the policy wins **14/15**; against the *deadband-free*
baseline it wins only **11/15** — three apparent wins (Retail-2997,
Restaurant-3999, Restaurant-3003) were artifacts of the slack baseline. The policy's
mean advantage over the baseline shrinks from +2114 to +669, i.e. roughly two-thirds
of its apparent margin was baseline slack, with a real ≈one-third edge surviving.
On two buildings (Retail-2999/3000) the tighter tracking slightly *hurt* (mild
chatter), so −29 % is a net figure. We therefore report the RBC as a *standard*
reactive controller and note that a comfort-tightened baseline is a strictly harder
bar. (n=1 seed; the deadband-free config is a band-collapse, not itself tuned, so it
is a lower bound on the achievable comfort baseline — see the Optuna-tuned RBC.)
