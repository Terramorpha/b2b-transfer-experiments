"""Quick look at a b2b environment: action/obs sizes, actuator + observation names.

    python scripts/inspect_env.py [BUILDING_TYPE] [SPLIT] [INDEX]
    python scripts/inspect_env.py OfficeMedium test 0

Defaults: OfficeSmall test 0. Prints action-space size (grouped by actuator kind),
observation-space size (grouped), the resolved building id, and the Amorpheus
morphology's per-node action dims.
"""
import collections
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("vendor/morel", "vendor/Building2Building", "scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import numpy as np

import building2building as b2b
from building2building.data.registry import get_registry

BT = sys.argv[1] if len(sys.argv) > 1 else "OfficeSmall"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def group(names):
    g = collections.Counter()
    for n in names:
        nl = n.lower()
        if "mass flow" in nl or "flow rate" in nl or "flow_fraction" in nl:
            g["fan/flow"] += 1
        elif "temp setpoint" in nl or "supply_temp" in nl:
            g["supply-temp setpoint"] += 1
        elif "heating_setpoint" in nl or "reheat" in nl:
            g["heating/reheat setpoint"] += 1
        elif nl.startswith("target_temperature"):
            g["target_temperature (obs)"] += 1
        elif nl.startswith("zone air temperature"):
            g["zone air temp (obs)"] += 1
        elif nl.startswith("zone_occupancy"):
            g["zone occupancy (obs)"] += 1
        elif "schedule value" in nl:
            g["schedule value"] += 1
        else:
            g[n.split("::")[1] if "::" in n else "other"] += 1
    return g


bid = get_registry().get_building_by_index(BT, SPLIT, IDX).building_id
env = b2b.make_env(BT, split=SPLIT, index=IDX, task="task_occ_e0", run_period="full_year")
an = list(env.metadata["action_names"])
on = list(env.metadata["observation_names"])

print(f"\n{bid}  ({BT} / {SPLIT} / index {IDX})")
print(f"  action_space: {env.action_space.shape}  ({int(np.prod(env.action_space.shape))} dims)")
for k, v in group(an).most_common():
    print(f"     {v:3d}  {k}")
print(f"  observation_space: {env.observation_space.shape}  "
      f"({int(np.prod(env.observation_space.shape))} dims)")
for k, v in group(on).most_common():
    print(f"     {v:3d}  {k}")
if "hvac_equipment" in env.metadata:
    eq = env.metadata["hvac_equipment"]
    print(f"  hvac_equipment: {len(eq)} systems")
env.close()
