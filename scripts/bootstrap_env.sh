#!/bin/bash
# Build/refresh the experiment venv with uv from pyproject.toml's commit pins.
#   bash scripts/bootstrap_env.sh          # reproducible: uv sync from uv.lock
#   bash scripts/bootstrap_env.sh --dev    # + editable overrides from siblings
# Run inside the guix env (claude-env.sh / guix-load python uv ...).
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
if [ "${1:-}" = "--dev" ]; then
  SIB="$(cd .. && pwd)"
  uv pip install -e "$SIB/minergym" -e "$SIB/Building2Building" \
    -e "$SIB/morel" \
    -e "$SIB/morel/third_party/morel_amorpheus" \
    -e "$SIB/morel/third_party/morel_b2b" \
    -e "$SIB/morel/third_party/morel_b2b_amorpheus" \
    -e "$SIB/morel/third_party/morel_b2b_modumorph" \
    -e "$SIB/morel/third_party/morel_modumorph"
  echo "dev mode: sibling checkouts override the pins in this venv"
fi
echo "smoke: ENERGYPLUS_PATH=... .venv/bin/python scripts/check_supply_symmetry.py --steps 5"
