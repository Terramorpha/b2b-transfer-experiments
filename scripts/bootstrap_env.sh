#!/bin/bash
# Build the experiment venv from PINNED COMMITS (requirements-pinned.txt).
# Default: reproducible install from github pins.
# --dev:   editable installs from the SIBLING CHECKOUTS
#          (~/code_sync/maitrise/{morel,Building2Building,minergym}) so local
#          edits take effect immediately -- commits still govern what's pushed.
#
#   bash scripts/bootstrap_env.sh [--dev]
#
# Run inside the guix env (guix-load python uv ... / claude-env.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-pinned}"
VENV=.venv
python3 -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip >/dev/null
if [ "$MODE" = "--dev" ]; then
  SIB="$(cd .. && pwd)"
  "$PIP" install -e "$SIB/minergym" -e "$SIB/Building2Building" \
    -e "$SIB/morel" \
    -e "$SIB/morel/third_party/morel_amorpheus" \
    -e "$SIB/morel/third_party/morel_b2b" \
    -e "$SIB/morel/third_party/morel_b2b_amorpheus" \
    -e "$SIB/morel/third_party/morel_b2b_modumorph" \
    -e "$SIB/morel/third_party/morel_modumorph"
else
  "$PIP" install -r requirements-pinned.txt
fi
echo "venv ready: $VENV"
echo "smoke: ENERGYPLUS_PATH=... $VENV/bin/python scripts/check_supply_symmetry.py --steps 5"
