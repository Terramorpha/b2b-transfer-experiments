# Reproduction pipeline for the cross-morphology transfer experiment.
#
# Pinned dependencies live in the submodules (both tagged transfer-experiment-v1):
#   vendor/morel              @ 139c79c  Terramorpha/morel
#   vendor/Building2Building  @ b435015  vtaboga/Building2Building (feature/adjacency, thermal-adjacency)
# The Python env is morel's own uv workspace; the HF dataset revision is pinned
# by scripts/_pin_dataset.py.

REPO  := $(CURDIR)
MOREL := $(REPO)/vendor/morel
PY    := $(MOREL)/.venv/bin/python
# The scripts self-insert the vendor/* paths they need (morel root modules /
# b2b's baselines/); building2building itself comes from the installed env, so
# we deliberately do NOT put vendor/Building2Building on PYTHONPATH globally --
# that would shadow the installed package with the un-built source tree.

SEEDS := 0 1 2

.PHONY: setup baseline eval figure all train clean help

help:
	@grep -E '^[a-z].*:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:            ## init submodules + build morel's pinned env (needs uv + guix runtime)
	git submodule update --init --recursive
	cd $(MOREL) && uv sync --all-packages   # --all-packages: install every third_party/* workspace member
	cd $(MOREL) && uv pip install pyarrow   # b2b reads its dataset metadata as parquet but doesn't declare pyarrow

# --- reproduce the FIGURE from the archived checkpoints (fast, no training) ---
baseline:         ## RBC baseline on the 672-step chunk -> data/baseline_chunk.csv
	$(PY) scripts/baseline_chunk.py
eval:             ## eval the 3 checkpoints on held-out test buildings -> data/transfer_port_eval.csv
	$(PY) scripts/eval_transfer_port.py
figure:           ## build the house-style figures -> figures_out/
	$(PY) figures/build_transfer_port_figure.py
all: baseline eval figure   ## full figure reproduction from checkpoints/

# --- retrain from scratch (slow; writes runs/, then copy the .eqx into checkpoints/) ---
train:            ## train all 3 seeds
	@for s in $(SEEDS); do \
	  $(PY) scripts/train_transfer_port.py --seed $$s --out runs/transfer_port_s$$s ; \
	done

clean:            ## remove regenerated intermediates (keeps committed artifacts)
	rm -rf runs figures_out/transfer_port_preview.png
