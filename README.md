# Cross-morphology transfer experiment

A frozen, reproducible bundle for the transfer figure: one Amorpheus policy
trained across **four heterogeneous building types** (retail, fast-food
restaurant, medium office, small office) with the faithful **cold-start +
resampling** loop, evaluated zero-shot on **held-out test-split buildings** of
those same types over the 672-step chunk it trained on. Ours vs the reactive
(RBC) baseline, in normalized return.

Everything needed to regenerate the figure — and to retrain — is pinned by
commit. `git clone --recursive` gives you the exact code; the archived
checkpoints let you rebuild the figure without retraining.

## Layout

```
scripts/    train_transfer_port.py  eval_transfer_port.py  baseline_chunk.py  _pin_dataset.py
figures/    build_transfer_port_figure.py  pub_style.py  validate_palette.py
data/       transfer_port_eval.csv  baseline_chunk.csv        # figure inputs (regenerable)
checkpoints/ model_s{0,1,2}.eqx                               # the trained policies (source of truth)
figures_out/ transfer_{retail,restaurant,officemed,officesmall,combined}.pdf
vendor/     morel/  Building2Building/  rl_x/                  # pinned submodules
MANIFEST.json   Makefile   .gitignore
```

## Pins (all in `MANIFEST.json`)

| Dependency | Pin | Tag |
|---|---|---|
| `vendor/morel` | `139c79c` (Terramorpha/morel) | `transfer-experiment-v1` |
| `vendor/Building2Building` | `b435015` (vtaboga, `feature/adjacency`, thermal-adjacency) | `transfer-experiment-v1` |
| `vendor/rl_x` | `70e220a` (nico-bohlinger/RL-X) | — (morel needs it as an editable `../rl_x` dep) |
| HF dataset `vtaboga/building2building_dataset` | `b879e179…` | via `scripts/_pin_dataset.py` |
| EnergyPlus | **25.1.0** via guix (`ENERGYPLUS_PATH`) | — |

**Runtime system libs.** The env also needs guix-provided native libraries
(EnergyPlus 25.1.0 + the `python-pytorch`/`libX11` stack RL-X pulls in). On the
machine these ran on that comes from `direnv` loading morel's `.envrc`
(`guix-load energyplus python python-numpy python-pandas python-pytorch ty uv`,
then `ENERGYPLUS_PATH=$GUIX_LOAD_PROFILE`). Reproduce inside an equivalent guix
profile; the repo does not pin guix itself.

Both submodule commits are protected by annotated `transfer-experiment-v1` tags,
so they survive branch rebases/deletions (a tag is a permanent gc-root).

## Reproduce

```sh
git clone --recursive <this repo>          # or: git submodule update --init --recursive
# enter the guix runtime first (energyplus + python-pytorch/libX11 + uv), e.g. via
# direnv on vendor/morel/.envrc, or:  guix shell energyplus python python-pytorch uv
make setup                                 # uv sync --all-packages + pyarrow, inside vendor/morel
make all                                   # baseline -> eval -> figure, from checkpoints/
```

Do **not** force `LD_LIBRARY_PATH` to the guix profile lib — the guix shell
already resolves libX11 for torch, and overriding it shadows pyarrow's native
libs. Verified end-to-end via a 1-building smoke (`RetailStandalone`, 48 steps).

`make all` rebuilds `data/*.csv` and `figures_out/*.pdf` from the **archived
checkpoints** (fast, no training). `make figure` alone rebuilds just the PDFs
from the committed CSVs. `make train` retrains all three seeds from scratch
(slow) into `runs/`.

## Notes

- **Checkpoints are the source of truth, not retraining.** JAX/XLA is not
  guaranteed bit-identical across versions/hardware, so `make train` reproduces
  the *method* but not necessarily the exact `.eqx`. The figure is reproduced
  from the committed checkpoints.
- **Dataset pin.** `building2building` at the pinned commit reads
  `REVISION = "main"` (a moving target); `scripts/_pin_dataset.py` rebinds it to
  the frozen SHA at import time. If/when the revision is frozen inside b2b
  itself, that shim becomes redundant and can be removed.
- **`vendor/*` are not modified here.** They are read-only pinned sources; the
  scripts add them to `PYTHONPATH` for the repo-root modules (`evaluate.py`,
  b2b's `baselines/`) that aren't shipped in the installed wheels.
