"""Freeze the building2building HF dataset to the exact revision this
experiment was produced on.

The dataset (``vtaboga/building2building_dataset``) is a *moving target*
separate from the pinned code: ``building2building`` at the pinned commit reads
``REVISION = "main"``, so a re-run would silently pull whatever HEAD is current.
Importing this module first rebinds that module global to the frozen SHA, so
every download in this repo fetches identical data.

This lives in the experiments repo (not b2b) on purpose: the dataset revision is
experiment-specific config, and b2b is pinned/immutable. Remove this once the
revision is frozen inside b2b itself.
"""
from __future__ import annotations

import building2building.data.download as _download

DATASET_REVISION = "b879e1794a0751f91c0558a0b98cf7a84c702cbd"
_download.REVISION = DATASET_REVISION
