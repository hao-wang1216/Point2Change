"""Resource locations shared by source runs and future frozen distributions."""

import os
from pathlib import Path
import sys


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def model_checkpoint() -> Path:
    override = os.environ.get("CHANGE_LABELER_MODEL")
    if override:
        return Path(override).expanduser().resolve()
    return application_root() / "models" / "sam2_hiera_large.pt"
