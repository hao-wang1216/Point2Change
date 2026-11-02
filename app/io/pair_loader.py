from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_pair(t1_path: str | Path, t2_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a co-registered image pair as matching HWC RGB uint8 arrays."""
    t1_path, t2_path = Path(t1_path), Path(t2_path)
    with Image.open(t1_path) as image:
        t1 = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(t2_path) as image:
        t2 = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if t1.shape[:2] != t2.shape[:2]:
        raise ValueError(
            "T1 and T2 must have the same dimensions; "
            f"got T1={t1.shape[1]}x{t1.shape[0]}, T2={t2.shape[1]}x{t2.shape[0]}"
        )
    return t1, t2
