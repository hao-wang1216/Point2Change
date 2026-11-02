from __future__ import annotations

import math


def display_to_image(
    position: tuple[float, float],
    *,
    scale: float,
    offset: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[int, int] | None:
    """Map a display-space coordinate to an image pixel.

    ``image_size`` is ``(width, height)`` and ``offset`` is the display-space
    location of image pixel (0, 0).
    """
    if scale <= 0:
        raise ValueError("scale must be greater than zero")

    x = int(math.floor((position[0] - offset[0]) / scale + 0.5))
    y = int(math.floor((position[1] - offset[1]) / scale + 0.5))
    width, height = image_size
    if not (0 <= x < width and 0 <= y < height):
        return None
    return x, y
