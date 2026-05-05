"""
Rectangular (toroidal) tiling implementation

Wraps coordinates modularly: right edge wraps to left, bottom wraps to top.
"""

import math


def rect_tiling(
    x: int,
    y: int,
    padded_size: tuple[int, int],
    work_w: float,
    work_h: float,
) -> tuple[int, int]:
    """Rectangular tiling with float-point dimensions.

    Uses float modular arithmetic for wrapping, with rounding only at final output.
    Pixels inside the float rectangle map to themselves.
    Pixels outside wrap to the opposite side.

    :param x: X coordinate in padded space
    :param y: Y coordinate in padded space
    :param padded_size: (width, height) of padded tensor
    :param work_w: Working rectangle width (pre-computed at appropriate resolution)
    :param work_h: Working rectangle height (pre-computed at appropriate resolution)
    :return: (new_x, new_y) source coordinates in padded space
    """
    pw, ph = padded_size

    cx = pw / 2.0
    cy = ph / 2.0

    rel_x = x - cx
    rel_y = y - cy

    new_x = cx + ((rel_x + work_w / 2) % work_w) - work_w / 2
    new_y = cy + ((rel_y + work_h / 2) % work_h) - work_h / 2

    return (math.floor(new_x + 0.5), math.floor(new_y + 0.5))


def rect_tiling_at(x, y, padded_size, resolved):
    """Rectangular tiling with automatic resolution scaling from ResolvedSettings.

    Scales working area to the tensor resolution of padded_size,
    then delegates to rect_tiling().
    """
    work_w, work_h = resolved.work_at(padded_size[0], padded_size[1])
    return rect_tiling(x, y, padded_size, work_w, work_h)
