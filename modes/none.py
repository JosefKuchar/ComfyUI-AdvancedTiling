"""
Default generation mode - no tiling
"""

from . import Settings


def none_tiling(
    x: int,
    y: int,
    _original_size: tuple[int, int],
    _padded_size: tuple[int, int],
    _settings: Settings,
) -> tuple[int, int]:
    """
    Just return the same coordinates

    :param x: X coordinate
    :param y: Y coordinate
    :return (x, y): Coordinates
    """

    return (x, y)
