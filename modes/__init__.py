"""
Collection of tiling modes
"""

# ruff: noqa: E402
# pylint: disable=wrong-import-position
# This is to solve circular imports


class Settings:
    """
    For representing tiling settings
    """

    def __init__(self, mode, rotation):
        self.mode = mode
        self.tiling_fn = modes[mode]
        self.rotation = rotation

    def __hash__(self):
        # We don't care about the tiling function, because it's determined by the mode
        return hash((self.mode, self.rotation))


from .hex import hex_tiling
from .none import none_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
}


__all__ = ["modes", "Settings"]
