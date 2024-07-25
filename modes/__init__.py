"""
Collection of tiling modes
"""

from .hex import hex_tiling
from .none import none_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
}


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


__all__ = ["modes", "Settings"]
