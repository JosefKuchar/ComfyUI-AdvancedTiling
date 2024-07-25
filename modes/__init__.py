"""
Collection of tiling modes
"""

from .hex import hex_tiling
from .none import none_tiling

modes = {
    "None": none_tiling,
    "Hexagon": hex_tiling,
}

__all__ = ["modes"]
