"""
ComfyUI Node entry point
"""

# pylint: disable=invalid-name

from .advanced_tiling import (
    AdvancedTilingSettings,
    AdvancedTiling,
    AdvancedTilingVAEDecode,
)

NODE_CLASS_MAPPINGS = {
    "AdvancedTilingSettings": AdvancedTilingSettings,
    "AdvancedTiling": AdvancedTiling,
    "AdvancedTilingVAEDecode": AdvancedTilingVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedTilingSettings": "Advanced Tiling Settings",
    "AdvancedTiling": "Advanced Tiling",
    "AdvancedTilingVAEDecode": "Advanced Tiling VAE Decode",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
