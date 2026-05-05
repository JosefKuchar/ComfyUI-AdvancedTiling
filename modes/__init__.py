"""
Collection of tiling modes
"""


MODE_NAMES = ["None", "Hexagon", "Rectangular"]

# Sentinel values: scale=0 means auto, min_margin=-1 means auto.
# After _resolve_auto(), these are replaced with concrete values.
AUTO_SCALE = 0.0
AUTO_MARGIN = -1


class Settings:
    """Raw tiling settings from user input. May contain sentinel values.

    Sentinel values:
      scale=0.0      -> auto (resolved based on model type)
      min_margin=-1  -> auto (resolved based on model type)

    Use _resolve_auto() to produce a ResolvedSettings with all values concrete
    and resolution-dependent properties pre-computed.
    """

    __slots__ = ("mode", "rotation", "scale", "min_margin", "divisible_by",
                 "conv2d_attention_wrapping")

    def __init__(self, mode, rotation, scale, min_margin, divisible_by,
                 conv2d_attention_wrapping):
        self.mode = mode
        self.rotation = rotation
        self.scale = scale
        self.min_margin = min_margin
        self.divisible_by = divisible_by
        self.conv2d_attention_wrapping = conv2d_attention_wrapping

    def _resolve_auto(self, is_conv2d: bool, vae_factor: int, patch_size: int,
                      img_w: int, img_h: int) -> "ResolvedSettings":
        """Resolve auto-sentinel values and pre-compute all resolution-dependent properties.

        Margin pipeline (floor semantics):
          base_margin = W * (1 - scale) / 2
          margin = max(base_margin, min_margin_img)
          work = W - 2 * margin
          work = work // divisible_by * divisible_by
          margin = (W - work) / 2

        Returns a ResolvedSettings with working area and margins at image,
        latent, and patch resolutions — no downstream computation needed.
        """
        scale = self.scale
        if scale == AUTO_SCALE:
            scale = 1.0 if (is_conv2d or self.mode == "Hexagon") else 7 / 8

        min_margin = self.min_margin
        if min_margin == AUTO_MARGIN:
            min_margin = 4 if (not is_conv2d and self.mode == "Rectangular") else 0

        patch_vae_factor = vae_factor * patch_size

        if self.mode == "Rectangular":
            base_margin_w = img_w * (1 - scale) / 2
            base_margin_h = img_h * (1 - scale) / 2
            min_margin_img = min_margin * vae_factor
            margin_w = max(base_margin_w, min_margin_img)
            margin_h = max(base_margin_h, min_margin_img)
            work_w = img_w - 2 * margin_w
            work_h = img_h - 2 * margin_h
            if self.divisible_by >= 1:
                work_w = work_w // self.divisible_by * self.divisible_by
                work_h = work_h // self.divisible_by * self.divisible_by
            margin_w = (img_w - work_w) / 2
            margin_h = (img_h - work_h) / 2

            work_patch_w = work_w / patch_vae_factor
            work_patch_h = work_h / patch_vae_factor
            hex_size_img = hex_size_patch = 0.0

        elif self.mode == "Hexagon":
            min_dim = min(img_w, img_h)
            base_margin = min_dim / 2 * (1 - scale)
            min_margin_img = min_margin * vae_factor
            margin = max(base_margin, min_margin_img)
            hex_size = min_dim / 2 - margin
            if self.divisible_by >= 1:
                hex_height = 2 * hex_size
                hex_height = hex_height // self.divisible_by * self.divisible_by
                hex_size = hex_height / 2
            hex_size_img = hex_size
            hex_size_patch = hex_size / patch_vae_factor

            work_w = work_h = 2 * hex_size
            margin_w = (img_w - work_w) / 2
            margin_h = (img_h - work_h) / 2
            work_patch_w = work_patch_h = 2 * hex_size_patch

        else:  # "None"
            work_w = work_h = margin_w = 0.0
            work_patch_w = work_patch_h = 0.0
            hex_size_img = hex_size_patch = 0.0

        return ResolvedSettings(
            self.mode, self.rotation,
            work_w, work_h, margin_w,
            work_patch_w, work_patch_h,
            hex_size_img, hex_size_patch,
            img_w, img_h,
            self.conv2d_attention_wrapping,
        )

    def __eq__(self, other):
        if not isinstance(other, Settings):
            return NotImplemented
        return (self.mode, self.rotation, self.scale, self.min_margin,
                self.divisible_by, self.conv2d_attention_wrapping) == (
                    other.mode, other.rotation, other.scale,
                    other.min_margin, other.divisible_by,
                    other.conv2d_attention_wrapping)

    def __hash__(self):
        return hash((self.mode, self.rotation, self.scale, self.min_margin,
                     self.divisible_by, self.conv2d_attention_wrapping))


class ResolvedSettings:
    """Fully resolved tiling settings with all resolution-dependent properties.

    Produced by Settings._resolve_auto(). Guaranteed to have concrete values
    for all fields — no sentinel values, no None properties.

    work fields: bounding box of the working area.
      Rectangular: the working area IS the rectangle.
      Hexagon:     the working area is the bounding square of the hex.
      None:        all zeros (handled by early return in consumers).

    hex_size fields: hex radius (center to vertex). Populated for Hexagon, zero otherwise.
    """

    __slots__ = (
        "mode", "rotation",
        "work_img_w", "work_img_h", "margin_img_w",
        "work_patch_w", "work_patch_h",
        "hex_size_img", "hex_size_patch",
        "img_w", "img_h",
        "conv2d_attention_wrapping",
    )

    def __init__(self, mode, rotation,
                 work_img_w, work_img_h, margin_img_w,
                 work_patch_w, work_patch_h,
                 hex_size_img, hex_size_patch,
                 img_w, img_h,
                 conv2d_attention_wrapping):
        self.mode = mode
        self.rotation = rotation
        self.work_img_w = work_img_w
        self.work_img_h = work_img_h
        self.margin_img_w = margin_img_w
        self.work_patch_w = work_patch_w
        self.work_patch_h = work_patch_h
        self.hex_size_img = hex_size_img
        self.hex_size_patch = hex_size_patch
        self.img_w = img_w
        self.img_h = img_h
        self.conv2d_attention_wrapping = conv2d_attention_wrapping

    def __eq__(self, other):
        if not isinstance(other, ResolvedSettings):
            return NotImplemented
        return (self.mode, self.rotation,
                self.work_img_w, self.work_img_h, self.margin_img_w,
                self.work_patch_w, self.work_patch_h,
                self.hex_size_img, self.hex_size_patch,
                self.img_w, self.img_h,
                self.conv2d_attention_wrapping) == (
                    other.mode, other.rotation,
                    other.work_img_w, other.work_img_h, other.margin_img_w,
                    other.work_patch_w, other.work_patch_h,
                    other.hex_size_img, other.hex_size_patch,
                    other.img_w, other.img_h,
                    other.conv2d_attention_wrapping)

    def __hash__(self):
        return hash((self.mode, self.rotation,
                     self.work_img_w, self.work_img_h, self.margin_img_w,
                     self.work_patch_w, self.work_patch_h,
                     self.hex_size_img, self.hex_size_patch,
                     self.img_w, self.img_h,
                     self.conv2d_attention_wrapping))

    def work_at(self, tensor_w, tensor_h):
        """Scale working area dimensions to tensor resolution."""
        return (tensor_w * self.work_img_w / self.img_w,
                tensor_h * self.work_img_h / self.img_h)

    def hex_size_at(self, tensor_w, tensor_h):
        """Scale hex size to tensor resolution."""
        min_dim = min(self.img_w, self.img_h)
        return min(tensor_w, tensor_h) * self.hex_size_img / min_dim


__all__ = ["MODE_NAMES", "Settings", "ResolvedSettings", "AUTO_SCALE", "AUTO_MARGIN"]
