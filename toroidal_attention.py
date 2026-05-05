"""
Toroidal attention patches for hex and rectangular tiling in DiT models.

Injects wrapped-neighbor K/V entries into every attention layer via attn1_patch,
so boundary patches structurally "see" opposite-edge content as spatially adjacent.
Analogous to how Conv2d circular padding works for UNet models.
"""

import functools

import torch

from .modes import ResolvedSettings
from .modes.hex import hex_tiling
from .modes.rect import rect_tiling


def _factorize(n: int) -> tuple[int, int]:
    """Factorize n into h * w, preferring square.

    For prime n, returns (1, n) — callers should assume latent sizes are
    highly composite (powers of 2) in practice.
    """
    s = int(n ** 0.5)
    while s > 0:
        if n % s == 0:
            return s, n // s
        s -= 1
    return 1, n


# ---------------------------------------------------------------------------
# Hex toroidal attention
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _compute_hex_boundary_pairs(
    h_patches: int,
    w_patches: int,
    resolved: ResolvedSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hex boundary neighbor relationships at patch granularity.

    For each patch inside the hex that has a 4-connected neighbor outside
    the hex, record the boundary patch index, the wrapped source index,
    and the direction offset.

    Cached by (h_patches, w_patches, resolved).

    :param h_patches: Number of patch rows
    :param w_patches: Number of patch columns
    :param resolved: Resolved tiling settings with pre-computed values
    :return: (boundary_idx, source_idx, off_h, off_w) as LongTensors
    """
    boundary_idx = []
    source_idx = []
    offsets_h = []
    offsets_w = []

    for h in range(h_patches):
        for w in range(w_patches):
            src_w, src_h = hex_tiling(
                w, h,
                (w_patches, h_patches),
                resolved.hex_size_patch,
                resolved.rotation,
            )
            if src_w != w or src_h != h:
                continue

            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw

                n_src_w, n_src_h = hex_tiling(
                    nw, nh,
                    (w_patches, h_patches),
                    resolved.hex_size_patch,
                    resolved.rotation,
                )

                if n_src_w != nw or n_src_h != nh:
                    boundary_idx.append(h * w_patches + w)
                    source_idx.append(n_src_h * w_patches + n_src_w)
                    offsets_h.append(dh)
                    offsets_w.append(dw)

    if not boundary_idx:
        empty = torch.tensor([], dtype=torch.long)
        result = (empty, empty.clone(), empty.clone(), empty.clone())
    else:
        result = (
            torch.tensor(boundary_idx, dtype=torch.long),
            torch.tensor(source_idx, dtype=torch.long),
            torch.tensor(offsets_h, dtype=torch.long),
            torch.tensor(offsets_w, dtype=torch.long),
        )

    return result


# ---------------------------------------------------------------------------
# Rectangular toroidal attention
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _compute_rect_boundary_pairs(
    h_patches: int,
    w_patches: int,
    resolved: ResolvedSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rectangular boundary pairs using pre-computed working area.

    Uses rect_tiling() for boundary detection, matching hex pattern.

    Cached by (h_patches, w_patches, resolved).
    """
    boundary_idx = []
    source_idx = []
    offsets_h = []
    offsets_w = []

    for h in range(h_patches):
        for w in range(w_patches):
            src_w, src_h = rect_tiling(
                w, h,
                (w_patches, h_patches),
                resolved.work_patch_w,
                resolved.work_patch_h,
            )
            if src_w != w or src_h != h:
                continue

            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nh, nw = h + dh, w + dw

                n_src_w, n_src_h = rect_tiling(
                    nw, nh,
                    (w_patches, h_patches),
                    resolved.work_patch_w,
                    resolved.work_patch_h,
                )

                if n_src_w != nw or n_src_h != nh:
                    boundary_idx.append(h * w_patches + w)
                    source_idx.append(n_src_h * w_patches + n_src_w)
                    offsets_h.append(dh)
                    offsets_w.append(dw)

    if not boundary_idx:
        empty = torch.tensor([], dtype=torch.long)
        result = (empty, empty.clone(), empty.clone(), empty.clone())
    else:
        result = (
            torch.tensor(boundary_idx, dtype=torch.long),
            torch.tensor(source_idx, dtype=torch.long),
            torch.tensor(offsets_h, dtype=torch.long),
            torch.tensor(offsets_w, dtype=torch.long),
        )

    return result


# ---------------------------------------------------------------------------
# Shared base for attention patches
# ---------------------------------------------------------------------------

class _BaseToroidalAttentionPatch:
    """Shared logic for hex and rectangular toroidal attention patches."""

    def __init__(self, pe_embedder, resolved: ResolvedSettings):
        self.pe_embedder = pe_embedder
        self.resolved = resolved
        self._initialized = False
        self._boundary_idx = None
        self._source_idx = None
        self._n_extra = 0
        self._synthetic_pe = None

    def _compute_boundary_pairs(self, h_patches, w_patches):
        raise NotImplementedError

    def _initialize(self, n_img: int):
        h_patches, w_patches = _factorize(n_img)

        boundary_idx, source_idx, off_h, off_w = self._compute_boundary_pairs(
            h_patches, w_patches,
        )

        self._boundary_idx = boundary_idx
        self._source_idx = source_idx
        self._n_extra = len(boundary_idx)

        if self._n_extra == 0:
            self._initialized = True
            return

        boundary_h = boundary_idx // w_patches
        boundary_w = boundary_idx % w_patches

        h_center = h_patches // 2
        w_center = w_patches // 2

        n_axes = len(self.pe_embedder.axes_dim)
        syn_ids = torch.zeros(1, self._n_extra, n_axes, dtype=torch.float32)
        syn_ids[0, :, 0] = 0
        syn_ids[0, :, 1] = (boundary_h + off_h).float() - h_center
        syn_ids[0, :, 2] = (boundary_w + off_w).float() - w_center

        with torch.no_grad():
            self._synthetic_pe = self.pe_embedder(syn_ids)

        self._initialized = True

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        if extra_options is None or "img_slice" not in extra_options:
            return {}

        if attn_mask is not None:
            return {}

        if not self._initialized:
            img_slice = extra_options["img_slice"]
            n_img = img_slice[1] - img_slice[0]
            self._initialize(n_img)

        if self._n_extra == 0:
            return {}

        n_txt = extra_options["img_slice"][0]

        src_joint_idx = self._source_idx.to(k.device) + n_txt
        extra_k = k[:, :, src_joint_idx, :]
        extra_v = v[:, :, src_joint_idx, :]

        new_k = torch.cat([k, extra_k], dim=2)
        new_v = torch.cat([v, extra_v], dim=2)

        synthetic_pe = self._synthetic_pe.to(device=pe.device, dtype=pe.dtype)
        expand_shape = list(pe.shape)
        expand_shape[2] = synthetic_pe.shape[2]
        new_pe = torch.cat([
            pe,
            synthetic_pe.expand(expand_shape),
        ], dim=2)

        return {
            "k": new_k,
            "v": new_v,
            "pe": new_pe,
        }


class HexToroidalAttentionPatch(_BaseToroidalAttentionPatch):
    """attn1_patch for hex tiling: injects wrapped K/V for hex boundary patches."""

    def __init__(self, resolved: ResolvedSettings, pe_embedder):
        super().__init__(pe_embedder, resolved)

    def _compute_boundary_pairs(self, h_patches, w_patches):
        return _compute_hex_boundary_pairs(h_patches, w_patches, self.resolved)


class RectToroidalAttentionPatch(_BaseToroidalAttentionPatch):
    """attn1_patch for rectangular tiling: injects wrapped K/V from opposite edges."""

    def __init__(self, resolved: ResolvedSettings, pe_embedder):
        super().__init__(pe_embedder, resolved)

    def _compute_boundary_pairs(self, h_patches, w_patches):
        return _compute_rect_boundary_pairs(h_patches, w_patches, self.resolved)


# ---------------------------------------------------------------------------
# Lumina waste/margin token reset patch
# ---------------------------------------------------------------------------

class LuminaWastePatch:
    """double_block patch that replaces waste/margin tokens with their
    mapped source content after each transformer block, preventing garbage
    accumulation from polluting attention for working-area patches.

    For Hexagon mode: uses hex_tiling to identify waste positions (outside hex)
    and their hex source positions.
    For Rectangular mode: identifies margin positions (outside working rectangle)
    and maps them to the opposite edge of the working rectangle.
    """

    def __init__(self, patch_size: int, resolved: ResolvedSettings):
        self.patch_size = patch_size
        self.resolved = resolved
        self._initialized = False
        self._waste_idx = None
        self._waste_source_idx = None

    def _initialize(self, x: torch.Tensor):
        _, _, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        waste_indices = []
        waste_source_indices = []

        if self.resolved.mode == "Hexagon":
            for h in range(h_patches):
                for w in range(w_patches):
                    src_w, src_h = hex_tiling(
                        w, h,
                        (w_patches, h_patches),
                        self.resolved.hex_size_patch,
                        self.resolved.rotation,
                    )
                    if src_w != w or src_h != h:
                        waste_indices.append(h * w_patches + w)
                        waste_source_indices.append(src_h * w_patches + src_w)
        else:  # Rectangular
            for h in range(h_patches):
                for w in range(w_patches):
                    src_w, src_h = rect_tiling(
                        w, h,
                        (w_patches, h_patches),
                        self.resolved.work_patch_w,
                        self.resolved.work_patch_h,
                    )
                    if src_w != w or src_h != h:
                        waste_indices.append(h * w_patches + w)
                        waste_source_indices.append(src_h * w_patches + src_w)

        if waste_indices:
            self._waste_idx = torch.tensor(waste_indices, dtype=torch.long)
            self._waste_source_idx = torch.tensor(waste_source_indices, dtype=torch.long)

        self._initialized = True

    def __call__(self, data: dict) -> dict:
        if not self._initialized:
            self._initialize(data["x"])

        if self._waste_idx is None:
            return {}

        img = data["img"]
        img[:, self._waste_idx, :] = img[:, self._waste_source_idx, :]
        return {"img": img}
