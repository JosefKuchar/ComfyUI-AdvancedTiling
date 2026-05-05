"""
Test that wrapping preserves position-within-cell consistency.

For Hexagon mode, two independent checks:
  1. Path agreement: calculate_mapping (vectorized Conv2d path) agrees with
     hex_tiling (per-pixel production function) when given the same hex_size.
  2. Hex-distance preservation: every wrapped pixel's max-norm distance from
     the nearest hex center is preserved from source to destination (tolerance
     0.01).  This is the exact check from the original debug script that
     found the hex-to-pixel rounding bug — it catches regressions even if
     both production paths break equally.

For Rectangular mode: verify that calculate_mapping agrees with rect_tiling()
and that displacements are clean multiples of the working-area period.

All checks use ONLY production code paths — no re-implemented math.
"""

import sys
import os

_test_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_test_dir)
_custom_nodes = os.path.dirname(_pkg_dir)
sys.path.insert(0, _custom_nodes)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ComfyUI_AdvancedTiling",
    os.path.join(_pkg_dir, "__init__.py"),
    submodule_search_locations=[_pkg_dir],
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["ComfyUI_AdvancedTiling"] = _pkg
sys.modules["ComfyUI-AdvancedTiling"] = _pkg
_spec.loader.exec_module(_pkg)

import numpy as np
from ComfyUI_AdvancedTiling.modes import Settings
from ComfyUI_AdvancedTiling.modes.hex import hex_tiling, hex_distance_grid
from ComfyUI_AdvancedTiling.modes.rect import rect_tiling
import ComfyUI_AdvancedTiling.advanced_tiling as at_mod

VAE_FACTOR = 8
IMG_W = 1024
IMG_H = 1024


def _resolve(mode, scale, min_margin, div_by, is_conv2d=False):
    s = Settings(mode, 0.0, scale=scale, min_margin=min_margin,
                 divisible_by=div_by, conv2d_attention_wrapping=False)
    return s._resolve_auto(is_conv2d, VAE_FACTOR, 1, IMG_W, IMG_H)


def check_hex_mapping_agreement(resolved, res=128, pad=1):
    """Every pixel mapped by calculate_mapping must agree with hex_tiling.

    Both use the same hex_size_at(original_size) and operate in padded
    coordinate space.  calculate_mapping is the vectorized Conv2d path;
    hex_tiling is the per-pixel production function used by the attention
    path.  They must produce identical results.
    """
    pw, ph = res + 2 * pad, res + 2 * pad
    hex_size = resolved.hex_size_at(res, res)
    rotation = resolved.rotation

    mapping = at_mod.calculate_mapping((res, res), (pw, ph), resolved)
    n_mapped = len(mapping[0])
    if n_mapped == 0:
        return

    sx = mapping[0].numpy()
    sy = mapping[1].numpy()
    nx = mapping[2].numpy()
    ny = mapping[3].numpy()

    n_mismatch = 0
    for i in range(n_mapped):
        dx, dy = int(sx[i]), int(sy[i])
        expected_nx, expected_ny = hex_tiling(dx, dy, (pw, ph), hex_size, rotation)
        if expected_nx != int(nx[i]) or expected_ny != int(ny[i]):
            n_mismatch += 1

    assert n_mismatch == 0, (
        f"hex_mapping_agreement: {n_mismatch}/{n_mapped} pixels disagree "
        f"between calculate_mapping and hex_tiling "
        f"(hex_size={hex_size:.2f}, res={res}, pad={pad})"
    )


def check_hex_distance_preservation(resolved, res=128, pad=1):
    """Every wrapped pixel must preserve its hex-distance-from-center.

    For each mapped pixel (dx, dy) -> (nx, ny), the max-norm distance
    from the nearest hex center at (dx, dy) must equal the distance at
    (nx, ny).  This is the exact check from the original debug script
    that found the hex-to-pixel rounding bug.

    Uses production hex_distance_grid — no re-implemented math.
    """
    pw, ph = res + 2 * pad, res + 2 * pad
    hex_size = resolved.hex_size_at(res, res)
    rotation = resolved.rotation

    dist = hex_distance_grid((pw, ph), hex_size, rotation)

    mapping = at_mod.calculate_mapping((res, res), (pw, ph), resolved)
    n_mapped = len(mapping[0])
    if n_mapped == 0:
        return

    sx = mapping[0].numpy()
    sy = mapping[1].numpy()
    nx = mapping[2].numpy()
    ny = mapping[3].numpy()

    tolerance = 0.7 / hex_size

    n_mismatch = 0
    max_diff = 0.0
    for i in range(n_mapped):
        x, y = int(sx[i]), int(sy[i])
        src_x, src_y = int(nx[i]), int(ny[i])

        diff = abs(dist[y, x] - dist[src_y, src_x])
        if diff > max_diff:
            max_diff = diff
        if diff >= tolerance:
            n_mismatch += 1

    assert n_mismatch == 0, (
        f"hex_distance_preservation: {n_mismatch}/{n_mapped} pixels have "
        f"non-preserved hex-distance (max_diff={max_diff:.6f}, "
        f"tolerance={tolerance:.4f}, hex_size={hex_size:.2f}, res={res}, pad={pad})"
    )


def check_rect_mapping_agreement(resolved, res=128, pad=1):
    """Every pixel mapped by calculate_mapping must agree with rect_tiling.

    Uses production rect_tiling() with the same coordinate convention
    as calculate_mapping (work scaled to original resolution, grid over
    padded size).
    """
    pw, ph = res + 2 * pad, res + 2 * pad
    work_w, work_h = resolved.work_at(res, res)

    mapping = at_mod.calculate_mapping((res, res), (pw, ph), resolved)
    n_mapped = len(mapping[0])
    if n_mapped == 0:
        return

    sx = mapping[0].numpy()
    sy = mapping[1].numpy()
    nx = mapping[2].numpy()
    ny = mapping[3].numpy()

    n_mismatch = 0
    for i in range(n_mapped):
        dx, dy = int(sx[i]), int(sy[i])
        expected_nx, expected_ny = rect_tiling(dx, dy, (pw, ph), work_w, work_h)
        if expected_nx != int(nx[i]) or expected_ny != int(ny[i]):
            n_mismatch += 1

    assert n_mismatch == 0, (
        f"rect_mapping_agreement: {n_mismatch}/{n_mapped} pixels disagree "
        f"between calculate_mapping and rect_tiling "
        f"(work=({work_w:.1f},{work_h:.1f}), res={res}, pad={pad})"
    )

    # Also verify periodicity: displacement / work should be near-integer
    disp_x = (nx - sx).astype(np.float64)
    disp_y = (ny - sy).astype(np.float64)
    ratio_x = disp_x / work_w
    ratio_y = disp_y / work_h
    frac_x = np.abs(ratio_x - np.round(ratio_x))
    frac_y = np.abs(ratio_y - np.round(ratio_y))
    max_frac = max(frac_x.max(), frac_y.max())

    assert max_frac < 0.01, (
        f"rect_periodic: displacement not periodic — "
        f"max fractional period error={max_frac:.6f} "
        f"(work=({work_w:.1f},{work_h:.1f}), res={res}, pad={pad})"
    )


# ---------------------------------------------------------------------------
# Hex: sweep margins 0-10, paddings 0-2, and divisible_by values
# ---------------------------------------------------------------------------

def _make_hex_test(margin, div_by, pad):
    def test_fn():
        resolved = _resolve("Hexagon", 1.0, margin, div_by, is_conv2d=True)
        check_hex_mapping_agreement(resolved, res=128, pad=pad)
        check_hex_distance_preservation(resolved, res=128, pad=pad)
    test_fn.__name__ = f"test_hex_m{margin}_d{div_by}_p{pad}"
    return test_fn


for _m in range(0, 11):
    for _pad in [0, 1, 2]:
        _fn = _make_hex_test(_m, 1, _pad)
        globals()[_fn.__name__] = _fn

for _div in [2, 4, 8, 16]:
    for _pad in [0, 1]:
        _fn = _make_hex_test(0, _div, _pad)
        globals()[_fn.__name__] = _fn
        _fn = _make_hex_test(4, _div, _pad)
        globals()[_fn.__name__] = _fn


# ---------------------------------------------------------------------------
# Rect: sweep margins and divisible_by values
# ---------------------------------------------------------------------------

def _make_rect_test(margin, div_by, pad):
    def test_fn():
        resolved = _resolve("Rectangular", 1.0, margin, div_by, is_conv2d=True)
        check_rect_mapping_agreement(resolved, res=128, pad=pad)
    test_fn.__name__ = f"test_rect_m{margin}_d{div_by}_p{pad}"
    return test_fn


for _m in [0, 4, 8]:
    for _pad in [0, 1, 2]:
        _fn = _make_rect_test(_m, 1, _pad)
        globals()[_fn.__name__] = _fn

for _div in [2, 4, 8, 16]:
    _fn = _make_rect_test(4, _div, 1)
    globals()[_fn.__name__] = _fn


# ---------------------------------------------------------------------------
# Cross-resolution: latent and VAE decoder resolutions
# ---------------------------------------------------------------------------

def test_hex_cross_resolution():
    """Hex path agreement and distance preservation at all UNet resolutions."""
    for margin in [0, 2, 4, 6]:
        resolved = _resolve("Hexagon", 1.0, margin, 1, is_conv2d=True)
        for res in [128, 64, 32, 16]:
            check_hex_mapping_agreement(resolved, res=res, pad=1)
            check_hex_distance_preservation(resolved, res=res, pad=1)


def test_rect_cross_resolution():
    """Rect mapping agreement at all UNet resolutions."""
    for margin in [0, 4, 8]:
        resolved = _resolve("Rectangular", 1.0, margin, 1, is_conv2d=True)
        for res in [128, 64, 32, 16]:
            check_rect_mapping_agreement(resolved, res=res, pad=1)


# ---------------------------------------------------------------------------
# Production image resolutions
# ---------------------------------------------------------------------------

def _resolve_img(mode, scale, min_margin, div_by, img_w, img_h):
    s = Settings(mode, 0.0, scale=scale, min_margin=min_margin,
                 divisible_by=div_by, conv2d_attention_wrapping=False)
    return s._resolve_auto(True, VAE_FACTOR, 1, img_w, img_h)


def test_hex_image_resolutions():
    """Hex checks at production image resolutions."""
    for img_w, img_h in [(1024, 1024), (1328, 1328)]:
        for margin in [0, 4]:
            resolved = _resolve_img("Hexagon", 1.0, margin, 1, img_w, img_h)
            res = img_w // VAE_FACTOR
            check_hex_mapping_agreement(resolved, res=res, pad=1)
            check_hex_distance_preservation(resolved, res=res, pad=1)


def test_rect_image_resolutions():
    """Rect checks at production image resolutions."""
    for img_w, img_h in [(1024, 1024), (1328, 1328)]:
        for margin in [0, 4]:
            resolved = _resolve_img("Rectangular", 1.0, margin, 1, img_w, img_h)
            res = img_w // VAE_FACTOR
            check_rect_mapping_agreement(resolved, res=res, pad=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
