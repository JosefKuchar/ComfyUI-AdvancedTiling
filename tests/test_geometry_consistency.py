"""
Test geometry consistency across all tiling subsystems.

Compares four subsystems for each mode and settings combo:
1. Latent wrapping (calculate_mapping) -- vectorized Conv2d path
2. Per-pixel tiling (rect_tiling / hex_tiling) -- used by toroidal attention
3. Toroidal attention boundary (_compute_*_boundary_pairs) -- DiT K/V injection
4. VAE decode crop (create_crop_mask) -- post-decode cropping

Tests the full parameter matrix:
  scale = {1.0, 0.9}
  min_margin = {-1, 0, 4}  (-1 is auto sentinel, resolved before use)
  divisible_by = {1, 16}
  mode = {Rectangular, Hexagon}

Prints ASCII visualizations and asserts all four agree on the working area.
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

import torch
import numpy as np
from ComfyUI_AdvancedTiling.modes import Settings, ResolvedSettings
from ComfyUI_AdvancedTiling.modes.rect import rect_tiling_at
from ComfyUI_AdvancedTiling.modes.hex import hex_tiling_at
import ComfyUI_AdvancedTiling.advanced_tiling as at_mod
import ComfyUI_AdvancedTiling.toroidal_attention as ta_mod

VAE_FACTOR = 8
PATCH_SIZE = 2
IMG_W = 1024
IMG_H = 1024


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve(scale, min_margin, divisible_by, mode, is_conv2d=False,
             vae_factor=VAE_FACTOR, patch_size=PATCH_SIZE,
             img_w=IMG_W, img_h=IMG_H):
    """Create Settings and resolve."""
    s = Settings(mode, 0.0, scale=scale, min_margin=min_margin, divisible_by=divisible_by, conv2d_attention_wrapping=True)
    return s._resolve_auto(is_conv2d, vae_factor, patch_size, img_w, img_h)


def build_wrapping_identity(W, H, resolved):
    """2D bool array: True where pixel maps to itself (inside working area)."""
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping(
        (W, H), (W, H), resolved,
    )
    identity = torch.ones((H, W), dtype=torch.bool)
    if len(src_x) > 0:
        identity[src_y, src_x] = False
    return identity


def build_tiling_identity(W, H, resolved, mode):
    """2D bool array from per-pixel tiling: True where pixel maps to itself."""
    identity = np.ones((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            if mode == "Rectangular":
                nx, ny = rect_tiling_at(x, y, (W, H), resolved)
            else:
                nx, ny = hex_tiling_at(x, y, (W, H), resolved)
            if nx != x or ny != y:
                identity[y, x] = False
    return identity


def build_crop_inside(W, H, resolved):
    """2D bool array from crop mask: True where mask is 1."""
    mask = at_mod.create_crop_mask(W, H, resolved)
    return mask[0, :, :, 0].bool()


def build_boundary_map(H, W, resolved, mode):
    """2D bool array: True for patches on the toroidal attention boundary."""
    if mode == "Rectangular":
        fn = ta_mod._compute_rect_boundary_pairs
    else:
        fn = ta_mod._compute_hex_boundary_pairs

    boundary_idx, source_idx, off_h, off_w = fn(H, W, resolved)
    boundary = np.zeros((H, W), dtype=bool)
    if len(boundary_idx) > 0:
        for idx in boundary_idx.tolist():
            by, bx = divmod(idx, W)
            if 0 <= by < H and 0 <= bx < W:
                boundary[by, bx] = True
    return boundary


def ascii_grid(bool_2d, inside='#', outside='.'):
    """Convert 2D bool array to ASCII string."""
    lines = []
    for row in bool_2d:
        lines.append(''.join(inside if v else outside for v in row))
    return '\n'.join(lines)


def ascii_composite(identity, boundary, inside='#', boundary_ch='B', outside='.'):
    """Show identity map with boundary patches highlighted."""
    lines = []
    for y in range(identity.shape[0]):
        row = []
        for x in range(identity.shape[1]):
            if boundary[y, x]:
                row.append(boundary_ch)
            elif bool(identity[y, x]):
                row.append(inside)
            else:
                row.append(outside)
        lines.append(''.join(row))
    return '\n'.join(lines)


def ascii_overlay(a, b):
    """Show mismatch between two bool maps. 1=a-only, 2=b-only, ==agree."""
    lines = []
    for y in range(a.shape[0]):
        row = []
        for x in range(a.shape[1]):
            va, vb = bool(a[y, x]), bool(b[y, x])
            if va == vb:
                row.append('=')
            elif va:
                row.append('1')
            else:
                row.append('2')
        lines.append(''.join(row))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------

def check_geometry(label, mode, scale, min_margin, div_by, W, H, vae_factor,
                   patch_size=PATCH_SIZE, is_conv2d=False,
                   print_ascii=True, assert_consistency=True):
    """Run full geometry consistency check for one mode/settings combo."""
    img_w = W * vae_factor
    img_h = H * vae_factor
    resolved = _resolve(scale, min_margin, div_by, mode, is_conv2d,
                        vae_factor, patch_size, img_w, img_h)

    if mode == "Rectangular":
        dim_info = f"work_img_w={resolved.work_img_w:.2f}, work_img_h={resolved.work_img_h:.2f}"
    else:
        dim_info = f"hex_size_img={resolved.hex_size_img:.2f}, hex_height={resolved.hex_size_img*2:.2f}"

    wrap_id = build_wrapping_identity(W, H, resolved)
    tile_id = build_tiling_identity(W, H, resolved, mode)
    crop_id = build_crop_inside(W, H, resolved)
    boundary = build_boundary_map(H, W, resolved, mode)

    wrap_np = wrap_id.numpy() if isinstance(wrap_id, torch.Tensor) else wrap_id
    tile_np = tile_id.numpy() if isinstance(tile_id, torch.Tensor) else tile_id
    crop_np = crop_id.numpy() if isinstance(crop_id, torch.Tensor) else crop_id

    wrap_count = int(wrap_np.sum())
    tile_count = int(tile_np.sum())
    crop_count = int(crop_np.sum())
    boundary_count = int(boundary.sum())

    if print_ascii:
        print(f"\n{'='*60}")
        print(f"{label}: {mode} scale={scale} min_margin={min_margin} div_by={div_by}")
        print(f"Grid: {W}x{H} (latent), vae_factor={vae_factor}, {dim_info}")
        print(f"Inside counts: wrap={wrap_count} tile={tile_count} crop={crop_count} boundary={boundary_count}")
        print(f"{'='*60}")

        print(f"\n1. Wrapping identity:  #={wrap_count} .={W*H - wrap_count}")
        print(ascii_grid(wrap_np))

        print(f"\n2. Per-pixel tiling:  #={tile_count}")
        print(ascii_grid(tile_np))

        print(f"\n3. Crop mask:  #={crop_count}")
        print(ascii_grid(crop_np))

        print(f"\n4. Toroidal boundary:  B={boundary_count}")
        print(ascii_composite(wrap_np, boundary))

        if not np.array_equal(wrap_np, crop_np):
            print(f"\nDIFF wrapping vs crop:  1=wrap-only  2=crop-only")
            print(ascii_overlay(wrap_np, crop_np))
        else:
            print(f"\nwrapping == crop: MATCH")

        if not np.array_equal(wrap_np, tile_np):
            print(f"\nDIFF wrapping vs tiling:  1=wrap-only  2=tile-only")
            print(ascii_overlay(wrap_np, tile_np))
        else:
            print(f"\nwrapping == tiling: MATCH")

    errors = []

    if not np.array_equal(wrap_np, tile_np):
        diff = int((wrap_np != tile_np).sum())
        errors.append(f"wrapping != tiling: {diff} pixels differ")

    if not np.array_equal(wrap_np, crop_np):
        diff = int((wrap_np != crop_np).sum())
        errors.append(f"wrapping != crop: {diff} pixels differ")

    boundary_outside = boundary & ~wrap_np
    if boundary_outside.any():
        count = int(boundary_outside.sum())
        errors.append(f"boundary not subset of wrapping inside: {count} boundary patches outside")

    has_outside = int((~wrap_np).sum()) > 0
    if has_outside and boundary_count == 0:
        errors.append("has outside pixels but no toroidal boundary patches")

    if mode == "Rectangular":
        fn = ta_mod._compute_rect_boundary_pairs
    else:
        fn = ta_mod._compute_hex_boundary_pairs
    boundary_idx, source_idx, _, _ = fn(H, W, resolved)
    if has_outside and len(source_idx) == 0:
        errors.append("has outside pixels but no toroidal source pairs")

    if assert_consistency and errors:
        raise AssertionError('\n'.join(errors))

    return errors


# ---------------------------------------------------------------------------
# Parameter matrix at latent resolution
# ---------------------------------------------------------------------------

PARAM_GRID = [
    (1.0, 0, 1),
    (1.0, 0, 16),
    (1.0, 4, 1),
    (1.0, 4, 16),
    (0.9, 0, 1),
    (0.9, 0, 16),
    (0.9, 4, 1),
    (0.9, 4, 16),
]


# Generate test functions for the full matrix at 128x128 latent
for _mode in ["Rectangular", "Hexagon"]:
    for _scale, _margin, _div in PARAM_GRID:
        _tag = f"{_mode[:4].lower()}_s{_scale}_m{_margin}_d{_div}"
        def _make_test(mode, scale, margin, div_by):
            def test_fn():
                label = f"{mode[:4]}-s{scale}-m{margin}-d{div_by}"
                check_realistic_borders(label, mode, scale, margin, div_by)
                errors = check_geometry(
                    label, mode, scale, margin, div_by,
                    W=128, H=128, vae_factor=1,
                    print_ascii=False, assert_consistency=True,
                )
                assert not errors, '\n'.join(errors)
            return test_fn
        _fn = _make_test(_mode, _scale, _margin, _div)
        _fn.__name__ = f"test_{_tag}"
        globals()[_fn.__name__] = _fn


# ---------------------------------------------------------------------------
# Auto-resolution tests
# ---------------------------------------------------------------------------

def test_auto_resolve_dit_rect():
    """min_margin=-1, scale=0 resolves for DiT Rect."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=False, vae_factor=VAE_FACTOR,
                                 patch_size=PATCH_SIZE, img_w=IMG_W, img_h=IMG_H)

    # scale=7/8, min_margin=4
    assert resolved.work_img_w == 896.0
    assert resolved.margin_img_w == 64.0

    check_realistic_borders("RECT-auto-DiT", "Rectangular",
                            7/8, 4, 1)
    errors = check_geometry(
        "RECT-auto-DiT", "Rectangular",
        scale=7/8, min_margin=4, div_by=1,
        W=128, H=128, vae_factor=1,
        print_ascii=False, assert_consistency=True,
    )
    assert not errors, '\n'.join(errors)


def test_auto_resolve_conv2d_rect():
    """min_margin=-1, scale=0 resolves for Conv2d Rect."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=True, vae_factor=VAE_FACTOR,
                                 patch_size=1, img_w=IMG_W, img_h=IMG_H)

    # scale=1.0, min_margin=0
    assert resolved.work_img_w == 1024.0
    assert resolved.margin_img_w == 0.0

    check_realistic_borders("RECT-auto-Conv2d", "Rectangular",
                            1.0, 0, 1)
    errors = check_geometry("RECT-auto-Conv2d", "Rectangular",
                            1.0, 0, 1,
                            W=128, H=128, vae_factor=1, is_conv2d=True,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_dit_hex():
    """min_margin=-1, scale=0 resolves for DiT Hex."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=False, vae_factor=VAE_FACTOR,
                                 patch_size=PATCH_SIZE, img_w=IMG_W, img_h=IMG_H)

    # scale=1.0, min_margin=0
    assert resolved.hex_size_img == 512.0

    check_realistic_borders("HEX-auto-DiT", "Hexagon",
                            1.0, 0, 1)
    errors = check_geometry("HEX-auto-DiT", "Hexagon",
                            1.0, 0, 1,
                            W=128, H=128, vae_factor=1,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_conv2d_hex():
    """min_margin=-1, scale=0 resolves for Conv2d Hex."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=True, vae_factor=VAE_FACTOR,
                                 patch_size=1, img_w=IMG_W, img_h=IMG_H)

    assert resolved.hex_size_img == 512.0

    check_realistic_borders("HEX-auto-Conv2d", "Hexagon",
                            1.0, 0, 1)
    errors = check_geometry("HEX-auto-Conv2d", "Hexagon",
                            1.0, 0, 1,
                            W=128, H=128, vae_factor=1, is_conv2d=True,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_dit_rect_div16():
    """min_margin=-1, scale=0, div_by=16 resolves for DiT Rect."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=16, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=False, vae_factor=VAE_FACTOR,
                                 patch_size=PATCH_SIZE, img_w=IMG_W, img_h=IMG_H)

    assert resolved.work_img_w == 896.0

    check_realistic_borders("RECT-auto-DiT-d16", "Rectangular",
                            7/8, 4, 16)
    errors = check_geometry("RECT-auto-DiT-d16", "Rectangular",
                            7/8, 4, 16,
                            W=128, H=128, vae_factor=1,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_dit_hex_div16():
    """min_margin=-1, scale=0, div_by=16 resolves for DiT Hex."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=16, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=False, vae_factor=VAE_FACTOR,
                                 patch_size=PATCH_SIZE, img_w=IMG_W, img_h=IMG_H)

    assert resolved.hex_size_img == 512.0

    check_realistic_borders("HEX-auto-DiT-d16", "Hexagon",
                            1.0, 0, 16)
    errors = check_geometry("HEX-auto-DiT-d16", "Hexagon",
                            1.0, 0, 16,
                            W=128, H=128, vae_factor=1,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


# ---------------------------------------------------------------------------
# Auto-resolution with explicit scale=0.9 + min_margin=-1
# ---------------------------------------------------------------------------

def test_auto_resolve_dit_rect_s09_mneg1():
    """scale=0.9, min_margin=-1, div_by=16 for DiT Rect: margin resolves -1->4."""
    raw = Settings("Rectangular", 0.0, scale=0.9, min_margin=-1, divisible_by=16, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=False, vae_factor=VAE_FACTOR,
                                 patch_size=PATCH_SIZE, img_w=IMG_W, img_h=IMG_H)

    assert resolved.margin_img_w > 0

    check_realistic_borders("RECT-s0.9-auto-DiT", "Rectangular",
                            0.9, 4, 16)
    errors = check_geometry("RECT-s0.9-auto-DiT", "Rectangular",
                            0.9, 4, 16,
                            W=128, H=128, vae_factor=1,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_conv2d_rect_s09_mneg1():
    """scale=0.9, min_margin=-1, div_by=16 for Conv2d Rect: margin resolves -1->0."""
    raw = Settings("Rectangular", 0.0, scale=0.9, min_margin=-1, divisible_by=16, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=True, vae_factor=VAE_FACTOR,
                                 patch_size=1, img_w=IMG_W, img_h=IMG_H)

    assert resolved.margin_img_w > 0

    check_realistic_borders("RECT-s0.9-auto-Conv2d", "Rectangular",
                            0.9, 0, 16)
    errors = check_geometry("RECT-s0.9-auto-Conv2d", "Rectangular",
                            0.9, 0, 16,
                            W=128, H=128, vae_factor=1, is_conv2d=True,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_hex_s09_mneg1_dit():
    """HEX scale=0.9, min_margin=-1, div_by=16 for DiT: margin resolves -1->0."""
    raw = Settings("Hexagon", 0.0, scale=0.9, min_margin=-1, divisible_by=16, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=False, vae_factor=VAE_FACTOR,
                                 patch_size=PATCH_SIZE, img_w=IMG_W, img_h=IMG_H)

    assert resolved.hex_size_img > 0

    check_realistic_borders("HEX-s0.9-auto-DiT", "Hexagon",
                            0.9, 0, 16)
    errors = check_geometry("HEX-s0.9-auto-DiT", "Hexagon",
                            0.9, 0, 16,
                            W=128, H=128, vae_factor=1,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


def test_auto_resolve_hex_s09_mneg1_conv2d():
    """HEX scale=0.9, min_margin=-1, div_by=16 for Conv2d: margin resolves -1->0."""
    raw = Settings("Hexagon", 0.0, scale=0.9, min_margin=-1, divisible_by=16, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(is_conv2d=True, vae_factor=VAE_FACTOR,
                                 patch_size=1, img_w=IMG_W, img_h=IMG_H)

    assert resolved.hex_size_img > 0

    check_realistic_borders("HEX-s0.9-auto-Conv2d", "Hexagon",
                            0.9, 0, 16)
    errors = check_geometry("HEX-s0.9-auto-Conv2d", "Hexagon",
                            0.9, 0, 16,
                            W=128, H=128, vae_factor=1, is_conv2d=True,
                            print_ascii=False)
    assert not errors, '\n'.join(errors)


# ---------------------------------------------------------------------------
# Realistic resolution: border visualization
# ---------------------------------------------------------------------------

def ascii_border_region(bool_2d, border_px=5):
    """Show only the border region of a large grid as ASCII."""
    H, W = bool_2d.shape
    lines = []

    for y in range(min(border_px, H)):
        lines.append(f"  y={y:3d}  " + ''.join('#' if v else '.' for v in bool_2d[y]))

    if H > 2 * border_px:
        lines.append(f"  ... ({H - 2*border_px} rows omitted)")
        for y in range(max(border_px, H - border_px), H):
            lines.append(f"  y={y:3d}  " + ''.join('#' if v else '.' for v in bool_2d[y]))

    return '\n'.join(lines)


def ascii_boundary_zoom(bool_2d, zoom_px=10):
    """Zoom into the boundary transition zone of a large bool array."""
    H, W = bool_2d.shape
    lines = []

    for y in range(H):
        row = bool_2d[y]
        if row.any() and not row.all():
            x_trans = 0
            for x in range(W):
                if bool(row[x]) != bool(row[0]):
                    x_trans = x
                    break
            y0 = max(0, y - 2)
            x0 = max(0, x_trans - zoom_px // 2)
            lines.append(f"  Top-left boundary (row {y0}):")
            for ry in range(y0, min(H, y0 + zoom_px)):
                lines.append(f"    y={ry:4d}  " + ''.join(
                    '#' if bool_2d[ry, cx] else '.'
                    for cx in range(x0, min(W, x0 + zoom_px * 2))
                ))
            break

    for y in range(H - 1, -1, -1):
        row = bool_2d[y]
        if row.any() and not row.all():
            x_trans = W - 1
            for x in range(W - 1, -1, -1):
                if bool(row[x]) != bool(row[-1]):
                    x_trans = x
                    break
            y1 = min(H, y + 3)
            x1 = max(0, x_trans - zoom_px)
            lines.append(f"  Bottom-right boundary (row {y1 - zoom_px}):")
            for ry in range(max(0, y1 - zoom_px), y1):
                lines.append(f"    y={ry:4d}  " + ''.join(
                    '#' if bool_2d[ry, cx] else '.'
                    for cx in range(x1, min(W, x1 + zoom_px * 2))
                ))
            break

    return '\n'.join(lines)


def check_realistic_borders(label, mode, scale, min_margin, div_by,
                             W_lat=128, H_lat=128, vae_factor=VAE_FACTOR):
    """Check geometry at realistic latent resolution with border visualization."""
    img_w = W_lat * vae_factor
    img_h = H_lat * vae_factor
    resolved = _resolve(scale, min_margin, div_by, mode,
                        vae_factor=vae_factor, img_w=img_w, img_h=img_h)

    if mode == "Rectangular":
        dim_info = f"work_img_w={resolved.work_img_w:.2f}, work_img_h={resolved.work_img_h:.2f}"
    else:
        dim_info = f"hex_size_img={resolved.hex_size_img:.2f}, hex_height={resolved.hex_size_img*2:.2f}"

    print(f"\n{'='*70}")
    print(f"{label}: {mode} scale={scale} min_margin={min_margin} div_by={div_by}")
    print(f"Latent: {W_lat}x{H_lat}, vae_factor={vae_factor}, {dim_info}")
    print(f"{'='*70}")

    # Wrapping identity at latent resolution
    wrap_id = build_wrapping_identity(W_lat, H_lat, resolved)
    wrap_np = wrap_id.numpy() if isinstance(wrap_id, torch.Tensor) else wrap_id

    inside_count = int(wrap_np.sum())
    outside_count = W_lat * H_lat - inside_count
    print(f"  Inside: {inside_count}/{W_lat*H_lat} ({inside_count/(W_lat*H_lat)*100:.1f}%)")

    # Border visualization at latent resolution
    print(f"\n  Latent wrapping border (top/bottom 5 rows of {W_lat}x{H_lat}):")
    print(ascii_border_region(wrap_np, border_px=5))

    # Crop dimensions from ResolvedSettings
    rmin, rmax, cmin, cmax = at_mod.compute_crop_bounds(img_w, img_h, resolved)
    crop_w = cmax - cmin + 1
    crop_h = rmax - rmin + 1

    print(f"\n  Image crop: {crop_w}x{crop_h} (from {img_w}x{img_h})")

    # Image-resolution boundary visualization (crop mask)
    mask = at_mod.create_crop_mask(img_w, img_h, resolved)
    crop_mask_np = mask[0, :, :, 0].bool().numpy()
    if outside_count > 0:
        print(f"\n  Image crop mask boundary zoom ({img_w}x{img_h}, vae_factor={vae_factor}):")
        print(ascii_boundary_zoom(crop_mask_np, zoom_px=8))

    if mode == "Hexagon":
        assert crop_w == crop_h, f"hex crop not square: {crop_w}x{crop_h}"
        print(f"  Hex square crop: {crop_w}x{crop_h}")
        if div_by > 1:
            assert crop_w % div_by == 0, f"hex crop {crop_w} not div by {div_by}"
            print(f"  div_by={div_by} check: PASS ({crop_w}%{div_by}==0)")
    elif div_by > 1:
        assert crop_w % div_by == 0, f"crop_w={crop_w} not div by {div_by}"
        assert crop_h % div_by == 0, f"crop_h={crop_h} not div by {div_by}"
        print(f"  div_by={div_by} check: PASS ({crop_w}%{div_by}==0, {crop_h}%{div_by}==0)")

    # Verify wrapping and crop agree on inside count
    crop_id = mask[0, :, :, 0].bool()
    crop_inside = int(crop_id.sum())
    expected_crop_inside = inside_count * (vae_factor ** 2)
    tolerance = 2 * vae_factor * (W_lat + H_lat)
    assert abs(crop_inside - expected_crop_inside) <= tolerance, (
        f"crop inside ({crop_inside}) too far from expected ({expected_crop_inside}"
        f" +- {tolerance})"
    )
    print(f"  Crop inside: {crop_inside} (expected ~{expected_crop_inside}, tolerance {tolerance})")

    return True


# ---------------------------------------------------------------------------
# Realistic resolution with auto-resolved settings
# ---------------------------------------------------------------------------

def test_rect_realistic_dit_auto():
    """DiT Rect auto-resolved at 1024x1024."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(False, VAE_FACTOR, PATCH_SIZE, IMG_W, IMG_H)
    check_realistic_borders("RECT-DiT-auto", "Rectangular",
                            7/8, 4, 1)


def test_rect_realistic_dit_auto_d16():
    """DiT Rect auto-resolved + div_by=16 at 1024x1024."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=16, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(False, VAE_FACTOR, PATCH_SIZE, IMG_W, IMG_H)
    check_realistic_borders("RECT-DiT-auto-d16", "Rectangular",
                            7/8, 4, 16)


def test_hex_realistic_dit_auto():
    """DiT Hex auto-resolved at 1024x1024."""
    raw = Settings("Hexagon", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(False, VAE_FACTOR, PATCH_SIZE, IMG_W, IMG_H)
    check_realistic_borders("HEX-DiT-auto", "Hexagon",
                            1.0, 0, 1)


def test_conv2d_realistic_auto():
    """Conv2d auto-resolved at 1024x1024."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(True, VAE_FACTOR, 1, IMG_W, IMG_H)
    assert resolved.work_img_w == 1024.0

    mask = at_mod.create_crop_mask(1024, 1024, resolved)
    assert mask.sum().item() == 1024 * 1024, "Conv2d auto-resolved should cover full image"
    print(f"\nConv2d RECT auto: work_img_w={resolved.work_img_w}, full image")


# ---------------------------------------------------------------------------
# Subsystem boundary comparison at image resolution (1024x1024)
# ---------------------------------------------------------------------------

def _compare_subsystems_at_boundary(label, mode, scale, min_margin, div_by, is_conv2d):
    """Compare all subsystems at 1024x1024 image resolution."""
    raw = Settings(mode, 0.0, scale=scale, min_margin=min_margin, divisible_by=div_by, conv2d_attention_wrapping=True)
    vf = VAE_FACTOR
    ps = 1 if is_conv2d else PATCH_SIZE
    resolved = raw._resolve_auto(is_conv2d, vf, ps, IMG_W, IMG_H)

    W_img, H_img = IMG_W, IMG_H
    W_lat, H_lat = W_img // vf, H_img // vf
    model_type = 'Conv2d' if is_conv2d else 'DiT (patch_size=2)'

    print(f"\n{'='*80}")
    print(f"{label}: {mode} scale={scale} min_margin={min_margin} div_by={div_by}")
    print(f"Image: {W_img}x{H_img}, vae_factor={vf}, {model_type}")
    print(f"{'='*80}")

    # 1. Wrapping at LATENT resolution (matching real Conv2d path), upscaled
    wrap_lat = build_wrapping_identity(W_lat, H_lat, resolved)
    wrap_np = wrap_lat.numpy() if isinstance(wrap_lat, torch.Tensor) else wrap_lat
    wrap_img = np.repeat(np.repeat(wrap_np, vf, axis=0), vf, axis=1)

    # 2. Attention/tiling at PATCH resolution, upscaled to image
    W_p, H_p = W_lat // ps, H_lat // ps
    attn_tile = build_tiling_identity(W_p, H_p, resolved, mode)
    sf = ps * vf
    attn_img = np.repeat(np.repeat(attn_tile, sf, axis=0), sf, axis=1)

    # Attention boundary patches
    attn_bnd = build_boundary_map(H_p, W_p, resolved, mode)
    attn_bnd_img = np.repeat(np.repeat(attn_bnd, sf, axis=0), sf, axis=1)

    # 3. Mask at image resolution
    mask = at_mod.create_crop_mask(W_img, H_img, resolved)
    mask_img = mask[0, :, :, 0].bool().numpy()

    # 4. Crop using ResolvedSettings dimensions
    cr_rmin, cr_rmax, cr_cmin, cr_cmax = at_mod.compute_crop_bounds(W_img, H_img, resolved)
    cr = (cr_rmin, cr_rmax, cr_cmin, cr_cmax)
    crop_img = np.zeros((H_img, W_img), dtype=bool)
    crop_img[cr[0]:cr[1]+1, cr[2]:cr[3]+1] = mask_img[cr[0]:cr[1]+1, cr[2]:cr[3]+1]

    # --- Compare ---
    wrap_mask_diff = int(np.sum(wrap_img != mask_img))
    total = W_img * H_img

    wrap_inside = int(wrap_img.sum())
    mask_inside = int(mask_img.sum())
    crop_inside = int(crop_img.sum())
    attn_inside = int(attn_img.sum())

    print(f"\n  Inside counts:")
    print(f"    Wrapping:  {wrap_inside} ({wrap_inside/total*100:.1f}%)")
    print(f"    Mask:      {mask_inside} ({mask_inside/total*100:.1f}%)")
    print(f"    Crop:      {crop_inside} ({crop_inside/total*100:.1f}%)")
    print(f"    Attention: {attn_inside} ({attn_inside/total*100:.1f}%)")
    print(f"    Wrap vs Mask: {wrap_mask_diff} px differ")

    # Hex mode: wrapping at latent resolution (upscaled) vs mask at image resolution
    # have inherent rounding differences on diagonal edges. Allow tolerance.
    if mode == "Hexagon":
        # Tolerance: hex perimeter * vae_factor (one row of blocks along boundary)
        hex_perimeter = 2 * np.pi * resolved.hex_size_img if resolved.hex_size_img > 0 else 0
        tolerance = int(hex_perimeter * vf)
        assert wrap_mask_diff <= tolerance, (
            f"Wrapping != Mask: {wrap_mask_diff} pixels differ (tolerance {tolerance}). "
            f"This indicates a logic error in how settings are applied."
        )
        print(f"    Wrap vs Mask: within tolerance ({wrap_mask_diff} <= {tolerance})")
    else:
        assert wrap_mask_diff == 0, (
            f"Wrapping != Mask: {wrap_mask_diff} pixels differ. "
            f"This indicates a logic error in how settings are applied."
        )


def test_subsystem_hex_s1_dit():
    """HEX scale=1.0, min_margin=-1 (auto->0), div_by=1, DiT."""
    _compare_subsystems_at_boundary(
        "HEX-DiT-s1.0", "Hexagon", scale=1.0, min_margin=-1, div_by=1, is_conv2d=False)


def test_subsystem_hex_s1_conv2d():
    """HEX scale=1.0, min_margin=-1 (auto->0), div_by=1, Conv2d."""
    _compare_subsystems_at_boundary(
        "HEX-Conv2d-s1.0", "Hexagon", scale=1.0, min_margin=-1, div_by=1, is_conv2d=True)


def test_subsystem_hex_s09_dit():
    """HEX scale=0.9, min_margin=-1 (auto->0), div_by=1, DiT."""
    _compare_subsystems_at_boundary(
        "HEX-DiT-s0.9", "Hexagon", scale=0.9, min_margin=-1, div_by=1, is_conv2d=False)


def test_subsystem_hex_s09_conv2d():
    """HEX scale=0.9, min_margin=-1 (auto->0), div_by=1, Conv2d."""
    _compare_subsystems_at_boundary(
        "HEX-Conv2d-s0.9", "Hexagon", scale=0.9, min_margin=-1, div_by=1, is_conv2d=True)


# ---------------------------------------------------------------------------
# ResolvedSettings type check
# ---------------------------------------------------------------------------

def test_resolved_is_resolved_settings():
    """_resolve_auto returns a ResolvedSettings instance."""
    raw = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    resolved = raw._resolve_auto(False, VAE_FACTOR, PATCH_SIZE, IMG_W, IMG_H)
    assert isinstance(resolved, ResolvedSettings)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
