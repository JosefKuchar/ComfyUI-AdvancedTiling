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

from ComfyUI_AdvancedTiling.modes import Settings, ResolvedSettings
from ComfyUI_AdvancedTiling.modes.rect import rect_tiling

VAE_FACTOR = 8
PATCH_SIZE = 2
IMG_W = 1024
IMG_H = 1024


def _resolve(scale, min_margin, divisible_by, mode="Rectangular",
             is_conv2d=False, vae_factor=VAE_FACTOR, patch_size=PATCH_SIZE,
             img_W=IMG_W, img_H=IMG_H):
    """Helper: create Settings and resolve."""
    s = Settings(mode, 0.0, scale=scale, min_margin=min_margin, divisible_by=divisible_by, conv2d_attention_wrapping=True)
    return s._resolve_auto(is_conv2d, vae_factor, patch_size, img_W, img_H)


# --- ResolvedSettings tests ---


def test_resolve_floor_margin_scale_dominates():
    """scale=0.875 gives base_margin=64 > min_margin_img=32, so margin=64."""
    r = _resolve(scale=0.875, min_margin=4, divisible_by=1)
    # base_margin = 1024 * 0.125 / 2 = 64, min_margin_img = 4 * 8 = 32
    # margin = max(64, 32) = 64, work = 1024 - 128 = 896
    assert r.work_img_w == 896.0
    assert r.margin_img_w == 64.0


def test_resolve_floor_margin_min_dominates():
    """scale=0.98 gives base_margin=10.24 < min_margin_img=32, so margin=32."""
    r = _resolve(scale=0.98, min_margin=4, divisible_by=1)
    # base_margin = 1024 * 0.02 / 2 = 10.24, min_margin_img = 32
    # margin = max(10.24, 32) = 32, work = 1024 - 64 = 960
    assert r.work_img_w == 960.0
    assert r.margin_img_w == 32.0


def test_resolve_divisible_by():
    """divisible_by=9 rounds working area down."""
    r = _resolve(scale=0.98, min_margin=4, divisible_by=9)
    # margin=32, work=960, 960//9*9=954, final_margin=(1024-954)/2=35
    assert r.work_img_w == 954.0
    assert r.margin_img_w == 35.0


def test_resolve_divisible_by_zero_rect():
    """divisible_by=0 disables rounding — work area stays as-is (float)."""
    r = _resolve(scale=0.98, min_margin=4, divisible_by=0)
    # margin=32, work=960 — no rounding applied
    assert r.work_img_w == 960.0
    assert r.work_img_h == 960.0
    assert r.margin_img_w == 32.0


def test_resolve_divisible_by_zero_hex():
    """divisible_by=0 disables rounding for hex — hex_size stays as-is (float)."""
    r = _resolve(scale=0.8, min_margin=0, divisible_by=0, mode="Hexagon")
    # base_margin = 512 * 0.2 = 102.4, hex_size = 409.6 — no rounding
    assert r.hex_size_img == 409.6


def test_resolve_divisible_by_one_floors_rect():
    """divisible_by=1 floors work area to integer."""
    r = _resolve(scale=0.98, min_margin=3, divisible_by=1)
    # base_margin = 1024*0.02/2 = 10.24, min_margin_img = 24
    # margin = 24, work = 1024 - 48 = 976 — already integer, stays 976
    assert r.work_img_w == 976.0


def test_resolve_divisible_by_one_floors_hex():
    """divisible_by=1 floors hex height to integer."""
    r = _resolve(scale=0.8, min_margin=0, divisible_by=1, mode="Hexagon")
    # hex_size = 409.6, hex_height = 819.2 -> 819, hex_size = 409.5
    assert r.hex_size_img == 409.5


def test_resolve_patch_downscale():
    """Patch values are image values / (vae_factor * patch_size)."""
    r = _resolve(scale=0.98, min_margin=4, divisible_by=1)
    assert r.work_patch_w == r.work_img_w / (VAE_FACTOR * PATCH_SIZE)


def test_resolve_rectangular_image():
    """Non-square image: margins differ per dimension."""
    r = _resolve(scale=0.98, min_margin=4, divisible_by=9, img_W=1024, img_H=768)
    assert r.work_img_w == 954.0
    # 768: base=768*0.02/2=7.68, min_img=32, margin=32, work=704, 704//9*9=702
    assert r.work_img_h == 702.0
    assert r.margin_img_w == 35.0


def test_resolve_hex_mode():
    """Hex mode computes hex_size from min(W,H) with divisible_by=0 (no rounding)."""
    r = _resolve(scale=0.8, min_margin=0, divisible_by=0, mode="Hexagon")
    # base_margin = 512 * 0.2 = 102.4, min_margin_img = 0
    # margin = 102.4, hex_size = 512 - 102.4 = 409.6 — no rounding
    assert r.hex_size_img == 409.6
    assert r.hex_size_patch == r.hex_size_img / (VAE_FACTOR * PATCH_SIZE)


def test_resolve_none_mode():
    """None mode has all zeros."""
    r = _resolve(scale=0.8, min_margin=4, divisible_by=1, mode="None")
    assert r.mode == "None"
    assert r.work_img_w == 0.0
    assert r.hex_size_img == 0.0


def test_resolve_auto_scale_conv2d():
    """scale=0.0 resolves to 1.0 for Conv2d models."""
    r = _resolve(scale=0.0, min_margin=-1, divisible_by=1, is_conv2d=True)
    # Conv2d Rect: scale=1.0, min_margin=0
    # base_margin = 0, min_margin_img = 0, work = 1024
    assert r.work_img_w == 1024.0
    assert r.margin_img_w == 0.0


def test_resolve_auto_scale_dit_rect():
    """scale=0.0 resolves to 7/8 for DiT Rectangular."""
    r = _resolve(scale=0.0, min_margin=-1, divisible_by=1, is_conv2d=False)
    # scale=7/8, min_margin=4
    # base_margin = 1024*0.125/2 = 64, min_margin_img = 32
    # margin = 64, work = 896
    assert r.work_img_w == 896.0
    assert r.margin_img_w == 64.0


def test_resolve_auto_scale_dit_hex():
    """scale=0.0 resolves to 1.0 for DiT Hexagon."""
    r = _resolve(scale=0.0, min_margin=-1, divisible_by=1, mode="Hexagon", is_conv2d=False)
    # scale=1.0, min_margin=0
    # base_margin = 0, hex_size = 512
    assert r.hex_size_img == 512.0


def test_resolve_no_mutation():
    """_resolve_auto returns a new ResolvedSettings, original Settings is unchanged."""
    s = Settings("Rectangular", 0.0, scale=0.0, min_margin=-1, divisible_by=1, conv2d_attention_wrapping=True)
    r = s._resolve_auto(False, VAE_FACTOR, PATCH_SIZE, IMG_W, IMG_H)
    assert s.scale == 0.0
    assert s.min_margin == -1
    assert isinstance(r, ResolvedSettings)


# --- rect_tiling tests (new API) ---


def test_rect_identity_inside():
    """Pixels inside the float rectangle map to themselves."""
    # work_w=10, work_h=10 at 10x10 → no margin, all identity
    result = rect_tiling(5, 5, (10, 10), 10.0, 10.0)
    assert result == (5, 5)


def test_rect_wraps_right_to_left():
    """Pixel at right edge wraps to left."""
    result = rect_tiling(10, 5, (10, 10), 10.0, 10.0)
    assert result == (0, 5)


def test_rect_wraps_bottom_to_top():
    """Pixel at bottom edge wraps to top."""
    result = rect_tiling(5, 10, (10, 10), 10.0, 10.0)
    assert result == (5, 0)


def test_rect_scaled_wraps_outside():
    """Pixel outside scaled rectangle wraps to inside."""
    # work_w=5.0, center=5.0, x=0 is outside [2.5, 7.5)
    result = rect_tiling(0, 0, (10, 10), 5.0, 5.0)
    assert result != (0, 0)


def test_rect_scaled_identity_inside():
    """Pixel inside scaled rectangle maps to itself."""
    # work_w=5.0, center=5.0, rect spans [2.5, 7.5), x=5 is inside
    result = rect_tiling(5, 5, (10, 10), 5.0, 5.0)
    assert result == (5, 5)


# --- calculate_mapping tests ---

import ComfyUI_AdvancedTiling.advanced_tiling as at_mod


def test_calculate_mapping_rect_identity():
    """All pixels inside rect map to themselves -> empty mapping."""
    r = _resolve(scale=1.0, min_margin=0, divisible_by=1)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((10, 10), (10, 10), r)
    assert len(src_x) == 0


def test_calculate_mapping_rect_scaled_has_remapping():
    """Scaled rect produces non-empty mapping for outside pixels."""
    r = _resolve(scale=0.5, min_margin=0, divisible_by=1, img_W=10, img_H=10)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((10, 10), (10, 10), r)
    assert len(src_x) > 0


# --- create_crop_mask tests ---


def test_crop_mask_rect_full():
    """scale=1.0, no margin -> all pixels in mask."""
    r = _resolve(scale=1.0, min_margin=0, divisible_by=1)
    mask = at_mod.create_crop_mask(10, 10, r)
    assert mask.shape == (1, 10, 10, 1)
    assert mask.sum().item() == 100


def test_crop_mask_rect_scaled():
    """scale=0.5 -> mask has fewer pixels than total."""
    r = _resolve(scale=0.5, min_margin=0, divisible_by=1, img_W=10, img_H=10)
    mask = at_mod.create_crop_mask(10, 10, r)
    assert mask.shape == (1, 10, 10, 1)
    assert mask.sum().item() < 100
    assert mask.sum().item() > 0


# --- integration tests ---


def test_end_to_end_scale_and_divisible():
    """Full pipeline: resolve -> wrapping -> crop mask."""
    W_lat, H_lat = 128, 128
    r = _resolve(scale=0.8, min_margin=4, divisible_by=64)

    # 1. Working area is valid
    work_lat_w, work_lat_h = r.work_at(W_lat, H_lat)
    assert work_lat_w > 0 and work_lat_h > 0

    # 2. Wrapping: center pixel is identity
    result = rect_tiling(W_lat // 2, H_lat // 2, (W_lat, H_lat),
                         work_lat_w, work_lat_h)
    assert result == (W_lat // 2, H_lat // 2)

    # 3. Wrapping: corner pixel wraps
    result = rect_tiling(0, 0, (W_lat, H_lat), work_lat_w, work_lat_h)
    assert result != (0, 0)

    # 4. Mapping is non-empty
    mapping = at_mod.calculate_mapping((W_lat, H_lat), (W_lat, H_lat), r)
    assert len(mapping[0]) > 0

    # 5. Crop mask covers fewer pixels than total
    mask = at_mod.create_crop_mask(IMG_W, IMG_H, r)
    assert mask.sum().item() < IMG_W * IMG_H
    assert mask.sum().item() > 0


# --- Settings equality tests ---


def test_settings_eq():
    """Settings with same values are equal."""
    s1 = Settings("Rectangular", 0.0, scale=0.8, min_margin=4, divisible_by=64, conv2d_attention_wrapping=True)
    s2 = Settings("Rectangular", 0.0, scale=0.8, min_margin=4, divisible_by=64, conv2d_attention_wrapping=True)
    assert s1 == s2
    assert hash(s1) == hash(s2)


def test_settings_eq_different():
    """Settings with different values are not equal."""
    s1 = Settings("Rectangular", 0.0, scale=0.8, min_margin=4, divisible_by=1, conv2d_attention_wrapping=True)
    s2 = Settings("Rectangular", 0.0, scale=0.9, min_margin=4, divisible_by=1, conv2d_attention_wrapping=True)
    assert s1 != s2


# --- None mode tests ---


def test_none_calculate_mapping_empty():
    """None mode produces empty mapping."""
    r = _resolve(scale=1.0, min_margin=0, divisible_by=1, mode="None")
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((10, 10), (10, 10), r)
    assert len(src_x) == 0


def test_none_create_crop_mask_full():
    """None mode mask covers all pixels."""
    r = _resolve(scale=1.0, min_margin=0, divisible_by=1, mode="None")
    mask = at_mod.create_crop_mask(10, 10, r)
    assert mask.shape == (1, 10, 10, 1)
    assert mask.sum().item() == 100


# --- Hex mode basic tests ---


def test_hex_calculate_mapping_nonempty():
    """Hex mode with scale < 1 produces non-empty mapping."""
    r = _resolve(scale=0.8, min_margin=0, divisible_by=1, mode="Hexagon", img_W=32, img_H=32)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((4, 4), (4, 4), r)
    assert len(src_x) > 0


def test_hex_create_crop_mask():
    """Hex mode mask covers fewer pixels than total."""
    r = _resolve(scale=0.8, min_margin=0, divisible_by=1, mode="Hexagon", img_W=32, img_H=32)
    mask = at_mod.create_crop_mask(32, 32, r)
    assert mask.shape == (1, 32, 32, 1)
    assert mask.sum().item() < 32 * 32
    assert mask.sum().item() > 0


def test_hex_crop_mask_with_margin():
    """Hex mode with min_margin reduces mask area further."""
    r1 = _resolve(scale=0.8, min_margin=0, divisible_by=1, mode="Hexagon", img_W=32, img_H=32)
    r2 = _resolve(scale=0.8, min_margin=2, divisible_by=1, mode="Hexagon", img_W=32, img_H=32)
    mask1 = at_mod.create_crop_mask(32, 32, r1)
    mask2 = at_mod.create_crop_mask(32, 32, r2)
    assert mask2.sum().item() < mask1.sum().item()


# --- tiling vs mapping consistency ---


def test_tiling_vs_mapping_consistency():
    """Per-pixel rect_tiling and vectorized calculate_mapping produce same results."""
    W_lat, H_lat = 16, 16
    r = _resolve(scale=0.6, min_margin=2, divisible_by=1, img_W=W_lat * VAE_FACTOR, img_H=H_lat * VAE_FACTOR)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((W_lat, H_lat), (W_lat, H_lat), r)

    work_lat_w, work_lat_h = r.work_at(W_lat, H_lat)
    for i in range(len(src_x)):
        sx, sy = src_x[i].item(), src_y[i].item()
        nx, ny = new_x[i].item(), new_y[i].item()
        px, py = rect_tiling(sx, sy, (W_lat, H_lat), work_lat_w, work_lat_h)
        assert (px, py) == (nx, ny), f"Mismatch at ({sx},{sy}): rect_tiling={px},{py} vs mapping={nx},{ny}"


# --- non-square tests ---


def test_non_square_calculate_mapping():
    """calculate_mapping works for non-square dimensions."""
    r = _resolve(scale=0.5, min_margin=0, divisible_by=1, img_W=20, img_H=10)
    src_x, src_y, new_x, new_y = at_mod.calculate_mapping((20, 10), (20, 10), r)
    assert len(src_x) > 0
    assert len(src_x) == len(src_y) == len(new_x) == len(new_y)


def test_non_square_create_crop_mask():
    """create_crop_mask works for non-square dimensions."""
    r = _resolve(scale=0.5, min_margin=0, divisible_by=1, img_W=20, img_H=10)
    mask = at_mod.create_crop_mask(20, 10, r)
    assert mask.shape == (1, 10, 20, 1)
    assert mask.sum().item() < 20 * 10
    assert mask.sum().item() > 0


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
