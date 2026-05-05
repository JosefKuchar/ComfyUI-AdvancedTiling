"""
Test: Central hex + 6 surrounding hexes tile seamlessly at exact floating-point offsets.

Verifies that the hex crop mask produces tiles that can be placed at exact
floating-point hex grid positions without gaps or overlaps.  This is how
game engines (Godot, Unity) place hex tiles — using Vector2 positions
computed from hex_size * basis_matrix * [q, r], not integer-rounded offsets.

The key identity: pixel_to_hex(P - offset) = pixel_to_hex(P) - [q, r]
where offset = hex_size * mat @ [q, r].  This holds exactly in float64,
so cube_round produces perfectly complementary cell assignments.

Checks:
  1. Crop mask agrees with hex cell membership
  2. Zero overlap between central mask and each neighbor's exact-float mask
  3. Content reassembly: wrapped image split into 7 cells matches original
"""

import sys
import os
import time

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
from ComfyUI_AdvancedTiling.modes.hex import get_inverse_matrix, get_matrix, _cube_round_vectorized
import ComfyUI_AdvancedTiling.advanced_tiling as at_mod

VAE_FACTOR = 8
HEX_NEIGHBORS = np.array([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)], dtype=np.float64)
IMG_W, IMG_H = 1024, 1024


def _resolve(scale, div_by):
    s = Settings("Hexagon", 0.0, scale=scale, min_margin=0,
                 divisible_by=div_by, conv2d_attention_wrapping=False)
    return s._resolve_auto(True, VAE_FACTOR, 1, IMG_W, IMG_H)


def _hex_coords_grid(hex_size, rotation):
    """Fractional hex coords (q, r) for every pixel in the image."""
    inv_mat = get_inverse_matrix(rotation)
    cx = np.arange(IMG_W, dtype=np.float64) - IMG_W // 2
    cy = np.arange(IMG_H, dtype=np.float64) - IMG_H // 2
    gx, gy = np.meshgrid(cx, cy, indexing='xy')
    pts = np.stack([gx.ravel(), gy.ravel()], axis=0)
    qr = (inv_mat @ pts) / hex_size
    return qr[0].reshape(IMG_H, IMG_W), qr[1].reshape(IMG_H, IMG_W)


def run_checks(scale, div_by):
    """Run all 3 checks for one (scale, div_by) config, sharing hex grid."""
    t_start = time.perf_counter()

    # --- Shared setup ---
    resolved = _resolve(scale, div_by)
    hex_size = resolved.hex_size_at(IMG_W, IMG_H)
    t_resolve = time.perf_counter()

    q, r = _hex_coords_grid(hex_size, resolved.rotation)
    rq, rr = _cube_round_vectorized(q, r)
    central = (rq == 0) & (rr == 0)
    t_grid = time.perf_counter()

    # --- Check 1: mask agreement ---
    mask_tensor = at_mod.create_crop_mask(IMG_W, IMG_H, resolved)
    crop_mask = mask_tensor[0, :, :, 0].numpy() > 0.5
    t_mask = time.perf_counter()

    n_disagree = int((crop_mask != central).sum())
    assert n_disagree == 0, (
        f"mask_agreement: {n_disagree} pixels disagree "
        f"(scale={scale}, div={div_by})"
    )

    # --- Check 2: no overlap (batched — one cube_round for all 6 neighbors) ---
    # q_flat/r_flat: (N,), shifts: (6,2) → shifted_q/shifted_r: (6, N)
    q_flat = q.ravel()
    r_flat = r.ravel()
    shifted_q = q_flat[np.newaxis, :] - HEX_NEIGHBORS[:, 0, np.newaxis]  # (6, N)
    shifted_r = r_flat[np.newaxis, :] - HEX_NEIGHBORS[:, 1, np.newaxis]  # (6, N)
    srq, srr = _cube_round_vectorized(shifted_q, shifted_r)  # (6, N) each
    for i in range(6):
        neighbor = (srq[i].reshape(IMG_H, IMG_W) == 0) & (srr[i].reshape(IMG_H, IMG_W) == 0)
        overlap = int((central & neighbor).sum())
        assert overlap == 0, (
            f"no_overlap: ({HEX_NEIGHBORS[i, 0]:.0f},{HEX_NEIGHBORS[i, 1]:.0f}) "
            f"has {overlap} pixel overlap (scale={scale}, div={div_by})"
        )
    t_overlap = time.perf_counter()

    # --- Check 3: content reassembly ---
    img = np.zeros((IMG_H, IMG_W, 2), dtype=np.int64)
    img[:, :, 0] = np.arange(IMG_W)[np.newaxis, :]
    img[:, :, 1] = np.arange(IMG_H)[:, np.newaxis]

    mapping = at_mod.calculate_mapping((IMG_W, IMG_H), (IMG_W, IMG_H), resolved)
    wrapped = img.copy()
    if len(mapping[0]) > 0:
        sx, sy = mapping[0].numpy(), mapping[1].numpy()
        nx, ny = mapping[2].numpy(), mapping[3].numpy()
        wrapped[sy, sx] = img[ny, nx]
    t_mapping = time.perf_counter()

    # 4-neighbor search — batched into single matmul
    mat = get_matrix(resolved.rotation)
    inv_mat = get_inverse_matrix(resolved.rotation)
    dq = q - rq
    dr = r - rr
    src_x = hex_size * (float(mat[0, 0]) * dq + float(mat[0, 1]) * dr)
    src_y = hex_size * (float(mat[1, 0]) * dq + float(mat[1, 1]) * dr)
    base_x = np.floor(src_x).astype(np.int64)
    base_y = np.floor(src_y).astype(np.int64)

    # Stack all 4 candidates: (4, 2, N)
    offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    bx_flat = base_x.ravel()
    by_flat = base_y.ravel()
    all_x = (bx_flat[np.newaxis, :] + offsets[:, 0, np.newaxis]).ravel()
    all_y = (by_flat[np.newaxis, :] + offsets[:, 1, np.newaxis]).ravel()
    all_pts = np.stack([all_x, all_y], axis=0)  # (2, 4*N)
    all_qr = (inv_mat @ all_pts) / hex_size       # (2, 4*N)
    aq = all_qr[0].reshape(4, IMG_H, IMG_W)
    ar = all_qr[1].reshape(4, IMG_H, IMG_W)
    a_rq, a_rr = _cube_round_vectorized(aq, ar)
    a_dq = aq - a_rq
    a_dr = ar - a_rr
    # Reshape to (4, H, W) for comparison with dq/dr (H, W)
    dq_exp = np.broadcast_to(dq, (4, IMG_H, IMG_W))
    dr_exp = np.broadcast_to(dr, (4, IMG_H, IMG_W))
    errs = np.maximum(np.abs(a_dq - dq_exp), np.abs(a_dr - dr_exp))
    best_idx = np.argmin(errs, axis=0)  # (H, W)
    best_x = base_x + (best_idx % 2)
    best_y = base_y + (best_idx // 2)
    t_search = time.perf_counter()

    src_cx = best_x + IMG_W // 2
    src_cy = best_y + IMG_H // 2
    valid = (src_cx >= 0) & (src_cx < IMG_W) & (src_cy >= 0) & (src_cy < IMG_H)

    cells = [(0, 0)] + [(int(n[0]), int(n[1])) for n in HEX_NEIGHBORS]
    for cq, cr in cells:
        cell_mask = (rq == cq) & (rr == cr)
        if not cell_mask.any():
            continue
        v = valid & cell_mask
        if not v.any():
            continue
        expected = img[src_cy[v], src_cx[v]]
        actual = wrapped[np.where(v)[0], np.where(v)[1]]
        n_mismatch = int((expected[:, 0] != actual[:, 0]).sum() +
                         (expected[:, 1] != actual[:, 1]).sum())
        assert n_mismatch == 0, (
            f"content_reassembly: cell ({cq},{cr}) has {n_mismatch} pixel mismatches "
            f"(scale={scale}, div={div_by})"
        )
    t_verify = time.perf_counter()

    return {
        "resolve": t_resolve - t_start,
        "hex_grid": t_grid - t_resolve,
        "create_crop_mask": t_mask - t_grid,
        "overlap_check": t_overlap - t_mask,
        "calculate_mapping": t_mapping - t_overlap,
        "4-neigh_search": t_search - t_mapping,
        "cell_verify": t_verify - t_search,
    }


# ---------------------------------------------------------------------------
# Test configurations — div_by=0 omitted at scale=1.0 (identical to div_by=1)
# ---------------------------------------------------------------------------

CASES = [
    (0.9, 0), (0.9, 1), (0.9, 8), (0.9, 9),
    (1.0, 1), (1.0, 8), (1.0, 9),
]


def _run_case(scale, div_by):
    label = f"s={scale} d={div_by}"
    try:
        timings = run_checks(scale, div_by)
        return label, timings, None
    except Exception as e:
        return label, None, str(e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    passed = 0
    failed = 0
    all_timings = {}
    n_workers = min(len(CASES), os.cpu_count() or 4)

    t_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_case, s, d): (s, d) for s, d in CASES}
        for fut in as_completed(futures):
            label, timings, err = fut.result()
            if err is None:
                all_timings[label] = timings
                passed += 1
            else:
                print(f"  FAIL  {label}: {err}")
                failed += 1
    t_total = time.perf_counter() - t_total

    print(f"\n{passed} passed, {failed} failed  ({t_total:.1f}s)\n")

    phase_names = ["resolve", "hex_grid", "create_crop_mask", "overlap_check",
                   "calculate_mapping", "4-neigh_search", "cell_verify"]
    for label in [f"s={s} d={d}" for s, d in CASES]:
        t = all_timings.get(label)
        if t is None:
            continue
        parts = " | ".join(f"{p}={t[p]:.3f}s" for p in phase_names)
        print(f"  {label}: {sum(t.values()):.3f}s  [{parts}]")

    sys.exit(1 if failed else 0)
