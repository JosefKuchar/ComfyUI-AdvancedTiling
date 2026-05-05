"""
Hexagonal tiling implementation

Some of this code is taken from excelent guide https://www.redblobgames.com/grids/hexagons/
"""

import math
import functools

from .utils import rotation_matrix
import numpy as np


def cube_to_axial(cube_coords: tuple[int, int, int]) -> tuple[int, int]:
    """
    Convert cube coordinates to axial coordinates

    :param cube_coords: Cube coordinates
    :return: Axial coordinates
    """

    return (cube_coords[0], cube_coords[1])


def axial_to_cube(axial_coords: tuple[int, int]) -> tuple[int, int, int]:
    """
    Convert axial coordinates to cube coordinates

    :param axial_coords: Axial coordinates
    :return: Cube coordinates
    """

    q = axial_coords[0]
    r = axial_coords[1]
    s = -q - r

    return (q, r, s)


def axial_round(frac_coords: tuple[float, float]) -> tuple[int, int]:
    """
    Round fractional axial coordinates to nearest axial coordinate

    :param frac_coords: Fractional axial coordinates
    :return: Axial coordinates
    """

    return cube_to_axial(cube_round(axial_to_cube(frac_coords)))


def cube_round(frac_coords: tuple[float, float, float]) -> tuple[int, int, int]:
    """
    Round fractional cube coordinates to nearest cube coordinate

    :param frac_coords: Fractional cube coordinates
    :return: Cube coordinates
    """

    q = math.floor(frac_coords[0] + 0.5)
    r = math.floor(frac_coords[1] + 0.5)
    s = math.floor(frac_coords[2] + 0.5)

    q_diff = abs(q - frac_coords[0])
    r_diff = abs(r - frac_coords[1])
    s_diff = abs(s - frac_coords[2])

    if q_diff > r_diff and q_diff > s_diff:
        q = -r - s
    elif r_diff > s_diff:
        r = -q - s
    else:
        s = -q - r

    return (q, r, s)


@functools.cache
def get_matrix(rotation: float) -> np.ndarray:
    """
    Get rotation matrix

    :param rotation: Rotation angle in degrees
    :return: Rotation matrix
    """

    return np.matmul(
        rotation_matrix(rotation),
        # Hexagon basis vectors
        np.array([[math.sqrt(3), math.sqrt(3) / 2], [0, 3 / 2]]),
    )


@functools.cache
def get_inverse_matrix(rotation: float) -> np.ndarray:
    """
    Get inverse rotation matrix

    :param rotation: Rotation angle in degrees
    :return: Inverse rotation matrix
    """

    return np.linalg.inv(get_matrix(rotation))


def hex_to_pixel(
    hex_coords: tuple[int, int], size: int, rotation: float
) -> tuple[int, int]:
    """
    Convert hexagonal coordinates to pixel coordinates

    :param hex_coords: Hexagonal coordinates
    :param size: Size of hexagon
    :param rotation: Rotation angle in degrees
    :return: Pixel coordinates
    """

    (x, y) = (
        size
        * np.matmul(
            get_matrix(rotation),
            np.array([[hex_coords[0]], [hex_coords[1]]]),
        ).flatten()
    )

    return (int(math.floor(x + 0.5)), int(math.floor(y + 0.5)))


def pixel_to_hex(
    pixel_coords: tuple[int, int], size: int, rotation: float
) -> tuple[float, float]:
    """
    Convert pixel coordinates to fractional hexagonal coordinates

    :param pixel_coords: Pixel coordinates
    :param size: Size of hexagon
    :param rotation: Rotation angle in degrees
    :return: Fractional hexagonal coordinates
    """

    (q, r) = (
        np.matmul(
            get_inverse_matrix(rotation),
            np.array([[pixel_coords[0]], [pixel_coords[1]]]),
        ).flatten()
        / size
    )

    return (q, r)


def cube_round_offsets(q: float, r: float):
    """Cube-round (q, r) and return the fractional offsets (dq, dr)."""
    s = -q - r
    rq = math.floor(q + 0.5)
    rr = math.floor(r + 0.5)
    rs = math.floor(s + 0.5)
    q_diff = abs(rq - q)
    r_diff = abs(rr - r)
    s_diff = abs(rs - s)
    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs
    else:
        rs = -rq - rr
    return q - rq, r - rr


def _cube_round_vectorized(q, r):
    """Vectorized cube-round using floor(x+0.5) half-up rounding.

    Returns (rq, rr) — the rounded integer coordinates.
    """
    s = -q - r
    rq = np.floor(q + 0.5)
    rr = np.floor(r + 0.5)
    rs = np.floor(s + 0.5)
    q_diff = np.abs(rq - q)
    r_diff = np.abs(rr - r)
    s_diff = np.abs(rs - s)
    mask_q = (q_diff > r_diff) & (q_diff > s_diff)
    mask_r = ~mask_q & (r_diff > s_diff)
    rq = np.where(mask_q, -rr - rs, rq)
    rr = np.where(mask_r, -rq - rs, rr)
    return rq, rr


@functools.cache
def hex_tiling(
    x: int,
    y: int,
    padded_size: tuple[int, int],
    hex_size: float,
    rotation: float,
) -> tuple[int, int]:
    """Hexagonal tiling with pre-computed hex radius.

    Uses 4-neighbor search to find the integer source pixel whose hex offset
    best matches the destination's hex offset, eliminating rounding mismatches.

    :param x: X coordinate in padded space
    :param y: Y coordinate in padded space
    :param padded_size: (width, height) of padded tensor
    :param hex_size: Hex radius (pre-computed at appropriate resolution)
    :param rotation: Rotation angle in degrees
    :return: (new_x, new_y) source coordinates in padded space
    """
    inv_mat = get_inverse_matrix(rotation)
    mat = get_matrix(rotation)

    q, r = pixel_to_hex(
        (x - padded_size[0] // 2, y - padded_size[1] // 2),
        hex_size,
        rotation,
    )
    target_dq, target_dr = cube_round_offsets(q, r)

    # Continuous pixel position of the target offset
    pixel = hex_size * (mat @ np.array([[target_dq], [target_dr]])).flatten()
    base_x = int(math.floor(pixel[0]))
    base_y = int(math.floor(pixel[1]))

    # 4-neighbor search: pick integer pixel with closest hex offset
    best_x, best_y = base_x, base_y
    best_err = float('inf')
    for dx in (0, 1):
        for dy in (0, 1):
            cx, cy = base_x + dx, base_y + dy
            cq, cr = (
                inv_mat @ np.array([[cx], [cy]])
            ).flatten() / hex_size
            cand_dq, cand_dr = cube_round_offsets(float(cq), float(cr))
            err = max(abs(cand_dq - target_dq), abs(cand_dr - target_dr))
            if err < best_err:
                best_err = err
                best_x, best_y = cx, cy

    new_x = (best_x + padded_size[0] // 2) % padded_size[0]
    new_y = (best_y + padded_size[1] // 2) % padded_size[1]

    return (new_x, new_y)


def hex_tiling_at(x, y, padded_size, resolved):
    """Hexagonal tiling with automatic resolution scaling from ResolvedSettings.

    Scales hex size to the tensor resolution of padded_size,
    then delegates to hex_tiling().
    """
    size = resolved.hex_size_at(padded_size[0], padded_size[1])
    return hex_tiling(x, y, padded_size, size, resolved.rotation)


def hex_remap_batch(
    centers_x, centers_y, size, wrap_w, wrap_h, rotation,
):
    """Vectorized hex coordinate remapping with 4-neighbor search.

    Uses 4-neighbor search to find the integer source pixel whose hex offset
    best matches the destination's hex offset, eliminating rounding mismatches
    that occur when the hex-to-pixel round-trip lands on a suboptimal integer.

    :param centers_x: X coordinates centered at grid origin (any shape, raveled)
    :param centers_y: Y coordinates centered at grid origin (any shape, raveled)
    :param size: Hex radius at this resolution
    :param wrap_w: Width for modulo wrapping
    :param wrap_h: Height for modulo wrapping
    :param rotation: Rotation angle in degrees
    :return: (new_x, new_y) as int64 arrays, same shape as input
    """
    orig_shape = np.asarray(centers_x).shape
    flat_x = np.asarray(centers_x).ravel().astype(np.float64)
    flat_y = np.asarray(centers_y).ravel().astype(np.float64)

    inv_mat = get_inverse_matrix(rotation)
    mat = get_matrix(rotation)

    # pixel_to_hex (batch): inverse matrix multiply + divide by size
    pts = np.stack([flat_x, flat_y], axis=0)
    qr = (inv_mat @ pts) / size
    q, r = qr[0], qr[1]

    # cube_round — floor(x+0.5) for consistent half-up rounding
    rq, rr = _cube_round_vectorized(q, r)

    # Target hex offsets (desired relative position within hex cell)
    target_dq = q - rq
    target_dr = r - rr

    # Continuous pixel position of the target offset
    pixel = size * (mat @ np.stack([target_dq, target_dr], axis=0))
    base_x = np.floor(pixel[0]).astype(np.int64)
    base_y = np.floor(pixel[1]).astype(np.int64)

    # 4-neighbor search: pick the integer pixel whose hex offset
    # is closest to the target offset (Chebyshev distance in hex space).
    best_x = np.empty_like(base_x)
    best_y = np.empty_like(base_y)
    best_err = np.full(len(q), np.inf)

    for dx in (0, 1):
        for dy in (0, 1):
            cx = (base_x + dx).astype(np.float64)
            cy = (base_y + dy).astype(np.float64)
            c_pts = np.stack([cx, cy], axis=0)
            c_qr = (inv_mat @ c_pts) / size
            cq, cr = c_qr[0], c_qr[1]
            crq, crr = _cube_round_vectorized(cq, cr)
            cand_dq = cq - crq
            cand_dr = cr - crr
            err = np.maximum(np.abs(cand_dq - target_dq), np.abs(cand_dr - target_dr))
            better = err < best_err
            best_x = np.where(better, base_x + dx, best_x)
            best_y = np.where(better, base_y + dy, best_y)
            best_err = np.where(better, err, best_err)

    new_x = (best_x + wrap_w // 2) % wrap_w
    new_y = (best_y + wrap_h // 2) % wrap_h
    return new_x.reshape(orig_shape), new_y.reshape(orig_shape)


def hex_distance_grid(padded_size, hex_size, rotation):
    """Compute max-norm distance from nearest hex center for every pixel.

    Uses the same cube_round as hex_remap_batch (np.floor-based),
    ensuring bit-identical results with the Conv2d wrapping path.

    :param padded_size: (width, height) of the tensor
    :param hex_size: Hex radius at this resolution
    :param rotation: Rotation angle in degrees
    :return: 2D numpy array (height, width) of distances
    """
    pw, ph = padded_size
    inv_mat = get_inverse_matrix(rotation)

    cx = np.arange(pw, dtype=np.float64) - pw // 2
    cy = np.arange(ph, dtype=np.float64) - ph // 2
    grid_cx, grid_cy = np.meshgrid(cx, cy, indexing='xy')

    pts = np.stack([grid_cx.ravel(), grid_cy.ravel()], axis=0)
    qr = (inv_mat @ pts) / hex_size
    hq, hr = qr[0], qr[1]
    hs = -hq - hr

    rq, rr = _cube_round_vectorized(hq, hr)
    rs = -rq - rr

    dist = np.maximum(np.maximum(np.abs(hq - rq), np.abs(hr - rr)), np.abs(hs - rs))
    return dist.reshape(ph, pw)
