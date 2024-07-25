"""
Hexagonal tiling implementation

Some of this code is taken from excelent guide https://www.redblobgames.com/grids/hexagons/
"""

import math


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

    q = round(frac_coords[0])
    r = round(frac_coords[1])
    s = round(frac_coords[2])

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


def hex_tiling(
    x: int, y: int, original_size: tuple[int, int], padded_size: tuple[int, int]
) -> tuple[int, int]:
    ssize = padded_size[0] // 2
    size = original_size[0] // 2

    q = (math.sqrt(3) / 3 * (x - ssize) - 1 / 3 * (y - ssize)) / size
    r = (2 / 3 * (y - ssize)) / size

    rounded = axial_round((q, r))
    q -= rounded[0]
    r -= rounded[1]

    xx = round(size * (math.sqrt(3) * q + (math.sqrt(3) / 2) * r))
    yy = round(size * ((3 / 2) * r))

    xx = (xx + ssize) % padded_size[0]
    yy = (yy + ssize) % padded_size[1]

    return (xx, yy)
