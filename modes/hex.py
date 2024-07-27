"""
Hexagonal tiling implementation

Some of this code is taken from excelent guide https://www.redblobgames.com/grids/hexagons/
"""

import math
import functools

from . import Settings
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


@functools.cache
def get_matrix(settings: Settings) -> np.ndarray:
    """
    Get rotation matrix

    :param settings: Tiling settings
    :return: Rotation matrix
    """

    return np.matmul(
        rotation_matrix(settings.rotation),
        # Hexagon basis vectors
        np.array([[math.sqrt(3), math.sqrt(3) / 2], [0, 3 / 2]]),
    )


@functools.cache
def get_inverse_matrix(settings: Settings) -> np.ndarray:
    """
    Get inverse rotation matrix

    :param settings: Tiling settings
    :return: Inverse rotation matrix
    """

    return np.linalg.inv(get_matrix(settings))


def hex_to_pixel(
    hex_coords: tuple[int, int], size: int, settings: Settings
) -> tuple[int, int]:
    """
    Convert hexagonal coordinates to pixel coordinates

    :param hex_coords: Hexagonal coordinates
    :param size: Size of hexagon
    :return: Pixel coordinates
    """

    (x, y) = (
        size
        * np.matmul(
            get_matrix(settings),
            np.array([[hex_coords[0]], [hex_coords[1]]]),
        ).flatten()
    )

    # We need to round!
    return (round(x), round(y))


def pixel_to_hex(
    pixel_coords: tuple[int, int], size: int, settings: Settings
) -> tuple[float, float]:
    """
    Convert pixel coordinates to fractional hexagonal coordinates

    :param pixel_coords: Pixel coordinates
    :param size: Size of hexagon
    :return: Fractional hexagonal coordinates
    """

    (q, r) = (
        np.matmul(
            get_inverse_matrix(settings),
            np.array([[pixel_coords[0]], [pixel_coords[1]]]),
        ).flatten()
        / size
    )

    return (q, r)


@functools.cache
def hex_tiling(
    x: int,
    y: int,
    original_size: tuple[int, int],
    padded_size: tuple[int, int],
    settings: Settings,
) -> tuple[int, int]:
    """
    Hexagonal tiling function

    :param x: X coordinate
    :param y: Y coordinate
    :param original_size: Original size of tensor
    :param padded_size: Padded size of tensor
    :param settings: Tiling settings
    :return (x, y): Coordinates
    """

    # Hexagon size - it needs to fit in the image
    size = min(original_size[0], original_size[1]) // 2
    # Shift the origin to the center of the image and convert to fractional hexagon coordinates
    q, r = pixel_to_hex(
        (x - padded_size[0] // 2, y - padded_size[1] // 2),
        size,
        settings,
    )
    # Round to nearest hexagon
    rounded = axial_round((q, r))
    # Get fractional part of hexagon coordinates
    q -= rounded[0]
    r -= rounded[1]
    # Convert back to pixel coordinates
    new_x, new_y = hex_to_pixel((q, r), size, settings)
    # Calculated coordinates are relative, so we need to shift them back
    new_x = (new_x + padded_size[0] // 2) % padded_size[0]
    new_y = (new_y + padded_size[1] // 2) % padded_size[1]

    return (new_x, new_y)
