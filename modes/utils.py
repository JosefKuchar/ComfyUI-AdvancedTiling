"""
Various utility functions used across multiple tiling modes
"""

import math
import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    """
    Generate a 2D rotation matrix

    :param angle: Angle in radians
    :return: Rotation matrix
    """

    # Convert angle to radians
    angle = math.radians(angle)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
