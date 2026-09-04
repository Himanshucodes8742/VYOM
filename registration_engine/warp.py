"""Perspective warping utilities for lunar image registration."""

import numpy as np
import cv2


def warp_image(
    source_img: np.ndarray,
    transform_matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp the source image onto the reference frame using a 3x3 homography.

    Args:
        source_img: Grayscale source image (2D uint8 numpy array).
        transform_matrix: 3x3 homography / perspective transformation matrix.
        output_shape: (height, width) of the output warped image — should match
                      the reference image dimensions.

    Returns:
        Warped image as a 2D uint8 numpy array with shape == output_shape.
    """
    height, width = output_shape[:2]
    warped = cv2.warpPerspective(
        source_img,
        transform_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped
