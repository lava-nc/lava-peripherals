# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


def crop(data: np.ndarray,
        input_indices: np.ndarray,
        cam_shape: tuple,
        crop_params: tuple,
        out_shape: tuple
        ) -> ty.Tuple[np.ndarray, np.ndarray]:
    """
    Spatial cropping of incoming event indices.

    The function also adjusts the indices to map to a new output shape starting
    from (0, 0) in the cropped area.

    Parameters:
    -----------
        data: np.ndarray: 
            Array of polarities of incoming events
        input_indices: np.ndarray
            Flat indices of events in the camera-shaped array.
        cam_shape: tuple
            A tuple (y, x) defining the shape of the camera array.
        crop_params: tuple
            A tuple (crop_l, crop_r, crop_t, crop_b) specifying the
            left, right, top, and bottom cropping boundaries.
            Margin definitions of cropping:
            - crop_x_l is the left boundary (inclusive),
            - crop_x_r is the right boundary (exclusive),
            - crop_y_t is the top boundary (inclusive),
            - crop_y_b is the bottom boundary (exclusive).
        out_shape: tuple
            The shape (y, x) of the output array indices after cropping.

    Returns:
        output_data (np.ndarray): Cropped array of polarities
        output_indices (np.ndarray): Flat indices of events in the new out_shape,
            adjusted to start at (0, 0) in the cropped frame.

    Notes:
        - If there are no valid events in the new crop area, outputs will be empty
        arrays of dtype=int.
    """
    height, width = cam_shape
    crop_l, crop_r, crop_t, crop_b = crop_params
    max_x = width - crop_r
    max_y = height - crop_b

    # Unravel flat indices to 2D camera shape indices
    multi_indices = np.unravel_index(input_indices, cam_shape)
    multi_indices = np.vstack(multi_indices).T  # [row, col] format or [y, x]

    # Create boolean mask for cropping
    x_con = (multi_indices[:, 1] >= crop_l) & (multi_indices[:, 1] < max_x)
    y_con = (multi_indices[:, 0] >= crop_t) & (multi_indices[:, 0] < max_y)
    combined_con = x_con & y_con

    # Apply the filter to remove events outside of the crop region
    cropped_indices = multi_indices[combined_con]
    output_data = data[combined_con]

    # Adjust indices to start from (0, 0) in the new output shape
    adjusted_indices = cropped_indices.copy()
    adjusted_indices[:, 0] -= crop_t
    adjusted_indices[:, 1] -= crop_l

    # Convert adjusted indices back to flat indices within the new out_shape
    if len(adjusted_indices) > 0:
        output_indices = np.ravel_multi_index(
            (adjusted_indices[:, 0], adjusted_indices[:, 1]), out_shape)
    else:
        output_indices = np.array([], dtype=int)

    return output_data, output_indices
