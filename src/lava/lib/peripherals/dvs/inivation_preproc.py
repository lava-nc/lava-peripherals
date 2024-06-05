import numpy as np

def crop(data, input_indices, cam_shape, crop_params, out_shape):
    """
    Spatial cropping of incoming event indices.

    The function also adjusts the indices to map to a new output shape, effectively resetting the indices to start
    from (0, 0) in the cropped area.

    Parameters:
    -----------
        data (np.ndarray): Array of polarities of incoming events
        input_indices (np.ndarray): Flat indices into the dataset, where each index corresponds
            to the flattened position of an event in the original camera-shaped array.
        cam_shape (tuple): A tuple (height, width) defining the shape of the camera array
            from which indices are derived.
        crop_params (tuple): A tuple (crop_x_l, crop_x_r, crop_y_t, crop_y_b) specifying the
            left, right, top, and bottom cropping boundaries. These values define the margins
            of cropping:
            - crop_x_l is the left boundary (inclusive),
            - crop_x_r is the right boundary (exclusive),
            - crop_y_t is the top boundary (inclusive),
            - crop_y_b is the bottom boundary (exclusive).
        out_shape (tuple): The shape (height, width) of the array for the output indices after cropping.
            This shape should reflect the new dimensions of the array after it has been cropped.

    Returns:
        output_data (np.ndarray): Cropped array of polarities
        output_indices (np.ndarray): Flat indices of the cropped elements within the context of
            the new out_shape, adjusted to start at (0, 0) in the cropped frame.

    Notes:
        - The indices are adjusted to reflect their new positions in a cropped array that starts
          at (0, 0), making them suitable for indexing into arrays shaped according to out_shape.
        - If there are no data points within the specified crop area, output_data will be an empty
          array and output_indices will be an empty array of dtype=int.
    """
    height, width = cam_shape
    crop_x_l, crop_x_r, crop_y_t, crop_y_b = crop_params
    max_x = width - crop_x_r
    max_y = height - crop_y_b

    # Unravel flat indices to 2D camera shape indices
    multi_indices = np.unravel_index(input_indices, cam_shape)
    multi_indices = np.vstack(multi_indices).T  # [row, col] format or [y, x]

    # Create boolean mask for cropping
    x_condition = (multi_indices[:, 1] >= crop_x_l) & (multi_indices[:, 1] < max_x)
    y_condition = (multi_indices[:, 0] >= crop_y_t) & (multi_indices[:, 0] < max_y)
    combined_condition = x_condition & y_condition

    # Apply the filter to remove events outside of the crop region
    cropped_indices = multi_indices[combined_condition]
    output_data = data[combined_condition]

    # Adjust indices to start from (0, 0) in the new output shape
    adjusted_indices = cropped_indices.copy()
    adjusted_indices[:, 0] -= crop_y_t  
    adjusted_indices[:, 1] -= crop_x_l

    # Convert adjusted indices back to flat indices within the new out_shape
    if len(adjusted_indices) > 0:
        output_indices = np.ravel_multi_index(
            (adjusted_indices[:, 0], adjusted_indices[:, 1]), out_shape)
    else:
        output_indices = np.array([], dtype=int)

    return output_data, output_indices

