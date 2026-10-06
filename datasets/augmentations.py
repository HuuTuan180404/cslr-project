import numpy as np


def gaussian_jitter(keypoints: np.ndarray, std: float = 0.003) -> np.ndarray:
    """
    Add Gaussian noise to keypoint coordinates.

    Args:
        keypoints: [T, J, 2]
        std: standard deviation of Gaussian noise.

    Returns:
        [T, J, 2]
    """

    noise = np.random.normal(loc=0.0, scale=std, size=keypoints.shape).astype(
        keypoints.dtype
    )

    return keypoints + noise


# right = gaussian_jitter(right, std=0.003)
# left = gaussian_jitter(left, std=0.003)
# body = gaussian_jitter(body, std=0.003)


def small_rotation(keypoints: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate keypoints around origin.

    Args:
        keypoints: [T, J, 2]
        angle: rotation angle in degrees.

    Returns:
        [T, J, 2]
    """

    theta = np.deg2rad(angle)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=keypoints.dtype
    )

    # [T, J, 2] @ [2, 2]
    return keypoints @ rotation_matrix.T


# angle = np.random.uniform(-5.0, 5.0)

# right = small_rotation(right, angle)
# left = small_rotation(left, angle)
# body = small_rotation(body, angle)


def temporal_resample(keypoints: np.ndarray, new_indices: np.ndarray) -> np.ndarray:
    """
    Temporal interpolation.

    Args:
        keypoints:
            [T, J, 2]

        new_indices:
            [T_new]

    Returns:
        [T_new, J, 2]
    """

    T, J, C = keypoints.shape

    old_indices = np.arange(T)

    output = np.empty((len(new_indices), J, C), dtype=keypoints.dtype)

    for j in range(J):
        for c in range(C):
            output[:, j, c] = np.interp(new_indices, old_indices, keypoints[:, j, c])

    return output


def generate_temporal_indices(T: int, scale_range=(0.9, 1.1)) -> np.ndarray:
    """
    Generate temporal sampling indices.

    Args:
        T:
            Original sequence length.

        scale_range:
            Temporal stretch/compression range.

            0.9 -> sequence ngắn hơn
            1.1 -> sequence dài hơn

    Returns:
        [T_new]
    """

    scale = np.random.uniform(scale_range[0], scale_range[1])

    T_new = max(2, int(round(T * scale)))

    old_indices = np.arange(T)

    new_indices = np.linspace(0, T - 1, T_new)

    return new_indices
