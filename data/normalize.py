import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize_lips(lips):
    """
    lips: [T, 19, 2]
    return: [T, 19, 2]
    """
    center = lips.mean(dim=1, keepdim=True)  # [T, 1, 2]

    lips = lips - center

    min_xy = lips.min(dim=1, keepdim=True).values
    max_xy = lips.max(dim=1, keepdim=True).values

    scale = (max_xy - min_xy).amax(dim=-1, keepdim=True)
    scale = scale.clamp_min(1e-6)

    lips = lips / scale

    return lips


def normalize_hand_bbox_01(hand, eps=1e-6):
    """
    Normalize hand keypoints using bounding box to [0, 1].

    Args:
        hand: [T, 21, 2]

    Returns:
        [T, 21, 2]
    """

    min_xy = hand.amin(dim=1, keepdim=True)  # [T, 1, 2]
    max_xy = hand.amax(dim=1, keepdim=True)  # [T, 1, 2]

    bbox_size = (max_xy - min_xy).clamp_min(eps)

    hand = (hand - min_xy) / bbox_size

    return hand


def normalize_hand_bbox_center(hand, eps=1e-6):
    """
    Normalize hand keypoints using bbox center and
    a shared bbox scale.

    Args:
        hand: [T, 21, 2]

    Returns:
        [T, 21, 2]
    """

    min_xy = hand.amin(dim=1, keepdim=True)  # [T, 1, 2]
    max_xy = hand.amax(dim=1, keepdim=True)  # [T, 1, 2]

    # Bbox center
    center = (min_xy + max_xy) / 2.0

    # One scale for both x and y
    bbox_size = max_xy - min_xy
    scale = bbox_size.amax(dim=-1, keepdim=True)
    scale = scale.clamp_min(eps)

    hand = (hand - center) / scale

    return hand


# Wrist + scale
def normalize_hand_wrist(hand, eps=1e-6):
    """
    Normalize hand keypoints using wrist as origin
    and wrist -> middle MCP as scale.

    Args:
        hand: [T, 21, 2]

    Returns:
        [T, 21, 2]
    """

    # Point 0 = wrist
    wrist = hand[:, 0:1, :]  # [T, 1, 2]

    # Translate wrist to origin
    hand = hand - wrist

    # Point 9 = middle finger MCP
    scale = torch.linalg.vector_norm(hand[:, 9:10, :], dim=-1, keepdim=True)
    # [T, 1, 1]

    scale = scale.clamp_min(eps)

    hand = hand / scale

    return hand


def visualize_hand_normalizations(hand, frame_idx=0):
    """
    Visualize original hand and 3 normalization methods.

    Args:
        hand: Tensor [T, 21, 2] or numpy array [T, 21, 2]
        frame_idx: frame to visualize
    """

    if isinstance(hand, torch.Tensor):
        hand = hand.detach().cpu().numpy()

    # --------------------------------------------------
    # Original frame
    # --------------------------------------------------
    original = hand[frame_idx]

    # --------------------------------------------------
    # Apply normalizations
    # --------------------------------------------------
    hand_tensor = torch.from_numpy(hand).float()

    bbox_01 = normalize_hand_bbox_01(hand_tensor)
    bbox_center = normalize_hand_bbox_center(hand_tensor)
    wrist = normalize_hand_wrist(hand_tensor)

    bbox_01 = bbox_01[frame_idx].numpy()
    bbox_center = bbox_center[frame_idx].numpy()
    wrist = wrist[frame_idx].numpy()

    # --------------------------------------------------
    # MediaPipe hand connections
    # --------------------------------------------------
    connections = [
        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        # Middle
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        # Ring
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        # Palm
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    # --------------------------------------------------
    # Plot helper
    # --------------------------------------------------
    def plot_hand(ax, points, title):
        ax.scatter(points[:, 0], points[:, 1], s=30)

        for i, j in connections:
            ax.plot(
                [points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], linewidth=1
            )

        # Label keypoints
        for i, (x, y) in enumerate(points):
            ax.text(x, y, str(i), fontsize=7)

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    plot_hand(axes[0], original, f"Original - Frame {frame_idx}")

    plot_hand(axes[1], bbox_01, "BBox [0, 1]")

    plot_hand(axes[2], bbox_center, "BBox Center + Scale")

    plot_hand(axes[3], wrist, "Wrist + Scale")

    plt.tight_layout()
    plt.show()


def normalize_body(body, left_shoulder=11, right_shoulder=12, eps=1e-6):
    """
    Normalize body keypoints.

    Args:
        body: np.ndarray, shape (T, 25, 2)

    Returns:
        normalized_body: np.ndarray, shape (T, 25, 2)

    Body indices:
        11 = left shoulder
        12 = right shoulder
    """

    body = np.asarray(body, dtype=np.float32)

    assert body.ndim == 3
    assert body.shape[1:] == (25, 2), f"Expected (T, 25, 2), got {body.shape}"

    # --------------------------------------------------
    # Shoulders
    # --------------------------------------------------
    ls = body[:, left_shoulder, :]  # (T, 2)
    rs = body[:, right_shoulder, :]  # (T, 2)

    # --------------------------------------------------
    # Center = midpoint between shoulders
    # --------------------------------------------------
    center = (ls + rs) / 2.0  # (T, 2)

    body = body - center[:, None, :]  # (T, 25, 2)

    # --------------------------------------------------
    # Scale = shoulder distance
    # --------------------------------------------------
    shoulder_dist = np.linalg.norm(ls - rs, axis=-1)  # (T,)

    # Sequence-level scale
    scale = np.median(shoulder_dist)

    scale = max(scale, eps)

    body = body / scale

    return body


hand = [
    [
        [772.7068, 1703.3921],
        [735.4481, 1671.1747],
        [696.3009, 1664.8661],
        [655.5599, 1666.8396],
        [618.5415, 1663.9404],
        [719.9688, 1745.0983],
        [682.7225, 1786.0150],
        [659.0396, 1811.0056],
        [639.9303, 1835.0105],
        [735.4855, 1781.5366],
        [683.0637, 1839.1013],
        [651.7462, 1872.3978],
        [626.9128, 1899.7186],
        [741.2290, 1802.9294],
        [691.5715, 1855.5927],
        [660.4941, 1885.9414],
        [635.1526, 1909.9105],
        [741.1803, 1812.2493],
        [702.7294, 1853.8060],
        [674.7792, 1880.0085],
        [651.5970, 1900.9788],
    ]
]

body = [
    [
        [927.6439476, 300.93548298],
        [954.37042236, 267.52262056],
        [974.60552216, 269.22638118],
        [993.79772186, 272.03644037],
        [901.20077133, 270.36647558],
        [883.11985016, 274.37948942],
        [867.4546051, 279.86516476],
        [1029.39079285, 304.1558075],
        [848.97542953, 316.32623434],
        [965.10807037, 350.72466373],
        [892.22082138, 354.0107131],
        [1135.17997742, 532.39970326],
        [746.90477371, 550.3910923],
        [1257.01572418, 792.47663498],
        [630.69580078, 830.53711653],
        [1093.93867493, 881.79110527],
        [802.93218613, 819.63303566],
        [1024.88113403, 935.15022039],
        [852.74831772, 863.30195189],
        [1022.84477234, 878.14694881],
        [881.93029404, 817.67467976],
        [1037.22724915, 858.52437973],
        [876.67888641, 804.91659164],
        [1059.05239105, 1037.34154701],
        [826.3410759, 1048.07034016],
    ]
]


def visualize_body_before_after(body, frame_idx=0, show_index=True):
    """
    body: (T, 25, 2)
    """

    BODY_CONNECTIONS = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 17),
        (15, 19),
        (15, 21),
        (16, 18),
        (16, 20),
        (16, 22),
        (11, 23),
        (12, 24),
        (23, 24),
    ]

    if isinstance(body, torch.Tensor):
        body = body.detach().cpu().numpy()

    body = np.asarray(body)

    assert body.shape[1:] == (25, 2)

    normalized_body = normalize_body(body, left_shoulder=11, right_shoulder=12)

    original = body[frame_idx]
    normalized = normalized_body[frame_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, pts, title in [
        (axes[0], original, "Before normalization"),
        (axes[1], normalized, "After normalization"),
    ]:
        # Skeleton
        for a, b in BODY_CONNECTIONS:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], linewidth=2)

        # Keypoints
        ax.scatter(pts[:, 0], pts[:, 1], s=45)

        # Index
        if show_index:
            for i, (x, y) in enumerate(pts):
                ax.text(x + 0.01, y + 0.01, str(i), fontsize=9)

        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # visualize_hand_normalizations(hand=np.array(hand))
    body = np.array(body)
    print(body.shape)
    visualize_body_before_after(body=body)
