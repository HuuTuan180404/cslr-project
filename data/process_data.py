import torch
import pickle
import normalize
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


INPUT_DIR = Path("data/isharah500/pose")
OUTPUT_DIR = Path("data/isharah500/normalized_pose")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hand topology
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

# Expected upper-body subset, if present in the old PKL body section
UPPER_BODY_IDXS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
]

UPPER_BODY_CONNECTIONS_ORIG = [
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


def build_local_connections(selected_indices, original_connections):
    idx_map = {
        orig_idx: local_idx for local_idx, orig_idx in enumerate(selected_indices)
    }
    local_connections = []
    for a, b in original_connections:
        if a in idx_map and b in idx_map:
            local_connections.append((idx_map[a], idx_map[b]))
    return local_connections


BODY_CONNECTIONS = build_local_connections(UPPER_BODY_IDXS, UPPER_BODY_CONNECTIONS_ORIG)


def draw_connections(ax, pts, connections, color, linewidth=2):
    for a, b in connections:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            ax.plot(
                [pts[a, 0], pts[b, 0]],
                [pts[a, 1], pts[b, 1]],
                color=color,
                linewidth=linewidth,
            )


def render_frame_matplotlib(frame_xy, title=None):
    rh = frame_xy[0:21]
    lh = frame_xy[21:42]
    lip = frame_xy[42:61]
    bd = frame_xy[61:]

    fig, ax = plt.subplots(figsize=(8, 8))

    body_has_connections = len(bd) == len(UPPER_BODY_IDXS)
    if body_has_connections:
        draw_connections(ax, bd, BODY_CONNECTIONS, color="limegreen", linewidth=2)

    draw_connections(ax, lh, HAND_CONNECTIONS, color="blue", linewidth=2)
    draw_connections(ax, rh, HAND_CONNECTIONS, color="red", linewidth=2)

    if len(bd) > 0:
        ax.scatter(bd[:, 0], bd[:, 1], c="limegreen", s=35, label="Body")
    ax.scatter(lh[:, 0], lh[:, 1], c="blue", s=20, label="Left hand")
    ax.scatter(rh[:, 0], rh[:, 1], c="red", s=20, label="Right hand")
    ax.scatter(lip[:, 0], lip[:, 1], c="#c8b400", s=16, label="Lips")

    ax.invert_yaxis()  # match image coordinate system
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    if title is not None:
        ax.set_title(title)
    plt.show()


def process_samples():
    samples = [p.name for p in INPUT_DIR.iterdir() if p.is_file()]
    for sample_file in tqdm(samples, desc="Processing"):
        sample_path = INPUT_DIR / sample_file
        with open(sample_path, "rb") as f:
            data = pickle.load(f)
        data = np.array(data["keypoints"])

        right_hand_seq = data[:, 0:21, :]
        left_hand_seq = data[:, 21:42, :]
        lips_seq = data[:, 42:61, :]
        body_seq = data[:, 61:, :]

        normalized_right_hand_seq = normalize.normalize_hand_wrist(
            torch.from_numpy(right_hand_seq)
        )
        normalized_left_hand_seq = normalize.normalize_hand_wrist(
            torch.from_numpy(left_hand_seq)
        )

        normalized_lips_seq = normalize.normalize_lips(torch.from_numpy(lips_seq))
        normalized_body_seq = torch.from_numpy(normalize.normalize_body(body_seq))

        full_body = (
            torch.cat(
                [
                    normalized_right_hand_seq,
                    normalized_left_hand_seq,
                    normalized_lips_seq,
                    normalized_body_seq,
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        data = {"keypoints": full_body}

        with open(OUTPUT_DIR / sample_file, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    process_samples()
