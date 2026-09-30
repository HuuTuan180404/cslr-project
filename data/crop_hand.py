import cv2
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = Path("data/isharah500")

INPUT_DIR = DATASET_NAME / "02"
OUTPUT_DIR = DATASET_NAME / "02_cropped"

IMAGE_SIZE = (112, 112)

PADDING = 0.10

# Smoothing factor
ALPHA = 0.7

# Placeholder color (BGR)
PLACEHOLDER_COLOR = (128, 128, 128)

# MediaPipe
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

IS_MIRRORED = True  # Set to True if the input images are mirrored (e.g., from a webcam)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands


# ============================================================
# UTILS
# ============================================================


def get_bbox_from_landmarks(hand_landmarks, image_width, image_height):
    xs = [landmark.x * image_width for landmark in hand_landmarks.landmark]

    ys = [landmark.y * image_height for landmark in hand_landmarks.landmark]

    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32,
    )


def add_padding(bbox, image_width, image_height, padding=0.10):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    pad_x = w * padding
    pad_y = h * padding

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(image_width, x2))
    y2 = int(min(image_height, y2))

    return x1, y1, x2, y2


def smooth_bbox(current_bbox, previous_bbox, alpha=0.7):
    if previous_bbox is None:
        return current_bbox

    return alpha * current_bbox + (1.0 - alpha) * previous_bbox


def crop_bbox(image, bbox):
    x1, y1, x2, y2 = bbox

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def create_placeholder():
    """
    Create 112x112 neutral gray image.
    """

    width, height = IMAGE_SIZE

    return np.full(
        (height, width, 3),
        PLACEHOLDER_COLOR,
        dtype=np.uint8,
    )


def save_image(image, save_path):
    """
    Resize and save image.
    """

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(save_path), image)


# ============================================================
# MAIN
# ============================================================


def process_frames():

    sample_ids = sorted(
        [p.name for p in INPUT_DIR.iterdir() if p.is_dir()],
        key=lambda x: tuple(map(int, x.split("_"))),
    )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        for sample_idx, sample_id in enumerate(tqdm(sample_ids)):
            # for sample_idx, sample_id in enumerate(sample_ids):
            input_dir = INPUT_DIR / sample_id
            output_dir = OUTPUT_DIR / sample_id

            left_dir = output_dir / "left"
            right_dir = output_dir / "right"

            left_dir.mkdir(parents=True, exist_ok=True)

            right_dir.mkdir(parents=True, exist_ok=True)

            frame_paths = sorted(
                [
                    p
                    for p in input_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )

            if not frame_paths:
                print(f"No frames found: {input_dir}")
                continue

            # ------------------------------------------------
            # Reset for every sample
            # ------------------------------------------------

            previous_bbox = {"left": None, "right": None}

            # ------------------------------------------------
            # Process frames
            # ------------------------------------------------

            for frame_idx, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))

                if frame is None:
                    print(f"Cannot read: {frame_path}")
                    continue

                image_height, image_width = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb)

                # Track which hand was detected
                detected_hands = {"left": False, "right": False}

                # ====================================================
                # DETECTED HANDS
                # ====================================================

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        hand_label = handedness.classification[0].label.lower()
                        # Swap because input images are NOT mirrored
                        if IS_MIRRORED:
                            hand_label = "right" if hand_label == "left" else "left"

                        detected_hands[hand_label] = True

                        # --------------------------------------------
                        # BBox
                        # --------------------------------------------

                        bbox = get_bbox_from_landmarks(
                            hand_landmarks,
                            image_width,
                            image_height,
                        )

                        # --------------------------------------------
                        # Smoothing
                        # --------------------------------------------

                        bbox = smooth_bbox(bbox, previous_bbox[hand_label], ALPHA)

                        previous_bbox[hand_label] = bbox.copy()

                        # --------------------------------------------
                        # Padding
                        # --------------------------------------------

                        crop_box = add_padding(bbox, image_width, image_height, PADDING)

                        # --------------------------------------------
                        # Crop
                        # --------------------------------------------

                        crop = crop_bbox(frame, crop_box)

                        if crop is None:
                            continue

                        # --------------------------------------------
                        # Save
                        # --------------------------------------------
                        if hand_label == "left":
                            save_path = left_dir / f"{frame_path.stem}.jpg"
                        else:
                            save_path = right_dir / f"{frame_path.stem}.jpg"

                        save_image(crop, save_path)

                # ====================================================
                # MISSING HANDS
                # ====================================================
                for hand_label in ["left", "right"]:
                    if detected_hands[hand_label]:
                        continue

                    # ------------------------------------------------
                    # Previous bbox exists
                    # ------------------------------------------------
                    if previous_bbox[hand_label] is not None:
                        crop_box = add_padding(
                            previous_bbox[hand_label],
                            image_width,
                            image_height,
                            PADDING,
                        )

                        crop = crop_bbox(frame, crop_box)

                        if crop is not None:
                            if hand_label == "left":
                                save_path = left_dir / f"{frame_path.stem}.jpg"
                            else:
                                save_path = right_dir / f"{frame_path.stem}.jpg"

                            save_image(crop, save_path)

                    # ------------------------------------------------
                    # No previous bbox
                    # ------------------------------------------------

                    else:
                        placeholder = create_placeholder()

                        if hand_label == "left":
                            save_path = left_dir / f"{frame_path.stem}.jpg"
                        else:
                            save_path = right_dir / f"{frame_path.stem}.jpg"

                        cv2.imwrite(str(save_path), placeholder)

            if len(frame_paths) != len(list(left_dir.iterdir())):
                print(
                    f"Warning: Left hand frames count mismatch: "
                    f"{len(frame_paths)} vs {len(list(left_dir.iterdir()))}"
                )
                return

            if len(frame_paths) != len(list(right_dir.iterdir())):
                print(
                    f"Warning: Right hand frames count mismatch: "
                    f"{len(frame_paths)} vs {len(list(right_dir.iterdir()))}"
                )
                return


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    process_frames()
