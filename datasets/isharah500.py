"""Dataset adapter for the iSharah500 benchmark."""

from pathlib import Path
from typing import Any, Dict
import warnings

import cv2
import pickle
import numpy as np
import pandas as pd

from .base_dataset import BaseSignLanguageDataset

class ISharah500Dataset(BaseSignLanguageDataset):
    """iSharah500 dataset built on :class:`BaseSignLanguageDataset`."""

    def __init__(self, args: Any, phase: str):
        super().__init__(args=args, phase=phase)

    def _load_pose(self, path: Path) -> Dict[str, np.ndarray]:
        _key = "keypoints"

        with path.open("rb") as file:
            pose_data = pickle.load(file)

        pose_data = pose_data[_key]

        pose = {
            'right': pose_data[:, 0:21, :],
            'left': pose_data[:, 21:42, :],
            'face': pose_data[:, 42:61, :],
            'body': pose_data[:, 61:, :]
        }

        return pose


    def _load_rgb(self, path: Path) -> Any:
        """Examples
        --------
        Given the directory structure::

            path/
            |-- left/
            |   |-- frame0001.jpg
            |   `-- frame0002.jpg
            `-- right/
                `-- frame0001.jpg

        Load both hands with::

            rgb = dataset._load_rgb(Path("sample_dir"))
            left_frames = rgb["left"]
            right_frames = rgb["right"]
        """

        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"RGB sample directory does not exist: {path}. "
                "If you do not want to use RGB data, set use_rgb=False."
            )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        rgb = {}

        for hand in ("left", "right"):
            hand_path = path / hand
            if not hand_path.is_dir():
                rgb[f'rgb_{hand}'] = []
                continue

            frames = []
            frame_paths = sorted(
                frame_path
                for frame_path in hand_path.iterdir()
                if frame_path.is_file() and frame_path.suffix.lower() in image_extensions
            )

            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Unable to read RGB frame: {frame_path}")
                frames.append(frame)

            rgb[f'rgb_{hand}'] = self.resize_images(frames)

        return rgb