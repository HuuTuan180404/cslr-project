# datasets/base_dataset.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import cv2
import warnings

import pandas as pd
from torch.utils.data import Dataset


class BaseSignLanguageDataset(Dataset, ABC):
    """
    Base Dataset cho các bài toán Sign Language Recognition.

    Dataset hierarchy:

        Benchmark
            └── Split
                  └── Sample
                        ├── Pose
                        └── RGB

    Ví dụ:

        data/
        ├── WLASL100/
        │   ├── train/
        │   ├── dev/
        │   └── test/
        │
        └── LSA64/
            ├── train/
            ├── dev/
            └── test/
    """

    SUPPORTED_PHASES = {"train", "dev", "test"}

    def __init__(self, args, phase: str):
        """
        Args:
            root:
                Root directory chứa data.

                Ví dụ:
                    ./data

            benchmark:
                Tên benchmark.

                Ví dụ:
                    WLASL100
                    LSA64
                    CSL_Daily

            split:
                train / dev / test

            metadata_root:
                Root directory chứa metadata.

                Nếu None:
                    metadata sẽ nằm trong ./metadata
        """

        super().__init__()
        self.args = args
        self.phase = phase

        self.data_root = Path(args.dataset.root)
        self.benchmark = args.dataset.benchmark
        self.metadata_dir = Path(args.dataset.metadata_root)
        self.use_rgb = args.dataset.use_rgb

        self._validate()

        self.pose_transform = args.pose.transform
        self.pose_normalize = args.pose.normalize
        self.pose_augmentation = args.pose.augmentation
        self.pose_extention = args.pose.format

        self.rgb_transform = args.rgb.transform
        self.rgb_normalize = args.rgb.normalize
        self.rgb_augmentation = args.rgb.augmentation
        self.rgb_extention = args.rgb.format
        self.image_size = args.rgb.image_size

        # Metadata sẽ được load bởi class con
        self.metadata = self._load_metadata()

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate(self):
        """Kiểm tra cấu hình Dataset."""

        if self.phase not in self.SUPPORTED_PHASES:
            raise ValueError(
                f"Unsupported split: '{self.phase}'. "
                f"Expected one of {self.SUPPORTED_PHASES}"
            )

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Benchmark directory does not exist: " f"{self.data_root}"
            )


    # ==========================================================
    # Metadata
    # ==========================================================

    def _load_metadata(self) -> pd.DataFrame:
        """
        Load metadata của benchmark.

        Mặc định:

            metadata/
            └── WLASL100/
                ├── train.csv
                ├── dev.csv
                └── test.csv
        """

        metadata_path = self.metadata_dir / f"{self.phase}.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

        metadata = pd.read_csv(metadata_path)

        if len(metadata) == 0:
            raise ValueError(f"Metadata is empty: {metadata_path}")

        if "sample_id" not in metadata.columns:
            raise ValueError(
                f"Metadata must contain 'sample_id' column. "
                f"Found: {list(metadata.columns)}"
            )

        return metadata
    # ==========================================================
    # Dataset interface
    # ==========================================================
    def __len__(self) -> int:
        """Number of samples."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load một sample.

        Returns:
            Dictionary, ví dụ:

            {
                "id": "sample_000001",
                "gloss": [...],
                "text": "...",
                "use_rgb": True/False,

                'right': <(T, J, C)>,
                'left': <(T, J, C)>,
                'face': <(T, J, C)>,
                'body': <(T, J, C)>

                "rgb_left": <(T, H, W, 3)> / None,
                "rgb_right": <(T, H, W, 3)> / None

                "length": 120, ?
            }
        """
        row = self.metadata.iloc[index]

        sample_id = self.get_sample_id(row.name)

        sample: Dict[str, Any] = {
            "id": sample_id,
            "gloss": self.get_optional_value(row, "gloss", default=[]),
            "text": self.get_optional_value(row, "text", default=""),
            "use_rgb": self.use_rgb,
        }

        pose_path = self.get_pose_path(sample_id)
        if pose_path is not None:
            sample = sample | self._load_pose(pose_path)

        if self.use_rgb:
            rgb_path = self.get_rgb_path(sample_id)
            return sample | self._load_rgb(rgb_path)

        return sample

    # ==========================================================
    # Sample loading
    # ==========================================================
    @abstractmethod
    def _load_rgb(self, path: Path) -> Dict[str, np.ndarray]:
        """
        Returns:
            Dictionary, ví dụ:

            {
                "rgb_left": <(T, H, W, 3)>,
                "rgb_right": <(T, H, W, 3)>
            }
        """
        raise NotImplementedError


    @abstractmethod
    def _load_pose(self, path: Path) -> Dict[str, np.ndarray]:
        """
        Returns:
            Dictionary, ví dụ:
            {
                'right': <(T, J, C)>,
                'left': <(T, J, C)>,
                'face': <(T, J, C)>,
                'body': <(T, J, C)>
            }
        """
        raise NotImplementedError


    # ==========================================================
    # Path helpers
    # ==========================================================
    def get_pose_path(self, sample_id: str) -> Path:
        """
        Trả về path tới pose của sample.

        Ví dụ:

            data/WLASL100/pose/sample_000001.npz
        """

        path = self.data_root / "pose" / f"{sample_id}.{self.pose_extention}"

        if not path.exists():
            warnings.warn(
                f"Sample '{sample_id}' does not exist: {path}",
                UserWarning
            )
            return None
        return path


    def get_rgb_path(self, sample_id: str) -> Path:
        """
        Trả về root directory của RGB sample.

        Ví dụ:

            data/WLASL100/rgb/sample_000001/
        """

        path = self.data_root / "rgb" / sample_id

        return path


    def get_rgb_hand_path(self, sample_id: str, hand: str) -> Path:
        """
        Trả về directory RGB của một bàn tay.

        Args:
            sample_id:
                sample_000001

            hand:
                left / right

        Returns:

            data/WLASL100/rgb/
                sample_000001/
                    left/
                        image1
                        image2
                        ...
                    right/
                        ...

        """

        if hand not in {"left", "right"}:
            raise ValueError(f"Invalid hand: {hand}. " f"Expected 'left' or 'right'.")

        return self.get_rgb_path(sample_id) / hand


    # ==========================================================
    # Metadata helpers
    # ==========================================================

    def get_sample_id(self, index: int) -> str:
        """Lấy sample_id theo index."""

        return str(self.metadata.iloc[index]["sample_id"])


    def get_metadata(self, index: int) -> pd.Series:
        """Lấy metadata của một sample."""

        return self.metadata.iloc[index]


    def resize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Resize a sequence of images.

        Parameters
        ----------
        frames : np.ndarray
            Shape: (T, H, W, C)
        size : tuple[int, int]
            Target size: (width, height)

        Returns
        -------
        np.ndarray
            Shape: (T, target_H, target_W, C)
        """
        size = (self.image_size, self.image_size)

        resized_frames = np.stack(
            [cv2.resize(image, size) for image in images],
            axis=0,
        )

        return resized_frames

    # ==========================================================
    # Debug
    # ==========================================================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"benchmark='{self.benchmark}', "
            f"split='{self.phase}', "
            f"samples={len(self)}"
            f")"
        )

    @staticmethod
    def get_optional_value(row: pd.Series, column: str, default: Any = None) -> Any:
        """Return a metadata value without leaking pandas NaN values."""

        if column not in row.index or pd.isna(row[column]):
            return default
        return row[column]