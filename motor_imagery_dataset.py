import numpy as np
from typing import Tuple


class MotorImageryDataset:
    """Handles loading and accessing EEG motor imagery data from C3 and C4 channels."""

    def __init__(self, filepath: str):
        """Initializes the dataset and loads data from the given file path."""
        self.filepath = filepath
        self._load_data()
        self.full_names = ['C3-Left', 'C3-Right', 'C4-Left', 'C4-Right']
        self.channel_names = {0: 'C3', 1: 'C4'}

    def _load_data(self):
        """Loads EEG signals, labels, and trial metadata from the .npy file."""
        with open(self.filepath, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            self.data = data[:, :, :2]  # Only C3, C4 channels

            labels = np.load(f, allow_pickle=True)
            self.left_flags = labels[2]
            self.right_flags = labels[3]

            self.left_indices = np.where(self.left_flags)[0]
            self.right_indices = np.where(self.right_flags)[0]

            self.data_left = self.data[self.left_indices]
            self.data_right = self.data[self.right_indices]

            self.labels_name = np.load(f, allow_pickle=True)
            fs = np.load(f, allow_pickle=True)
            self.fs = fs.item() if isinstance(fs, np.ndarray) else fs

            # Extract channels: 0 - C3, 1 - C4
            self.c3_left = self.data_left[:, :, 0]
            self.c3_right = self.data_right[:, :, 0]
            self.c4_left = self.data_left[:, :, 1]
            self.c4_right = self.data_right[:, :, 1]

            self.c3_data = self.data[:, :, 0]
            self.c4_data = self.data[:, :, 1]

    def get_all(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Returns all raw data arrays, labels, and sampling frequency."""
        return (self.data_left, self.data_right,
                self.c3_left, self.c3_right,
                self.c4_left, self.c4_right,
                self.labels_name, self.fs)

    def get_windowed_data(self, t_min: float, t_max: float) -> dict:
        """Returns a dictionary of data cropped to the specified time window for each channel."""
        if not (0 <= t_min < t_max):
            raise ValueError("t_min must be >= 0 and less than t_max")

        start_idx = int(t_min * self.fs)
        end_idx = int(t_max * self.fs)

        def crop(data: np.ndarray) -> np.ndarray:
            return data[:, start_idx:end_idx]

        return {
            'C3-Left': crop(self.c3_left),
            'C3-Right': crop(self.c3_right),
            'C4-Left': crop(self.c4_left),
            'C4-Right': crop(self.c4_right),
            'fs': self.fs,
            'labels_name': self.labels_name,
        }

    def __repr__(self):
        """Returns a summary string representation of the dataset."""
        return (f"MotorImageryDataset(\n"
                f"  filepath={self.filepath},\n"
                f"  fs={self.fs} Hz,\n"
                f"  left_trials={len(self.left_indices)}, right_trials={len(self.right_indices)}\n"
                f")")
