import numpy as np
from typing import Tuple


class MotorImageryDataset:
    """Handles loading and accessing EEG motor imagery data from C3 and C4 channels."""

    def __init__(self, filepath: str,train_mode=True,fs=None):
        """Initializes the dataset and loads data from the given file path."""
        self.filepath = filepath
        self.train_mode = train_mode
        self._load_data(train_mode=self.train_mode)
        self.full_names = ['C3-Left', 'C3-Right', 'C4-Left', 'C4-Right']
        self.channel_names = {0: 'C3', 1: 'C4'}
        self.fs = 128

    def _load_data(self,train_mode=True):
        """Loads EEG signals, labels, and trial metadata from the .npy file."""
        with open(self.filepath, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            self.data = data[:, :, :2]  # Only C3, C4 channels
            if train_mode:
                self.labels = np.load(f, allow_pickle=True)
                self.left_flags = self.labels[2]
                self.right_flags = self.labels[3]

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
            'C3': crop(self.c3_data),
            'C4': crop(self.c4_data),
            'fs': self.fs,
        }
    
    def reduce_baseline(self,baseline: np.ndarray,train_mode=True) -> np.ndarray:
        """Computes the mean mV for each channel in the baseline and returns the reduced baseline.
        Args: baseline: int - time (seconds)"""

        t = baseline * self.fs
        if t < 0 or t >= self.data.shape[1]:
            raise ValueError(f"Baseline time must be within the data range: 0 to {self.data.shape[1] / self.fs} seconds")
        baseline_data_c3 = self.data[:, :int(t),0]
        baseline_data_c4 = self.data[:, :int(t),1]
        baseline_c3 = np.mean(baseline_data_c3, axis=1)
        baseline_c4 = np.mean(baseline_data_c4, axis=1)
        self.data[:, :, 0] -= baseline_c3[:, np.newaxis]
        self.data[:, :, 1] -= baseline_c4[:, np.newaxis]
        
        if train_mode:
            self.data_left = self.data[self.left_indices]
            self.data_right = self.data[self.right_indices]

            self.c3_left = self.data_left[:, :, 0]
            self.c3_right = self.data_right[:, :, 0]
            self.c4_left = self.data_left[:, :, 1]
            self.c4_right = self.data_right[:, :, 1]

        self.c3_data = self.data[:, :, 0]
        self.c4_data = self.data[:, :, 1]
        
    def __repr__(self):
        """Returns a summary string representation of the dataset."""
        return (f"MotorImageryDataset(\n"
                f"  filepath={self.filepath},\n"
                f"  fs={self.fs} Hz,\n"
                f"  left_trials={len(self.left_indices)}, right_trials={len(self.right_indices)}\n"
                f")")
