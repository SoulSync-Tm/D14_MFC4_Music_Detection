import numpy as np


class RingBuffer:
    """A ring buffer to hold audio samples and yield fixed-size windows."""

    def __init__(self, sample_rate: int = 8000, duration: float = 2):
        self.size = int(sample_rate * duration)
        self.buffer = np.zeros(self.size, dtype=np.float32)
        self.index = 0
        self.is_full = False

    def append(self, samples: np.ndarray) -> None:
        """Append new samples to the buffer, overwriting old data if necessary."""
        for sample in samples:
            self.buffer[self.index] = sample
            self.index = (self.index + 1) % self.size
            if self.index == 0:
                self.is_full = True

    def get_window(self) -> np.ndarray:
        """Return the current window of samples, ordered from oldest to newest."""
        if not self.is_full:
            return self.buffer[: self.index].copy()
        return np.concatenate((self.buffer[self.index :], self.buffer[: self.index]))
