import librosa
import numpy as np


class AudioFingerprinter:

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def preprocess(self, file_path):
        """
        Load audio and standardize it.
        Returns:
            y  -> normalized mono waveform
            sr -> fixed sample rate
        """

        y, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            mono=True
        )


        y = librosa.util.normalize(y)

        y = y - np.mean(y)

        return y, sr