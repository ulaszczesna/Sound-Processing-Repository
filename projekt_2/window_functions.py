import numpy as np
from scipy.fft import fft, fftfreq

class WindowFunction:
    @staticmethod
    def rectangular(N):
        return np.ones(N)

    @staticmethod
    def hamming(N):
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

    @staticmethod
    def hann(N):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

    @staticmethod
    def triangular(N):
        return 1 - np.abs((np.arange(N) - (N - 1) / 2) / ((N - 1) / 2))
    
    @staticmethod
    def blackman(N):
        return 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)) + 0.08 * np.cos(4 * np.pi * np.arange(N) / (N - 1))


class SignalProcessor:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate

    def apply_window(self, window_type, frame_start=0, frame_end=None):
        if frame_end is None:
            frame = self.data[frame_start:]
        else:
            frame = self.data[frame_start:frame_end]

        N = len(frame)

        window = self._get_window(window_type, N)

        windowed_signal = frame * window
        return windowed_signal, frame

    def _get_window(self, window_type, N):
        window_type = window_type.lower()
        if window_type == 'rectangular':
            return WindowFunction.rectangular(N)
        elif window_type == 'hamming':
            return WindowFunction.hamming(N)
        elif window_type == 'hann':
            return WindowFunction.hann(N)
        elif window_type == 'triangular':
            return WindowFunction.triangular(N)
        elif window_type == 'blackman':
            return WindowFunction.blackman(N)
        else:
            raise ValueError("Not valid window type! Choose: rectangular, triangular, hamming, hann")
        
    

