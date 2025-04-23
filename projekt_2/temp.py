import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

from window_functions import WindowFunction


class SignalProcessor:
    def __init__(self, data, sample_rate, start, end, frame_ms, hop_size=None):
        self.data = data
        self.sample_rate = sample_rate
        self.start = start
        self.end = end
        self.frame_ms = frame_ms
        self.frame_size = int(frame_ms * sample_rate / 1000)
        self.hop_size = hop_size if hop_size else self.frame_size // 2
        self.audio = data[start:end]
    
    def split_into_frames(self):
        frames = []
        for start in range(0, len(self.audio) - self.frame_size, self.hop_size):
            end = start + self.frame_size
            frame = self.audio[start:end]
            frames.append(frame)
        return frames
    
    def apply_window(self, window_type):
        frames = self.split_into_frames()
        window = self._get_window(window_type, self.frame_size)
        windowed_frames = [frame * window for frame in frames]
        return windowed_frames
    
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
        elif window_type == 'triangular':
            return WindowFunction.triangular(N)
        else:
            raise ValueError("Not valid window type! Choose: rectangular, triangular, hamming, hann, triangular, blackman")
    
    # sum window
    def sum_window(self, window_type):
        window = self._get_window(window_type, self.frame_size)
        window_sum = np.sum(window)
        return window_sum
        

class SpectrogramGenerator:
    def __init__(self, signal_processor):
        self.signal_processor = signal_processor

    def generate(self, window_type='rectangular'):
        windowed_frames = self.signal_processor.apply_window(window_type)
        spectrogram = []
        for frame in windowed_frames:
            ft = fft(frame, n=self.signal_processor.frame_size)
            magnitude = np.abs(ft)[:self.signal_processor.frame_size // 2]
            spectrogram.append(magnitude)
        self.spectrogram = np.array(spectrogram).T
        self.frequencies = np.fft.fftfreq(self.signal_processor.frame_size, 1 / self.signal_processor.sample_rate)[:self.signal_processor.frame_size // 2]
        self.times = np.arange(len(windowed_frames)) * (self.signal_processor.hop_size / self.signal_processor.sample_rate)
        return self.spectrogram, self.frequencies, self.times
    
    def plot_spectrogram(self, db_scale=True):
        if not hasattr(self, 'spectrogram'):
            raise ValueError("Spectrogram data not generated. Call generate() first.")
        
        spectrogram_to_plot = self.spectrogram
        ref_spectrum = np.max(spectrogram_to_plot)
        spectrogram_db = 10 * np.log10(spectrogram_to_plot + 1e-10)
  
        fig, ax = plt.subplots()
        if db_scale:
            
            c = ax.pcolormesh(self.times, self.frequencies, spectrogram_db, cmap='viridis', shading='auto')
            ax.set_title('Spectrogram (dB)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            fig.colorbar(c, ax=ax, label='Magnitude (dB)')
        else:
         
            c = ax.pcolormesh(self.times, self.frequencies, spectrogram_to_plot, cmap='viridis', shading='auto')
            ax.set_title('Spectrogram')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            fig.colorbar(c, ax=ax, label='Magnitude')
        
        return fig
        