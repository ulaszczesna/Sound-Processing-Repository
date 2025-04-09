import numpy as np
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go

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
        
    

class SpectogramGenerator:
    def __init__(self, signal_processor):
        self.signal_processor = signal_processor
    
    def generate(self, start_sample, end_sample, frame_length, hop_length, window_type='hann'):
        signal = self.signal_processor.data
        sample_rate = self.signal_processor.sample_rate
        fragment = signal[start_sample:end_sample]
        if len(fragment) < frame_length:
            raise ValueError("Fragment length is less than frame length.")
        n_samples = len(fragment)
        n_frames = (n_samples - frame_length) // hop_length + 1
        spectogram = []

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end > n_frames:
                end = n_samples
            if end - start < frame_length:
                break

            
            windowed_frame = self.signal_processor.apply_window(window_type, frame_start=start, frame_end=end)[0]
            fft_result = fft(windowed_frame)
            magnitude_spectrum = np.abs(fft_result[:frame_length // 2 + 1])
            spectogram.append(magnitude_spectrum)
        
        self.spectogram_data = np.array(spectogram).T
        self.frequencies = fftfreq(frame_length, 1 / sample_rate)[:frame_length // 2 + 1]
        self.times = np.arange(n_frames) * hop_length / sample_rate
        
        return self.spectogram_data, self.frequencies, self.times
    
    def plot_spectrogram(self, db_scale=True):
        if not hasattr(self, 'spectogram_data'):
            raise ValueError("Spectrogram data not generated. Call generate() first.")
        
        spectrogram_to_plot = self.spectogram_data
        if db_scale:
            spectrogram_db = 10 * np.log10(spectrogram_to_plot + 1e-10)
            data = go.Heatmap(
                z=spectrogram_db,
                x=self.times,
                y=self.frequencies,
                colorscale='Viridis',
                colorbar=dict(title='Magnitude (dB)'),
            )
        else:
            data = go.Heatmap(
                z=spectrogram_to_plot,
                x=self.times,
                y=self.frequencies,
                colorscale='Viridis',
                colorbar=dict(title='Magnitude'),
            )
        layout = go.Layout(
            title='Spectrogram',
            xaxis=dict(title='Time (s)'),
            yaxis=dict(title='Frequency (Hz)')
        )
        fig = go.Figure(data=[data], layout=layout)
        return fig