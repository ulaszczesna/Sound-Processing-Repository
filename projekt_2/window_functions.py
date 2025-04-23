import numpy as np
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    
    @staticmethod
    def triangular(N):
        return 1 - np.abs((np.arange(N) - (N - 1) / 2) / ((N - 1) / 2))



# class SignalProcessor:
#     def __init__(self, data, sample_rate):
#         self.data = data
#         self.sample_rate = sample_rate

#     def apply_window(self, window_type, frame=None, frame_start=0, frame_end=None):
#         if frame is None:
#             if frame_end is None:
#                 frame = self.data[frame_start:]
#             else:
#                 frame = self.data[frame_start:frame_end]

#         N = len(frame)

#         window = self._get_window(window_type, N)

#         windowed_signal = frame * window
      
#         return windowed_signal, frame

#     def _get_window(self, window_type, N):
#         window_type = window_type.lower()
#         if window_type == 'rectangular':
#             return WindowFunction.rectangular(N)
#         elif window_type == 'hamming':
#             return WindowFunction.hamming(N)
#         elif window_type == 'hann':
#             return WindowFunction.hann(N)
#         elif window_type == 'triangular':
#             return WindowFunction.triangular(N)
#         elif window_type == 'blackman':
#             return WindowFunction.blackman(N)
#         else:
#             raise ValueError("Not valid window type! Choose: rectangular, triangular, hamming, hann")
        
    

# class SpectogramGenerator:
#     def __init__(self, signal_processor):
#         self.signal_processor = signal_processor

#     def split_into_frames_with_hop(self, fragment, frame_length, hop_length):
        
#         if len(fragment) < frame_length:
#             raise ValueError("Fragment length is less than frame length.")
#         n_samples = len(fragment)
#         n_frames = (n_samples - frame_length) // hop_length + 1
#         frames = []

#         for i in range(n_frames):
#             start = i * hop_length
#             end = start + frame_length
#             if end > n_samples:
#                 end = n_samples
#             if end - start < frame_length:
#                 break

#             frames.append(fragment[start:end])
        
#         return np.array(frames)    
    
#     def generate(self, start_sample, end_sample, frame_length, hop_length, window_type='hann'):
#         signal = self.signal_processor.data
#         sample_rate = self.signal_processor.sample_rate
#         fragment = signal[start_sample:end_sample]
#         if len(fragment) < frame_length:
#             raise ValueError("Fragment length is less than frame length.")

#         spectogram = []
#         frames = self.split_into_frames_with_hop(fragment, frame_length, hop_length)
#         for frame in frames:
#             windowed_frame, _ = self.signal_processor.apply_window(window_type, frame=frame)
#             ft = fft(windowed_frame)
#             magnitude_spectrum = np.abs(ft)[:len(ft) // 2]
#             spectogram.append(magnitude_spectrum)

        
#         self.spectogram_data = np.array(spectogram).T
#         self.frequencies = fftfreq(frame_length, 1 / sample_rate)[:frame_length // 2 ]
#         self.times = np.arange(len(frames)) * hop_length / sample_rate
        
#         return self.spectogram_data, self.frequencies, self.times
    
#     def plot_spectrogram(self, db_scale=True):
#         if not hasattr(self, 'spectogram_data'):
#             raise ValueError("Spectrogram data not generated. Call generate() first.")
        
#         spectrogram_to_plot = self.spectogram_data
#         ref_spectrum = np.max(spectrogram_to_plot)
#         spectrogram_db = 10 * np.log10(spectrogram_to_plot + 1e-10)
  
#         fig, ax = plt.subplots()
#         if db_scale:
            
#             c = ax.pcolormesh(self.times, self.frequencies, spectrogram_db, cmap='viridis', shading='auto')
#             ax.set_title('Spectrogram (dB)')
#             ax.set_xlabel('Time (s)')
#             ax.set_ylabel('Frequency (Hz)')
#             fig.colorbar(c, ax=ax, label='Magnitude (dB)')
#         else:
         
#             c = ax.pcolormesh(self.times, self.frequencies, spectrogram_to_plot, cmap='viridis', shading='auto')
#             ax.set_title('Spectrogram')
#             ax.set_xlabel('Time (s)')
#             ax.set_ylabel('Frequency (Hz)')
#             fig.colorbar(c, ax=ax, label='Magnitude')
        
#         return fig