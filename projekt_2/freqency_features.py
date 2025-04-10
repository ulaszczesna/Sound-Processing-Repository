import numpy as np
from scipy.stats import gmean
import plotly.graph_objects as go

class FreqencyDomainFeatures:
    def __init__(self, data, rate, start=0, end=None, frame_ms=20):
        if end is None:
            end = len(data)
        self.data = data
        self.rate = rate
        self.start = start
        self.end = end
        self.frame_ms = frame_ms
        self.frame_size = int(frame_ms * rate / 1000)
        self.frames = self._split_into_frames(data[self.start:self.end])
        self.spectrum = np.array([np.abs(np.fft.fft(frame))[:self.frame_size // 2 + 1] for frame in self.frames])
        self.frequencies = np.fft.fftfreq(self.frame_size, d=1/rate)[:self.frame_size // 2 + 1]

    def _split_into_frames(self, data):
        num_frames = len(data) // self.frame_size
        frames = np.array([data[i * self.frame_size: (i + 1) * self.frame_size] for i in range(num_frames)])
        return frames
    
    def volume(self):
        return np.array([np.sum(np.abs(frame_spectrum) ** 2) / self.frame_size for frame_spectrum in self.spectrum])
    
    def frequency_centroid(self):
        fc = []
        for frame_spectrum in self.spectrum:
            numerator = np.sum(self.frequencies * frame_spectrum)
            denominator = np.sum(frame_spectrum)
            if denominator > 0:
                fc.append(numerator / denominator)
            else:
                fc.append(0)
        return np.array(fc)
    
    def effective_bandwith(self):
        ebw = []
        fc = self.frequency_centroid()
        for i, frame_spectrum in enumerate(self.spectrum):
            numerator = np.sum((self.frequencies - fc[i]) ** 2 * frame_spectrum)
            denominator = np.sum(frame_spectrum)
            if denominator > 0:
                ebw.append(np.sqrt(numerator / denominator))
            else:
                ebw.append(0)
        return np.array(ebw)
    
    def spectral_flatness_meassure(self):
        sfm = []
        epsilon = 1e-8
        for frame_spectrum in self.spectrum:
            geometric_mean = gmean(frame_spectrum + epsilon)
            arithmetic_mean = np.mean(frame_spectrum + epsilon)
            if arithmetic_mean > 0:
                sfm.append(geometric_mean / arithmetic_mean)
            else:
                sfm.append(0.0)

        return np.array(sfm)

class FrequencyDomainPlotter:
    def __init__(self, feature_extractor):
        self.extractor = feature_extractor
        self.times = np.linspace(0, len(self.extractor.frames) * self.extractor.frame_ms / 1000, len(self.extractor.frames), endpoint=False)

    def _plot_line(self, y_values, title, y_title, name='Data'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.times, y=y_values, mode='lines', name=name))
        fig.update_layout(title=title, xaxis_title='Time (s)', yaxis_title=y_title)
        return fig  
    
    def plot_volume(self, title='Volume over Time'):
        if hasattr(self.extractor, 'volume'):
            volume = self.extractor.volume()
            return self._plot_line(volume, title, 'Volume (Energy)', 'Volume')
           
        else:
            raise ValueError("Volume feature not available. Please compute volume first.")
        
    def plot_frequency_centroid(self, title='Frequency Centroid over Time'):
        if hasattr(self.extractor, 'frequency_centroid'):
            fc = self.extractor.frequency_centroid()
            return self._plot_line(fc, title, 'Frequency Centroid (Hz)', 'Frequency Centroid')
            
        else:
            raise ValueError("Frequency centroid feature not available. Please compute frequency centroid first.")

    def plot_effective_bandwidth(self, title='Effective Bandwitdth Over Time'):
        if hasattr(self.extractor, 'effective_bandwith'):
            ebw = self.extractor.effective_bandwith()
            return self._plot_line(ebw, title, 'Effective Bandwidth (Hz)', 'Effective Bandwidth')
        else:
            raise ValueError("Effective bandwidth feature not available. Please compute effective bandwidth first.")
        
    def plot_spectral_flatness_measure(self, title='Spectral Flatness Measure Over Time'):
        if hasattr(self.extractor, 'spectral_flatness_meassure'):
            sfm = self.extractor.spectral_flatness_meassure()
            return self._plot_line(sfm, title, 'Spectral Flatness Measure', 'Spectral Flatness Measure')
        else:
            raise ValueError("Spectral flatness measure feature not available. Please compute spectral flatness measure first.")
     