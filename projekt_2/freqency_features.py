import numpy as np
from scipy.stats import gmean
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class FreqencyDomainFeatures:
    def __init__(self, signalprocessor, window_type='rectangular'):
        self.data = signalprocessor.audio
        self.rate = signalprocessor.sample_rate
        self.frame_ms = signalprocessor.frame_ms
        self.frame_size = signalprocessor.frame_size
        self.frames = signalprocessor.split_into_frames()
        self.spectrum = np.array([np.abs(np.fft.fft(frame))[:self.frame_size // 2] for frame in self.frames])
        self.frequencies = np.fft.fftfreq(self.frame_size, d=1/self.rate)[:self.frame_size // 2]
        self.windowed_frames = signalprocessor.apply_window(window_type)
        self.window_sum = signalprocessor.sum_window(window_type)

    
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
    
    def band_energies(self):
        bands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
        band_energies = {i: [] for i in range(len(bands))}
        windowed_frames = self.windowed_frames
        window_sum = self.window_sum

        for frame in windowed_frames:

            ft = np.fft.fft(frame, n=self.frame_size)
            magnitude_spectrum = np.abs(ft)[:self.frame_size // 2]
            freqs = np.fft.fftfreq(self.frame_size, d=1/self.rate)[:self.frame_size // 2]
            power = magnitude_spectrum ** 2
            for i, (low, high) in enumerate(bands):
                band_mask = (freqs >= low) & (freqs < high)
                band_energy = np.sum(power[band_mask]) / window_sum
                band_energies[i].append(band_energy)
        return band_energies

    def band_energy_ratios(self):
        band_energies = self.band_energies()
        num_frames = len(self.frames)
        num_bands = len(band_energies)
        volumes = []
        for i in range(num_frames):
            total_energy = sum(band_energies[j][i] for j in range(num_bands))
            volumes.append(total_energy)

        band_energy_ratios = {i: [] for i in range(num_bands)}
        for i in range(num_frames):
            vol = volumes[i]
            for idx in range(num_bands):
                ratio = band_energies[idx][i] / vol if vol != 0 else 0
                band_energy_ratios[idx].append(ratio)
        return band_energy_ratios

    def spectral_flatness_meassure(self):
        bands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
        sfm = {i: [] for i in range(len(bands))}
        epsilon = 1e-8
        for frame_spectrum in self.spectrum:
            for i, (low, high) in enumerate(bands):
                band_mask = (self.frequencies >= low) & (self.frequencies < high)
                band_spectrum = frame_spectrum[band_mask]
                if np.sum(band_spectrum) > 0:
                    geometric_mean = gmean(band_spectrum + epsilon)
                    arithmetic_mean = np.mean(band_spectrum + epsilon)
                    sfm[i].append(geometric_mean / arithmetic_mean)
                else:
                    sfm[i].append(0)

        return {band: np.array(sfm[idx]) for idx, band in enumerate(bands)}
    
    def spectral_crest(self):
        bands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
        band_crest = {i: [] for i in range(len(bands))}
        for frame in self.windowed_frames:
            ft = np.fft.fft(frame, n=self.frame_size)
            magnitude_spectrum = np.abs(ft)[:self.frame_size // 2]
            power = magnitude_spectrum ** 2
            for i, (low, high) in enumerate(bands):
                band_mask = (self.frequencies >= low) & (self.frequencies < high)
                band_power = power[band_mask]
                nominator = np.max(band_power)
                denominator = np.mean(band_power)
                band_crest[i].append(nominator / denominator if denominator > 0 else 0)
        return band_crest

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
        
    def plot_spectral_flatness_measure(self):
        # Pobieramy wyniki płaskości widma
        sfm_results = self.extractor.spectral_flatness_meassure()
        
        # Oś czasu
        time = np.linspace(0, len(self.extractor.frames) * self.extractor.frame_ms / 1000, len(self.extractor.frames), endpoint=False)

        # Kolory i etykiety pasm
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']
        
        # Tworzymy wykres
        fig = go.Figure()

        # Dodajemy każdą linię dla pasm
        for idx, (band, sfm_values) in enumerate(sfm_results.items()):
            fig.add_trace(go.Scatter(x=time, y=sfm_values, mode='lines', name=labels[idx], line=dict(color=colors[idx])))
        
        # Ustawiamy tytuł i etykiety osi
        fig.update_layout(
            title="Spectral Flatness Measure for Different Frequency Bands",
            xaxis_title="Time [s]",
            yaxis_title="Spectral Flatness Measure",
            template="plotly_dark"
        )
        
       
        return fig
        
    def plot_band_energies(self):
        band_energies = self.extractor.band_energies()
        frame_numbers = np.arange(len(self.extractor.frames))
        times = np.linspace(0, len(self.extractor.frames) * self.extractor.frame_ms / 1000, len(self.extractor.frames), endpoint=False)
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']

        # Rysowanie wykresów dla poszczególnych pasm
        for idx in band_energies:
            fig.add_trace(go.Scatter(x=times, y=band_energies[idx], mode='lines', name=labels[idx], line=dict(color=colors[idx])))

        # Aktualizacja układu wykresu
        fig.update_layout(
            title="Band Energies Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Energy",
            showlegend=True
        )

  

        return fig
    
    def plot_band_energy_ratios(self):
        band_energy_ratios = self.extractor.band_energy_ratios()
        frame_numbers = np.arange(len(self.extractor.frames))
        times = np.linspace(0, len(self.extractor.frames) * self.extractor.frame_ms / 1000, len(self.extractor.frames), endpoint=False)
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']

        # Rysowanie wykresów dla poszczególnych pasm
        for idx in band_energy_ratios:
            fig.add_trace(go.Scatter(x=times, y=band_energy_ratios[idx], mode='lines', name=labels[idx], line=dict(color=colors[idx])))

        # Aktualizacja układu wykresu
        fig.update_layout(
            title="Band Energies Ratios Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Energy Ratio",
            showlegend=True
        )
        return fig
    
    def plot_spectral_crest(self):
        spectral_crest = self.extractor.spectral_crest()
        frame_numbers = np.arange(len(self.extractor.frames))
        times = np.linspace(0, len(self.extractor.frames) * self.extractor.frame_ms / 1000, len(self.extractor.frames), endpoint=False)
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']

        # Rysowanie wykresów dla poszczególnych pasm
        for idx in spectral_crest:
            fig.add_trace(go.Scatter(x=times, y=spectral_crest[idx], mode='lines', name=labels[idx], line=dict(color=colors[idx])))

        # Aktualizacja układu wykresu
        fig.update_layout(
            title="Spectral Crest Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Spectral Crest",
            showlegend=True 
        )
        return fig

     