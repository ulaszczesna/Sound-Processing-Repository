import numpy as np
import librosa
import plotly.graph_objects as go

def load_audio(file):
    data, rate = librosa.load(file, sr=None)
    return rate, data

def normalize_audio(data):
    return data / max(data)

def continous_spectrum(data, rate, frame_ms=20, frame_start=0, frame_end=None):
    if frame_end is None:
        frame_end = len(data)
    data = data[frame_start:frame_end]
    frame_size = int(frame_ms * rate / 1000)
    num_frames = len(data) // frame_size
    frames = np.array([data[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)])
    spectrum = np.abs(np.fft.fft(frames, axis=1))
    return spectrum, frame_size

def plot_spectrum(spectrum, rate, frame_size):
    freq = np.fft.fftfreq(frame_size, d=1/rate)
    fig = go.Figure()
    for i in range(spectrum.shape[0]):
        fig.add_trace(go.Scatter(x=freq[:frame_size // 2], y=spectrum[i, :frame_size // 2], mode='lines', name=f"Frame {i}"))
    fig.update_layout(title="Continuous Spectrum", xaxis_title="Frequency [Hz]", yaxis_title="Magnitude")
    return fig

def plot_fft_signal(data, rate):
    fft_result = np.fft.fft(data)
    freq = np.fft.fftfreq(len(data), d=1/rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq[:len(data) // 2], y=np.abs(fft_result[:len(data) // 2]), mode='lines', name="FFT Signal"))
    fig.update_layout(title="FFT Signal", xaxis_title="Frequency [Hz]", yaxis_title="Magnitude")
    return fig

def plot_waveform(data, rate, start=0, end=None):
    if end is None:
        end = len(data)
    duration = len(data) / rate
    time = np.linspace(0, duration, len(data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name="Audio Waveform"))
    fig.add_shape(
        type="rect", x0=start / rate, y0=-1, x1=end / rate, y1=1,
        fillcolor="purple", opacity=0.2, line_width=0, layer="below",
        name="Selected Range"
    )
    fig.update_layout(title="Waveform", xaxis_title="Time [s]", yaxis_title="Amplitude")
    return fig

def plot_waveform_window(full_date, windowed_data, start, end, rate):
    if end is None:
        end = len(full_date)
    duration = len(full_date) / rate
    time_full = np.linspace(0, duration, len(full_date))
    time_windowed = np.linspace(start / rate, end / rate,  len(windowed_data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_full, y=full_date, mode='lines', name="Audio Waveform", opacity=0.8))
    fig.add_trace(go.Scatter(x=time_windowed, y=windowed_data, mode='lines', name="Signal with Window Function", opacity=0.8))
    fig.add_shape(
        type="rect", x0=start / rate, y0=-1, x1=end / rate, y1=1,
        fillcolor="purple", opacity=0.2, line_width=0, layer="below",
        name="Selected Range"
    )
    fig.update_layout(title="Waveform", xaxis_title="Time [s]", yaxis_title="Amplitude")
    return fig

def split_into_frames(data, rate, frame_ms=20):
    frame_size = int(frame_ms * rate / 1000)
    num_frames = len(data) // frame_size
    frames = np.array([data[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)])
    return frames, frame_size



def volume(data, rate, frame_ms=20, plot=False):
    frames, frame_size = split_into_frames(data, rate, frame_ms)
    volume = []
    for frame in frames:
        spectrum = np.abs(np.fft.fft(frame))
        vol = np.sum(spectrum ** 2) / frame_size
        volume.append(vol)
    volume = np.array(volume)
    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(volume)), y=volume, mode='lines', name="Volume"))
        fig.update_layout(title="Volume", xaxis_title="Frame", yaxis_title="Volume")
        
    return volume, frame_size, fig if plot else None

