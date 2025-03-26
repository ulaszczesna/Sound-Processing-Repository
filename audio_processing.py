import numpy as np
import plotly.graph_objects as go
import librosa
import streamlit as st



def load_audio(file):
    data, rate = librosa.load(file, sr=None)
    return rate, data

def normalize_audio(data):
    return data / max(data)


def plot_waveform(data, rate):
    duration = len(data) / rate
    time = np.linspace(0, duration, len(data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name="Audio Waveform"))
    fig.update_layout(title="Waveform", xaxis_title="Time [s]", yaxis_title="Amplitude")
    return fig

def split_into_frames(data, rate, frame_ms=20):
    frame_size = int(frame_ms * rate / 1000)
    num_frames = len(data) // frame_size
    frames = np.array([data[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)])
    return frames, frame_size



def detect_silence(data, rate, frame_ms=20, percentage=5, zcr_threshold=0.3):
    frames, frame_size = split_into_frames(data, rate, frame_ms)
    silence_frames = []
    
    threshold = percentage* 0.01 * np.max(np.sqrt(np.mean(data ** 2)))
    for i, frame in enumerate(frames):
        avg_loudnes = np.sqrt(np.mean(frame ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) 
        if avg_loudnes < threshold and zcr < zcr_threshold:
            silence_frames.append(i * frame_size)
   
    
    

    return silence_frames, frame_size

def loudness(data, rate, frame_ms=20):
    frames, frame_size = split_into_frames(data, rate, frame_ms)
    frame_loudnes = []
    frames = np.array(frames)
    for frame in frames:
        energy = np.sum(frame ** 2)
        mean_energy = energy / frame_size
        loudness = np.sqrt(mean_energy)
        frame_loudnes.append(loudness)
    
    
    return frame_loudnes, frame_size

def plot_loudness(data, rate, frame_ms=20):
    duration = len(data) / rate
    
    frame_loudnes, frame_size = loudness(data, rate, frame_ms)
    time_loudness = np.arange(len(frame_loudnes)) * (frame_size / rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_loudness, y=frame_loudnes, mode='lines', name="loudness"))
    fig.update_layout(title="Loudness", xaxis_title="Time [s]", yaxis_title="Loudness")
    return fig

def plot_silence(data, rate, silence_frames, frame_size):
    duration = len(data) / rate
    time = np.linspace(0., duration, len(data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name="Audio Waveform"))

    for frame_start in silence_frames:
        start_time = frame_start / rate
        end_time = (frame_start + frame_size) / rate
        fig.add_shape(
            type="rect", x0=start_time, x1=end_time, y0=-1, y1=1,  
            fillcolor="red", opacity=0.3, layer="below", line_width=0
        )

    fig.update_layout(title="Silence Detection", xaxis_title="Time [s]", yaxis_title="Amplitude")
    return fig

def short_time_energy(data, rate, frame_ms = 20):
    frames, frame_size = split_into_frames(data, rate, frame_ms)
    frame_energy = []
    for frame in frames:
        energy = np.sum(frame ** 2)
        mean_energy = energy / frame_size
        frame_energy.append(mean_energy)

    return frame_energy, frame_size


def plot_short_time_energy(data, rate, frame_ms=20):

    frame_energy, frame_size = short_time_energy(data, rate, frame_ms)

    time_energy = np.arange(len(frame_energy)) * (frame_size / rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_energy, y=frame_energy, mode='lines', name="Short-Time Energy"))
    fig.update_layout(title="Short-Time Energy", xaxis_title="Time [s]", yaxis_title="Energy")
    return fig

def zero_crossing_rate(data, rate, frame_ms=20):
    frames, frame_size = split_into_frames(data, rate, frame_ms)
    zcr_values = []
    for frame in frames:
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) 
        zcr_values.append(zcr)

    return zcr_values, frame_size

def plot_zero_crossing_rate(data, rate, frame_ms=20):
    zcr_values, frame_size = zero_crossing_rate(data, rate, frame_ms)
    time = np.arange(len(zcr_values)) * (frame_size / rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=zcr_values, mode='lines', name="Zero Crossing Rate", line=dict(color='purple')))
    fig.update_layout(title="Zero Crossing Rate", xaxis_title="Time [s]", yaxis_title="ZCR")
    return fig

def plot_frame_features(rms_values, mean_values, var_values, frame_size, rate):
    time = np.arange(len(rms_values)) * (frame_size / rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=rms_values, mode='lines', name="RMS", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time, y=mean_values, mode='lines', name="Mean", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=time, y=var_values, mode='lines', name="Variance", line=dict(color='red')))
    
    fig.update_layout(title="Frame Features (RMS, Mean, Variance)",
                      xaxis_title="Time [s]", yaxis_title="Value")
    return fig

def compute_frame_features(data, rate, frame_ms=20):
    # """ Oblicza parametry w dziedzinie czasu na poziomie ramki """
    frame_size = int(frame_ms * rate / 1000)  
    num_frames = len(data) // frame_size

    rms_values = []
    mean_values = []
    var_values = []
    amplitude_values = []

    for i in range(num_frames):
        frame = data[i * frame_size: (i + 1) * frame_size]
        
        amplitude_values.append(np.sqrt(np.mean(frame ** 2)))

    return rms_values, mean_values, var_values, frame_size

def autocorrelation(frame):
    corr = np.correlate(frame, frame, mode='full')
    return corr[len(corr) // 2:]

def compute_f0_autocorrelation(data, rate, frame_ms = 20, min_f0=50, max_f0=400):
    frame_size = int(frame_ms * rate / 1000)
    num_frames = len(data) // frame_size
    f0_values = []

    for i in range(num_frames):
        frame = data[i * frame_size: (i + 1) * frame_size]
        corr = autocorrelation(frame)
        peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
        
        if len(peaks) > 0:
            peak_index = peaks[0]
            f0 = rate / peak_index
            if min_f0 <= f0 <= max_f0:
                f0_values.append(f0)
            else:
                f0_values.append(0)
        else:
            f0_values.append(0)

    return f0_values, frame_size

def plot_f0(f0_values, frame_size, rate):
    time = np.arange(len(f0_values)) * (frame_size / rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=f0_values, mode='lines', name="F0", line=dict(color='purple')))
    
    fig.update_layout(title="Fundamental Frequency (F0)",
                      xaxis_title="Time [s]", yaxis_title="Frequency [Hz]")
    return fig

def compute_voiced_unvoiced(data, rate, frame_ms=20, ste_threshold=0.02, zcr_threshold=0.1):
    ste, frame_size = short_time_energy(data, rate, frame_ms)
    zcr, _ = zero_crossing_rate(data, rate, frame_ms)

    if len(ste) == 0 or len(zcr) == 0:
        return np.array([]), frame_size

    # Klasyfikacja na podstawie stałych progów
    voiced = np.array([
        1 if ste[i] > ste_threshold and zcr[i] < zcr_threshold else 0
        for i in range(len(ste))
    ])

    return voiced, frame_size
    

def plot_voiced_unvoiced(data, rate, frame_ms=20, ste_threshold= 0.02, zcr_threshold=0.1):
    duration = len(data) / rate
    time = np.linspace(0., duration, len(data))
    
    voiced, frame_size = compute_voiced_unvoiced(data, rate, frame_ms, ste_threshold, zcr_threshold)  # Zamiana F0 na 0/1 (voiced/unvoiced)
  
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name="Audio Waveform"))

    for i, is_voiced in enumerate(voiced):
        if is_voiced:  # Jeśli fragment jest voiced
            start_time = (i * frame_size) / rate
            end_time = ((i + 1) * frame_size) / rate
            fig.add_shape(
                type="rect", x0=start_time, x1=end_time, y0=-1, y1=1,
                fillcolor="blue", opacity=0.2, layer="below", line_width=0
            )

    # Dodanie "fałszywego" trace do legendy dla Voiced
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color="blue", opacity=0.2),
        name="Voiced Region"
    ))

    fig.update_layout(title="Voiced/Unvoiced Segmentation", 
                      xaxis_title="Time [s]", yaxis_title="Amplitude")  # Pozycja legendy w lewym górnym rogu
    return fig


# clip features
    

def compute_LSTER(data, rate, frame_ms=20, window_s=1):
    frame_size = int(frame_ms * rate / 1000)
    window_size = int(window_s * rate/ frame_size)

    num_frames = len(data) // frame_size
    ste_values = []

    for i in range(num_frames):
        frame = data[i * frame_size: (i + 1) * frame_size]
        ste_values.append(np.sum(frame ** 2))

    lster_values = []
    for i in range(num_frames - window_size + 1):
        window_ste = ste_values[i: i + window_size]
        ste_mean = np.mean(window_ste)
        low_energy_count = sum(ste < 0.5 * ste_mean for ste in window_ste)
        lster = low_energy_count / window_size
        lster_values.append(lster)

    return lster_values, frame_size

def Zstd(data, rate, frame_ms=20):
    frame_size = int(frame_ms * rate / 1000)
    num_frames = len(data) // frame_size
    std_values = []
    for i in range(num_frames):
        frame = data[i * frame_size: (i + 1) * frame_size]
        std_values.append(np.std(frame))
    return std_values

def compute_clip_features(data, rate, frame_ms=20):
    frame_size = int(frame_ms * rate / 1000)
    num_frames = len(data) // frame_size
    volumes = []
    for i in range(num_frames):
        frame = data[i * frame_size: (i + 1) * frame_size]
        volumes.append(np.sqrt(np.mean(frame ** 2)))
    max_volume = np.max(volumes)
    min_volume = np.min(volumes)
    vstd = np.std(volumes) / max_volume if max_volume != 0 else 0
    vdr = (max_volume - min_volume) / max_volume if max_volume != 0 else 0
    
    # Volume Undulation (VU)
    frame_rms = [
        np.sqrt(np.mean(data[i : i + frame_size] ** 2))
        for i in range(0, len(data) - frame_size, frame_size)
    ]
    vu = np.std(frame_rms) / np.mean(frame_rms) if np.mean(frame_rms) != 0 else 0
    
    # Low Short-Time Energy Ratio (LSTER)
    lster_values, frame_size = compute_LSTER(data, rate, frame_ms=frame_ms)



    return vstd, vdr, vu, lster_values, frame_size


import matplotlib.pyplot as plt

def display_clip_features(data, rate, frame_ms=20):
    vstd, vdr, vu, lster_values, frame_size = compute_clip_features(data, rate, frame_ms)


    # Jeśli mamy wartości LSTER, narysujmy je
    if lster_values:
        time_axis = np.arange(len(lster_values)) * (frame_size / rate)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=lster_values, mode='lines', name="LSTER"))
        fig.update_layout(title="Low Short-Time Energy Ratio (LSTER)", xaxis_title="Time [s]", yaxis_title="LSTER")
        
        return fig



def classify_speech_music(f0_values):
    speech_threshold = 150
    music_threshold = 300
    classified = []

    for f0 in f0_values:
        if f0 > music_threshold:
            classified.append("Music")
        elif f0 > speech_threshold:
            classified.append("Speech")
        else:
            classified.append("Unclear")
    
    return classified

import csv

def save_to_csv(filename, data_table):
    if filename.endswith('.csv'):
        data_table.to_csv(filename, index=False)
    elif filename.endswith('.txt'):
        data_table.to_csv(filename, sep='\t', index=False)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .txt")
    