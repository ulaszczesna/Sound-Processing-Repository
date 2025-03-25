from audio_processing import *
import librosa
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks

# volume based
def vstd(data, rate, frame_ms = 1000):
    frame_loudnes, frame_size = loudness(data, rate, frame_ms)
    std = np.std(frame_loudnes)
    max_loudness = np.max(frame_loudnes)
    return std/max_loudness


def vdr(data, rate, frame_ms = 1000):
    frame_loudnes, frame_size = loudness(data, rate, frame_ms)
    max_loudness = np.max(frame_loudnes)
    min_loudness = np.min(frame_loudnes)
    print(max_loudness, min_loudness)
    return (max_loudness - min_loudness)/max_loudness


def vu(data, rate, frame_ms = 1000):
    frame_loudnes, frame_size = loudness(data, rate, frame_ms)
    dif = np.diff(frame_loudnes)
    changes = dif[1:] * dif[:-1] < 0
    dif = dif[1:] # usunięcie pierwszego elementu dif.
    vu = np.sum(np.abs(dif[changes]))
    return vu
    


# energy based


def lster(data, rate, frame_ms = 20):
    ste_1s, frame_size = short_time_energy(data, rate, 1000)
    lster = []
    for i, frame in enumerate(ste_1s):
        lst = 0
        av_ste = np.average(frame)
        # chose only data in this frame
        data_frame = data[i * frame_size:(i + 1) * frame_size]
        ste = short_time_energy(data_frame, rate, frame_ms)[0]
        for i in range(len(ste)):
            lst += np.sign(0.5 * av_ste - ste[i]) + 1
        lster.append(lst / 2 * len(ste))
    
    return lster 

def plot_lster(data, rate, frame_ms = 20):
    lste = lster(data, rate, frame_ms)
    time = np.linspace(0, len(data) / rate, len(lste))
    fig = go.Figure(data=go.Scatter(x=time, y=lste)) # Poprawka tutaj
    fig.update_layout(title='LSTER', xaxis_title='Time [s]', yaxis_title='LSTER')
    return fig

def energy_entropy(data, rate, frame_ms = 1000):
    ste, frame_size = short_time_energy(data, rate, frame_ms)
    K = 441
    normalized_ste =[] 
    for i in range(len(ste)):
        frame_start = i * frame_size
        frame_end = (i + 1) * frame_size
        frame = data[frame_start:frame_end]
        total_energy = ste[i]
        for start in range(0, len(frame), K):
            segment_end = min(start+K, len(ste))
            segment = frame[start:segment_end]
            segment_energy = np.sum(segment**2)

            if total_energy > 0:  #dzielenie przez 0 
                normalized_energy = segment_energy / total_energy
            else:
                normalized_energy = 0
            normalized_ste.append(normalized_energy)
    #clipping to avoid log(0)
    normalized_ste = np.clip(normalized_ste, 1e-10, 1)
    entropy = -np.sum(normalized_ste * np.log(normalized_ste))
    return entropy

# zcr based
def zcr_dev(data, rate, frame_ms = 1000):
    zcr, frame_size = zero_crossing_rate(data, rate, frame_ms)
    zcr_dev = np.std(zcr)
    return zcr_dev

def hzcrr(data, rate, frame_ms = 20):
    zcr_1s, frame_size = zero_crossing_rate(data, rate, 1000)
    hzcrr = []
    for i, frame in enumerate(zcr_1s):
        hzcr = 0
        av_zcr = np.average(frame)
        data_frame = data[i * frame_size:(i + 1) * frame_size]
        zcr = zero_crossing_rate(data_frame, rate, frame_ms)[0]
        for i in range(len(zcr)):
            hzcr += np.sign(zcr[i] - 1.5 * av_zcr) + 1
        hzcrr.append(hzcr / 2 * len(zcr))
    return hzcrr

def plot_hzcrr(data, rate, frame_ms = 20):
    hzcr = hzcrr(data, rate, frame_ms)
    time = np.linspace(0, len(data) / rate, len(hzcr))
    fig = go.Figure(data=go.Scatter(x=time, y=hzcr))
    fig.update_layout(title='HZCRR', xaxis_title='Time [s]', yaxis_title='HZCRR')
    return fig

def plot_speach_music(data, rate, frame_ms = 20):
    zcr_values, frame_size = zero_crossing_rate(data, rate, frame_ms)
    speech_frames = []
    music_frames = []
    for i, frame in enumerate(zcr_values):
        if 0.08 < frame:
            start_time = i * frame_size / rate
            end_time = (i + 1) * frame_size / rate
            speech_frames.append((start_time, end_time))
        elif 0.01 < frame < 0.08:
            start_time = i * frame_size / rate
            end_time = (i + 1) * frame_size / rate
            music_frames.append((start_time, end_time))

    time = np.linspace(0, len(data) / rate, len(data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Waveform'))

    # Zaznaczenie fragmentów mowy
    for start, end in speech_frames:
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=min(data),
            y1=max(data),
            fillcolor="red",
            opacity=0.3,
            line_width=0,
        )

    # Zaznaczenie fragmentów muzyki
    for start, end in music_frames:
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=min(data),
            y1=max(data),
            fillcolor="green",
            opacity=0.3,
            line_width=0,
        )

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red', opacity=0.3), name='Speech'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='green', opacity=0.3), name='Music'))

    fig.update_layout(title='Audio Waveform with Speech/Music Segmentation', xaxis_title='Time [s]', yaxis_title='Amplitude')

    return fig
   

