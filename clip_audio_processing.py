from audio_processing import *
import librosa
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks

# volume based
def vstd(data, rate, frame_ms = 100):
    frame_loudnes, frame_size = loudness(data, rate, frame_ms)
    std = np.std(frame_loudnes)
    max_loudness = np.max(frame_loudnes)
    return std/max_loudness

# do poprawy
def vdr(data, rate, frame_ms = 100):
    frame_loudnes, frame_size = loudness(data, rate, frame_ms)
    max_loudness = np.max(frame_loudnes)
    min_loudness = np.min(frame_loudnes)
    return (max_loudness - min_loudness)/max_loudness

# do poprawy
def vu(data, rate, frame_ms = 100):
    frame_loudnes, frame_size = loudness(data, rate, frame_ms)
    print(frame_loudnes)
    frame_loudnes = np.array(frame_loudnes, dtype=np.float64)  # Convert to numeric type

    if frame_loudnes.ndim != 1:  # Check if the array is 1D
        raise ValueError("frame_loudnes must be a 1D array.")
    peaks, _ = find_peaks(frame_loudnes)
    valleys, _ = find_peaks(-frame_loudnes)
    extrema = np.sort(np.concatenate((peaks, valleys)))
    diffs = np.diff(frame_loudnes[extrema])
    vu = np.sum(np.abs(diffs))
    return vu

# energy based

# do poprawy
def lster(data, rate, frame_ms = 10):
    ste = short_time_energy(data, rate, frame_ms)
    avg_ste = np.average(ste)
    sum = 0
    for s in ste:
        sum += np.sign(0.5 * avg_ste - s)+1
    return sum/(2*len(ste))

def energy_entropy(data, rate, frame_ms = 100):
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