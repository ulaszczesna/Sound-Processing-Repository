import numpy as np
import plotly.graph_objects as go

def apply_window(data, window_type, frame_start=0, frame_size=None):
    if frame_size is None:
        frame = data[frame_start:]
    else:
        frame = data[frame_start:frame_start + frame_size]

    N = len(frame)

    # wybór funkcji okna
    if window_type == 'rectangular':
        window = np.ones(N)
    elif window_type == 'hamming':
        window = hamming_window(N)
    elif window_type == 'hann':
        window = Hann_window(N)
    elif window_type == 'triangular':
        window = traingular_window(N)
    else:
        raise ValueError("Not valid window type! Choose: rectangular, triangular, hamming, hann")

    # zastosowanie okna 
    windowed_signal = frame * window

    return windowed_signal, 

def hamming_window(N):
    w = []
    for n in range(N):
        w.append(0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)))
    return np.array(w)

def Hann_window(N):
    w = []
    for n in range(N):
        w.append(0.5(1 - np.cos((2 * np.pi * n)/(N-1))))
    return np.array(w)

def traingular_window(N):
    w = []
    for n in range(N):
        w.append(1 - np.abs((n - (N-1)/2) / ((N-1)/2)))
    return w