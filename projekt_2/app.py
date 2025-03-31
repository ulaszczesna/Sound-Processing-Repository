import streamlit as st
from audio_processing import *

st.title('🎵 Audio Analysis App')

uploaded_file = st.file_uploader("📂 Wybierz plik WAV", type="wav")

if uploaded_file is not None:
    rate, data = load_audio(uploaded_file)

    

    st.sidebar.header('Audio analysis options:')
    normalize = st.sidebar.checkbox('Normalize audio', False)
    if normalize:
        data = normalize_audio(data)
    frame_ms = st.sidebar.slider('Frame size (ms)', 10, 50, 20)
    frame_size = int(frame_ms * rate / 1000)
    
    st.write(f'📌 **Częstotliwość próbkowania:** {rate} Hz')
    st.audio(uploaded_file)

    st.subheader('📊 Wavefrom')
    st.plotly_chart(plot_waveform(data, rate))

    st.subheader('📊 Continuous Spectrum')
    spectrum, _ = continous_spectrum(data, rate, frame_ms)
    st.plotly_chart(plot_fft_signal(data, rate))

    volume_plot = st.sidebar.checkbox('Volume plot', False)
    if volume_plot:
        st.subheader('📊 Volume')
        volume, frame_size, volume_plot = volume(data, rate, frame_ms, plot=True)
        st.plotly_chart(volume_plot)
    #st.plotly_chart(plot_spectrum(spectrum, rate, frame_size))

