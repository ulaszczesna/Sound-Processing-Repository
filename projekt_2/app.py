import streamlit as st
from audio_processing import *
from window_functions import *
from freqency_features import *

st.title('🎵 Audio Analysis App')

uploaded_file = st.file_uploader("📂 Wybierz plik WAV", type="wav")

if uploaded_file is not None:
    rate, data = load_audio(uploaded_file)
    audio_duration = len(data) / rate

    

    st.sidebar.header('Audio analysis options:')
    normalize = st.sidebar.checkbox('Normalize audio', False)
    if normalize:
        data = normalize_audio(data)
    frame_ms = st.sidebar.slider('Frame size (ms)', 10, 50, 20)
    frame_size = int(frame_ms * rate / 1000)
    time_range = st.sidebar.slider("Select time range (s)", 0.0, float(audio_duration), (0.0, float(audio_duration)), step=0.1)

    start_sample = int(time_range[0] * rate)
    end_sample = int(time_range[1] * rate)
    
    st.write(f'📌 **Częstotliwość próbkowania:** {rate} Hz')
    st.audio(uploaded_file)

    st.subheader('📊 Wavefrom')
    st.plotly_chart(plot_waveform(data, rate, start_sample, end_sample))

    st.subheader('📊 Continuous Spectrum')
    spectrum, _ = continous_spectrum(data, rate, frame_ms, start_sample, end_sample)
    st.plotly_chart(plot_spectrum(spectrum, rate, frame_size))

    fequency_features = st.sidebar.selectbox('Frequency features', options=('Volume', 'Frequency Centroid', 'Effective Bandwidth', 'Spectral Flatness Measure'), key='features')
    frequency_feature = FreqencyDomainFeatures(data, rate, start=start_sample, end=end_sample, frame_ms=frame_ms)

    st.subheader(f'📊 {fequency_features}')
    if fequency_features == 'Volume':
        volume = frequency_feature.volume()
        volume_ploter = FrequencyDomainPlotter(frequency_feature)
        st.plotly_chart(volume_ploter.plot_volume())

    elif fequency_features == 'Frequency Centroid':
        frequency_centroid = frequency_feature.frequency_centroid()
        frequency_centroid_plotter = FrequencyDomainPlotter(frequency_feature)
        st.plotly_chart(frequency_centroid_plotter.plot_frequency_centroid())

    elif fequency_features == 'Effective Bandwidth':
        effective_bandwidth = frequency_feature.effective_bandwith()
        effective_bandwidth_plotter = FrequencyDomainPlotter(frequency_feature)
        st.plotly_chart(effective_bandwidth_plotter.plot_effective_bandwidth())

    elif fequency_features == 'Spectral Flatness Measure':
        flatness_measure = frequency_feature.spectral_flatness_meassure()
        flatness_measure_plotter = FrequencyDomainPlotter(frequency_feature)
        st.plotly_chart(flatness_measure_plotter.plot_spectral_flatness_measure())

 
    # if volume_plot:
    #     st.subheader('📊 Volume')
    #     volume, frame_size, volume_plot = volume(data, rate, frame_ms, plot=True)
    #     st.plotly_chart(volume_plot)
    #st.plotly_chart(plot_spectrum(spectrum, rate, frame_size))

    st.sidebar.subheader('Window Functions')
    
    window_on = st.sidebar.toggle('Window function', False)
    if window_on:
        window_function = st.sidebar.selectbox('Choose window function', options=('rectangular', 'triangular', 
                                                                'hamming', 'hann', 'blackman'), key='window')
        window_processor = SignalProcessor(data, rate)
        windowed_signal, frame = window_processor.apply_window(window_function, frame_start=start_sample, frame_end=end_sample)
        st.subheader('📊 Windowed Signal')
        st.plotly_chart(plot_fft_signal(windowed_signal, rate))
        st.plotly_chart(plot_waveform_window(data, windowed_signal, start_sample, end_sample, rate))
    
    spectrogram_on = st.sidebar.toggle('Spectrogram', False)
    if spectrogram_on:
        st.subheader('📊 Spectrogram')
        window_processor = SignalProcessor(data, rate)
        window_function_spect = st.sidebar.selectbox('Choose window function', options=('rectangular', 'triangular', 'hamming', 'hann', 'blackman'), key='spectrogram')
        frame_length = int(frame_ms * rate / 1000)
        hop = st.sidebar.slider('Hop size (ms)', 1, 50, 10)
        hop_size = int(hop * rate / 1000)
        spectrogram_generator = SpectogramGenerator(window_processor)
        spectrogram_data, frequencies, times = spectrogram_generator.generate(start_sample, end_sample, frame_length, hop_size, window_type=window_function_spect)
        db = st.sidebar.checkbox('dB scale', True)
        # st.plotly_chart(spectrogram_generator.plot_spectrogram(db_scale=db))
        st.pyplot(spectrogram_generator.plot_spectrogram(db_scale=db))
