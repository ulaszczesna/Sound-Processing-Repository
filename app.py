import streamlit as st
import soundfile as sf
import pandas as pd
from audio_processing import classify_speech_music, load_audio, plot_waveform, detect_silence, plot_silence, plot_frame_features, plot_f0, plot_voiced_unvoiced, compute_f0_autocorrelation, save_to_csv
from audio_processing import compute_frame_features, compute_voiced_unvoiced, compute_clip_features, normalize_audio

st.title('🎵 Audio Analysis App')

uploaded_file = st.file_uploader("📂 Wybierz plik WAV", type="wav")

if uploaded_file is not None:
    rate, data = load_audio(uploaded_file)

    

    st.sidebar.header('Audio analysis options:')
    normalize = st.sidebar.checkbox('Normalize audio', False)
    if normalize:
        data = normalize_audio(data)

    
    st.write(f'📌 **Częstotliwość próbkowania:** {rate} Hz')
    st.audio(uploaded_file)

    st.subheader('📊 Wavefrom')
    st.plotly_chart(plot_waveform(data, rate))


    silence = st.sidebar.toggle('Silence detection', False)
    if silence:
        silence_frames, frame_size = detect_silence(data, rate)
        st.subheader('🔇 Silence Detection')
        st.plotly_chart(plot_silence(data, rate, silence_frames, frame_size))
    

    paremeters = st.sidebar.toggle('Frame parameters', False)
    rms_values, mean_values, var_values, frame_size = compute_frame_features(data, rate)
    if paremeters:
        
        st.subheader('📈 Frame parameters (RMS, Mean, Variance)')
        st.plotly_chart(plot_frame_features(rms_values, mean_values, var_values, frame_size, rate))
    
    fundamental_frequency = st.sidebar.toggle('Fundamental Frequency (F0)', False)
    f0_values, frame_size = compute_f0_autocorrelation(data, rate)
    if fundamental_frequency:
        
        st.subheader('🎶 Fundamental Frequency (F0)')
        st.plotly_chart(plot_f0(f0_values, frame_size, rate))
    
    voiced_unvoiced = st.sidebar.toggle('Voiced/Unvoiced detection', False)
    if voiced_unvoiced:
        voiced = compute_voiced_unvoiced(f0_values)
        st.subheader('🎤 Voiced/Unvoiced')
        st.plotly_chart(plot_voiced_unvoiced(voiced, frame_size, rate))

    st.sidebar.header('Detailed information:')  
    details = st.sidebar.checkbox('Show detailed information about the audio clip', False)
    if details:
        rms, mean, var, max_amplitude, min_amplitude, energy = compute_clip_features(data)
         
        st.subheader('📊 Detailed information about the audio clip')
        data_table = pd.DataFrame({
            'Parameter': ['RMS', 'Mean', 'Variance', 'Max amplitude', 'Min amplitude', 'Energy'],
            'Value': [rms, mean, var, max_amplitude, min_amplitude, energy]
        })


        # Wyświetlanie tabeli
        st.table(data_table)


        st.text('Save detailed information to CSV file or txt file')
        
        save_csv = st.button('Save to CSV')
        save_txt = st.button('Save to TXT')
        if save_csv:
            save_to_csv('audio_details.csv', data_table)
            st.success('Data saved to audio_details.csv')
        if save_txt:
            save_to_csv('audio_details.txt', data_table)
            st.success('Data saved to audio_details.txt')
    
    

