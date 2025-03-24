import streamlit as st
import soundfile as sf
import pandas as pd
from audio_processing import classify_speech_music, load_audio, plot_waveform, detect_silence, plot_silence, plot_frame_features, plot_f0, plot_voiced_unvoiced, compute_f0_autocorrelation, save_to_csv
from audio_processing import compute_frame_features, compute_voiced_unvoiced, compute_clip_features, normalize_audio
from audio_processing import compute_clip_features, display_clip_features

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


    silence = st.sidebar.toggle('Silence detection', False)
    if silence:
        percentage = st.sidebar.slider('Threshold (%)', 1, 10, 5)
        zcr_threshold = st.sidebar.slider('ZCR threshold', 0.1, 1.0, 0.3)
        silence_frames, frame_size = detect_silence(data, rate, frame_ms, percentage, zcr_threshold)
        st.subheader('🔇 Silence Detection')
        st.plotly_chart(plot_silence(data, rate, silence_frames, frame_size))
    

    paremeters = st.sidebar.toggle('Frame parameters', False)
    rms_values, mean_values, var_values, frame_size = compute_frame_features(data, rate, frame_ms)
    if paremeters:
        
        st.subheader('📈 Frame parameters (RMS, Mean, Variance)')
        st.plotly_chart(plot_frame_features(rms_values, mean_values, var_values, frame_size, rate))
    
    fundamental_frequency = st.sidebar.toggle('Fundamental Frequency (F0)', False)
    f0_values, frame_size = compute_f0_autocorrelation(data, rate, frame_ms)
    print('f0')
    #f0_values, frame_size = compute_f0_amdf(data, rate, frame_ms)
    if fundamental_frequency:
        
        st.subheader('🎶 Fundamental Frequency (F0)')
        print('chart')
        print(f0_values)
        st.plotly_chart(plot_f0(f0_values, frame_size, rate))
    
    voiced_unvoiced = st.sidebar.toggle('Voiced/Unvoiced detection', False)
    if voiced_unvoiced:
        #voiced = compute_voiced_unvoiced(f0_values)
        st.subheader('🎤 Voiced/Unvoiced')
        st.plotly_chart(plot_voiced_unvoiced(data, rate, f0_values, frame_size))

    st.sidebar.header('Detailed information:')  
    details = st.sidebar.checkbox('Show detailed information about the audio clip', False)
    if details:
        st.subheader('📊 Detailed information about the audio clip')
       
        vstd, vdr, vu, lster_values, frame_size = compute_clip_features(data, rate, frame_ms)
        
        
        
        st.write(f'Volume Standard Deviation (VSTD): {vstd:.4f}')
        st.write(f'Volume Dynamic Range (VDR): {vdr:.4f}')
        st.write(f'Volume Undulation (VU): {vu:.4f}')
       

        st.plotly_chart(display_clip_features(data, rate, frame_ms))
        


        st.text('Save detailed information to CSV file or txt file')
        
        save_csv = st.button('Save to CSV')
        save_txt = st.button('Save to TXT')
        if save_csv:
            #save_to_csv('audio_details.csv', )
            st.success('Data saved to audio_details.csv')
        if save_txt:
            #save_to_csv('audio_details.txt', data_table)
            st.success('Data saved to audio_details.txt')
    
    

