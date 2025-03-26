import io
import streamlit as st
import soundfile as sf
import pandas as pd
from audio_processing import load_audio, plot_waveform, detect_silence, plot_silence, plot_f0, plot_voiced_unvoiced, compute_f0_autocorrelation, save_to_csv
from audio_processing import  normalize_audio
from audio_processing import  plot_loudness, plot_short_time_energy, plot_zero_crossing_rate, estimate_f0_amdf
from clip_audio_processing import plot_hzcrr, plot_lster, plot_speach_music, vstd, vdr, vu, energy_entropy

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



    zcr = st.sidebar.toggle('Zero Crossing Rate (ZCR)', False)
    if zcr:
        st.subheader('🔀 Zero Crossing Rate (ZCR)')
        st.plotly_chart(plot_zero_crossing_rate(data, rate, frame_ms))

    silence = st.sidebar.toggle('Silence detection', False)
    if silence:
        percentage = st.sidebar.slider('Threshold (%)', 1, 10, 5)
        zcr_threshold = st.sidebar.slider('ZCR threshold', 0.1, 1.0, 0.3)
        silence_frames, frame_size = detect_silence(data, rate, frame_ms, percentage, zcr_threshold)
        st.subheader('🔇 Silence Detection')
        st.plotly_chart(plot_silence(data, rate, silence_frames, frame_size))
    
    loudness = st.sidebar.toggle('Loudness', False)
    if loudness:
        st.subheader('🔊 Loudness')
        st.plotly_chart(plot_loudness(data, rate, frame_ms))

    ste = st.sidebar.toggle('Short-Time Energy (STE)', False)
    if ste:
        st.subheader('🔋 Short-Time Energy (STE)')
        st.plotly_chart(plot_short_time_energy(data, rate, frame_ms))

    speach_music = st.sidebar.toggle('Speech/Music detection', False)
    if speach_music:
        st.subheader('🎤🎵 Speech/Music detection')
        st.plotly_chart(plot_speach_music(data, rate, frame_ms))

    
    

    
    fundamental_frequency = st.sidebar.toggle('Fundamental Frequency (F0)', False)
    

    #f0_values, frame_size = compute_f0_amdf(data, rate, frame_ms)
    if fundamental_frequency:
        # choose method
        st.sidebar.write('Choose method:')
        option = st.sidebar.selectbox('Method:', ['Autocorrelation', 'AMDF'])

        if option == 'Autocorrelation':
            f0_values, frame_size = compute_f0_autocorrelation(data, rate, frame_ms)
        elif option == 'AMDF':
            f0_values, frame_size = estimate_f0_amdf(data, rate, frame_ms)
        st.subheader('🎶 Fundamental Frequency (F0)')
   
        st.plotly_chart(plot_f0(f0_values, frame_size, rate))
    
    voiced_unvoiced = st.sidebar.toggle('Voiced/Unvoiced detection', False)
    if voiced_unvoiced:
        #voiced = compute_voiced_unvoiced(f0_values)
        ste_threshold = st.sidebar.slider('STE threshold ',0.01, 0.1, 0.05)
        zcr_treshold = st.sidebar.slider('ZCR threshold', 0.05, 0.5, 0.2)
        st.subheader('🎤 Voiced/Unvoiced')
        st.plotly_chart(plot_voiced_unvoiced(data, rate, frame_ms, ste_threshold, zcr_treshold))

    st.sidebar.header('Detailed information:')  
    details = st.sidebar.checkbox('Show detailed information about the audio clip', False)
    # length of audio
    time = len(data) / rate
    st.write(f'🕒 **Czas trwania:** {time:.2f} s')
    

    if details and time > 2:
        st.subheader('📊 Detailed information about the audio clip')
        vstd_value = vstd(data, rate)
        vdr_value = vdr(data, rate)
        vu_value = vu(data, rate)
        entropy = energy_entropy(data, rate)
        data_table = pd.DataFrame({
            'Volume Standard Deviation (VSTD)': [vstd_value],
            'Volume Dynamic Range (VDR)': [vdr_value],
             'Volume Undulation (VU)': [vu_value],
            'Energy Entropy': [entropy]
        })
        st.table(data_table)
        st.text('Save detailed information to CSV file or txt file')
        
        file_format = st.radio("Wybierz format pliku do zapisania:", ('.csv', '.txt'))

# Przycisk do zapisu tabeli
        if st.button("📥 Pobierz tabelę"):
            # Zapisujemy tabelę w wybranym formacie
            file_name = f"audio_analysis_results{file_format}"
            save_to_csv(file_name, data_table)
            st.success(f"Plik {file_name} został zapisany!")
            with open(file_name, "rb") as file:
                st.download_button(label="Kliknij, aby pobrać plik", data=file, file_name=file_name, mime="text/csv" if file_format == '.csv' else "text/plain")
        st.plotly_chart(plot_lster(data, rate, frame_ms))
        st.plotly_chart(plot_hzcrr(data, rate, frame_ms))
       

    

        
    elif details and time <= 2:
        st.error('Audio clip is too short to calculate detailed information')
       
      
        


    
        

