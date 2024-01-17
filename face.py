import cv2
import streamlit as st
from deepface import DeepFace

# Fungsi untuk memproses video dan mengambil frame
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

# Fungsi untuk menganalisis ekspresi wajah
def analyze_expression(frames):
    results = []

    for frame in frames:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            results.append(result)
        except Exception as e:
            print("Error in frame analysis:", e)

    return results

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Analisis Ekspresi Wajah")

    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    if uploaded_file is not None:
        with st.spinner('Mengolah video...'):
            frames = process_video(uploaded_file)
        
        with st.spinner('Menganalisis ekspresi...'):
            results = analyze_expression(frames)

        st.success('Analisis Selesai!')

        for result in results:
            st.write(result)

if __name__ == '__main__':
    main()
