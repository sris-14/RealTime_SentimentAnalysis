import streamlit as st 
import cv2
import numpy as np 
from PIL import Image
from tensorflow.keras.models import load_model
model = load_model('sequential_saved_model.h5')


def live_capture_pg(model):
    st.title("Live Capture Analysis")
    st.write("Capture and analyze real-time video or images.")
    

    

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)  # Open webcam

    run = st.checkbox("Start Camera")
    if run:
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame. Please check your camera.")
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            # FRAME_WINDOW.image(frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame_gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.reshape(1, 48, 48, 1) / 255.0
                prediction = model.predict(face)
                label = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"][np.argmax(prediction)]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
            FRAME_WINDOW.image(frame_rgb)
            
    else:
        camera.release()
        st.write('Camera stopped!')
        
live_capture_pg(model)        