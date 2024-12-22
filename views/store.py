import streamlit as st 
import cv2
from PIL import Image  
# from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('sequential_saved_model.h5')

def stored_data_pg(model):
    st.title("Stored Data Analysis")
    st.write("Upload and analyze stored data here.")

    file = st.file_uploader("Choose a file (Image or MP4)...", type=["jpg", "jpeg", "png", "mp4"])
    if file is not None:
            file_type = file.type
            if "image" in file_type:
                # Display and analyze image
                st.image(file, caption="Uploaded Image", use_column_width=True)
                st.write("Analyzing sentiment for the image...")
                
                # Convert image to grayscale and predict
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (48, 48)).reshape(1, 48, 48, 1) / 255.0
                prediction = model.predict(img_resized)
                sentiment = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"][np.argmax(prediction)]
                st.write(f"Sentiment: {sentiment}")
                
            elif "mp4" in file_type:
                # Display and analyze video
                st.video(file)
                st.write("Analyzing sentiment for the video...")
                # Process the video frame by frame (example below)
                # analyze_video(file)
                
                # Process the video frame by frame
                temp_file = "temp_video.mp4"
                with open(temp_file, "wb") as f:
                    f.write(file.getbuffer())
    
                cap = cv2.VideoCapture(temp_file)
                st.write("Processing video...")
    
    
                 # Ensure the video was opened successfully
                if not cap.isOpened():
                    st.error("Failed to open video file.")
                    return

                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_resized = cv2.resize(frame_gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0
                        prediction = model.predict(frame_resized)
                        sentiment = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"][np.argmax(prediction)]
                        st.image(frame, caption=f"Frame Sentiment: {sentiment}", use_column_width=True)
                finally:
                    

                    cap.release()
                    st.write("Video processing complete.")
    
stored_data_pg(model)
# analyze_video()    