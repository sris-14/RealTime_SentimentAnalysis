
import streamlit as st 


def home_pg():
    st.title("Welcome to the Real-Time Sentiment Analysis WebApp!")

    st.write("""
              This application performs sentiment analysis using visual content.
        - **Home Page:** Overview of the application.
        - **Stored Data:** Analyze previously stored data for sentiment.
        - **Live Capture:** Perform real-time analysis using your webcam.
            """)
       
    st.write("[CODE LINK](https://github.com/sris-14/RealTime_SentimentAnalysis)")


home_pg()