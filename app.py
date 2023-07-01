import streamlit as st
import pickle
from src.pipelines.prediction_pipeline import predict

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button('Predict'):
    result = predict(input_sms)  
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
