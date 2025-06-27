# app.py
import streamlit as st
from utils import load_model, predict_image

st.set_page_config(page_title="Cat vs Dog Classifier 🐱🐶")
st.title("🐶🐱 Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Classifying..."):
        model = load_model("model/binary_classifier.pth")
        label, confidence = predict_image(uploaded_file, model)
        st.success(f"Prediction: **{label}** ({confidence*100:.2f}% confidence)")
