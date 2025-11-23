import streamlit as st
import requests
from PIL import Image

st.title("Cats vs Dogs Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://backend:8000/predict", files=files)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: **{result['prediction']}**")
            st.write(f"Raw score: {result['raw']}")
        else:
            st.error("Prediction failed.")
