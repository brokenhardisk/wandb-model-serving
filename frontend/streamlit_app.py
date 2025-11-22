import os
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import requests
import dotenv

dotenv.load_dotenv('.env')
API_URL = os.environ['API_URL']

st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("Animal Image Classifier")

st.markdown("Upload an image to get predictions from the TensorFlow model served via TensorFlow Serving.")
CLASS_LABELS = ["Bird", "Cat", "Dog"] 
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width='stretch')

    if st.button("Run Prediction"):
        with st.spinner("Sending image to backend..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(f"{API_URL}/predict", files=files)
                resp.raise_for_status()
                data = resp.json()
                preds = np.array(data["predictions"][0])
                top_idx = int(preds.argmax())
                top_label = CLASS_LABELS[top_idx]
                top_prob = preds[top_idx]
                st.success(f"Prediction complete! The model predicts it is a {top_label} with {top_prob:.2%} confidence.")
                df = pd.DataFrame({"Class": CLASS_LABELS, "Probability": preds})
                df["Probability"] = (df["Probability"] * 100).map("{:.2f}%".format)
                st.table(df)
            except Exception as e:
                st.error(f"Error during prediction: {e}")