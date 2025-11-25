import os
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import requests
import dotenv
import time

dotenv.load_dotenv('.env')
API_URL = os.environ['API_URL']

# Initialize session state for storing results
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

st.set_page_config(page_title="Animal Classifier", layout="wide")
st.title("🐾 Multi-Model Image Classification")

# Model selection
col1, col2 = st.columns([1, 2])
with col1:
    model = st.selectbox(
        "Select Model",
        ["animals"],
        index=0
    )

with col2:
    versions = st.multiselect(
        "Select Version(s) for Comparison",
        ["v1", "v2"],
        default=["v1", "v2"]
    )

# Add model info and uploaded image in sidebar
with st.sidebar:
    st.header("ℹ️ Model Information")
    if st.button("📊 Show Model Versions", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/models")
            model_info = response.json()
            
            st.success("✅ Model Server Status")
            
            # Display version status
            if 'model_version_status' in model_info:
                st.subheader("Available Versions")
                for version_status in model_info['model_version_status']:
                    version = version_status.get('version', 'Unknown')
                    state = version_status.get('state', 'Unknown')
                    
                    # Use color coding for status
                    if state == 'AVAILABLE':
                        st.success(f"**Version {version}**: {state} ✅")
                    else:
                        st.warning(f"**Version {version}**: {state}")
                    
                    # Show error if exists
                    error_msg = version_status.get('status', {}).get('error_message', '')
                    if error_msg:
                        st.error(f"Error: {error_msg}")
            else:
                st.json(model_info)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    **Model Versions:**
    - **v1**: 3-class (Bird/Cat/Dog)
    - **v2**: Binary (Cat/Dog)
    
    **Features:**
    - Multi-model comparison
    - Redis queue processing
    - W&B inference tracking
    """)
    
    # Display uploaded image in sidebar if available
    if st.session_state.uploaded_image:
        st.markdown("---")
        st.subheader("📷 Uploaded Image")
        st.image(st.session_state.uploaded_image, use_container_width=True)

if not versions:
    st.warning("⚠️ Please select at least one model version")
    st.stop()

st.markdown("""
Upload an image to get predictions from multiple TensorFlow model versions simultaneously.
Results will be compared side-by-side.
""")

# Different models have different output formats
CLASS_LABELS_V1 = ["Bird", "Cat", "Dog"]
CLASS_LABELS_V2 = ["Cat", "Dog"]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.uploaded_image = image

    if st.button("🚀 Run Prediction", type="primary"):
        # Clear previous results
        st.session_state.prediction_results = None
        
        with st.spinner("Submitting prediction task..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {
                "model": model,
                "versions": ",".join(versions)
            }
            
            try:
                # Submit task
                resp = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    params=params,
                    timeout=10
                )
                resp.raise_for_status()
                task_data = resp.json()
                task_id = task_data.get("task_id")
                
                if not task_id:
                    st.error(f"Error: {task_data}")
                    st.stop()
                
                st.success(f"✅ Task queued! ID: `{task_id}`")
                
                # Poll for results
                max_attempts = 30
                attempt = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while attempt < max_attempts:
                    status_text.text(f"Processing... ({attempt + 1}/{max_attempts})")
                    progress_bar.progress((attempt + 1) / max_attempts)
                    
                    result_resp = requests.get(f"{API_URL}/result/{task_id}")
                    result_data = result_resp.json()
                    
                    if result_data.get("status") == "completed":
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success("🎉 Predictions complete!")
                        
                        # Store results in session state
                        st.session_state.prediction_results = result_data.get("results", {})
                        
                        break
                    
                    time.sleep(1)
                    attempt += 1
                
                if attempt >= max_attempts:
                    progress_bar.empty()
                    status_text.empty()
                    st.warning("⏱️ Prediction is taking longer than expected. Task ID: " + task_id)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {e}")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# Display stored results if they exist
if st.session_state.prediction_results:
    results = st.session_state.prediction_results
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    if len(results) > 1:
        # Side-by-side comparison
        cols = st.columns(len(results))
        
        for idx, (version_key, result) in enumerate(sorted(results.items())):
            with cols[idx]:
                st.subheader(f"Model {version_key.upper()}")
                
                if not result.get("success"):
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    continue
                
                preds = np.array(result["predictions"][0])
                
                # Handle different output formats
                if version_key == "v1":
                    CLASS_LABELS = CLASS_LABELS_V1
                    top_idx = int(preds.argmax())
                    top_label = CLASS_LABELS[top_idx]
                    top_prob = preds[top_idx]
                    
                    st.metric("Prediction", top_label, f"{top_prob:.1%}")
                    
                    df = pd.DataFrame({
                        "Class": CLASS_LABELS,
                        "Probability": preds
                    })
                    df["Probability"] = (df["Probability"] * 100).map("{:.2f}%".format)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                else:  # v2
                    CLASS_LABELS = CLASS_LABELS_V2
                    dog_prob = float(preds)
                    cat_prob = 1.0 - dog_prob
                    probs = [cat_prob, dog_prob]
                    top_label = "Dog" if dog_prob > 0.5 else "Cat"
                    top_prob = max(cat_prob, dog_prob)
                    
                    st.metric("Prediction", top_label, f"{top_prob:.1%}")
                    
                    df = pd.DataFrame({
                        "Class": CLASS_LABELS,
                        "Probability": probs
                    })
                    df["Probability"] = (df["Probability"] * 100).map("{:.2f}%".format)
                    st.dataframe(df, use_container_width=True, hide_index=True)
    
    else:
        # Single model result
        for version_key, result in results.items():
            st.subheader(f"Model {version_key.upper()} Results")
            
            if not result.get("success"):
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                continue
            
            preds = np.array(result["predictions"][0])
            
            if version_key == "v1":
                CLASS_LABELS = CLASS_LABELS_V1
                top_idx = int(preds.argmax())
                top_label = CLASS_LABELS[top_idx]
                top_prob = preds[top_idx]
                st.success(f"**{top_label}** with {top_prob:.2%} confidence")
                df = pd.DataFrame({"Class": CLASS_LABELS, "Probability": preds})
            else:
                CLASS_LABELS = CLASS_LABELS_V2
                dog_prob = float(preds)
                cat_prob = 1.0 - dog_prob
                probs = [cat_prob, dog_prob]
                top_label = "Dog" if dog_prob > 0.5 else "Cat"
                top_prob = max(cat_prob, dog_prob)
                st.success(f"**{top_label}** with {top_prob:.2%} confidence")
                df = pd.DataFrame({"Class": CLASS_LABELS, "Probability": probs})
            
            df["Probability"] = (df["Probability"] * 100).map("{:.2f}%".format)
            st.table(df)