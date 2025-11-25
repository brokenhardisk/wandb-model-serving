import os
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import dotenv
import time
import base64
import io

dotenv.load_dotenv('.env')
API_URL = os.environ['API_URL']

# Initialize session state
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

st.set_page_config(page_title="App", layout="wide")
st.title("Multi-Model AI Prediction Platform")

# Model selection
model_type = st.selectbox(
    "Select Model Type",
    ["animals", "sketch"],
    index=0,
    help="Choose between animal classification or sketch recognition"
)

# Sketch categories
SKETCH_CATEGORIES = [
    'apple', 'banana', 'cat', 'dog', 'house',
    'tree', 'car', 'fish', 'bird', 'clock',
    'book', 'chair', 'cup', 'star', 'heart',
    'smiley face', 'sun', 'moon', 'key', 'hammer'
]

# Different models have different output formats
CLASS_LABELS_V1 = ["Bird", "Cat", "Dog"]
CLASS_LABELS_V2 = ["Cat", "Dog"]

# ====================
# ANIMALS MODEL UI
# ====================
if model_type == "animals":
    # Version selector for animals
    versions = st.multiselect(
        "Select Version(s) for Comparison",
        ["v1", "v2"],
        default=["v1", "v2"]
    )
    
    # Sidebar for animals model
    with st.sidebar:
        st.header("ℹ️ Animal Model Information")
        if st.button("📊 Show Model Versions", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/models")
                model_info = response.json()
                
                st.success("Model Server Status")
                
                if 'model_version_status' in model_info:
                    st.subheader("Available Versions")
                    for version_status in model_info['model_version_status']:
                        version = version_status.get('version', 'Unknown')
                        state = version_status.get('state', 'Unknown')
                        
                        if state == 'AVAILABLE':
                            st.success(f"**Version {version}**: {state} ✅")
                        else:
                            st.warning(f"**Version {version}**: {state} ❌")
                        
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
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        
        if st.button("🚀 Run Prediction", type="primary"):
            st.session_state.prediction_results = None
            
            with st.spinner("Submitting prediction task..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {
                    "model": "animals",
                    "versions": ",".join(versions)
                }
                
                try:
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
                            st.session_state.prediction_results = result_data.get("results", {})
                            break
                        
                        time.sleep(1)
                        attempt += 1
                    
                    if attempt >= max_attempts:
                        progress_bar.empty()
                        status_text.empty()
                        st.warning(f"⏱️ Prediction is taking longer than expected.")
                        st.info(f"**Task ID:** `{task_id}`")
                        st.markdown("""**Your prediction is still being processed.** You can:
                        - Wait and click the button below to check status
                        - The result will be available for 1 hour after completion
                        """)
                        
                        if st.button("🔄 Check Status Now", key="retry_animal"):
                            with st.spinner("Checking status..."):
                                result_resp = requests.get(f"{API_URL}/result/{task_id}")
                                result_data = result_resp.json()
                                if result_data.get("status") == "completed":
                                    st.session_state.prediction_results = result_data.get("results", {})
                                    st.success("🎉 Predictions complete!")
                                    time.sleep(1)  # Brief pause to show success message
                                    st.rerun()
                                else:
                                    st.info("Still processing... Please try again in a few seconds.")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {e}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    
    # Display animal results
    if st.session_state.prediction_results:
        results = st.session_state.prediction_results
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        if len(results) > 1:
            cols = st.columns(len(results))
            
            for idx, (version_key, result) in enumerate(sorted(results.items())):
                with cols[idx]:
                    st.subheader(f"Model {version_key.upper()}")
                    
                    if not result.get("success"):
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                        continue
                    
                    preds = np.array(result["predictions"][0])
                    
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

# ====================
# SKETCH MODEL UI
# ====================
else:  # model_type == "sketch"
    with st.sidebar:
        st.header("ℹ️ Sketch Model Information")
        st.markdown("""
        **Model:** Sketch Recognition CNN  
        **Dataset:** Google QuickDraw  
        **Categories:** 20 classes  
        **Input:** 28x28 grayscale  
        
        **Features:**  
        - Real-time drawing  
        - Top-5 predictions  
        - W&B tracking
        """)
        
        st.markdown("---")
        st.header("📝 Supported Categories")
        cols = st.columns(2)
        for i, cat in enumerate(SKETCH_CATEGORIES):
            with cols[i % 2]:
                st.write(f"• {cat.title()}")
    
    st.markdown("""
    Draw a sketch in the canvas below and click Predict to get Model predictions!
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✏️ Draw Your Sketch")
        
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=20,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if st.button("🚀 Predict Sketch", type="primary"):
            if canvas_result.image_data is not None:
                # Check if canvas has any drawing
                if np.sum(canvas_result.image_data[:,:,3]) > 0:  # Check alpha channel
                    st.session_state.prediction_results = None
                    
                    with st.spinner("Submitting sketch prediction task..."):
                        # Convert canvas to image
                        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                        img = img.convert('RGB')
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        try:
                            # Submit to backend
                            resp = requests.post(
                                f"{API_URL}/predict/sketch",
                                json={"image": img_b64},
                                timeout=10
                            )
                            resp.raise_for_status()
                            task_data = resp.json()
                            task_id = task_data.get("task_id")
                            
                            if not task_id:
                                st.error(f"Error: {task_data}")
                                st.stop()
                            
                            st.success(f"Task queued! ID: `{task_id}`")
                            
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
                                    st.session_state.prediction_results = result_data.get("results", {})
                                    break
                                
                                time.sleep(1)
                                attempt += 1
                            
                            if attempt >= max_attempts:
                                progress_bar.empty()
                                status_text.empty()
                                st.warning(f"⏱️ Sketch prediction is taking longer than expected.")
                                st.info(f"**Task ID:** `{task_id}`")
                                st.markdown("""**Your sketch is still being processed.** You can:
                                - Wait and click the button below to check status
                                - The result will be available for 1 hour after completion
                                """)
                                
                                if st.button("🔄 Check Status Now", key="retry_sketch"):
                                    with st.spinner("Checking status..."):
                                        result_resp = requests.get(f"{API_URL}/result/{task_id}")
                                        result_data = result_resp.json()
                                        if result_data.get("status") == "completed":
                                            st.session_state.prediction_results = result_data.get("results", {})
                                            st.success("🎉 Predictions complete!")
                                            time.sleep(1)  # Brief pause to show success message
                                            st.rerun()
                                        else:
                                            st.info("Still processing... Please try again in a few seconds.")
                                
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Network error: {e}")
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {e}")
                else:
                    st.warning("⚠️ Please draw something on the canvas first!")
            else:
                st.warning("⚠️ Canvas is empty. Please draw something first!")
    
    with col2:
        st.subheader("📊 Results")
        if st.session_state.prediction_results and 'sketch' in st.session_state.prediction_results:
            result = st.session_state.prediction_results['sketch']
            
            if result.get('success'):
                predictions = result.get('predictions', [])
                
                if predictions:
                    st.metric("Top Prediction", predictions[0]['category'].upper(), 
                             f"{predictions[0]['confidence']:.1f}%")
                    
                    st.subheader("Top 5 Predictions:")
                    for i, pred in enumerate(predictions[:5], 1):
                        st.write(f"{i}. **{pred['category'].title()}** - {pred['confidence']:.1f}%")
                        st.progress(pred['confidence'] / 100)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            st.info("Draw a sketch and click predict to see results here")
