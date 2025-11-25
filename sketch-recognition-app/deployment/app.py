import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deployment.inference import SketchPredictor
from deployment.wandb_utils import WandBVisualizer
from model_training.config import TRAIN_CONFIG, WANDB_CONFIG

# Page configuration
st.set_page_config(
    page_title="Sketch Recognition App",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sketch_model.h5')
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.warning(f"Model not found at {model_path}. Please train the model first.")
        return None
    
    try:
        predictor = SketchPredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_wandb_visualizer():
    """Load W&B visualizer"""
    try:
        visualizer = WandBVisualizer(
            #wandb_entity=WANDB_CONFIG['entity'],
            wandb_project=WANDB_CONFIG['project']
        )
        return visualizer
    except Exception as e:
        st.warning(f"Could not connect to W&B: {e}")
        return None

def draw_sketch_canvas():
    """Create a canvas for drawing sketches"""
    try:
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=8,  # Thicker strokes for better recognition
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        return canvas_result
    except ImportError:
        st.error("Please install streamlit-drawable-canvas: `pip install streamlit-drawable-canvas`")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">Sketch Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Draw a sketch and let AI recognize it!</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses a Custom CNN model trained on the QuickDraw dataset 
        to recognize hand-drawn sketches.
        
        **Features:**
        - Real-time sketch recognition
        - 20 different categories
        - W&B metrics visualization
        """)
        
        st.divider()
        
        st.header(" Categories")
        categories = TRAIN_CONFIG['categories']
        for i, cat in enumerate(categories, 1):
            st.write(f"{i}. {cat.title()}")
        
        st.divider()
        
        # Model info
        st.header(" Model Info")
        st.write(f"**Classes:** {len(categories)}")
        st.write("**Architecture:** Custom CNN")
        st.write("**Framework:** TensorFlow/Keras")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header(" Draw Your Sketch")
        
        # Canvas for drawing
        canvas_result = draw_sketch_canvas()
        
        # Control buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            predict_btn = st.button(" Predict", use_container_width=True, type="primary")
        with btn_col2:
            clear_btn = st.button(" Clear", use_container_width=True)
        with btn_col3:
            upload = st.checkbox(" Upload Image")
        
        # File uploader
        uploaded_file = None
        if upload:
            uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    with col2:
        st.header(" Predictions")
        
        # Load model
        predictor = load_model()
        
        if predictor is None:
            st.error("Model not loaded. Please ensure the model file exists.")
            st.info("To train the model, run the training notebook in `model_training/train_sketch_model.ipynb`")
        else:
            # Process image and predict
            image_data = None
            
            if uploaded_file is not None:
                # Use uploaded image
                image = Image.open(uploaded_file).convert('L')
                image_data = np.array(image)
                st.image(image, caption="Uploaded Image", width=280)
            elif canvas_result is not None and canvas_result.image_data is not None:
                # Use canvas drawing
                image_data = canvas_result.image_data
            
            if predict_btn and image_data is not None:
                # Convert to grayscale if needed
                if len(image_data.shape) == 3:
                    # Check if image has meaningful content (not all white)
                    if image_data[:, :, 0].mean() < 250:  # Not blank
                        image_data = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                    else:
                        st.warning(" Canvas is empty. Please draw something!")
                        image_data = None
                
                if image_data is not None:
                    with st.spinner(" Analyzing your sketch..."):
                        try:
                            # Show what the model sees (debug visualization)
                            with st.expander(" Model Input Preview (28x28)"):
                                preprocessed = cv2.resize(image_data, (28, 28))
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(image_data, caption="Your Drawing", width=140)
                                with col2:
                                    st.image(preprocessed, caption="Model Input (28x28)", width=140)
                            
                            predictions = predictor.predict(image_data)
                            
                            # Display results
                            st.success(" Prediction Complete!")
                            
                            # Top prediction with highlight
                            st.markdown("### Top Prediction")
                            top_pred = predictions[0]
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2 style='color: #FF6B6B; margin: 0;'>{top_pred['category'].title()}</h2>
                                <h3 style='color: #4ECDC4; margin: 10px 0 0 0;'>{top_pred['confidence']:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # All predictions
                            st.markdown("###  All Predictions")
                            for i, pred in enumerate(predictions, 1):
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.progress(pred['confidence'] / 100)
                                    st.write(f"**{i}. {pred['category'].title()}**")
                                with col_b:
                                    st.metric("", f"{pred['confidence']:.1f}%")
                        
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
            elif predict_btn:
                st.warning(" Please draw something or upload an image first!")
    
    # W&B Metrics Section
    st.divider()
    st.header(" Training Metrics (Weights & Biases)")
    
    wandb_viz = load_wandb_visualizer()
    
    if wandb_viz is not None:
        try:
            # Get latest run metrics
            history, config, summary = wandb_viz.get_latest_run_metrics()
            
            if history is not None and not history.empty:
                # Display training plots
                fig = wandb_viz.create_training_plots(history)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display summary metrics
                st.subheader(" Final Metrics")
                
                metric_cols = st.columns(4)
                
                metrics_to_show = [
                    ("Training Accuracy", "accuracy", ""),
                    ("Validation Accuracy", "val_accuracy", ""),
                    ("Training Loss", "loss", ""),
                    ("Validation Loss", "val_loss", "")
                ]
                
                for col, (label, key, emoji) in zip(metric_cols, metrics_to_show):
                    with col:
                        if key in summary:
                            value = summary[key]
                            if 'accuracy' in key:
                                st.metric(f"{emoji} {label}", f"{value:.2%}")
                            else:
                                st.metric(f"{emoji} {label}", f"{value:.4f}")
                        else:
                            st.metric(f"{emoji} {label}", "N/A")
                
                # Show config
                if config:
                    with st.expander(" Model Configuration"):
                        st.json(config)
            else:
                st.info("No training history found. Train the model to see metrics here.")
        
        except Exception as e:
            st.warning(f"Could not load W&B metrics: {e}")
            st.info("Make sure you have trained the model and logged metrics to W&B.")
    else:
        st.info("W&B integration not available. Check your W&B configuration in `model_training/config.py`")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Built with  using Streamlit, TensorFlow, and Weights & Biases</p>
        <p>Dataset: Google QuickDraw</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
