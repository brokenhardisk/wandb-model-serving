"""
Sketch Recognition Model Deployment using Modal
Python 3.12+ compatible deployment configuration for sketch recognition application

Features:
- FastAPI REST API for predictions
- Streamlit web interface
- Automatic model downloading from W&B
- Batch prediction support
- Health monitoring and validation
- Type-safe with modern Python features
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

# ============================================================================
# CONFIGURATION
# ============================================================================

MODAL_CONFIG = {
    "name": "sketch-recognition",
    "python_version": "3.12",
    "timeout": 600,
    "container_idle_timeout": 300,
}

# Create Modal app (modern API, replaces deprecated Stub)
app = modal.App(MODAL_CONFIG["name"])

# ============================================================================
# CONTAINER IMAGE DEFINITION
# ============================================================================

sketch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .pip_install(
        # Core dependencies with Python 3.12 compatibility
        "tensorflow==2.17.0",  # Updated for Python 3.12 (2.15 not available)
        "numpy==1.26.4",
        "opencv-python-headless==4.9.0.80",
        "scikit-learn==1.5.0",  # Updated for Python 3.12 compatibility
        # Web frameworks
        "streamlit==1.32.0",
        "streamlit-drawable-canvas==0.9.3",
        "fastapi==0.109.2",
        "uvicorn[standard]==0.27.1",
        "pydantic==2.6.1",
        # Visualization and tracking
        "plotly==5.19.0",
        "pandas==2.2.1",
        "wandb==0.16.3",
        # Utilities
        "requests==2.31.0",
        "Pillow==10.2.0",
    )
)

# Persistent storage for models
model_volume = modal.Volume.from_name("sketch-models", create_if_missing=True)
model_nfs = modal.NetworkFileSystem.from_name("sketch-model-nfs", create_if_missing=True)

# ============================================================================
# MODEL MANAGEMENT FUNCTIONS
# ============================================================================


@app.function(
    image=sketch_image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=600,
    retries=modal.Retries(
        max_retries=3,
        initial_delay=1.0,
        backoff_coefficient=2.0,
    ),
)
def download_model(model_path: str = "/models/sketch_model.h5") -> dict[str, Any]:
    """Download model from W&B with robust error handling
    
    Args:
        model_path: Path where model should be saved
        
    Returns:
        Dictionary with status and path information
    """
    import wandb

    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if model_path_obj.exists():
        size_mb = model_path_obj.stat().st_size / (1024**2)
        print(f"Model already exists at {model_path} ({size_mb:.2f} MB)")
        return {
            "status": "exists",
            "path": str(model_path),
            "size_mb": round(size_mb, 2)
        }
    
    # Try to download from W&B
    try:
        print(" Downloading model from W&B...")
        wandb.login()
        
        # Initialize W&B run
        with wandb.init(project="sketch-recognition", job_type="model-download") as run:
            # Replace with your actual artifact path
            artifact = run.use_artifact('sketch-recognition/sketch_model:latest', type='model')
            artifact_dir = artifact.download(root=str(model_path_obj.parent))
            
            # Find and move the model file
            for file in Path(artifact_dir).rglob("*.h5"):
                file.rename(model_path_obj)
                size_mb = model_path_obj.stat().st_size / (1024**2)
                print(f" Model downloaded to {model_path} ({size_mb:.2f} MB)")
                return {
                    "status": "downloaded",
                    "path": str(model_path),
                    "size_mb": round(size_mb, 2),
                }
            
            raise FileNotFoundError("No .h5 model file found in artifact")
                
    except Exception as e:
        error_msg = f"Failed to download from W&B: {e}"
        print(f"{error_msg}")
        print(" Hint: Upload model manually or update artifact path")
        return {
            "status": "error",
            "error": error_msg,
            "path": str(model_path)
        }


@app.function(
    image=sketch_image,
    volumes={"/models": model_volume},
    timeout=600,
)
def upload_model(local_path: str = "./models/sketch_model.h5") -> dict[str, Any]:
    """Upload a local model to Modal volume with validation
    
    Args:
        local_path: Path to local model file
        
    Returns:
        Upload status information
    """
    import shutil
    import tensorflow as tf

    # Note: local_path here refers to a path that will be accessible 
    # inside the Modal container, not your local machine!
    # We need to read the file and pass it as bytes
    
    print(f"  Note: This function needs to be called differently!")
    print(f"   Use: modal run model-deploy.py::upload_model_from_local")
    
    local_path = Path(local_path)
    dest_path = Path("/models/sketch_model.h5")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not local_path.exists():
        error_msg = f"Model file not found at {local_path} inside container"
        print(f" {error_msg}")
        return {"status": "error", "error": error_msg}
    
    # Copy to volume
    print(f" Copying model to Modal volume...")
    shutil.copy(str(local_path), str(dest_path))
    
    # Validate
    try:
        model = tf.keras.models.load_model(str(dest_path))
        size_mb = dest_path.stat().st_size / (1024**2)
        print(f" Model uploaded successfully ({size_mb:.2f} MB)")
        
        # Commit volume changes
        model_volume.commit()
        
        return {
            "status": "success",
            "path": str(dest_path),
            "size_mb": round(size_mb, 2),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }
    except Exception as e:
        return {"status": "error", "error": f"Model validation failed: {e}"}


# NEW: Function to upload from your local machine
@app.function(
    image=sketch_image,
    volumes={"/models": model_volume},
    timeout=600,
)
def upload_model_from_bytes(model_bytes: bytes) -> dict[str, Any]:
    """Upload model from bytes to Modal volume
    
    Args:
        model_bytes: Model file as bytes
        
    Returns:
        Upload status
    """
    import tensorflow as tf
    
    dest_path = Path("/models/sketch_model.h5")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write bytes to file
    print(f" Writing model to Modal volume...")
    with open(dest_path, "wb") as f:
        f.write(model_bytes)
    
    # Validate
    try:
        model = tf.keras.models.load_model(str(dest_path))
        size_mb = dest_path.stat().st_size / (1024**2)
        print(f" Model uploaded successfully ({size_mb:.2f} MB)")
        
        # Commit volume changes (important!)
        model_volume.commit()
        
        return {
            "status": "success",
            "path": str(dest_path),
            "size_mb": round(size_mb, 2),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }
    except Exception as e:
        return {"status": "error", "error": f"Model validation failed: {e}"}


@app.function(
    image=sketch_image,
    volumes={"/models": model_volume},
    timeout=120,
)
def check_model() -> dict[str, Any]:
    """Check if model exists in volume and validate it
    
    Returns:
        Model status information
    """
    import tensorflow as tf

    model_path = Path("/models/sketch_model.h5")
    exists = model_path.exists()
    
    result = {"path": str(model_path), "exists": exists}
    
    if exists:
        size = model_path.stat().st_size
        size_mb = size / (1024**2)
        result["size_mb"] = round(size_mb, 2)
        print(f" Model exists: {model_path}")
        print(f" Model size: {size_mb:.2f} MB")
        
        # Try to load model
        try:
            model = tf.keras.models.load_model(str(model_path))
            result["valid"] = True
            result["input_shape"] = str(model.input_shape)
            result["output_shape"] = str(model.output_shape)
            print(f" Model is valid: {model.input_shape} -> {model.output_shape}")
        except Exception as e:
            result["valid"] = False
            result["error"] = str(e)
            print(f" Model validation failed: {e}")
    else:
        print(f" Model not found at {model_path}")
        result["valid"] = False
    
    return result

# ============================================================================
# APPLICATION SETUP
# ============================================================================


@app.function(
    image=sketch_image,
    volumes={"/models": model_volume},
    network_file_systems={"/cache": model_nfs},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=900,
)
def setup_application() -> dict[str, Any]:
    """Setup the application environment with health checks
    
    Returns:
        Status dictionary with setup results
    """
    print(" Setting up application environment...")
    
    # Verify Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python version: {python_version}")
    
    # Download model if needed
    model_status = download_model.remote()
    
    # Verify model exists
    model_path = Path("/models/sketch_model.h5")
    if not model_path.exists():
        return {
            "status": "error",
            "message": "Model file not found after setup",
            "python_version": python_version,
        }
    
    print(" Application setup complete")
    return {
        "status": "success",
        "python_version": python_version,
        "model_status": model_status,
        "model_exists": True,
    }

# ============================================================================
# FASTAPI REST API
# ============================================================================


@app.function(
    image=sketch_image,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=300,
    max_containers=100,  # Renamed from concurrency_limit
    min_containers=1,    # Renamed from keep_warm
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI app for REST API predictions with validation and error handling"""
    import cv2
    import numpy as np
    import tensorflow as tf
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    # Inline SketchPredictor class (since we can't import from local files)
    class SketchPredictor:
        def __init__(self, model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.categories = [
                'apple', 'banana', 'cat', 'dog', 'house',
                'tree', 'car', 'fish', 'bird', 'clock',
                'book', 'chair', 'cup', 'star', 'heart',
                'smiley face', 'sun', 'moon', 'key', 'hammer'
            ]
            self.img_size = 28
        
        def preprocess_sketch(self, image):
            """Preprocess sketch image for prediction"""
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            image = image.astype(np.uint8)
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            image = image.reshape(1, self.img_size, self.img_size, 1)
            
            return image
        
        def predict(self, image):
            """Predict sketch category"""
            processed_image = self.preprocess_sketch(image)
            predictions = self.model.predict(processed_image, verbose=0)
            
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = [
                {
                    'category': self.categories[i],
                    'confidence': float(predictions[0][i]) * 100
                }
                for i in top_indices
            ]
            
            return top_predictions
    
    # Initialize FastAPI app first
    api = FastAPI(
        title="Sketch Recognition API",
        version="2.0.0",
        description="High-performance sketch recognition using CNN model trained on QuickDraw dataset",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure specific origins in production
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Global exception handler
    @api.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions"""
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
                "type": type(exc).__name__
            },
        )
    
    # Load model with error handling
    predictor = None
    try:
        model_path = Path("/models/sketch_model.h5")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        predictor = SketchPredictor(str(model_path))
        print(f" Model loaded successfully from {model_path}")
    except Exception as e:
        print(f" Failed to load model: {e}")
        predictor = None
    
    # ========================================================================
    # API ENDPOINTS
    # ========================================================================
    
    @api.get("/", tags=["Status"])
    async def root() -> dict[str, str]:
        """Root endpoint with API information"""
        return {
            "message": "Sketch Recognition API",
            "version": "2.0.0",
            "status": "running",
            "model_loaded": "yes" if predictor else "no",
            "docs": "/docs",
        }
    
    @api.get("/health", tags=["Status"])
    async def health_check() -> dict[str, Any]:
        """Health check endpoint for monitoring"""
        return {
            "status": "healthy" if predictor else "degraded",
            "model_loaded": predictor is not None,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "categories_count": len(predictor.categories) if predictor else 0,
            "timestamp": time.time(),
        }
    
    @api.post("/predict", tags=["Prediction"])
    async def predict_sketch(image_data: dict[str, str]) -> dict[str, Any]:
        """Predict sketch category from base64 encoded image
        
        Args:
            image_data: Dict with 'image' key containing base64 encoded image
            
        Returns:
            Prediction results with confidence scores and processing time
            
        Raises:
            HTTPException: If model not loaded, invalid image, or prediction fails
        """
        start_time = time.time()
        
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please contact administrator."
            )
        
        try:
            # Get image from dict
            if "image" not in image_data:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'image' field in request body"
                )
            
            # Decode base64 image
            image_b64 = image_data["image"].split(",")[-1] if "," in image_data["image"] else image_data["image"]
            
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 encoding: {e}"
                )
            
            # Decode image
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to decode image. Ensure image is valid PNG/JPEG."
                )
            
            # Make prediction
            predictions = predictor.predict(image)
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "success": True,
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None,
                "processing_time_ms": round(processing_time, 2),
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )
    
    @api.get("/categories", tags=["Info"])
    async def get_categories() -> dict[str, Any]:
        """Get list of supported sketch categories"""
        if predictor:
            return {
                "categories": predictor.categories,
                "count": len(predictor.categories)
            }
        
        # Fallback to config
        try:
            sys.path.insert(0, "/app")
            from model_training.config import TRAIN_CONFIG
            categories = TRAIN_CONFIG["categories"]
            return {"categories": categories, "count": len(categories)}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load categories: {e}"
            )
    
    @api.get("/model-info", tags=["Info"])
    async def get_model_info() -> dict[str, Any]:
        """Get detailed model information"""
        if not predictor:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        return {
            "input_shape": (28, 28, 1),
            "num_classes": len(predictor.categories),
            "categories": predictor.categories,
            "model_path": "./models/sketch_model.h5",
            "framework": "TensorFlow/Keras",
        }
    
    return api

# ============================================================================
# STREAMLIT WEB INTERFACE
# ============================================================================


@app.function(
    image=sketch_image,
    volumes={"/models": model_volume},
    timeout=86400,
    max_containers=10,
    min_containers=1,
)
@modal.web_server(8501, startup_timeout=180)
def streamlit_app():
    """Deploy Streamlit UI as a web service on Modal"""
    import subprocess
    import sys
    from pathlib import Path
    
    # Create streamlit app file inline
    streamlit_code = '''
import streamlit as st
import streamlit.components.v1 as components
import requests

# Page configuration
st.set_page_config(
    page_title="Sketch Recognition App",
    page_icon="><",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI endpoint
API_URL = "https://m024uma--sketch-recognition-fastapi-app.modal.run"

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
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header"> Sketch Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Draw a sketch and get instant AI predictions</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ About")
        st.write("""
        AI-powered sketch recognition using a CNN model trained on the QuickDraw dataset.
        
        **How to use:**
        1. Draw a sketch in the canvas
        2. Click Predict
        3. See results instantly
        4. Click Clear to start over
        """)
        
        st.divider()
        
        st.header(" Supported Categories")
        categories = [
            'apple', 'banana', 'cat', 'dog', 'house',
            'tree', 'car', 'fish', 'bird', 'clock',
            'book', 'chair', 'cup', 'star', 'heart',
            'smiley face', 'sun', 'moon', 'key', 'hammer'
        ]
        
        cols = st.columns(2)
        for i, cat in enumerate(categories):
            with cols[i % 2]:
                st.write(f"• {cat.title()}")
        
        st.divider()
        
        st.header(" Model Info")
        st.write("**Framework:** TensorFlow/Keras")
        st.write("**Classes:** 20 categories")
        st.write("**Platform:** Modal Serverless")
    
    # Main content - Full width canvas and predictions below
    st.header(" Draw Your Sketch")
    
    # HTML Canvas with direct API call
    canvas_html = f"""
    <div style="text-align: center;">
        <canvas id="canvas" width="400" height="400" style="border: 2px solid #333; cursor: crosshair; background: white;"></canvas>
        <br><br>
        <button onclick="clearCanvas()" style="padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; background: #ff4444; color: white; border: none; border-radius: 5px;">Clear Canvas</button>
        <button onclick="predictSketch()" style="padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 5px;"> Predict</button>
        <div id="status" style="margin-top: 20px; font-size: 14px; color: #666;"></div>
        <div id="results" style="margin-top: 20px; text-align: left; max-width: 400px; margin-left: auto; margin-right: auto;"></div>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        
        // Set up canvas
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Mouse events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch events
        canvas.addEventListener('touchstart', handleTouchStart);
        canvas.addEventListener('touchmove', handleTouchMove);
        canvas.addEventListener('touchend', stopDrawing);
        
        function startDrawing(e) {{
            drawing = true;
            const rect = canvas.getBoundingClientRect();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        }}
        
        function draw(e) {{
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
        }}
        
        function stopDrawing() {{
            drawing = false;
        }}
        
        function handleTouchStart(e) {{
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            drawing = true;
            ctx.beginPath();
            ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
        }}
        
        function handleTouchMove(e) {{
            e.preventDefault();
            if (!drawing) return;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
            ctx.stroke();
        }}
        
        function clearCanvas() {{
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('results').innerHTML = '';
            document.getElementById('status').innerHTML = '';
        }}
        
        async function predictSketch() {{
            const statusDiv = document.getElementById('status');
            const resultsDiv = document.getElementById('results');
            
            statusDiv.innerHTML = ' Analyzing your sketch...';
            resultsDiv.innerHTML = '';
            
            try {{
                // Get canvas data as base64
                const imageData = canvas.toDataURL('image/png').split(',')[1];
                
                // Call FastAPI endpoint
                const response = await fetch('{API_URL}/predict', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ image: imageData }})
                }});
                
                if (response.ok) {{
                    const data = await response.json();
                    statusDiv.innerHTML = ' Prediction Complete!';
                    
                    // Display results
                    let html = '<div style="background: #f0f2f6; padding: 20px; border-radius: 10px;">';
                    html += '<h3 style="color: #FF6B6B; margin-top: 0;"> Top Prediction</h3>';
                    html += `<h2 style="margin: 10px 0;">${{data.predictions[0].category.toUpperCase()}}</h2>`;
                    html += `<p style="font-size: 18px; color: #4CAF50;">Confidence: ${{data.predictions[0].confidence.toFixed(1)}}%</p>`;
                    html += '<hr style="margin: 20px 0;">';
                    html += '<h4> All Predictions:</h4>';
                    
                    data.predictions.slice(0, 5).forEach((pred, i) => {{
                        html += `<div style="margin: 10px 0;">`;
                        html += `<strong>${{i+1}}. ${{pred.category.charAt(0).toUpperCase() + pred.category.slice(1)}}</strong>: ${{pred.confidence.toFixed(1)}}%`;
                        html += `<div style="background: #ddd; height: 10px; border-radius: 5px; margin-top: 5px;"><div style="background: #4CAF50; height: 10px; border-radius: 5px; width: ${{pred.confidence}}%;"></div></div>`;
                        html += `</div>`;
                    }});
                    
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                }} else {{
                    statusDiv.innerHTML = ' Error: ' + response.status;
                    resultsDiv.innerHTML = '<p style="color: red;">Failed to get prediction. Please try again.</p>';
                }}
            }} catch (error) {{
                statusDiv.innerHTML = ' Error occurred';
                resultsDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
            }}
        }}
    </script>
    """
    
    # Display the canvas
    components.html(canvas_html, height=700, scrolling=False)
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Built with ❤️ using Streamlit, TensorFlow, and Modal</p>
        <p>Dataset: Google QuickDraw</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''
    
    # Write Streamlit app to file
    app_path = Path("/tmp/streamlit_app.py")
    with open(app_path, "w") as f:
        f.write(streamlit_code)
    
    # Create Streamlit config to handle file uploads properly
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.toml"
    
    config_content = """
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 10

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false
fastReruns = true
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    # Run Streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]
    
    subprocess.Popen(cmd)

# ============================================================================
# BATCH PREDICTION
# ============================================================================


@app.function(
    image=sketch_image,
    volumes={"/models": model_volume},
    timeout=300,
)
def batch_predict(images: list[str]) -> list[dict[str, Any]]:
    """Predict multiple sketches in batch for efficiency
    
    Args:
        images: List of base64 encoded images
        
    Returns:
        List of prediction results
    """
    import cv2
    import numpy as np
    import tensorflow as tf

    # Inline SketchPredictor class
    class SketchPredictor:
        def __init__(self, model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.categories = [
                'apple', 'banana', 'cat', 'dog', 'house',
                'tree', 'car', 'fish', 'bird', 'clock',
                'book', 'chair', 'cup', 'star', 'heart',
                'smiley face', 'sun', 'moon', 'key', 'hammer'
            ]
            self.img_size = 28
        
        def preprocess_sketch(self, image):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.astype(np.uint8)
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            image = image.reshape(1, self.img_size, self.img_size, 1)
            return image
        
        def predict(self, image):
            processed_image = self.preprocess_sketch(image)
            predictions = self.model.predict(processed_image, verbose=0)
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            return [
                {'category': self.categories[i], 'confidence': float(predictions[0][i]) * 100}
                for i in top_indices
            ]
    
    predictor = SketchPredictor("/models/sketch_model.h5")
    results = []
    
    for idx, img_data in enumerate(images):
        try:
            # Decode image
            img_b64 = img_data.split(",")[-1] if "," in img_data else img_data
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": "Failed to decode image"
                })
                continue
            
            # Predict
            predictions = predictor.predict(image)
            results.append({
                "index": idx,
                "success": True,
                "predictions": predictions
            })
            
        except Exception as e:
            results.append({
                "index": idx,
                "success": False,
                "error": str(e)
            })
    
    return results

# ============================================================================
# CLI ENTRYPOINT
# ============================================================================


@app.local_entrypoint()
def main(action: str = "help", model_path: str = "./models/sketch_model.h5") -> None:
    """Local entrypoint for testing and deployment
    
    Args:
        action: Action to perform (setup, deploy, test, check-model, upload-model)
        model_path: Path to local model file (for upload-model action)
    """
    if action == "setup":
        print(" Setting up application environment...")
        result = setup_application.remote()
        print(f"\n Setup result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
    elif action == "deploy":
        print(" Deploying to Modal...")
        print("\n Run one of these commands:")
        print("  modal deploy model-deploy.py  # Deploy all services")
        print("  modal serve model-deploy.py   # Deploy in dev mode")
        print("\n After deployment, you'll get URLs for:")
        print("  - FastAPI: https://TBT--sketch-recognition-fastapi-app.modal.run")
        print("  - Streamlit: https://TBT--sketch-recognition-streamlit-app.modal.run")
        
    elif action == "test":
        print(" Testing locally...")
        try:
            import streamlit.web.cli as stcli
            sys.argv = ["streamlit", "run", "deployment/app.py"]
            sys.exit(stcli.main())
        except Exception as e:
            print(f" Local test failed: {e}")
            print(" Make sure you're in the sketch-recognition-app directory")
            print(" Install dependencies: pip install streamlit")
            
    elif action == "check-model":
        print(" Checking model status...")
        result = check_model.remote()
        print(f"\n Model status:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
    elif action == "upload-model":
        print(" Uploading model from local machine to Modal...")
        
        # Read local model file
        local_path = Path(model_path)
        if not local_path.exists():
            print(f" Model not found at: {local_path}")
            print(f"\n Current directory: {Path.cwd()}")
            print(f" Looking for model at: {local_path.absolute()}")
            print(f"\n Available models:")
            for model_file in Path(".").rglob("*.h5"):
                print(f"   - {model_file}")
            return
        
        print(f" Found model at: {local_path}")
        print(f" Model size: {local_path.stat().st_size / (1024**2):.2f} MB")
        
        # Read model file as bytes
        with open(local_path, "rb") as f:
            model_bytes = f.read()
        
        print(f" Uploading to Modal volume...")
        result = upload_model_from_bytes.remote(model_bytes)
        
        print(f"\n Upload result:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        
        if result.get("status") == "success":
            print(f"\n Model successfully uploaded to Modal!")
            print(f"   You can now deploy: modal deploy model-deploy.py")
        else:
            print(f"\n Upload failed: {result.get('error')}")
        
    else:
        print("""
 Sketch Recognition Deployment Tool (Python 3.12)

Usage:
  modal run model-deploy.py::main --action=<action>

Actions:
  setup         Setup application environment on Modal
  deploy        Show deployment instructions
  test          Test Streamlit app locally
  check-model   Check if model exists and is valid
  upload-model  Show upload instructions
  help          Show this help message

  IMPORTANT: Upload Model First!
  Before the API works, you need to upload your trained model:
  
  Upload your local model:
    modal run model-deploy.py::main --action=upload-model --model-path=./models/sketch_model.h5

Examples:
  # 1. Upload your trained model
  modal run model-deploy.py::main --action=upload-model --model-path=./models/sketch_model.h5
  
  # 2. Check model uploaded successfully
  modal run model-deploy.py::main --action=check-model
  
  # 3. Deploy to Modal
  modal deploy model-deploy.py
  
  # 4. Test the API (visit the URL from deploy output)
  curl https://your-url.modal.run/health

 Documentation:
  - FastAPI Docs: <your-url>/docs
  - Health Check: <your-url>/health
  - Health Check: <your-url>/health

 Configuration:
  - Python: 3.12
  - TensorFlow: 2.15.0
  - Model: sketch_model.h5 (28x28 grayscale CNN)
  - Categories: 20 QuickDraw classes
        """)


if __name__ == "__main__":
    print("""
Sketch Recognition Deployment

Run with Modal CLI:
  modal run model-deploy.py::main --action=help
    """)
