import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import time
import sys
import os

# Define SketchPredictor with the improved preprocessing logic
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
        
        # Invert colors: Canvas is Black on White, Model expects White on Black
        image = cv2.bitwise_not(image)
        
        # Find bounding box of the sketch to crop empty space
        coords = cv2.findNonZero(image)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add padding to preserve stroke edges
            padding = 20
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(image.shape[1], x + w + padding)
            y_max = min(image.shape[0], y + h + padding)
            
            image = image[y_min:y_max, x_min:x_max]
        
        # Resize to 28x28 while maintaining aspect ratio
        # 1. Create a square canvas of the max dimension
        max_dim = max(image.shape[0], image.shape[1])
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        # 2. Center the image on the square canvas
        y_offset = (max_dim - image.shape[0]) // 2
        x_offset = (max_dim - image.shape[1]) // 2
        square_img[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        
        # 3. Resize to target size
        image = cv2.resize(square_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        # Apply thresholding to strengthen the strokes after resizing
        _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        
        image = image.astype('float32') / 255.0
        image = image.reshape(1, self.img_size, self.img_size, 1)
        return image
    
    def predict(self, image):
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

# Initialize FastAPI app
api = FastAPI(
    title="Sketch Recognition API",
    version="2.0.0",
    description="High-performance sketch recognition using CNN model trained on QuickDraw dataset",
)

# CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global exception handler
@api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "type": type(exc).__name__
        },
    )

# Load model
predictor = None
MODEL_PATH = "/models/sketch_model.h5"

try:
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}")
    else:
        predictor = SketchPredictor(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")

@api.get("/")
async def root():
    return {
        "message": "Sketch Recognition API",
        "status": "running",
        "model_loaded": "yes" if predictor else "no"
    }

@api.get("/health")
async def health_check():
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None
    }

@api.post("/predict")
async def predict_sketch(image_data: dict):
    start_time = time.time()
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        if "image" not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")
        
        image_b64 = image_data["image"].split(",")[-1] if "," in image_data["image"] else image_data["image"]
        
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
        
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        predictions = predictor.predict(image)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "processing_time_ms": round(processing_time, 2),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
