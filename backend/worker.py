import os
import json
import time
import redis
import requests
import numpy as np
from PIL import Image
import io
import base64
import wandb
from datetime import datetime
import cv2
import tensorflow as tf

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_BASE_URL = os.getenv("MODEL_URL", "http://model-server:8501/v1/models")
SKETCH_MODEL_PATH = os.getenv("SKETCH_MODEL_PATH", "/models/sketch_model.h5")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

# Sketch categories
SKETCH_CATEGORIES = [
    'apple', 'banana', 'cat', 'dog', 'house',
    'tree', 'car', 'fish', 'bird', 'clock',
    'book', 'chair', 'cup', 'star', 'heart',
    'smiley face', 'sun', 'moon', 'key', 'hammer'
]

# Load sketch model
sketch_model = None
try:
    if os.path.exists(SKETCH_MODEL_PATH):
        sketch_model = tf.keras.models.load_model(SKETCH_MODEL_PATH)
        print(f"✅ Sketch model loaded from {SKETCH_MODEL_PATH}")
    else:
        print(f"⚠️  Sketch model not found at {SKETCH_MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Failed to load sketch model: {e}")

# Initialize W&B if API key is provided
wandb_enabled = False
if WANDB_API_KEY:
    try:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project="model-serving",
            name=f"worker-{datetime.now().strftime('%Y%m%d')}",
            config={
                "architecture": "async-queue",
                "framework": "tensorflow-serving",
                "queue": "redis"
            }
        )
        wandb_enabled = True
        print("✅ W&B tracking enabled")
    except Exception as e:
        print(f"⚠️  W&B initialization failed: {e}")
        print("Continuing without W&B tracking...")
else:
    print("ℹ️  WANDB_API_KEY not set. W&B tracking disabled.")

def preprocess_image_version_1(image_bytes, img_size):
    """Preprocess image for model inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))
    img_arr = np.array(img, dtype=np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def preprocess_image_version_2(image_bytes, img_size):
    """Preprocess image for model inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))
    img_arr = np.array(img, dtype=np.float32)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def predict_with_model(image_bytes, model_name, version):
    """Send prediction request to TensorFlow Serving."""
    try:
        # Determine image size based on version
        img_size = 150 if version == "1" else 128
        

        # Preprocess image
        if version == "1":
            img_arr = preprocess_image_version_1(image_bytes, img_size)
        else:
            img_arr = preprocess_image_version_2(image_bytes, img_size)
        
        # Prepare payload
        payload = {"instances": img_arr.tolist()}
        
        # Send to TF Serving
        model_url = f"{MODEL_BASE_URL}/{model_name}/versions/{version}:predict"
        response = requests.post(model_url, json=payload, timeout=30)
        response.raise_for_status()
        
        predictions = response.json().get("predictions", [])
        return {"success": True, "predictions": predictions, "version": version}
    
    except Exception as e:
        return {"success": False, "error": str(e), "version": version}

def log_to_wandb(task_id, model_name, version, prediction_result, duration, image_bytes, task_type="animal"):
    """Log prediction to W&B."""
    if not wandb_enabled:
        return
    
    try:
        if task_type == "sketch":
            # Decode sketch image
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            img_cv = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            img = Image.fromarray(img_cv)
            
            log_data = {
                "task_id": task_id,
                "model": "sketch",
                "task_type": "sketch",
                "inference_time_ms": duration * 1000,
                "input_size": 28,
                "input_shape": "28x28x1",
                "success": prediction_result.get("success", False),
                "timestamp": datetime.now().isoformat()
            }
            
            if prediction_result.get("success"):
                preds = prediction_result.get("predictions", [])
                if preds:
                    log_data["top_prediction"] = preds[0]["category"]
                    log_data["top_confidence"] = preds[0]["confidence"]
            else:
                log_data["error"] = prediction_result.get("error", "Unknown error")
            
            log_data["image"] = wandb.Image(img, caption=f"Sketch - Task {task_id}")
        else:
            # Animal prediction logging
            img_size = 150 if version == "1" else 128
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            log_data = {
                "task_id": task_id,
                "model": model_name,
                "version": f"v{version}",
                "task_type": "animal",
                "inference_time_ms": duration * 1000,
                "input_size": img_size,
                "input_shape": f"{img_size}x{img_size}x3",
                "success": prediction_result.get("success", False),
                "timestamp": datetime.now().isoformat()
            }
            
            if prediction_result.get("success"):
                preds = prediction_result.get("predictions", [[]])[0]
                num_predictions = len(preds) if isinstance(preds, list) else 1
                log_data["num_predictions"] = num_predictions
                log_data["output_shape"] = num_predictions
            else:
                log_data["error"] = prediction_result.get("error", "Unknown error")
            
            log_data["image"] = wandb.Image(img, caption=f"v{version} - Task {task_id}")
        
        wandb.log(log_data)
        print(f"📊 Logged to W&B: {task_id} - {duration*1000:.2f}ms")
    except Exception as e:
        print(f"⚠️  W&B logging error: {e}")

def preprocess_sketch(image_bytes):
    """Preprocess sketch image for model."""
    try:
        # Decode image
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Invert colors (canvas is black on white, model expects white on black)
        img = cv2.bitwise_not(img)
        
        # Apply threshold to clean up the image
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Find bounding box of drawing
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add padding (10% on each side)
            padding = int(max(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # Crop to bounding box
            img = img[y:y+h, x:x+w]
        
        # Resize to 28x28 maintaining aspect ratio
        if img.shape[0] > img.shape[1]:
            new_h = 28
            new_w = int(28 * img.shape[1] / img.shape[0])
        else:
            new_w = 28
            new_h = int(28 * img.shape[0] / img.shape[1])
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create 28x28 canvas and center the image
        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
        
        # Normalize to [0, 1]
        canvas = canvas.astype(np.float32) / 255.0
        
        # Reshape for model (batch_size, height, width, channels)
        canvas = canvas.reshape(1, 28, 28, 1)
        
        return canvas
    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")

def predict_sketch(image_bytes):
    """Predict sketch category."""
    try:
        if sketch_model is None:
            return {
                "success": False,
                "error": "Sketch model not loaded"
            }
        
        # Preprocess
        processed_image = preprocess_sketch(image_bytes)
        
        # Predict
        predictions = sketch_model.predict(processed_image, verbose=0)
        predictions = predictions[0]  # Get first (and only) result
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        top_predictions = [
            {
                "category": SKETCH_CATEGORIES[idx],
                "confidence": float(predictions[idx] * 100)
            }
            for idx in top_indices
        ]
        
        return {
            "success": True,
            "predictions": top_predictions
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def process_task(task_data):
    """Process a single prediction task."""
    task_id = task_data.get("task_id")
    task_type = task_data.get("task_type", "animal")
    image_b64 = task_data.get("image")
    
    # Decode image
    image_bytes = base64.b64decode(image_b64)
    
    if task_type == "sketch":
        # Process sketch prediction
        print(f"Processing sketch task {task_id}")
        
        start_time = time.time()
        result = predict_sketch(image_bytes)
        duration = time.time() - start_time
        
        # Log to W&B
        log_to_wandb(task_id, "sketch", None, result, duration, image_bytes, task_type="sketch")
        
        # Store results in Redis
        result_key = f"result:{task_id}"
        redis_client.setex(
            result_key,
            3600,  # 1 hour TTL
            json.dumps({"sketch": result})
        )
        
        print(f"Sketch task {task_id} completed. Results stored.")
    else:
        # Process animal prediction
        model_name = task_data.get("model", "animals")
        versions = task_data.get("versions", ["2"])
        
        print(f"Processing animal task {task_id} for versions: {versions}")
        
        # Run predictions for each version
        results = {}
        for version in versions:
            start_time = time.time()
            result = predict_with_model(image_bytes, model_name, version)
            duration = time.time() - start_time
            
            results[f"v{version}"] = result
            
            # Log to W&B
            log_to_wandb(task_id, model_name, version, result, duration, image_bytes, task_type="animal")
        
        # Store results in Redis with expiration (1 hour)
        result_key = f"result:{task_id}"
        redis_client.setex(
            result_key,
            3600,  # 1 hour TTL
            json.dumps(results)
        )
        
        print(f"Animal task {task_id} completed. Results stored.")

def main():
    """Main worker loop."""
    print("Starting prediction worker...")
    print(f"Redis URL: {REDIS_URL}")
    print(f"Model URL: {MODEL_BASE_URL}")
    print(f"W&B Tracking: {'Enabled' if WANDB_API_KEY else 'Disabled'}")
    
    while True:
        try:
            # Block and wait for task (timeout 1 second)
            task = redis_client.blpop("prediction_queue", timeout=1)
            
            if task:
                _, task_json = task
                task_data = json.loads(task_json)
                process_task(task_data)
        
        except KeyboardInterrupt:
            print("\nShutting down worker...")
            break
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
