import io
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import os
import base64
from dotenv import load_dotenv
import wandb
import time

# Load environment variables from .env
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
MODEL_BASE_URL = os.getenv("MODEL_URL", "http://model-server:8501/v1/models")

wandb.login(key=WANDB_API_KEY)
wandb.init(
    project="model-serving",
    name="api_inference_logging",
    reinit=True
)


app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(default="animals", description="Model name"),
    version: str = Query(default="v1", description="Model version (v1, v2)")
    ):
    start_time = time.time()
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Empty file received"})

        # Build model URL based on version
        # Convert version label (v1, v2) to version number (1, 2)
        if version.startswith("v"):
            version_num = version[1:]  # Remove 'v' prefix
            model_url = f"{MODEL_BASE_URL}/{model}/versions/{version_num}:predict"
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid Model Version selected: {version}"}
            )

        print(f"Using model URL: {model_url}") 
        
        # Different versions expect different input sizes
        # v1: 150x150, v2: 128x128
        img_size = 150 if version == "v1" else 128
        
        # Open and convert image
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((img_size, img_size))
        img_arr = np.array(img, dtype=np.float32)
        img_arr = img_arr / 255.0  # Normalize to [0, 1] as done during training
        img_arr = np.expand_dims(img_arr, axis=0)
        payload = {"instances": img_arr.tolist()}
        try:
            response = requests.post(model_url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"TensorFlow Serving request failed: {str(e)}"}
            )

        predictions = response.json().get("predictions", [])
        inference_time = time.time() - start_time

        # Log to W&B
        wandb.log({
            "model": model,
            "version": version,
            "num_predictions": len(predictions),
            "inference_time_sec": inference_time,
            "input_size": img_size,
            "example_input": wandb.Image(img)
        })
        return JSONResponse(content={
            "predictions": predictions,
            "model": model,
            "version": version,
            "model_url": model_url
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        wandb.log({
            "error": str(e),
            "traceback": tb,
            "model": model,
            "version": version
        })
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}", "traceback": tb}
        )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_base_url": MODEL_BASE_URL}

@app.get("/models")
def list_models():
    """List available models and versions"""
    try:
        response = requests.get(f"{MODEL_BASE_URL.replace('/v1/models', '')}/v1/models/animals")
        return response.json()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.on_event("shutdown")
def shutdown_event():
    wandb.finish()