import io
import json
import uuid
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import redis
import os
import base64
from dotenv import load_dotenv
import time
from typing import List

# Load environment variables from .env
load_dotenv()

MODEL_BASE_URL = os.getenv("MODEL_URL", "http://model-server:8501/v1/models")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize Redis
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    print(f"Connected to Redis at {REDIS_URL}")
except Exception as e:
    print(f"Warning: Could not connect to Redis: {e}")
    redis_client = None

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
    versions: str = Query(default="v1,v2", description="Model versions comma-separated (v1, v2)")
    ):
    """
    Submit a prediction task to Redis queue.
    Supports multiple versions for comparison.
    """
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Empty file received"})

        if not redis_client:
            return JSONResponse(
                status_code=503,
                content={"error": "Redis queue unavailable. Service degraded."}
            )

        # Parse versions for animals model
        version_list = [v.strip() for v in versions.split(",")]
        version_nums = []
        for v in version_list:
            if v.startswith("v"):
                version_nums.append(v[1:])
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid version format: {v}. Use v1, v2, etc."}
                )

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Encode image to base64
        image_b64 = base64.b64encode(contents).decode('utf-8')

        # Create task
        task = {
            "task_id": task_id,
            "task_type": "animal",
            "image": image_b64,
            "model": model,
            "versions": version_nums,
            "filename": file.filename
        }

        # Push to Redis queue
        redis_client.rpush("prediction_queue", json.dumps(task))

        return JSONResponse(content={
            "task_id": task_id,
            "status": "queued",
            "versions": version_list,
            "message": f"Prediction task queued for versions: {', '.join(version_list)}"
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}", "traceback": tb}
        )


@app.post("/predict/sketch")
async def predict_sketch(image_data: dict = Body(...)):
    """
    Submit a sketch prediction task to Redis queue.
    Expects base64 encoded image from canvas.
    """
    try:
        if not redis_client:
            return JSONResponse(
                status_code=503,
                content={"error": "Redis queue unavailable. Service degraded."}
            )

        if "image" not in image_data:
            return JSONResponse(status_code=400, content={"error": "Missing 'image' field"})

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Get base64 image (remove data URL prefix if present)
        image_b64 = image_data["image"].split(",")[-1] if "," in image_data["image"] else image_data["image"]

        # Create task
        task = {
            "task_id": task_id,
            "task_type": "sketch",
            "image": image_b64,
            "model": "sketch"
        }

        # Push to Redis queue
        redis_client.rpush("prediction_queue", json.dumps(task))

        return JSONResponse(content={
            "task_id": task_id,
            "status": "queued",
            "message": "Sketch prediction task queued"
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}", "traceback": tb}
        )


@app.get("/result/{task_id}")
def get_result(task_id: str):
    """Retrieve prediction result from Redis."""
    try:
        if not redis_client:
            return JSONResponse(
                status_code=503,
                content={"error": "Redis unavailable"}
            )

        result_key = f"result:{task_id}"
        result = redis_client.get(result_key)

        if not result:
            # Check if task is still in queue
            return JSONResponse(content={
                "task_id": task_id,
                "status": "processing",
                "message": "Task is being processed or has expired"
            })

        # Parse and return result
        result_data = json.loads(result)
        return JSONResponse(content={
            "task_id": task_id,
            "status": "completed",
            "results": result_data
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
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