import io
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import os
import base64

app = FastAPI()

MODEL_URL = os.getenv("MODEL_URL", "http://model-server:8501/v1/models/animal_classifier:predict")

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Empty file received"})

        # 1. Open and convert
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((150, 150))
        img_arr = np.array(img, dtype=np.float32)
        img_arr /= 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        payload = {"instances": img_arr.tolist()}
        try:
            response = requests.post(MODEL_URL, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"TensorFlow Serving request failed: {str(e)}"}
            )

        predictions = response.json().get("predictions", [])
        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}", "traceback": tb}
        )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_url": MODEL_URL}