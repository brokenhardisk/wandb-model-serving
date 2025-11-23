from fastapi import FastAPI, UploadFile, File
import requests
from PIL import Image
import numpy as np
import io
import uvicorn

app = FastAPI()

# URL of TF Serving for your model
TF_SERVING_URL = "http://tfserving:8501/v1/models/cats_dogs_model:predict"

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    input_tensor = preprocess_image(img_bytes)

    response = requests.post(
        TF_SERVING_URL,
        json={"instances": input_tensor.tolist()}
    )
    pred = response.json()["predictions"][0][0]
    label = "dog" if pred > 0.5 else "cat"
    return {"prediction": label, "raw": float(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
