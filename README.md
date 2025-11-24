# WandB Model Serving

A complete machine learning model serving system for animal image classification using TensorFlow Serving, FastAPI, and Streamlit.

## 🏗️ Architecture

This project demonstrates a production-ready ML serving pipeline with:

- **TensorFlow Serving**: Serves trained models via REST API
- **FastAPI Backend**: Handles image preprocessing and model inference requests
- **Streamlit Frontend**: Interactive web UI for image classification
- **Docker Compose**: Orchestrates all services

## 📁 Project Structure

```
wandb-model-serving/
├── backend/              # FastAPI service
│   ├── app.py           # API endpoints and image preprocessing
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/            # Streamlit web app
│   ├── streamlit_app.py # Interactive UI
│   ├── Dockerfile
│   └── requirements.txt
├── models/              # TensorFlow SavedModel files
│   ├── models.config    # TF Serving configuration
│   └── animals/         # Animal classification models
│       ├── 1/           # v1: Bird/Cat/Dog (150x150 input)
│       └── 2/           # v2: Cat/Dog binary (128x128 input)
├── model_training/      # Jupyter notebooks for training
├── docker-compose.yml   # Service orchestration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended

### Running the Application

1. **Clone the repository**
   ```bash
   git clone https://github.com/brokenhardisk/wandb-model-serving.git
   cd wandb-model-serving
   ```

2. **Start all services**
   ```bash
   docker compose up -d
   ```

3. **Access the application**
   - **Streamlit UI**: http://localhost:8502
   - **FastAPI Backend**: http://localhost:8080
   - **TensorFlow Serving**: http://localhost:8501

### Using the Streamlit UI

1. Open http://localhost:8502 in your browser
2. Select a model version:
   - **v1**: 3-class classification (Bird, Cat, Dog) - 150x150 images
   - **v2**: Binary classification (Cat, Dog) - 128x128 images
3. Upload an image (JPG, JPEG, or PNG)
4. Click "Run Prediction" to get classification results

## 🔧 API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

### Make a Prediction

```bash
curl -X POST "http://localhost:8080/predict?model=animals&version=v2" \
  -F "file=@/path/to/your/image.jpg"
```

### List Available Models

```bash
curl http://localhost:8080/models
```

### Direct TensorFlow Serving Access

```bash
# Get model metadata
curl http://localhost:8501/v1/models/animals

# Get specific version metadata
curl http://localhost:8501/v1/models/animals/versions/2/metadata
```

## 📊 Model Versions

### Version 1 (v1)
- **Input**: 150x150 RGB images
- **Output**: 3 classes (Bird, Cat, Dog)
- **Format**: Multi-class probabilities `[bird_prob, cat_prob, dog_prob]`

### Version 2 (v2)
- **Input**: 128x128 RGB images
- **Output**: Binary classification (Cat vs Dog)
- **Format**: Single sigmoid value (0=Cat, 1=Dog)

## 🛠️ Development

### Rebuild Services

```bash
# Rebuild all images
docker compose build

# Rebuild specific service
docker compose build backend
docker compose build frontend
```

### View Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs model-server
docker compose logs backend
docker compose logs frontend

# Follow logs
docker compose logs -f
```

### Stop Services

```bash
docker compose down

# Remove volumes
docker compose down -v

# Remove images
docker compose down --rmi all
```

## 🧪 Testing

Use the included `test.ipynb` notebook to test the TensorFlow Serving API directly:

```python
import requests
import numpy as np

# Test model availability
response = requests.get('http://localhost:8501/v1/models/animals')
print(response.json())

# Make a prediction
# See test.ipynb for full examples
```

## 📝 Configuration

### TensorFlow Serving Configuration

Edit `models/models.config` to modify model serving behavior:

```protobuf
model_config_list {
  config {
    name: "animals"
    base_path: "/models/animals"
    model_platform: "tensorflow"
    model_version_policy {
      all {}  # Serve all versions
    }
  }
}
```

### Environment Variables

Create a `.env` file for custom configurations:

```bash
MODEL_URL=http://model-server:8501/v1/models
API_URL=http://backend
STREAMLIT_SERVER_PORT=8502
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Stop all containers and free ports
docker ps -q | xargs -r docker stop
for p in 8501 8502 8080; do lsof -tiTCP:$p -sTCP:LISTEN | xargs -r kill -9; done
```

### Model Not Loading

Check TensorFlow Serving logs:
```bash
docker logs model_server
```

Ensure model files exist:
```bash
ls -la models/animals/1/
ls -la models/animals/2/
```

### Backend Connection Issues

Verify all services are running:
```bash
docker compose ps
```

## 📚 Training New Models

See the `model_training/` directory for Jupyter notebooks demonstrating model training with W&B integration.

## 🤝 Contributers

- Leonhartsberger, Thomas
- Prakash, Umakanth
- Ruppert, Maximillian
- Wadhwani, Amar

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🔗 Related Technologies

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Weights & Biases](https://wandb.ai/)