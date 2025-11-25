# WandB Model Serving

A complete ML serving system for image classification with Redis queue-based async predictions, multi-model comparison, and W&B inference tracking.

## 🏗️ Architecture

This project demonstrates a complete ML serving pipeline with:

- **TensorFlow Serving**: Serves trained models via REST API
- **Redis Queue**: Async task queue for scalable predictions
- **Worker Service**: Background workers processing predictions with W&B tracking
- **FastAPI Backend**: API for task submission and result retrieval
- **Streamlit Frontend**: Interactive UI with multi-model comparison
- **Docker Compose**: Orchestrates all services

## 📁 Project Structure

```
wandb-model-serving/
├── backend/              # FastAPI service
│   ├── app.py           # API endpoints (task queue)
│   ├── worker.py        # Redis worker with W&B tracking
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/            # Streamlit web app
│   ├── streamlit_app.py # Interactive UI with multi-select
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

## ✨ Key Features

### 🔄 Async Prediction Queue
- Redis-based task queue for scalable inference
- Background workers process predictions asynchronously
- Configurable result caching (1-hour TTL)

### 📊 Multi-Model Comparison
- Select multiple model versions simultaneously
- Side-by-side result comparison in UI
- Support for different model architectures (3-class vs binary)

### 📈 W&B Inference Tracking
- Automatic logging of all predictions to Weights & Biases
- Track model performance, latency, and prediction distribution
- Distributed tracking per model version

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- (Optional) W&B account for inference tracking

### Running the Application

1. **Clone the repository**
   ```bash
   git clone https://github.com/brokenhardisk/wandb-model-serving.git
   cd wandb-model-serving
   ```

2. **(Optional) Configure W&B**
   ```bash
   echo "WANDB_API_KEY=your_api_key_here" > .env
   ```

3. **Start all services**
   ```bash
   docker compose up -d
   ```

4. **Access the application**
   - **Streamlit UI**: http://localhost:8502
   - **FastAPI Backend**: http://localhost:8080
   - **TensorFlow Serving**: http://localhost:8501
   - **Redis**: localhost:6379

### Using the Streamlit UI

1. Open http://localhost:8502 in your browser
2. Select one or more model versions (both selected by default):
   - **v1**: 3-class classification (Bird, Cat, Dog) - 150x150 images
   - **v2**: Binary classification (Cat, Dog) - 128x128 images
3. Upload an image (JPG, JPEG, or PNG)
4. Click "🚀 Run Prediction" 
5. View results side-by-side for easy comparison

## 🔧 API Usage

### Submit Prediction Task

```bash
# Single version
curl -X POST "http://localhost:8080/predict?versions=v2" \
  -F "file=@/path/to/image.jpg"

# Multiple versions for comparison
curl -X POST "http://localhost:8080/predict?versions=v1,v2" \
  -F "file=@/path/to/your/image.jpg"

# Response includes task_id
{
  "task_id": "uuid-here",
  "status": "queued",
  "versions": ["v1", "v2"]
}
```

### Get Prediction Results

```bash
# Poll for results using task_id
curl http://localhost:8080/result/{task_id}

# Response when completed
{
  "task_id": "uuid",
  "status": "completed",
  "results": {
    "v1": {
      "success": true,
      "predictions": [[0.1, 0.3, 0.6]],
      "version": "1"
    },
    "v2": {
      "success": true,
      "predictions": [[0.85]],
      "version": "2"
    }
  }
}
```

### Health Check

```bash
curl http://localhost:8080/health
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

# Specific services
docker compose logs redis
docker compose logs worker
docker compose logs model-server
docker compose logs backend
docker compose logs frontend

# Follow logs (useful for debugging)
docker compose logs -f worker
```

### Monitor Redis Queue

```bash
# Connect to Redis CLI
docker exec -it redis_queue redis-cli

# Check queue length
LLEN prediction_queue

# View recent results
KEYS result:*

# Get specific result
GET result:{task_id}
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

### Environment Variables

Create a `.env` file for custom configurations:

```bash
# W&B Configuration (optional)
WANDB_API_KEY=your_wandb_api_key_here

# Service URLs (defaults shown)
MODEL_URL=http://model-server:8501/v1/models
REDIS_URL=redis://redis:6379
API_URL=http://backend
STREAMLIT_SERVER_PORT=8502
```

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

## 📈 Weights & Biases Integration

The worker service automatically logs all predictions to W&B when `WANDB_API_KEY` is configured.

### Tracked Metrics

- **Per-prediction logging**:
  - Task ID and timestamp
  - Model name and version
  - Inference duration (ms)
  - Prediction probabilities
  - Predicted class
  - Success/failure status

### View Your Dashboard

1. Set `WANDB_API_KEY` in `.env`
2. Start services: `docker compose up -d`
3. Make predictions via the UI or API
4. View metrics at: https://wandb.ai/your-username/model-serving-inference

### Distributed Tracking

Each worker logs to the same W&B project, allowing you to:
- Compare performance across model versions
- Track inference latency trends
- Analyze prediction distribution
- Monitor error rates

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Stop all containers and free ports
docker ps -q | xargs -r docker stop
for p in 6379 8501 8502 8080; do lsof -tiTCP:$p -sTCP:LISTEN | xargs -r kill -9; done
```

### Redis Connection Issues

Check Redis is running:
```bash
docker compose ps redis
docker logs redis_queue
```

Test Redis connection:
```bash
docker exec -it redis_queue redis-cli ping
# Should return: PONG
```

### Worker Not Processing Tasks

Check worker logs:
```bash
docker compose logs worker
```

Verify queue has tasks:
```bash
docker exec -it redis_queue redis-cli LLEN prediction_queue
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

### Predictions Taking Too Long

- Check worker is running: `docker compose ps worker`
- Increase worker count by scaling: `docker compose up -d --scale worker=3`
- Monitor queue length: `docker exec -it redis_queue redis-cli LLEN prediction_queue`

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