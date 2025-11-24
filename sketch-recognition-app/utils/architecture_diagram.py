"""
Project Structure Visualization
"""

STRUCTURE = """
sketch-recognition-app/
│
├── 📱 FRONTEND (Streamlit UI)
│   └── deployment/
│       ├── app.py                  # Main Streamlit application
│       ├── inference.py            # Prediction logic
│       └── wandb_utils.py          # W&B visualization
│
├── 🧠 MODEL (Training & Architecture)
│   └── model_training/
│       ├── config.py               # Configuration (UPDATE WANDB ENTITY!)
│       ├── model.py                # CNN architecture
│       ├── data_loader.py          # QuickDraw data loader
│       └── train_sketch_model.ipynb # Training notebook
│
├── 🐳 DEPLOYMENT
│   ├── docker/
│   │   ├── Dockerfile              # Container definition
│   │   └── requirements.txt        # Python dependencies
│   │
│   └── model-deploy.py             # Modal deployment script
│
├── 🎨 CONFIGURATION
│   └── .streamlit/
│       └── config.toml             # UI theme & settings
│
├── 🔧 UTILITIES
│   └── utils/
│       └── test_setup.py           # Test all components
│
├── 📁 GENERATED (after training)
│   └── models/
│       └── sketch_model.h5         # Trained model
│
├── 📖 DOCUMENTATION
│   ├── README.md                   # Main documentation
│   ├── QUICKSTART.md              # 5-minute setup guide
│   ├── DEPLOYMENT.md              # Modal deployment guide
│   └── CORRECTIONS.md             # Changes made
│
└── 🚀 SCRIPTS
    └── run_app.sh                  # Quick start script
"""

ARCHITECTURE = """
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│  ┌──────────────────────┐      ┌─────────────────────────────┐ │
│  │  Drawing Canvas      │      │    Image Upload             │ │
│  │  (streamlit-canvas)  │      │    (file uploader)          │ │
│  └──────────┬───────────┘      └──────────┬──────────────────┘ │
│             │                              │                     │
└─────────────┼──────────────────────────────┼─────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IMAGE PREPROCESSING                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Convert to grayscale                                   │  │
│  │  • Resize to 28x28                                        │  │
│  │  • Invert colors (QuickDraw format)                       │  │
│  │  • Normalize (0-1)                                        │  │
│  └──────────────────────┬───────────────────────────────────┘  │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CNN MODEL (TensorFlow)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Conv2D(32) → BatchNorm → MaxPool → Dropout              │  │
│  │  Conv2D(64) → BatchNorm → MaxPool → Dropout              │  │
│  │  Conv2D(128) → BatchNorm → MaxPool → Dropout             │  │
│  │  Flatten                                                   │  │
│  │  Dense(128) → BatchNorm → Dropout                         │  │
│  │  Dense(64) → BatchNorm → Dropout                          │  │
│  │  Dense(20) → Softmax                                      │  │
│  └──────────────────────┬───────────────────────────────────┘  │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREDICTION OUTPUT                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Top 3 Predictions:                                       │  │
│  │  1. Category: "cat"      Confidence: 87.3%               │  │
│  │  2. Category: "dog"      Confidence: 8.2%                │  │
│  │  3. Category: "bird"     Confidence: 2.1%                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    W&B METRICS DISPLAY                           │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │  Training Curves    │      │  Final Metrics              │  │
│  │  • Loss over time   │      │  • Best accuracy: 92%       │  │
│  │  • Accuracy curve   │      │  • Best loss: 0.23          │  │
│  └─────────────────────┘      └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
"""

DATA_FLOW = """
TRAINING FLOW:
──────────────

1. QuickDraw Dataset (Google)
        ↓
2. Download via data_loader.py
        ↓
3. Preprocess (normalize, reshape)
        ↓
4. Train CNN (model.py)
        ↓ (log metrics)
5. Weights & Biases
        ↓
6. Save model → models/sketch_model.h5


INFERENCE FLOW:
───────────────

1. User draws sketch
        ↓
2. Canvas → base64 image
        ↓
3. Decode & preprocess
        ↓
4. Load model (inference.py)
        ↓
5. Predict → probabilities
        ↓
6. Format & display results


DEPLOYMENT FLOW:
────────────────

LOCAL:
  streamlit run deployment/app.py
        ↓
  http://localhost:8501

DOCKER:
  docker build -f docker/Dockerfile .
        ↓
  docker run -p 8501:8501
        ↓
  http://localhost:8501

MODAL:
  modal deploy model-deploy.py
        ↓
  https://your-app.modal.run
"""

TECHNOLOGIES = """
TECHNOLOGY STACK:
─────────────────

🧠 Machine Learning:
   • TensorFlow 2.13.0        → Deep learning framework
   • Keras                    → High-level neural network API
   • NumPy 1.24.3            → Numerical computing

🎨 Frontend:
   • Streamlit 1.28.0        → Web UI framework
   • streamlit-drawable-canvas → Drawing interface
   • Plotly 5.18.0           → Interactive charts

🖼️ Image Processing:
   • OpenCV 4.8.1            → Image manipulation
   • Pillow 10.1.0           → Image loading

📊 Experiment Tracking:
   • Weights & Biases 0.16.0 → Metric logging & visualization

☁️ Deployment:
   • Modal                    → Serverless deployment
   • Docker                   → Containerization

📦 Data & Utilities:
   • scikit-learn 1.3.0      → Data splitting
   • pandas 2.0.3            → Data manipulation
   • requests 2.31.0         → HTTP requests
"""

CATEGORIES = """
20 SKETCH CATEGORIES:
─────────────────────

🍎 Food:
   • apple
   • banana

🐾 Animals:
   • cat
   • dog
   • fish
   • bird

🏠 Objects:
   • house
   • tree
   • car
   • clock
   • book
   • chair
   • cup
   • key
   • hammer

⭐ Shapes & Fun:
   • star
   • heart
   • smiley face
   • sun
   • moon
"""

MODEL_SPECS = """
MODEL SPECIFICATIONS:
─────────────────────

Architecture:
  • Type: Custom CNN
  • Input: 28x28x1 (grayscale)
  • Output: 20 classes (softmax)

Layers:
  • Conv blocks: 3
  • Conv filters: [32, 64, 128]
  • Dense layers: 2
  • Dense units: [128, 64]
  • Dropout rate: 0.3

Training:
  • Optimizer: Adam
  • Learning rate: 0.001
  • Batch size: 64
  • Epochs: 10
  • Loss: Sparse categorical crossentropy
  • Metrics: Accuracy

Dataset:
  • Source: Google QuickDraw
  • Samples per class: 10,000
  • Train/Val/Test: 64%/16%/20%
  • Total samples: ~160,000

Performance (Expected):
  • Training accuracy: ~90-95%
  • Validation accuracy: ~85-92%
  • Inference time: <100ms
"""

DEPLOYMENT_OPTIONS = """
DEPLOYMENT OPTIONS:
───────────────────

1. LOCAL DEVELOPMENT:
   ├─ Best for: Testing, development
   ├─ Command: streamlit run deployment/app.py
   ├─ Access: http://localhost:8501
   └─ Cost: Free

2. DOCKER CONTAINER:
   ├─ Best for: Reproducibility, local deployment
   ├─ Command: docker run -p 8501:8501 sketch-app
   ├─ Access: http://localhost:8501
   └─ Cost: Free

3. MODAL (RECOMMENDED):
   ├─ Best for: Public deployment, scalability
   ├─ Command: modal deploy model-deploy.py
   ├─ Access: https://your-app.modal.run
   ├─ Cost: Free tier available
   └─ Features: Auto-scaling, public URL, serverless

4. OTHER OPTIONS:
   ├─ Streamlit Cloud (free for public apps)
   ├─ Heroku (containerized deployment)
   ├─ AWS/GCP/Azure (full control)
   └─ Hugging Face Spaces (ML-focused)
"""

if __name__ == "__main__":
    print("=" * 70)
    print("SKETCH RECOGNITION APP - ARCHITECTURE OVERVIEW")
    print("=" * 70)
    
    sections = [
        ("PROJECT STRUCTURE", STRUCTURE),
        ("ARCHITECTURE DIAGRAM", ARCHITECTURE),
        ("DATA FLOW", DATA_FLOW),
        ("TECHNOLOGY STACK", TECHNOLOGIES),
        ("CATEGORIES", CATEGORIES),
        ("MODEL SPECIFICATIONS", MODEL_SPECS),
        ("DEPLOYMENT OPTIONS", DEPLOYMENT_OPTIONS),
    ]
    
    for title, content in sections:
        print(f"\n\n{'=' * 70}")
        print(f"{title:^70}")
        print(f"{'=' * 70}")
        print(content)
    
    print("\n" + "=" * 70)
    print("For more details, see:")
    print("  • README.md - Full documentation")
    print("  • QUICKSTART.md - Get started in 5 minutes")
    print("  • DEPLOYMENT.md - Deploy to Modal")
    print("=" * 70)
