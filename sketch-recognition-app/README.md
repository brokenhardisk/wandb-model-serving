docker build -t sketch-recognition:latest .

docker run -p 8501:8501 -p 8000:8000 sketch-recognition:latest

modal deploy model-deploy.py

also for login to wandb we should first execute 
> wandb login

similarly for Modal
install modal (pip install modal)

modal run model-deploy.py::main --action=setup




3-Layer Convolutional Neural Network with the following structure:

Conv Blocks (x3)
Conv2D layer with ReLU activation
Batch Normalization
MaxPooling2D (2x2)
Dropout for regularization
Dense Layers (x2)
Fully connected layers with ReLU
Batch Normalization
Dropout
Output Layer
Softmax activation for multi-class classification
Key Features
Input: 28x28 grayscale images
Filter progression: 32 → 64 → 128 (from config)
Regularization: BatchNorm + Dropout throughout
Optimizer: Adam with configurable learning rate
Loss: Sparse categorical crossentropy
Configuration
Model hyperparameters are loaded from MODEL_CONFIG in config.py:

Number of classes
Filter sizes per conv layer
Dense layer units
Dropout rate
Input shape
