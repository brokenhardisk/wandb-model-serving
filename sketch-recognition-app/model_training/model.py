import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model_training.config import MODEL_CONFIG
except ImportError:
    from .config import MODEL_CONFIG

def create_cnn_model():
    """Create a custom CNN model for sketch recognition"""
    config = MODEL_CONFIG
    
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(config['conv_filters'][0], (3, 3), activation='relu', 
                     input_shape=config['input_shape']),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(config['dropout_rate']),
        
        # Second Conv Block
        layers.Conv2D(config['conv_filters'][1], (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(config['dropout_rate']),
        
        # Third Conv Block
        layers.Conv2D(config['conv_filters'][2], (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(config['dropout_rate']),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(config['dense_units'][0], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config['dropout_rate']),
        
        layers.Dense(config['dense_units'][1], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config['dropout_rate']),
        
        # Output Layer
        layers.Dense(config['num_classes'], activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """Compile the model with appropriate settings"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model