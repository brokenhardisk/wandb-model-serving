import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model_training.config import TRAIN_CONFIG
except ImportError:
    # Fallback for when config is not available
    TRAIN_CONFIG = {
        'categories': [
            'apple', 'banana', 'cat', 'dog', 'house',
            'tree', 'car', 'fish', 'bird', 'clock',
            'book', 'chair', 'cup', 'star', 'heart',
            'smiley face', 'sun', 'moon', 'key', 'hammer'
        ]
    }

class SketchPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.categories = TRAIN_CONFIG['categories']
        self.img_size = 28
    
    def preprocess_sketch(self, image):
        """Preprocess sketch image for prediction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ensure uint8 format
        image = image.astype(np.uint8)
        
        # Resize to 28x28 with anti-aliasing
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] range
        image = image.astype('float32') / 255.0
        
        # Reshape for model (batch_size, height, width, channels)
        image = image.reshape(1, self.img_size, self.img_size, 1)
        
        return image
    
    def predict(self, image):
        """Predict sketch category"""
        processed_image = self.preprocess_sketch(image)
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'category': self.categories[i],
                'confidence': float(predictions[0][i]) * 100
            }
            for i in top_indices
        ]
        
        return top_predictions