import numpy as np
import requests
import os
import sys
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model_training.config import DATA_CONFIG, TRAIN_CONFIG
except ImportError:
    from .config import DATA_CONFIG, TRAIN_CONFIG

class QuickDrawDataLoader:
    def __init__(self):
        self.data_dir = DATA_CONFIG['data_dir']
        self.categories = TRAIN_CONFIG['categories']
        self.img_size = DATA_CONFIG['img_size']
        self.max_samples = DATA_CONFIG['max_samples_per_class']
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_category(self, category):
        """Download QuickDraw data for a single category"""
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category.replace(' ', '%20')}.npy"
        file_path = os.path.join(self.data_dir, f"{category}.npy")
        
        if not os.path.exists(file_path):
            print(f"Downloading {category}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading {category}: {e}")
                return None
        return file_path
    
    def load_data(self):
        """Load and preprocess all categories"""
        X, y = [], []
        
        for label, category in enumerate(self.categories):
            file_path = self.download_category(category)
            if file_path is None:
                continue
                
            try:
                data = np.load(file_path)
                # Take only max_samples per class
                data = data[:self.max_samples]
                X.append(data)
                y.extend([label] * len(data))
                print(f"Loaded {len(data)} samples for {category}")
            except Exception as e:
                print(f"Error loading {category}: {e}")
                continue
        
        if not X:
            raise Exception("No data loaded!")
        
        X = np.vstack(X).astype('float32')
        y = np.array(y)
        
        # Normalize and reshape
        X = X / 255.0
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)