import os

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (28, 28, 1),
    'num_classes': 20,
    'conv_filters': [32, 64, 128],
    'dense_units': [128, 64],
    'dropout_rate': 0.3
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 128,  # Increased for faster training
    'epochs': 20,  # Increased from 10 to 20 for better learning
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'categories': [
        'apple', 'banana', 'cat', 'dog', 'house',
        'tree', 'car', 'fish', 'bird', 'clock',
        'book', 'chair', 'cup', 'star', 'heart',
        'smiley face', 'sun', 'moon', 'key', 'hammer'
    ]
}

# W&B Configuration
WANDB_CONFIG = {
    'project': 'sketch-recognition',
    #'entity': "wan-uma-personal",  # Set your W&B username
    'save_code': True
}

# Data Configuration
DATA_CONFIG = {
    'data_dir': 'quickdraw_data',
    'max_samples_per_class': 15000,  # Increased from 10000 to 15000
    'img_size': 28
}