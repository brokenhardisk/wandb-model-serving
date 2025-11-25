"""
Test script to verify all components are working
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print(f"✓ Streamlit")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import wandb
        print(f"✓ Weights & Biases")
    except ImportError as e:
        print(f"✗ W&B import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        from streamlit_drawable_canvas import st_canvas
        print(f"✓ Streamlit Drawable Canvas")
    except ImportError as e:
        print(f"✗ Drawable Canvas import failed: {e}")
        print("  Install with: pip install streamlit-drawable-canvas")
        return False
    
    return True

def test_config():
    """Test configuration files"""
    print("\nTesting configuration...")
    
    try:
        from model_training.config import MODEL_CONFIG, TRAIN_CONFIG, WANDB_CONFIG
        print(f"✓ Config loaded successfully")
        print(f"  - Number of classes: {MODEL_CONFIG['num_classes']}")
        print(f"  - Categories: {len(TRAIN_CONFIG['categories'])}")
        print(f"  - W&B project: {WANDB_CONFIG['project']}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_model_architecture():
    """Test model creation"""
    print("\nTesting model architecture...")
    
    try:
        from model_training.model import create_cnn_model, compile_model
        
        model = create_cnn_model()
        model = compile_model(model)
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {model.count_params():,}")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Output shape: {model.output_shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_inference():
    """Test inference module"""
    print("\nTesting inference module...")
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sketch_model.h5')
    
    if not os.path.exists(model_path):
        print(f"⚠ Model not found at {model_path}")
        print("  Train the model first to test inference")
        return True  # Not a failure, just not ready yet
    
    try:
        from deployment.inference import SketchPredictor
        import numpy as np
        
        predictor = SketchPredictor(model_path)
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        predictions = predictor.predict(dummy_image)
        
        print(f"✓ Inference working")
        print(f"  - Top prediction: {predictions[0]['category']}")
        print(f"  - Confidence: {predictions[0]['confidence']:.2f}%")
        
        return True
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

def test_wandb_connection():
    """Test W&B connection"""
    print("\nTesting W&B connection...")
    
    try:
        from deployment.wandb_utils import WandBVisualizer
        from model_training.config import WANDB_CONFIG
        
        viz = WandBVisualizer(
            wandb_entity=WANDB_CONFIG['entity'],
            wandb_project=WANDB_CONFIG['project']
        )
        
        print(f"✓ W&B visualizer created")
        print(f"  - Entity: {WANDB_CONFIG['entity']}")
        print(f"  - Project: {WANDB_CONFIG['project']}")
        
        # Try to get runs (may fail if no runs exist yet)
        try:
            runs = viz.get_project_runs()
            if runs:
                print(f"  - Found {len(runs)} training runs")
            else:
                print(f"  - No runs found yet (train model to see runs)")
        except:
            print(f"  - Could not fetch runs (check W&B credentials)")
        
        return True
    except Exception as e:
        print(f"✗ W&B connection failed: {e}")
        return False

def check_directories():
    """Check required directories"""
    print("\nChecking directories...")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    required_dirs = [
        'deployment',
        'model_training',
        'docker'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ not found")
            all_exist = False
    
    # Check if models directory exists or needs creation
    models_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(models_dir):
        print(f"⚠ models/ directory will be created when you train the model")
    else:
        print(f"✓ models/")
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 60)
    print("Sketch Recognition App - Component Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Directory Structure", check_directories()))
    results.append(("Python Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Model Architecture", test_model_architecture()))
    results.append(("Inference Module", test_inference()))
    results.append(("W&B Connection", test_wandb_connection()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    
    if all_passed:
        print("✓ All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("1. Train the model: jupyter notebook model_training/train_sketch_model.ipynb")
        print("2. Run the app: ./run_app.sh or streamlit run deployment/app.py")
    else:
        print("⚠ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install -r docker/requirements.txt")
        print("- Update W&B entity in model_training/config.py")
        print("- Train the model for full functionality")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
