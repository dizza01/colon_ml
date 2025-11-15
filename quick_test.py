"""
Quick Tests for Critical App Functionality
==========================================

This is a lightweight test suite focusing on the most critical components:
- Model architecture compatibility
- Checkpoint loading
- Basic preprocessing
- Core utility functions

Run with: python quick_test.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from utils import UNet, double_conv, preprocess_image, predict_segmentation
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_architecture():
    """Test UNet model creation and basic functionality"""
    try:
        from utils import UNet
        
        model = UNet(n_class=1)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {output.shape}"
        print("✅ Model architecture test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

def test_checkpoint_loading():
    """Test checkpoint loading functionality"""
    try:
        from utils import UNet
        
        checkpoint_path = "data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth"
        
        if not Path(checkpoint_path).exists():
            print("⚠️ Checkpoint file not found - skipping checkpoint test")
            return True
            
        model = UNet(n_class=1)
        device = torch.device('cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        # Test that model works after loading
        dummy_input = torch.randn(1, 3, 256, 256)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            
        assert not torch.isnan(output).any(), "Model output contains NaN values"
        print("✅ Checkpoint loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint loading test failed: {e}")
        return False

def test_image_preprocessing():
    """Test image preprocessing functionality"""
    try:
        from utils import preprocess_image
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        tensor, rgb_image = preprocess_image(dummy_image, target_size=(256, 256))
        
        # Validate tensor
        assert tensor.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {tensor.shape}"
        assert torch.all(tensor >= 0) and torch.all(tensor <= 1), "Tensor should be normalized to [0,1]"
        
        # Validate RGB image
        assert rgb_image.shape == (256, 256, 3), f"Expected (256, 256, 3), got {rgb_image.shape}"
        assert rgb_image.dtype == np.uint8, f"Expected uint8, got {rgb_image.dtype}"
        
        print("✅ Image preprocessing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Image preprocessing test failed: {e}")
        return False

def test_prediction_functionality():
    """Test model prediction functionality"""
    try:
        from utils import UNet, predict_segmentation
        
        model = UNet(n_class=1)
        device = torch.device('cpu')
        dummy_input = torch.randn(1, 3, 256, 256)
        
        prediction, binary_mask = predict_segmentation(model, dummy_input, device)
        
        # Validate prediction
        assert prediction.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {prediction.shape}"
        assert torch.all(prediction >= 0) and torch.all(prediction <= 1), "Prediction should be in [0,1]"
        
        # Validate binary mask
        assert binary_mask.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {binary_mask.shape}"
        assert torch.all((binary_mask == 0) | (binary_mask == 1)), "Binary mask should only contain 0s and 1s"
        
        print("✅ Prediction functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Prediction functionality test failed: {e}")
        return False

def test_streamlit_compatibility():
    """Test that app files can be imported without errors"""
    try:
        # Check if app files exist and are importable
        app_files = ['app.py', 'streamlit_app.py']
        
        for app_file in app_files:
            if Path(app_file).exists():
                # Basic syntax check by trying to compile
                with open(app_file, 'r') as f:
                    content = f.read()
                
                compile(content, app_file, 'exec')
                print(f"✅ {app_file} syntax check passed")
            else:
                print(f"⚠️ {app_file} not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Streamlit compatibility test failed: {e}")
        return False

def run_quick_tests():
    """Run all quick tests"""
    print("Running Quick Tests for Colon Polyp Detection App")
    print("=" * 55)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Architecture", test_model_architecture),
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Image Preprocessing", test_image_preprocessing),
        ("Prediction Functionality", test_prediction_functionality),
        ("Streamlit Compatibility", test_streamlit_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"  {test_name} failed")
        except Exception as e:
            print(f"  {test_name} crashed: {e}")
    
    print("\n" + "=" * 55)
    print(f"Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All quick tests passed! Your app is ready to run.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for issues.")
        return False

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
