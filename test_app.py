"""
Unit tests for the Colon Polyp Detection App
============================================

This test suite covers:
- Model architecture and checkpoint loading
- Image preprocessing
- Model predictions
- Explanation generation
- Utility functions

Run with: python -m pytest test_app.py -v
Or: python test_app.py
"""

import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image
import warnings

# Import modules to test
from utils import (
    UNet, double_conv, BatchedScalarModel, 
    preprocess_image, predict_segmentation, generate_explanations,
    tensor_to_numpy_img, calculate_metrics, evaluate_explanations,
    load_validation_data, sample_validation_images,
    create_comparison_visualization, create_explainability_comparison,
    calculate_validation_metrics
)

class TestUNetArchitecture(unittest.TestCase):
    """Test the UNet model architecture"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = UNet(n_class=1)
        
    def test_model_creation(self):
        """Test that UNet model can be created"""
        self.assertIsInstance(self.model, UNet)
        # Count parameters
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(param_count, 0, "Model should have trainable parameters")
        
    def test_model_forward_pass(self):
        """Test forward pass with dummy input"""
        # Create dummy input (batch_size=1, channels=3, height=256, width=256)
        dummy_input = torch.randn(1, 3, 256, 256)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
            
        # Check output shape
        self.assertEqual(output.shape, (1, 1, 256, 256))
        self.assertTrue(torch.is_tensor(output))
        
    def test_double_conv_function(self):
        """Test the double_conv helper function"""
        conv_block = double_conv(3, 64)
        
        # Check that it's a Sequential module with 4 layers (Conv-ReLU-Conv-ReLU)
        self.assertEqual(len(conv_block), 4)
        self.assertIsInstance(conv_block[0], torch.nn.Conv2d)
        self.assertIsInstance(conv_block[1], torch.nn.ReLU)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        output = conv_block(dummy_input)
        self.assertEqual(output.shape, (1, 64, 256, 256))


class TestModelCheckpointLoading(unittest.TestCase):
    """Test model checkpoint loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = UNet(n_class=1)
        self.checkpoint_path = "data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth"
        
    def test_checkpoint_exists(self):
        """Test that the checkpoint file exists"""
        self.assertTrue(Path(self.checkpoint_path).exists(), 
                       f"Checkpoint file not found: {self.checkpoint_path}")
        
    def test_checkpoint_loading(self):
        """Test that checkpoint can be loaded successfully"""
        if not Path(self.checkpoint_path).exists():
            self.skipTest("Checkpoint file not available")
            
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            self.fail(f"Failed to load checkpoint: {e}")
            
    def test_checkpoint_structure(self):
        """Test the structure of the checkpoint"""
        if not Path(self.checkpoint_path).exists():
            self.skipTest("Checkpoint file not available")
            
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Should be a state_dict (dictionary of layer names to tensors)
        self.assertIsInstance(checkpoint, dict)
        
        # Check for expected layer names
        expected_layers = ['dconv_down1.0.weight', 'dconv_down1.0.bias', 'conv_last.weight', 'conv_last.bias']
        for layer in expected_layers:
            self.assertIn(layer, checkpoint.keys(), f"Missing layer: {layer}")
            
    def test_model_state_after_loading(self):
        """Test that model is in correct state after loading checkpoint"""
        if not Path(self.checkpoint_path).exists():
            self.skipTest("Checkpoint file not available")
            
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Test that model can still perform forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = self.model(dummy_input)
            
        self.assertEqual(output.shape, (1, 1, 256, 256))
        self.assertFalse(torch.isnan(output).any(), "Model output contains NaN values")


class TestImagePreprocessing(unittest.TestCase):
    """Test image preprocessing functions"""
    
    def test_preprocess_numpy_image(self):
        """Test preprocessing with numpy array input"""
        # Create dummy RGB image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        tensor, rgb_image = preprocess_image(dummy_image, target_size=(256, 256))
        
        # Check tensor properties
        self.assertEqual(tensor.shape, (1, 3, 256, 256))
        self.assertTrue(torch.is_tensor(tensor))
        self.assertTrue((tensor >= 0).all() and (tensor <= 1).all(), "Tensor values should be normalized to [0,1]")
        
        # Check RGB image
        self.assertEqual(rgb_image.shape, (256, 256, 3))
        self.assertEqual(rgb_image.dtype, np.uint8)
        
    def test_preprocess_pil_image(self):
        """Test preprocessing with PIL Image input"""
        # Create dummy PIL image
        dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_array)
        
        tensor, rgb_image = preprocess_image(pil_image, target_size=(128, 128))
        
        # Check tensor properties
        self.assertEqual(tensor.shape, (1, 3, 128, 128))
        self.assertTrue((tensor >= 0).all() and (tensor <= 1).all())


class TestModelPrediction(unittest.TestCase):
    """Test model prediction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = UNet(n_class=1)
        self.model.eval()
        
    def test_predict_segmentation(self):
        """Test segmentation prediction"""
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        
        prediction, binary_mask = predict_segmentation(self.model, dummy_input, self.device)
        
        # Check prediction properties
        self.assertEqual(prediction.shape, (1, 1, 256, 256))
        self.assertTrue((prediction >= 0).all() and (prediction <= 1).all(), 
                       "Prediction values should be in [0,1] after sigmoid")
        
        # Check binary mask
        self.assertEqual(binary_mask.shape, (1, 1, 256, 256))
        self.assertTrue(torch.all((binary_mask == 0) | (binary_mask == 1)), 
                       "Binary mask should only contain 0s and 1s")


class TestBatchedScalarModel(unittest.TestCase):
    """Test the BatchedScalarModel wrapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_model = UNet(n_class=1)
        self.wrapped_model_mean = BatchedScalarModel(self.base_model, reduction='mean')
        self.wrapped_model_sum = BatchedScalarModel(self.base_model, reduction='sum')
        
    def test_wrapped_model_output_shape(self):
        """Test that wrapped model returns scalar per sample"""
        dummy_input = torch.randn(2, 3, 256, 256)  # batch size 2
        
        # Test mean reduction
        output_mean = self.wrapped_model_mean(dummy_input)
        self.assertEqual(output_mean.shape, (2,), "Mean reduction should return (batch,) shape")
        
        # Test sum reduction
        output_sum = self.wrapped_model_sum(dummy_input)
        self.assertEqual(output_sum.shape, (2,), "Sum reduction should return (batch,) shape")
        
    def test_wrapped_model_scalar_values(self):
        """Test that wrapped model outputs are scalars"""
        dummy_input = torch.randn(1, 3, 256, 256)
        
        output = self.wrapped_model_mean(dummy_input)
        self.assertEqual(len(output.shape), 1, "Output should be 1D")
        self.assertEqual(output.shape[0], 1, "Batch size should be preserved")


class TestExplanationGeneration(unittest.TestCase):
    """Test explanation generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = UNet(n_class=1)
        self.model.eval()
        
    def test_generate_explanations(self):
        """Test that explanations can be generated without errors"""
        dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)
        
        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanations = generate_explanations(self.model, dummy_input, self.device)
        
        # Check that we get some explanations
        self.assertIsInstance(explanations, dict)
        
        # Check that explanations have the right shape if they exist
        for method, explanation in explanations.items():
            self.assertIsInstance(explanation, np.ndarray, f"Explanation {method} should be numpy array")
            self.assertEqual(len(explanation.shape), 4, f"Explanation {method} should have 4D shape")
            
    def test_explanation_methods(self):
        """Test that expected explanation methods are attempted"""
        dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanations = generate_explanations(self.model, dummy_input, self.device)
        
        # At least one explanation method should work
        self.assertGreater(len(explanations), 0, "At least one explanation method should succeed")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_tensor_to_numpy_img(self):
        """Test tensor to numpy image conversion"""
        # Test 4D tensor (batch, channel, height, width)
        tensor_4d = torch.randn(1, 3, 64, 64)
        result = tensor_to_numpy_img(tensor_4d, normalize=True)
        
        self.assertEqual(len(result.shape), 3)  # Should be (H, W, C)
        self.assertEqual(result.shape[2], 3)    # 3 channels
        self.assertTrue((result >= 0).all() and (result <= 1).all())  # Normalized
        
        # Test 3D tensor (channel, height, width)
        tensor_3d = torch.randn(1, 64, 64)
        result = tensor_to_numpy_img(tensor_3d, normalize=True)
        
        self.assertEqual(len(result.shape), 2)  # Should be (H, W)
        
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Create dummy prediction and ground truth
        prediction = torch.sigmoid(torch.randn(1, 1, 64, 64))
        ground_truth = torch.randint(0, 2, (1, 1, 64, 64)).float()
        
        metrics = calculate_metrics(prediction, ground_truth)
        
        # Check that expected metrics are present
        expected_metrics = ['mean_confidence', 'polyp_area_percentage', 'dice_score', 'iou', 'pixel_accuracy']
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
            self.assertIsInstance(metrics[metric], (int, float, np.number))
            
        # Check metric ranges
        self.assertTrue(0 <= metrics['dice_score'] <= 1, "Dice score should be in [0,1]")
        self.assertTrue(0 <= metrics['iou'] <= 1, "IoU should be in [0,1]")
        self.assertTrue(0 <= metrics['pixel_accuracy'] <= 1, "Pixel accuracy should be in [0,1]")


class TestAppIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        
        # Only run integration tests if checkpoint exists
        checkpoint_path = "data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth"
        if Path(checkpoint_path).exists():
            self.model = UNet(n_class=1)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.checkpoint_available = True
        else:
            self.checkpoint_available = False
            
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from image to explanations"""
        if not self.checkpoint_available:
            self.skipTest("Checkpoint not available for integration test")
            
        # Create a dummy colonoscopy-like image
        dummy_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        
        try:
            # 1. Preprocess image
            input_tensor, image_rgb = preprocess_image(dummy_image)
            
            # 2. Make prediction
            prediction, binary_mask = predict_segmentation(self.model, input_tensor, self.device)
            
            # 3. Generate explanations (with warnings suppressed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explanations = generate_explanations(self.model, input_tensor, self.device)
            
            # 4. Calculate metrics
            metrics = calculate_metrics(prediction)
            
            # Verify pipeline completed successfully
            self.assertIsNotNone(prediction)
            self.assertIsInstance(explanations, dict)
            self.assertIsInstance(metrics, dict)
            
            print("✅ End-to-end pipeline test passed")
            
        except Exception as e:
            self.fail(f"End-to-end pipeline failed: {e}")


class TestValidationFunctionality(unittest.TestCase):
    """Test validation dataset functionality"""
    
    def setUp(self):
        """Set up test fixtures for validation tests"""
        self.device = torch.device('cpu')
        
    def test_validation_data_loading(self):
        """Test validation data loading functionality"""
        try:
            from utils import load_validation_data
            X_val, y_val, available = load_validation_data()
            
            if available:
                print("✅ Validation data is available for testing")
                self.assertIsInstance(X_val, torch.Tensor, "X_val should be a tensor")
                self.assertIsInstance(y_val, torch.Tensor, "y_val should be a tensor")
                self.assertEqual(len(X_val.shape), 4, "X_val should be 4D (N,C,H,W)")
                self.assertEqual(len(y_val.shape), 4, "y_val should be 4D (N,C,H,W)")
                self.assertEqual(X_val.shape[1], 3, "X_val should have 3 channels")
                self.assertEqual(y_val.shape[1], 1, "y_val should have 1 channel")
                print(f"   📊 Validation dataset shape: X={X_val.shape}, y={y_val.shape}")
            else:
                print("⚠️ Validation data not available - skipping validation tests")
                self.skipTest("Validation data not available")
                
        except ImportError as e:
            self.fail(f"Could not import validation functions: {e}")
    
    def test_validation_sampling(self):
        """Test validation dataset sampling functionality"""
        try:
            from utils import load_validation_data, sample_validation_images
            X_val, y_val, available = load_validation_data()
            
            if not available:
                self.skipTest("Validation data not available")
                
            # Test sampling
            n_samples = min(5, len(X_val))
            sampled_X, sampled_y, indices = sample_validation_images(X_val, y_val, n_samples)
            
            self.assertEqual(len(sampled_X), n_samples, "Should return requested number of samples")
            self.assertEqual(len(sampled_y), n_samples, "Should return same number of X and y samples")
            self.assertEqual(len(indices), n_samples, "Should return same number of indices")
            self.assertTrue(all(0 <= idx < len(X_val) for idx in indices), "Indices should be valid")
            
            print(f"✅ Validation sampling works: {n_samples} samples selected")
            
        except ImportError as e:
            self.fail(f"Could not import validation functions: {e}")
    
    def test_comparison_visualization(self):
        """Test validation comparison visualization creation"""
        try:
            from utils import create_comparison_visualization
            
            # Create dummy data
            image = np.random.rand(256, 256, 3)
            ground_truth = np.random.randint(0, 2, (256, 256))
            prediction = np.random.rand(256, 256)
            
            # Test 3-panel visualization
            fig = create_comparison_visualization(image, ground_truth, prediction)
            self.assertIsNotNone(fig, "Should create a matplotlib figure")
            
            # Test 4-panel visualization with probabilities
            fig_with_prob = create_comparison_visualization(image, ground_truth, prediction, prediction)
            self.assertIsNotNone(fig_with_prob, "Should create figure with probabilities")
            
            print("✅ Comparison visualization creation works")
            
        except ImportError as e:
            self.fail(f"Could not import visualization functions: {e}")
    
    def test_explainability_comparison(self):
        """Test explainability comparison visualization"""
        try:
            from utils import create_explainability_comparison
            
            # Create dummy data
            image = np.random.rand(256, 256, 3)
            ground_truth = np.random.randint(0, 2, (256, 256))
            
            # Create dummy explanations
            explanations = {
                'Method1': np.random.rand(1, 3, 256, 256),
                'Method2': np.random.rand(1, 1, 256, 256),
            }
            
            fig = create_explainability_comparison(image, ground_truth, explanations)
            self.assertIsNotNone(fig, "Should create explainability comparison figure")
            
            print("✅ Explainability comparison visualization works")
            
        except ImportError as e:
            self.fail(f"Could not import explainability functions: {e}")
    
    def test_validation_metrics_calculation(self):
        """Test validation metrics calculation"""
        try:
            from utils import calculate_validation_metrics
            
            # Create dummy validation data
            n_samples = 10
            predictions = np.random.rand(n_samples, 256, 256)
            ground_truths = np.random.randint(0, 2, (n_samples, 256, 256))
            
            metrics = calculate_validation_metrics(predictions, ground_truths)
            
            # Check that all expected metrics are present
            expected_keys = ['mean_dice', 'std_dice', 'mean_iou', 'std_iou', 
                           'mean_accuracy', 'std_accuracy', 'individual_dice', 
                           'individual_iou', 'individual_accuracy']
            
            for key in expected_keys:
                self.assertIn(key, metrics, f"Metrics should contain {key}")
            
            # Check value ranges
            self.assertTrue(0 <= metrics['mean_dice'] <= 1, "Dice should be in [0,1]")
            self.assertTrue(0 <= metrics['mean_iou'] <= 1, "IoU should be in [0,1]")
            self.assertTrue(0 <= metrics['mean_accuracy'] <= 1, "Accuracy should be in [0,1]")
            
            # Check individual metrics
            self.assertEqual(len(metrics['individual_dice']), n_samples, "Should have per-sample dice scores")
            
            print("✅ Validation metrics calculation works")
            
        except ImportError as e:
            self.fail(f"Could not import metrics functions: {e}")
    
    def test_end_to_end_validation_pipeline(self):
        """Test end-to-end validation pipeline"""
        try:
            from utils import (load_validation_data, sample_validation_images, 
                             predict_segmentation, calculate_metrics, UNet)
            
            X_val, y_val, available = load_validation_data()
            
            if not available:
                self.skipTest("Validation data not available")
            
            # Create a simple model for testing
            model = UNet(n_class=1)
            model.eval()
            
            # Sample one image
            sampled_X, sampled_y, indices = sample_validation_images(X_val, y_val, 1)
            
            # Run prediction
            image_tensor = sampled_X[0].unsqueeze(0).to(self.device)
            ground_truth = sampled_y[0].squeeze(0).cpu().numpy()
            
            with torch.no_grad():
                prediction, binary_mask = predict_segmentation(model, image_tensor, self.device)
            
            # Calculate metrics
            metrics = calculate_metrics(prediction, torch.tensor(ground_truth).unsqueeze(0).unsqueeze(0))
            
            # Verify metrics are calculated
            self.assertIn('dice_score', metrics, "Should calculate Dice score")
            self.assertIn('iou', metrics, "Should calculate IoU")
            self.assertIn('pixel_accuracy', metrics, "Should calculate pixel accuracy")
            
            print("✅ End-to-end validation pipeline works")
            print(f"   📊 Sample metrics: Dice={metrics['dice_score']:.3f}, IoU={metrics['iou']:.3f}")
            
        except ImportError as e:
            self.fail(f"Could not import validation pipeline functions: {e}")

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].strip()}")
            
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(':', 1)[-1].strip()}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Colon Polyp Detection App Tests")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the tests
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        exit(1)
