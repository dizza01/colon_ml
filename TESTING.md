# Testing Guide for Colon Polyp Detection App

This directory contains comprehensive tests for the colon polyp detection and explainability application.

## Test Files Overview

### 1. `quick_test.py` - Quick Smoke Tests
**Purpose**: Fast tests for critical functionality  
**Runtime**: ~30 seconds  
**Use case**: Before development, deployment, or when troubleshooting

```bash
python quick_test.py
```

**Tests included**:
- ✅ Module imports
- ✅ Model architecture
- ✅ Checkpoint loading
- ✅ Image preprocessing
- ✅ Basic predictions
- ✅ Streamlit app syntax

### 2. `test_app.py` - Comprehensive Test Suite
**Purpose**: Thorough testing of all components  
**Runtime**: ~2-5 minutes  
**Use case**: Before commits, releases, or major changes

```bash
python test_app.py
# OR with pytest
pytest test_app.py -v
```

**Test categories**:
- **UNet Architecture**: Model creation, forward pass, layer structure
- **Checkpoint Loading**: File existence, loading, compatibility
- **Image Preprocessing**: PIL/NumPy inputs, resizing, normalization
- **Model Prediction**: Segmentation output, sigmoid activation, binary masks
- **BatchedScalarModel**: Wrapper for explanation methods
- **Explanation Generation**: Integrated Gradients, Guided Backprop, Grad-CAM
- **Utility Functions**: Tensor conversion, metrics calculation
- **Integration Tests**: End-to-end pipeline testing

### 3. `run_tests.sh` - Automated Test Runner
**Purpose**: Run all tests with proper setup and reporting

```bash
./run_tests.sh
```

## Test Dependencies

### Required Packages
```bash
pip install pytest pytest-timeout  # Optional, for advanced testing
```

### Required Files
- `utils.py` - Core utility functions
- `app.py` or `streamlit_app.py` - Streamlit applications  
- `data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth` - Model checkpoint (optional)

## Running Tests

### Option 1: Quick Tests Only
```bash
python quick_test.py
```
Best for: Development workflow, CI/CD pipelines

### Option 2: Full Test Suite
```bash
python test_app.py
```
Best for: Thorough validation, before deployment

### Option 3: With pytest (if installed)
```bash
pytest test_app.py -v --tb=short
```
Best for: Detailed reporting, test filtering

### Option 4: Automated Runner
```bash
./run_tests.sh
```
Best for: One-command testing, CI/CD integration

## Test Categories

### 🟢 Unit Tests
Test individual components in isolation:
- Model architecture
- Individual functions
- Data preprocessing
- Utility functions

### 🟡 Integration Tests  
Test component interactions:
- Model + checkpoint loading
- Preprocessing + prediction pipeline
- Explanation generation workflow

### 🔴 End-to-End Tests
Test complete user workflows:
- Image upload → prediction → explanations
- Model loading → inference → visualization

## Common Test Scenarios

### ✅ Development Workflow
```bash
# Before making changes
python quick_test.py

# After making changes  
python test_app.py

# Before committing
./run_tests.sh
```

### ✅ Deployment Checklist
```bash
# 1. Verify environment
python quick_test.py

# 2. Full validation
python test_app.py

# 3. Streamlit app test
streamlit run app.py --server.headless true
```

### ✅ Troubleshooting Issues
```bash
# Test specific component
python -c "from test_app import TestModelCheckpointLoading; import unittest; unittest.main()"

# Test with detailed output
pytest test_app.py::TestUNetArchitecture -v -s
```

## Test Configuration

### pytest.ini
Configuration for pytest runner with:
- Test discovery patterns
- Output formatting  
- Warning filters
- Timeout settings

### Environment Variables
```bash
export PYTHONPATH=.  # Ensure imports work
export CUDA_VISIBLE_DEVICES=""  # Force CPU testing
```

## Expected Test Results

### ✅ All Tests Pass
```
Results: 6/6 tests passed (100.0%)
🎉 All quick tests passed! Your app is ready to run.
```

### ⚠️ Checkpoint Missing
```
⚠️ Checkpoint file not found - skipping checkpoint test
```
*Normal if you haven't downloaded the model checkpoint*

### ❌ Test Failures
Check for:
1. Missing dependencies (`pip install -r requirements.txt`)
2. Incorrect Python environment
3. Missing model checkpoint file
4. CUDA/GPU configuration issues

## Writing New Tests

### Test Structure
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model = UNet(n_class=1)
        
    def test_new_functionality(self):
        """Test description"""
        # Arrange
        input_data = create_test_data()
        
        # Act  
        result = function_to_test(input_data)
        
        # Assert
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(condition_to_check)
```

### Best Practices
1. **Isolation**: Each test should be independent
2. **Clarity**: Use descriptive test names and docstrings  
3. **Speed**: Keep tests fast (mock expensive operations)
4. **Reliability**: Tests should pass consistently
5. **Coverage**: Test both success and failure cases

## Continuous Integration

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    python quick_test.py
    python test_app.py
```

### Local Git Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
python quick_test.py || exit 1
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **CUDA Out of Memory**
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Force CPU
   ```

3. **Missing Checkpoint**
   - Download model checkpoint or skip checkpoint-dependent tests

4. **Streamlit Import Issues**
   ```bash
   pip install streamlit
   ```

For more help, run tests with verbose output:
```bash
python test_app.py --verbose
```
