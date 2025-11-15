#!/bin/bash
# Test runner script for the Colon Polyp Detection App

set -e  # Exit on any error

echo "🧪 Running Colon Polyp Detection App Test Suite"
echo "=================================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating your virtual environment first:"
    echo "   source .venv/bin/activate"
    echo ""
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Install test dependencies if needed
echo "📦 Checking test dependencies..."
if ! python -c "import pytest" 2>/dev/null; then
    echo "Installing pytest..."
    pip install pytest pytest-timeout
fi

echo ""

# Run quick tests first
echo "🚀 Running Quick Tests (Critical Functionality)"
echo "-------------------------------------------------"
if python quick_test.py; then
    echo "✅ Quick tests passed - proceeding with full test suite"
else
    echo "❌ Quick tests failed - check basic setup before running full tests"
    exit 1
fi

echo ""

# Run full test suite with pytest
echo "🔬 Running Full Test Suite"
echo "--------------------------"

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Using pytest..."
    pytest test_app.py -v --tb=short
else
    echo "Using unittest..."
    python test_app.py
fi

echo ""
echo "✅ Test suite completed!"
echo ""

# Optional: Generate a simple test report
echo "📊 Test Summary"
echo "==============="
echo "Quick Tests:     ✅ Passed"

# Check test results
if [ $? -eq 0 ]; then
    echo "Full Test Suite: ✅ Passed"
    echo ""
    echo "🎉 All tests passed! Your app is ready for deployment."
else
    echo "Full Test Suite: ❌ Failed"
    echo ""
    echo "🔧 Some tests failed. Please review the output above and fix any issues."
    exit 1
fi
