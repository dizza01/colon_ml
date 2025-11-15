#!/bin/bash

# Quick start script for validation comparison demo
echo "🚀 Starting Colon Polyp Validation Demo"
echo "==============================================="
echo ""
echo "🔬 This will open the Streamlit app with validation comparison tools"
echo ""
echo "📊 Available Features:"
echo "  • Individual Sample Analysis (Original + Ground Truth + Prediction)"
echo "  • Batch Performance Analysis (Multiple samples with metrics)"
echo "  • Explainability Comparison (Attribution vs Ground Truth)"
echo ""
echo "💡 Navigate to '🔬 Validation Comparison' section in the sidebar"
echo ""
echo "🎯 Sample workflow:"
echo "  1. Select 'Individual Sample Analysis'"
echo "  2. Choose 'Random samples' and set number to 5"
echo "  3. Click 'Generate Random Samples'"
echo "  4. Observe side-by-side comparisons like your notebook"
echo ""
echo "🧠 For explainability:"
echo "  1. Select 'Explainability Comparison'"
echo "  2. Choose a sample index"
echo "  3. Click 'Generate Explainability Analysis'"
echo "  4. See attribution methods vs ground truth"
echo ""
echo "Starting app in 3 seconds..."
sleep 3

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected."
    echo "💡 You may need to activate it first:"
    echo "   source .venv/bin/activate"
    echo ""
fi

# Start the Streamlit app
streamlit run app.py
