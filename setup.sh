#!/bin/bash

# Streamlit App Setup Script for Colon Polyp Detection
# ===================================================

echo "🔬 Setting up Colon Polyp Detection & Explainability App"
echo "========================================================="

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo "Python version: $python_version"

if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" = "3.8" ]; then
    echo "✅ Python 3.8+ detected"
else
    echo "❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv streamlit_env
source streamlit_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r streamlit_requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the app:"
echo "   1. Activate the environment: source streamlit_env/bin/activate"
echo "   2. Run the app: streamlit run app.py"
echo ""
echo "🌐 The app will be available at: http://localhost:8501"
echo ""
echo "📋 Optional: Place your trained model checkpoint at:"
echo "   data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth"
