#!/bin/bash

# Quick Start Script for Colon Polyp Detection App
echo "🔬 Starting Colon Polyp Detection & Explainability Platform..."
echo "=============================================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Installing Streamlit..."
    pip install streamlit
fi

# Launch the app
echo "🚀 Launching the application..."
echo "📍 The app will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
