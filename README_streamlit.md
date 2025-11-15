# 🔬 Colon Polyp Detection & Explainability Platform

An interactive Streamlit web application for colon polyp detection using deep learning with comprehensive explainable AI features.

## 🎯 Overview

This application demonstrates an end-to-end machine learning pipeline for medical image analysis, specifically focused on colon polyp detection with explainable AI. It includes:

- **Real-time polyp detection** using a trained U-Net model
- **Multiple explainability methods** (Integrated Gradients, Guided Backprop, Grad-CAM)
- **Quantitative evaluation** of both model performance and explanation quality
- **Interactive web interface** for easy use by medical professionals

## 📋 Features

### 🔍 Detection Capabilities
- Upload colonoscopy images for analysis
- Real-time segmentation with confidence scores
- Adjustable detection thresholds
- Comprehensive visualization of results

### 🧠 Explainability Methods
- **Integrated Gradients**: Most focused attribution method
- **Guided Backprop**: Balanced precision and coverage
- **Grad-CAM**: High-level spatial attention visualization

### 📊 Evaluation Metrics
- Model performance: Dice score, Accuracy, Sensitivity, Specificity
- Explanation quality: Sparseness, Localization metrics
- Quantitative assessment using Quantus library

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU support optional (CUDA compatible)

### Installation

1. **Clone/Navigate to the project directory**:
   ```bash
   cd /colon_ml
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

3. **Activate the environment**:
   ```bash
   source streamlit_env/bin/activate
   ```

4. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### Manual Installation (Alternative)

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv streamlit_env
source streamlit_env/bin/activate

# Install requirements
pip install -r streamlit_requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
colon_ml/
├── app.py                          # Main Streamlit application
├── utils.py                        # Utility functions and model definitions
├── streamlit_requirements.txt      # Python dependencies
├── setup.sh                       # Automated setup script
├── README_streamlit.md            # This file
├── data/
│   └── CVC-ClinicDB/
│       ├── checkpoints/
│       │   └── best_model_dice_0.7879_epoch_49.pth  # Trained model weights
│       └── PNG/                   # Sample images (optional)
└── colon_ml_detection.ipynb      # Original research notebook
```

## 🎮 Usage Guide

### 1. 🏠 Home Page
- Overview of the platform capabilities
- Key performance metrics
- Quick navigation guide

### 2. 🔍 Live Detection
- **Upload Image**: Choose a colonoscopy image (PNG, JPG, JPEG)
- **Set Threshold**: Adjust confidence threshold for detection
- **Run Analysis**: Get segmentation results and metrics
- **View Results**: See prediction overlays and confidence analysis

### 3. 🧠 Explainability Analysis
- **Method Comparison**: Compare different explanation techniques
- **Interactive Learning**: Understand each method's strengths
- **Quality Metrics**: Quantitative evaluation of explanations

### 4. 📊 Model Overview
- **Architecture Details**: U-Net model specifications
- **Training Configuration**: Hyperparameters and dataset info
- **Performance Summary**: Comprehensive metrics overview

### 5. 📈 Model Evaluation
- **Performance Metrics**: Dice score, accuracy, sensitivity, specificity
- **Explanation Quality**: Sparseness, localization precision/recall
- **Comparative Analysis**: Method-by-method evaluation

## 🔧 Configuration

### Model Checkpoint
Place your trained model checkpoint at:
```
data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth
```

The app will automatically load the checkpoint if available. Otherwise, it uses randomly initialized weights.

### GPU Support
The app automatically detects and uses GPU if available:
- CUDA-compatible GPU recommended for faster inference
- CPU fallback supported for systems without GPU

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM for optimal performance
- **GPU**: 2GB+ VRAM if using GPU acceleration

## 📊 Performance Metrics

Our model achieves the following performance on the CVC-ClinicDB dataset:

| Metric | Value |
|--------|-------|
| Dice Score | 0.7879 |
| Accuracy | 94.2% |
| Sensitivity | 88.5% |
| Specificity | 95.1% |

### Explanation Quality Metrics

| Method | Sparseness | Localization Precision | Localization Recall |
|--------|------------|----------------------|-------------------|
| Integrated Gradients | 0.901 | 0.574 | 0.941 |
| Guided Backprop | 0.880 | 0.339 | 0.555 |
| Grad-CAM | 0.678 | 0.285 | 0.423 |

## 🚀 Deployment Options

### Local Development
- Follow the Quick Start guide above
- Access at `http://localhost:8501`

### Streamlit Cloud Deployment
1. Push code to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from repository

### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Production Considerations
- Use environment variables for sensitive configuration
- Implement proper logging and error handling
- Add authentication for medical use cases
- Ensure HIPAA compliance for patient data

## 🛠️ Technical Stack

- **Framework**: Streamlit for web interface
- **Deep Learning**: PyTorch, U-Net architecture
- **Explainability**: Captum (Integrated Gradients, Guided Backprop, Grad-CAM)
- **Evaluation**: Quantus metrics library
- **Visualization**: Matplotlib, Plotly
- **Image Processing**: OpenCV, PIL

## 📚 Research Background

This project demonstrates the application of explainable AI to medical image analysis. Key research contributions include:

1. **Comprehensive XAI Evaluation**: Multiple explanation methods with quantitative assessment
2. **Medical Domain Application**: Real-world relevance to gastroenterology
3. **Interactive Platform**: Accessible interface for medical professionals
4. **Reproducible Pipeline**: Complete end-to-end workflow

## 🤝 Contributing

Contributions are welcome! Please consider:

1. **Code Quality**: Follow PEP 8 style guidelines
2. **Documentation**: Update documentation for new features
3. **Testing**: Add tests for new functionality
4. **Medical Accuracy**: Ensure medical relevance and safety

## 📄 License

This project is for research and educational purposes. Please ensure compliance with relevant medical data regulations when using with patient data.

## 📧 Contact

For questions, collaborations, or support:
- Create an issue in the repository
- Contact the development team

## 🔗 Related Resources

- [CVC-ClinicDB Dataset](http://mv.cvc.uab.es/projects/colon-qa/cvcdb)
- [Captum Documentation](https://captum.ai/)
- [Quantus Library](https://quantus.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

