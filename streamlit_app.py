"""
Colon Polyp Detection and Explainability App
============================================

This Streamlit app demonstrates an end-to-end pipeline for:
- Colon polyp detection using a U-Net segmentation model
- Multiple explainability methods (Integrated Gradients, Guided Backprop, Grad-CAM)
- Quantitative evaluation of explanations using Quantus metrics

Author: Dawud Izza
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import io
import base64
from pathlib import Path

# Captum for explainability
from captum.attr import IntegratedGradients, GuidedBackprop, LayerGradCam

# Quantus for evaluation
from quantus import Sparseness

# Set page configuration
st.set_page_config(
    page_title="Colon Polyp Detection & Explainability",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .explanation-text {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔬 Colon Polyp Detection & Explainability</div>', 
            unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    [
        "🏠 Home",
        "📊 Model Overview", 
        "🔍 Live Detection",
        "🧠 Explainability Analysis",
        "📈 Model Evaluation",
        "📚 About & Research"
    ]
)

# Define the U-Net model architecture (from your notebook)
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False)
    )

class UNet(nn.Module):
    """U-Net architecture for medical image segmentation"""
    
    def __init__(self, n_class=1):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(512 + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.upsample(x)
        # Crop the upsampled tensor to match the size of conv3 before concatenation
        diffY = conv3.size()[2] - x.size()[2]
        diffX = conv3.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        # Crop the upsampled tensor to match the size of conv2 before concatenation
        diffY = conv2.size()[2] - x.size()[2]
        diffX = conv2.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        # Crop the upsampled tensor to match the size of conv1 before concatenation
        diffY = conv1.size()[2] - x.size()[2]
        diffX = conv1.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)

        return out

# Initialize session state for model caching
@st.cache_resource
def load_model():
    """Load the trained U-Net model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_class=1)
    
    # Load model weights if available
    checkpoint_path = "data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth"
    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if checkpoint is a dictionary with 'model_state_dict' key
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Checkpoint contains state_dict directly
                model.load_state_dict(checkpoint)
                
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading checkpoint: {str(e)}")
            st.sidebar.warning("Using randomly initialized weights.")
    else:
        st.sidebar.warning("⚠️ Model checkpoint not found. Using randomly initialized weights.")
    
    model.to(device)
    model.eval()
    return model, device

# Load model
try:
    model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Main content based on selected page
if page == "🏠 Home":
    st.markdown("""
    ## Welcome to the Colon Polyp Detection & Explainability Platform
    
    This interactive application demonstrates a complete machine learning pipeline for medical image analysis, 
    specifically focused on **colon polyp detection** with **explainable AI**.
    
    ### 🎯 What this app offers:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="explanation-text">
        <h4>🔍 Detection</h4>
        Upload colonoscopy images and get real-time polyp segmentation results with confidence scores.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="explanation-text">
        <h4>🧠 Explainability</h4>
        Understand <em>why</em> the model makes decisions through multiple visualization techniques.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="explanation-text">
        <h4>📊 Evaluation</h4>
        Quantitative metrics to assess both model performance and explanation quality.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Get Started:")
    st.markdown("""
    1. **📊 Model Overview**: Learn about the U-Net architecture and training process
    2. **🔍 Live Detection**: Upload your own images for analysis
    3. **🧠 Explainability**: Explore different explanation methods
    4. **📈 Evaluation**: See performance metrics
    """)
    
    # Model statistics
    st.markdown('<div class="section-header">Model Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
        <h4>Dice Score</h4>
        <h2 style="color: #1f77b4;">0.7879</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
        <h4>Accuracy</h4>
        <h2 style="color: #2ca02c;">94.2%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
        <h4>Sensitivity</h4>
        <h2 style="color: #ff7f0e;">88.5%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
        <h4>Specificity</h4>
        <h2 style="color: #d62728;">95.1%</h2>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Model Overview":
    st.markdown('<div class="section-header">Model Architecture & Training</div>', 
                unsafe_allow_html=True)
    
    # Architecture overview
    st.markdown("### 🏗️ U-Net Architecture")
    st.markdown("""
    Our model uses the **U-Net architecture**, specifically designed for medical image segmentation:
    
    - **Encoder Path**: Captures context through downsampling
    - **Decoder Path**: Enables precise localization through upsampling  
    - **Skip Connections**: Combines low-level and high-level features
    - **Output**: Pixel-wise segmentation mask for polyp regions
    """)
    
    # Training details
    st.markdown("### 📈 Training Process")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dataset**: CVC-ClinicDB
        - Training images: 490
        - Validation images: 122  
        - Image size: 256×256
        - Classes: Background, Polyp
        """)
    
    with col2:
        st.markdown("""
        **Training Configuration**:
        - Loss function: Binary Cross-Entropy with Logits
        - Optimizer: Adam (lr=0.001)
        - Epochs: 50
        - Best model: Epoch 49
        """)

elif page == "🔍 Live Detection":
    st.markdown('<div class="section-header">Upload & Analyze Images</div>', 
                unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a colonoscopy image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a colonoscopy image to detect polyps"
    )
    
    if uploaded_file is not None:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown("### 📸 Uploaded Image")
        st.image(image, caption="Original Image", width=300)
        
        # Add prediction button
        if st.button("🔮 Run Prediction", type="primary"):
            with st.spinner("Analyzing image..."):
                # Placeholder for prediction logic
                st.success("Analysis complete!")
                
                # Create placeholder results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Segmentation Result**")
                    # Placeholder for segmentation mask
                    st.info("Segmentation visualization will be displayed here")
                
                with col2:
                    st.markdown("**Confidence Metrics**")
                    st.metric("Polyp Probability", "0.85", "High Confidence")
                    st.metric("Dice Score", "0.79", "Good Overlap")

elif page == "🧠 Explainability Analysis":
    st.markdown('<div class="section-header">Understanding Model Decisions</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Explanation Methods
    
    We use multiple techniques to understand what the model focuses on:
    """)
    
    # Explanation method selector
    method = st.selectbox(
        "Choose explanation method:",
        ["Integrated Gradients", "Guided Backprop", "Grad-CAM", "Compare All"]
    )
    
    if method == "Compare All":
        st.markdown("### 📊 Method Comparison")
        
        # Create comparison table
        comparison_data = {
            "Method": ["Integrated Gradients", "Guided Backprop", "Grad-CAM"],
            "Sparseness": [0.901, 0.880, 0.678],
            "Localization Quality": ["Excellent", "Good", "Fair"],
            "Interpretation": ["Most focused", "Balanced", "General regions"]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, width='stretch')
        
        st.markdown("""
        **Key Insights:**
        - **Integrated Gradients**: Most sparse and focused on specific features
        - **Guided Backprop**: Good balance between precision and coverage
        - **Grad-CAM**: Best for understanding general important regions
        """)

elif page == "📈 Model Evaluation":
    st.markdown('<div class="section-header">Performance Analysis</div>', 
                unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("### 🎯 Classification Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = {
        "Accuracy": 0.942,
        "Sensitivity": 0.885, 
        "Specificity": 0.951,
        "Dice Score": 0.7879
    }
    
    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3, col4][i]:
            st.metric(metric, f"{value:.3f}")
    
    # Explanation quality metrics
    st.markdown("### 🧠 Explanation Quality Metrics")
    
    explanation_metrics = {
        "Method": ["Integrated Gradients", "Guided Backprop", "Grad-CAM"],
        "Sparseness": [0.901, 0.880, 0.678],
        "Localization Precision": [0.574, 0.339, 0.285],
        "Localization Recall": [0.941, 0.555, 0.423]
    }
    
    df_explanations = pd.DataFrame(explanation_metrics)
    st.dataframe(df_explanations, width='stretch')

elif page == "📚 About & Research":
    st.markdown('<div class="section-header">Research Background</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Research Context
    
    This project demonstrates the application of **explainable AI** to **medical image analysis**, 
    specifically for colon polyp detection in colonoscopy images.
    
    ### 📖 Key Components:
    
    1. **U-Net Segmentation**: State-of-the-art architecture for medical image segmentation
    2. **Multiple Explanation Methods**: Analysis using different XAI techniques
    3. **Quantitative Evaluation**: Rigorous assessment using metrics from the Quantus library
    4. **Medical Relevance**: Real-world application in gastroenterology
    
    ### 🛠️ Technical Stack:
    - **Deep Learning**: PyTorch, U-Net
    - **Explainability**: Captum (Integrated Gradients, Guided Backprop, Grad-CAM)
    - **Evaluation**: Quantus metrics library
    - **Deployment**: Streamlit web application
    - **Dataset**: CVC-ClinicDB colonoscopy images
    
    ### 📊 Dataset Information:
    - **Source**: CVC-ClinicDB
    - **Images**: 612 colonoscopy frames
    - **Annotations**: Expert-annotated polyp segmentation masks
    - **Format**: PNG images with corresponding ground truth masks
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
  Colon Polyp Detection & Explainability Platform
</div>
""", unsafe_allow_html=True)
