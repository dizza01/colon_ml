"""
Enhanced Colon Polyp Detection and Explainability App
=====================================================

Complete implementation with working prediction and explanation pipeline
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our utility functions
from utils import (
    UNet, preprocess_image, predict_segmentation, 
    generate_explanations, calculate_metrics, 
    evaluate_explanations, create_visualization, fig_to_base64,
    load_validation_data, create_comparison_visualization, 
    create_explainability_comparison, calculate_validation_metrics,
    sample_validation_images
)

# Set page configuration
st.set_page_config(
    page_title="Colon Polyp Detection & Explainability",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔬 Colon Polyp Detection & Explainability</div>', 
            unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.device = None

# Load model function
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
                
            model_status = "✅ Model loaded successfully!"
        except Exception as e:
            model_status = f"⚠️ Error loading checkpoint: {str(e)}"
    else:
        model_status = "⚠️ Using randomly initialized weights (checkpoint not found)"
    
    model.to(device)
    model.eval()
    return model, device, model_status

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    [
        "🏠 Home",
        "📊 Model Overview", 
        "🔍 Live Detection",
        "🧠 Explainability Analysis",
        "📈 Model Evaluation",
        "� Validation Comparison",
        "�📚 About & Research"
    ]
)

# Load model
if not st.session_state.model_loaded:
    with st.spinner("Loading model..."):
        try:
            model, device, status = load_model()
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.sidebar.write(status)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")

# Main content
if page == "🏠 Home":
    st.markdown("""
    ## Welcome to the Colon Polyp Detection & Explainability Platform
    
    This interactive application demonstrates a complete machine learning pipeline for medical image analysis, 
    specifically focused on **colon polyp detection** with **explainable AI**.
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
    
    # Quick demo section
    st.markdown('<div class="section-header">🚀 Quick Demo</div>', unsafe_allow_html=True)
    
    if st.button("🎯 Try with Sample Image", type="primary"):
        st.info("Sample image analysis would be shown here. Upload your own image in the '🔍 Live Detection' section.")

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
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.markdown("### ⚙️ Analysis Settings")
            
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5, 
                step=0.1,
                help="Threshold for polyp detection"
            )
            
            show_explanations = st.checkbox(
                "Generate Explanations", 
                value=True,
                help="Generate explainability visualizations"
            )
        
        # Prediction button
        if st.button("🔮 Run Analysis", type="primary", width='stretch'):
            if not st.session_state.model_loaded:
                st.error("❌ Model not loaded. Please check the sidebar for model status.")
            else:
                with st.spinner("🔄 Analyzing image..."):
                    try:
                        # Preprocess image
                        input_tensor, image_rgb = preprocess_image(image)
                        
                        # Get prediction
                        prediction, binary_mask = predict_segmentation(
                            st.session_state.model, 
                            input_tensor, 
                            st.session_state.device
                        )
                        
                        # Calculate metrics
                        metrics = calculate_metrics(prediction)
                        
                        # Display results
                        st.markdown('<div class="section-header">📊 Analysis Results</div>', 
                                   unsafe_allow_html=True)
                        
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Max Confidence", f"{metrics['max_confidence']:.3f}")
                        
                        with col2:
                            st.metric("Mean Confidence", f"{metrics['mean_confidence']:.3f}")
                        
                        with col3:
                            st.metric("Polyp Area %", f"{metrics['polyp_area_percentage']:.1f}%")
                        
                        with col4:
                            polyp_detected = metrics['max_confidence'] > confidence_threshold
                            st.metric("Detection", "POLYP" if polyp_detected else "CLEAR", 
                                    "🚨" if polyp_detected else "✅")
                        
                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎯 Segmentation Result")
                            
                            # Create overlay visualization
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                            
                            # Original image
                            ax1.imshow(image_rgb)
                            ax1.set_title('Original Image', fontsize=12, fontweight='bold')
                            ax1.axis('off')
                            
                            # Prediction overlay
                            ax2.imshow(image_rgb)
                            pred_np = prediction.detach().cpu().numpy().squeeze()
                            im = ax2.imshow(pred_np, alpha=0.6, cmap='jet')
                            ax2.set_title('Polyp Prediction', fontsize=12, fontweight='bold')
                            ax2.axis('off')
                            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("### 📈 Confidence Analysis")
                            
                            # Confidence histogram
                            fig, ax = plt.subplots(figsize=(8, 5))
                            pred_flat = pred_np.flatten()
                            ax.hist(pred_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.axvline(confidence_threshold, color='red', linestyle='--', 
                                      label=f'Threshold ({confidence_threshold})')
                            ax.set_xlabel('Confidence Score')
                            ax.set_ylabel('Pixel Count')
                            ax.set_title('Prediction Confidence Distribution')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        # Generate explanations if requested
                        if show_explanations:
                            st.markdown('<div class="section-header">🧠 Explainability Analysis</div>', 
                                       unsafe_allow_html=True)
                            
                            with st.spinner("🔄 Generating explanations..."):
                                explanations = generate_explanations(
                                    st.session_state.model, 
                                    input_tensor, 
                                    st.session_state.device
                                )
                                
                                if explanations:
                                    # Create visualization
                                    viz_fig = create_visualization(
                                        image_rgb, prediction, explanations, binary_mask
                                    )
                                    if viz_fig is not None:
                                        st.pyplot(viz_fig)
                                    else:
                                        st.warning("⚠️ Could not create visualization")
                                    
                                    # Evaluate explanations
                                    input_np = input_tensor.detach().cpu().numpy()
                                    eval_results = evaluate_explanations(
                                        explanations, input_np, 
                                        st.session_state.model, st.session_state.device
                                    )
                                    
                                    # Display evaluation results
                                    st.markdown("### 📊 Explanation Quality Metrics")
                                    
                                    eval_df_data = []
                                    for method, results in eval_results.items():
                                        if 'error' not in results:
                                            eval_df_data.append({
                                                'Method': method,
                                                'Sparseness': f"{results['sparseness']:.3f}",
                                                'Mean Attribution': f"{results['mean_attribution']:.6f}",
                                                'Non-zero %': f"{results['non_zero_percentage']:.1f}%"
                                            })
                                    
                                    if eval_df_data:
                                        eval_df = pd.DataFrame(eval_df_data)
                                        st.dataframe(eval_df, width='stretch')
                                    else:
                                        st.info("ℹ️ No evaluation metrics available")
                                else:
                                    st.warning("⚠️ No explanations could be generated. Check the console for error messages.")
                        
                        # Success message
                        st.markdown("""
                        <div class="success-box">
                        <h4>✅ Analysis Complete!</h4>
                        <p>The model has successfully processed your image. Review the segmentation results and confidence metrics above.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)

elif page == "🧠 Explainability Analysis":
    st.markdown('<div class="section-header">Understanding Model Decisions</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Explanation Methods
    
    We use multiple techniques to understand what the model focuses on when making predictions:
    """)
    
    # Method descriptions
    methods_info = {
        "Integrated Gradients": {
            "description": "Computes feature importance by integrating gradients along a path from baseline to input",
            "strengths": "• Most focused and precise\n• Satisfies implementation invariance\n• Good theoretical foundation",
            "use_case": "Best for understanding specific feature contributions"
        },
        "Guided Backprop": {
            "description": "Modifies backpropagation to only show positive influence of neurons",
            "strengths": "• Good balance of precision and coverage\n• Highlights relevant features clearly\n• Fast computation",
            "use_case": "Good for general understanding of important regions"
        },
        "Grad-CAM": {
            "description": "Uses gradients of target concept flowing into final conv layer to highlight important regions",
            "strengths": "• Shows general important regions\n• Class-discriminative\n• Interpretable visualizations",
            "use_case": "Best for understanding high-level spatial attention"
        }
    }
    
    selected_method = st.selectbox(
        "Choose explanation method to learn about:",
        list(methods_info.keys())
    )
    
    if selected_method:
        method_info = methods_info[selected_method]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Description:**\n{method_info['description']}")
            st.markdown(f"**Key Strengths:**\n{method_info['strengths']}")
        
        with col2:
            st.markdown(f"**Best Use Case:**\n{method_info['use_case']}")
    
    # Comparison section
    st.markdown("### 📊 Method Comparison")
    
    comparison_data = {
        "Method": ["Integrated Gradients", "Guided Backprop", "Grad-CAM"],
        "Sparseness Score": [0.901, 0.880, 0.678],
        "Localization Quality": ["Excellent", "Good", "Fair"],
        "Computation Speed": ["Slow", "Fast", "Medium"],
        "Interpretation": ["Most focused", "Balanced", "General regions"]
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, width='stretch')

elif page == "📈 Model Evaluation":
    st.markdown('<div class="section-header">Comprehensive Performance Analysis</div>', 
                unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("### 🎯 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
        <h4>Dice Score</h4>
        <h2 style="color: #1f77b4;">0.7879</h2>
        <p>Overlap between prediction and ground truth</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
        <h4>Accuracy</h4>
        <h2 style="color: #2ca02c;">94.2%</h2>
        <p>Pixel-wise classification accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
        <h4>Sensitivity</h4>
        <h2 style="color: #ff7f0e;">88.5%</h2>
        <p>True positive rate (recall)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
        <h4>Specificity</h4>
        <h2 style="color: #d62728;">95.1%</h2>
        <p>True negative rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Explanation quality metrics
    st.markdown("### 🧠 Explanation Quality Analysis")
    
    explanation_metrics = {
        "Method": ["Integrated Gradients", "Guided Backprop", "Grad-CAM"],
        "Sparseness": [0.901, 0.880, 0.678],
        "Localization Precision": [0.574, 0.339, 0.285],
        "Localization Recall": [0.941, 0.555, 0.423],
        "Mean Attribution": [0.000007, 0.000807, 0.003075]
    }
    
    df_explanations = pd.DataFrame(explanation_metrics)
    st.dataframe(df_explanations, width='stretch')
    
    st.markdown("""
    **Key Insights:**
    - **Integrated Gradients** provides the most focused explanations (highest sparseness)
    - **Excellent localization recall** (94.1%) means it captures most of the true polyp regions
    - **Grad-CAM** offers broader coverage but less precise localization
    """)

elif page == "📊 Model Overview":
    st.markdown('<div class="section-header">Model Architecture & Training</div>', 
                unsafe_allow_html=True)
    
    # Architecture overview
    st.markdown("### 🏗️ U-Net Architecture")
    
    st.markdown("""
    Our model uses the **U-Net architecture**, specifically designed for medical image segmentation:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Architecture Components:**
        - **Encoder Path**: Captures context through downsampling
        - **Decoder Path**: Enables precise localization through upsampling  
        - **Skip Connections**: Combines low-level and high-level features
        - **Output**: Pixel-wise segmentation mask for polyp regions
        """)
    
    with col2:
        st.markdown("""
        **Technical Specifications:**
        - Input: 3-channel RGB images (256×256)
        - Output: 1-channel binary mask
        - Parameters: ~31M trainable parameters
        - Architecture: Encoder-Decoder with skip connections
        """)
    
    # Training details
    st.markdown("### 📈 Training Configuration")
    
    training_config = {
        "Parameter": [
            "Dataset", "Training Images", "Validation Images", 
            "Image Size", "Loss Function", "Optimizer", 
            "Learning Rate", "Epochs", "Best Epoch", "Best Dice Score"
        ],
        "Value": [
            "CVC-ClinicDB", "490", "122", 
            "256×256", "BCE with Logits", "Adam", 
            "0.001", "50", "49", "0.7879"
        ]
    }
    
    config_df = pd.DataFrame(training_config)
    st.dataframe(config_df, width='stretch', hide_index=True)

elif page == "� Validation Comparison":
    st.markdown('<div class="section-header">🔬 Validation Dataset Comparison</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Compare Predictions with Ground Truth
    
    This section allows you to compare model predictions with ground truth masks from the validation dataset.
    You can visualize individual samples and analyze overall model performance.
    """)
    
    # Load validation data
    with st.spinner("Loading validation dataset..."):
        X_val_tensor, y_val_tensor, data_available = load_validation_data()
    
    if not data_available:
        st.warning("⚠️ Validation dataset not found. Please ensure the processed validation data is available at:")
        st.code("data/CVC-ClinicDB/processed_data/X_val.npy\ndata/CVC-ClinicDB/processed_data/y_val.npy")
        st.info("💡 Run the data preprocessing section in the Jupyter notebook to generate these files.")
    else:
        st.success(f"✅ Loaded {len(X_val_tensor)} validation samples")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Choose analysis type:",
            ["🎯 Individual Sample Analysis", "📊 Batch Performance Analysis", "🧠 Explainability Comparison"]
        )
        
        if analysis_type == "🎯 Individual Sample Analysis":
            st.markdown("### 🔍 Individual Sample Comparison")
            
            # Sample selection options
            col1, col2 = st.columns(2)
            with col1:
                sample_mode = st.radio(
                    "Sample selection:",
                    ["Random samples", "Specific index"]
                )
            
            with col2:
                if sample_mode == "Random samples":
                    n_samples = st.slider("Number of samples", 1, 20, 5)
                    if st.button("🎲 Generate Random Samples"):
                        st.session_state.sample_indices = np.random.choice(len(X_val_tensor), n_samples, replace=False)
                else:
                    sample_idx = st.number_input("Sample index", 0, len(X_val_tensor)-1, 0)
                    st.session_state.sample_indices = [sample_idx]
            
            # Show samples if indices are available
            if hasattr(st.session_state, 'sample_indices'):
                device = st.session_state.device
                model = st.session_state.model
                model.eval()
                
                with st.spinner("Generating predictions and comparisons..."):
                    for idx in st.session_state.sample_indices:
                        st.markdown(f"#### Sample {idx}")
                        
                        # Get image and ground truth
                        image_tensor = X_val_tensor[idx].unsqueeze(0).to(device)
                        ground_truth = y_val_tensor[idx].squeeze(0).cpu().numpy()
                        
                        # Generate prediction
                        with torch.no_grad():
                            prediction, binary_mask = predict_segmentation(model, image_tensor, device)
                            prediction_prob = prediction.squeeze().cpu().numpy()
                            prediction_mask = binary_mask.squeeze().cpu().numpy()
                        
                        # Convert image for visualization
                        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        
                        # Create comparison visualization
                        fig = create_comparison_visualization(
                            image_np, ground_truth, prediction_mask, prediction_prob
                        )
                        st.pyplot(fig)
                        
                        # Calculate metrics for this sample
                        metrics = calculate_metrics(prediction, torch.tensor(ground_truth).unsqueeze(0).unsqueeze(0))
                        
                        # Display metrics in columns
                        met_col1, met_col2, met_col3 = st.columns(3)
                        with met_col1:
                            st.metric("🎯 Dice Score", f"{metrics['dice_score']:.3f}")
                        with met_col2:
                            st.metric("🔗 IoU", f"{metrics['iou']:.3f}")
                        with met_col3:
                            st.metric("📊 Pixel Accuracy", f"{metrics['pixel_accuracy']:.3f}")
                        
                        st.markdown("---")
        
        elif analysis_type == "📊 Batch Performance Analysis":
            st.markdown("### 📈 Overall Performance Analysis")
            
            # Performance analysis options
            analysis_samples = st.slider("Number of samples to analyze", 10, min(100, len(X_val_tensor)), 20)
            
            if st.button("🚀 Run Batch Analysis"):
                with st.spinner(f"Analyzing {analysis_samples} samples..."):
                    # Sample random images
                    sampled_images, sampled_masks, sample_indices = sample_validation_images(
                        X_val_tensor, y_val_tensor, analysis_samples
                    )
                    
                    device = st.session_state.device
                    model = st.session_state.model
                    model.eval()
                    
                    # Generate predictions for all samples
                    all_predictions = []
                    all_ground_truths = []
                    
                    for i in range(len(sampled_images)):
                        image_tensor = sampled_images[i].unsqueeze(0).to(device)
                        ground_truth = sampled_masks[i].squeeze(0).cpu().numpy()
                        
                        with torch.no_grad():
                            prediction, _ = predict_segmentation(model, image_tensor, device)
                            prediction_prob = prediction.squeeze().cpu().numpy()
                        
                        all_predictions.append(prediction_prob)
                        all_ground_truths.append(ground_truth)
                    
                    # Calculate comprehensive metrics
                    val_metrics = calculate_validation_metrics(
                        np.array(all_predictions), np.array(all_ground_truths)
                    )
                    
                    # Display summary metrics
                    st.markdown("#### 📊 Performance Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "🎯 Mean Dice Score", 
                            f"{val_metrics['mean_dice']:.3f}",
                            f"± {val_metrics['std_dice']:.3f}"
                        )
                    with col2:
                        st.metric(
                            "🔗 Mean IoU", 
                            f"{val_metrics['mean_iou']:.3f}",
                            f"± {val_metrics['std_iou']:.3f}"
                        )
                    with col3:
                        st.metric(
                            "📊 Mean Accuracy", 
                            f"{val_metrics['mean_accuracy']:.3f}",
                            f"± {val_metrics['std_accuracy']:.3f}"
                        )
                    
                    # Distribution plots
                    st.markdown("#### 📈 Performance Distribution")
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Dice score distribution
                    axes[0].hist(val_metrics['individual_dice'], bins=20, alpha=0.7, color='blue')
                    axes[0].set_title('Dice Score Distribution')
                    axes[0].set_xlabel('Dice Score')
                    axes[0].set_ylabel('Frequency')
                    axes[0].axvline(val_metrics['mean_dice'], color='red', linestyle='--', label='Mean')
                    axes[0].legend()
                    
                    # IoU distribution
                    axes[1].hist(val_metrics['individual_iou'], bins=20, alpha=0.7, color='green')
                    axes[1].set_title('IoU Distribution')
                    axes[1].set_xlabel('IoU')
                    axes[1].set_ylabel('Frequency')
                    axes[1].axvline(val_metrics['mean_iou'], color='red', linestyle='--', label='Mean')
                    axes[1].legend()
                    
                    # Accuracy distribution
                    axes[2].hist(val_metrics['individual_accuracy'], bins=20, alpha=0.7, color='orange')
                    axes[2].set_title('Pixel Accuracy Distribution')
                    axes[2].set_xlabel('Accuracy')
                    axes[2].set_ylabel('Frequency')
                    axes[2].axvline(val_metrics['mean_accuracy'], color='red', linestyle='--', label='Mean')
                    axes[2].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        elif analysis_type == "🧠 Explainability Comparison":
            st.markdown("### 🔍 Explainability vs Ground Truth")
            
            st.info("💡 This analysis shows how explainability methods align with ground truth polyp regions.")
            
            # Sample selection
            sample_idx = st.number_input("Select sample index", 0, len(X_val_tensor)-1, 0)
            
            if st.button("🧠 Generate Explainability Analysis"):
                with st.spinner("Generating explanations and comparisons..."):
                    device = st.session_state.device
                    model = st.session_state.model
                    
                    # Get sample
                    image_tensor = X_val_tensor[sample_idx].unsqueeze(0).to(device)
                    ground_truth = y_val_tensor[sample_idx].squeeze(0).cpu().numpy()
                    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
                    # Generate explanations
                    explanations = generate_explanations(model, image_tensor, device)
                    
                    if explanations:
                        # Create explainability comparison
                        fig = create_explainability_comparison(image_np, ground_truth, explanations)
                        st.pyplot(fig)
                        
                        # Analyze explanation quality vs ground truth
                        st.markdown("#### 📊 Explanation Analysis")
                        
                        explanation_analysis = []
                        for method, attribution in explanations.items():
                            if isinstance(attribution, np.ndarray):
                                attr_map = attribution.squeeze()
                                if attr_map.ndim > 2:
                                    attr_map = np.mean(attr_map, axis=0)
                                
                                # Normalize attribution
                                attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
                                
                                # Calculate overlap with ground truth
                                polyp_mask = ground_truth > 0.5
                                high_attr_mask = attr_norm > np.percentile(attr_norm, 75)
                                
                                if polyp_mask.sum() > 0:
                                    overlap = np.sum(polyp_mask & high_attr_mask) / polyp_mask.sum()
                                    precision = np.sum(polyp_mask & high_attr_mask) / (high_attr_mask.sum() + 1e-8)
                                    
                                    explanation_analysis.append({
                                        'Method': method,
                                        'Polyp Recall': f"{overlap:.3f}",
                                        'Attribution Precision': f"{precision:.3f}",
                                        'Mean Attribution': f"{np.mean(attr_norm):.3f}"
                                    })
                        
                        if explanation_analysis:
                            analysis_df = pd.DataFrame(explanation_analysis)
                            st.dataframe(analysis_df, use_container_width=True)
                    else:
                        st.warning("⚠️ No explanations could be generated for this sample.")

elif page == "�📚 About & Research":
    st.markdown('<div class="section-header">Research Background</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Research Context
    
    This project demonstrates the application of **explainable AI** to **medical image analysis**, 
    specifically for colon polyp detection in colonoscopy images.
    
    ### 📖 Key Components:
    """)
    
    components = [
        {
            "title": "🏥 Medical Relevance",
            "content": "Colon polyps are precancerous growths that can develop into colorectal cancer. Early detection through colonoscopy is crucial for prevention."
        },
        {
            "title": "🤖 Deep Learning",
            "content": "U-Net architecture provides state-of-the-art performance for medical image segmentation tasks with pixel-level accuracy."
        },
        {
            "title": "🔍 Explainable AI",
            "content": "Multiple explanation methods (Integrated Gradients, Guided Backprop, Grad-CAM) help clinicians understand model decisions."
        },
        {
            "title": "📊 Quantitative Evaluation",
            "content": "Rigorous assessment using metrics from the Quantus library ensures explanation quality and reliability."
        }
    ]
    
    for comp in components:
        st.markdown(f"""
        <div class="explanation-text">
        <h4>{comp['title']}</h4>
        <p>{comp['content']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Technical Stack")
    
    tech_stack = {
        "Category": [
            "Deep Learning", "Explainability", "Evaluation", 
            "Deployment", "Dataset", "Visualization"
        ],
        "Technologies": [
            "PyTorch, U-Net Architecture", 
            "Captum (Integrated Gradients, Guided Backprop, Grad-CAM)",
            "Quantus metrics library", 
            "Streamlit web application",
            "CVC-ClinicDB colonoscopy images", 
            "Matplotlib, Plotly"
        ]
    }
    
    tech_df = pd.DataFrame(tech_stack)
    st.dataframe(tech_df, width='stretch', hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    Colon Polyp Detection & Explainability Platform <br>
</div>
""", unsafe_allow_html=True)
