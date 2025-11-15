#!/usr/bin/env python3
"""
Demo script for Validation Comparison Features
==============================================

This script demonstrates the new validation dataset comparison functionality
that has been added to the Streamlit app.

Usage: python demo_validation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import (
    UNet, load_validation_data, sample_validation_images,
    create_comparison_visualization, predict_segmentation,
    calculate_validation_metrics
)

def demo_validation_comparison():
    """Demonstrate the validation comparison functionality"""
    print("🔬 Demo: Validation Dataset Comparison")
    print("="*50)
    
    # Load model
    print("📦 Loading model...")
    device = torch.device('cpu')  # Use CPU for demo
    model = UNet(n_class=1)
    
    try:
        checkpoint_path = "data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load model checkpoint: {e}")
        print("Using randomly initialized model for demo")
    
    # Load validation data
    print("\n📊 Loading validation data...")
    X_val, y_val, data_available = load_validation_data()
    
    if not data_available:
        print("❌ Validation data not found.")
        print("Please run the preprocessing section in the notebook first.")
        return
    
    print(f"✅ Loaded {len(X_val)} validation samples")
    print(f"   Image shape: {X_val.shape}")
    print(f"   Mask shape: {y_val.shape}")
    
    # Sample some images for comparison
    print("\n🎲 Sampling random validation images...")
    sampled_images, sampled_masks, sample_indices = sample_validation_images(
        X_val, y_val, n_samples=3, random_seed=42
    )
    print(f"Selected samples: {sample_indices}")
    
    # Generate predictions and create comparisons
    print("\n🔍 Generating predictions and comparisons...")
    
    all_predictions = []
    all_ground_truths = []
    
    for i, idx in enumerate(sample_indices):
        print(f"\nProcessing sample {idx}...")
        
        # Get image and ground truth
        image_tensor = sampled_images[i].unsqueeze(0).to(device)
        ground_truth = sampled_masks[i].squeeze(0).cpu().numpy()
        
        # Generate prediction
        with torch.no_grad():
            prediction, binary_mask = predict_segmentation(model, image_tensor, device)
            prediction_prob = prediction.squeeze().cpu().numpy()
            prediction_mask = binary_mask.squeeze().cpu().numpy()
        
        # Store for batch metrics
        all_predictions.append(prediction_prob)
        all_ground_truths.append(ground_truth)
        
        # Convert image for visualization
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Create and save comparison visualization
        fig = create_comparison_visualization(
            image_np, ground_truth, prediction_mask, prediction_prob
        )
        
        # Save the figure
        output_path = f"validation_comparison_sample_{idx}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   💾 Saved comparison to: {output_path}")
        
        # Calculate individual metrics
        from utils import calculate_metrics
        metrics = calculate_metrics(
            prediction, 
            torch.tensor(ground_truth).unsqueeze(0).unsqueeze(0)
        )
        
        print(f"   📊 Dice Score: {metrics['dice_score']:.3f}")
        print(f"   🔗 IoU: {metrics['iou']:.3f}")
        print(f"   📈 Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
    
    # Calculate batch metrics
    print("\n📈 Calculating batch performance metrics...")
    batch_metrics = calculate_validation_metrics(
        np.array(all_predictions), 
        np.array(all_ground_truths)
    )
    
    print(f"   🎯 Mean Dice Score: {batch_metrics['mean_dice']:.3f} ± {batch_metrics['std_dice']:.3f}")
    print(f"   🔗 Mean IoU: {batch_metrics['mean_iou']:.3f} ± {batch_metrics['std_iou']:.3f}")
    print(f"   📊 Mean Accuracy: {batch_metrics['mean_accuracy']:.3f} ± {batch_metrics['std_accuracy']:.3f}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nGenerated files:")
    for idx in sample_indices:
        print(f"   📁 validation_comparison_sample_{idx}.png")
    
    print(f"\n💡 To explore more features, run the Streamlit app:")
    print(f"   streamlit run app.py")
    print(f"   Navigate to: 🔬 Validation Comparison")

def demo_features_overview():
    """Show what features are available in the validation comparison"""
    print("\n🌟 Validation Comparison Features Overview")
    print("="*50)
    
    print("🎯 Individual Sample Analysis:")
    print("   • Side-by-side comparison: Original → Ground Truth → Prediction")
    print("   • Prediction probabilities visualization")
    print("   • Per-sample metrics (Dice, IoU, Pixel Accuracy)")
    print("   • Random sampling or specific index selection")
    
    print("\n📊 Batch Performance Analysis:")
    print("   • Performance metrics over multiple samples")
    print("   • Statistical summary (mean ± std)")
    print("   • Distribution histograms for all metrics")
    print("   • Configurable sample size")
    
    print("\n🧠 Explainability Comparison:")
    print("   • Explanations overlaid with ground truth")
    print("   • Attribution quality analysis")
    print("   • Polyp recall and attribution precision")
    print("   • Method comparison metrics")
    
    print("\n📱 Interactive Features (in Streamlit app):")
    print("   • Real-time sample selection")
    print("   • Interactive metric displays")
    print("   • Downloadable visualizations")
    print("   • Responsive layout with error handling")

if __name__ == "__main__":
    try:
        demo_features_overview()
        demo_validation_comparison()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Check that all dependencies are installed and validation data is available.")
