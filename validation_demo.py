#!/usr/bin/env python3
"""
Demo script showing validation visualization functionality
This demonstrates the same functionality from the notebook that's implemented in the Streamlit app
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    UNet, load_validation_data, predict_segmentation, 
    create_comparison_visualization, generate_explanations,
    create_explainability_comparison, calculate_metrics,
    tensor_to_numpy_img
)

def demo_validation_comparison():
    """Demo the validation comparison functionality"""
    print("🔬 Colon Polyp Validation Demo")
    print("=" * 50)
    
    # Load validation data (same as notebook)
    print("Loading validation dataset...")
    X_val_tensor, y_val_tensor, data_available = load_validation_data()
    
    if not data_available:
        print("❌ Validation dataset not found!")
        return
    
    print(f"✅ Loaded {len(X_val_tensor)} validation samples")
    print(f"   X_val shape: {X_val_tensor.shape}")
    print(f"   y_val shape: {y_val_tensor.shape}")
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet(n_class=1)
    
    # Try to load checkpoint
    try:
        checkpoint_path = "data/CVC-ClinicDB/checkpoints/best_model_dice_0.7879_epoch_49.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        print("Using random initialized model for demo...")
        model.to(device)
        model.eval()
    
    # Demo 1: Individual sample comparison (like notebook)
    print(f"\n📊 Demo 1: Individual Sample Comparison")
    print("-" * 40)
    
    # Select random samples (same as notebook approach)
    num_samples_to_visualize = 3
    sample_indices = np.random.choice(len(X_val_tensor), num_samples_to_visualize, replace=False)
    
    print(f"Displaying {num_samples_to_visualize} sample images, ground truth masks, and predictions:")
    
    with torch.no_grad():
        for i in sample_indices:
            print(f"\nProcessing sample {i}...")
            
            # Get sample image and mask (same as notebook)
            image = X_val_tensor[i].unsqueeze(0).to(device)  # Add batch dimension
            mask = y_val_tensor[i].squeeze(0).cpu().numpy()  # Remove channel dim and move to CPU
            
            # Get model prediction (same as notebook)
            prediction, binary_mask = predict_segmentation(model, image, device)
            prediction_prob = prediction.squeeze().cpu().numpy()
            prediction_mask = binary_mask.squeeze().cpu().numpy()
            
            # Move image back to CPU and change channel order for plotting (CxHxW to HxWxC)
            image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(prediction, torch.tensor(mask).unsqueeze(0).unsqueeze(0))
            
            print(f"  🎯 Dice Score: {metrics['dice_score']:.3f}")
            print(f"  🔗 IoU: {metrics['iou']:.3f}")
            print(f"  📊 Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
            
            # Create visualization using our utility function
            fig = create_comparison_visualization(image_np, mask, prediction_mask, prediction_prob)
            plt.savefig(f'validation_comparison_sample_{i}.png', dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
    
    # Demo 2: Explainability comparison (like notebook)
    print(f"\n🧠 Demo 2: Explainability vs Ground Truth")
    print("-" * 40)
    
    # Select one sample for explainability
    sample_idx = sample_indices[0]
    print(f"Generating explanations for sample {sample_idx}...")
    
    # Get sample (same as notebook setup)
    image_tensor = X_val_tensor[sample_idx].unsqueeze(0).to(device)
    ground_truth = y_val_tensor[sample_idx].squeeze(0).cpu().numpy()
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Generate explanations (same methods as notebook)
    image_tensor.requires_grad_(True)  # Same as notebook
    explanations = generate_explanations(model, image_tensor, device)
    
    if explanations:
        print(f"✅ Generated {len(explanations)} explanation methods:")
        for method in explanations.keys():
            print(f"   - {method}")
        
        # Create explainability comparison visualization
        fig = create_explainability_comparison(image_np, ground_truth, explanations)
        plt.savefig(f'explainability_comparison_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Analyze explanation quality (same as Streamlit app)
        print("\n📈 Explanation Quality Analysis:")
        for method, attribution in explanations.items():
            if isinstance(attribution, np.ndarray):
                attr_map = attribution.squeeze()
                if attr_map.ndim > 2:
                    attr_map = np.mean(attr_map, axis=0)
                
                # Normalize attribution (same as notebook)
                attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
                
                # Calculate overlap with ground truth
                polyp_mask = ground_truth > 0.5
                high_attr_mask = attr_norm > np.percentile(attr_norm, 75)
                
                if polyp_mask.sum() > 0:
                    overlap = np.sum(polyp_mask & high_attr_mask) / polyp_mask.sum()
                    precision = np.sum(polyp_mask & high_attr_mask) / (high_attr_mask.sum() + 1e-8)
                    
                    print(f"  {method}:")
                    print(f"    - Polyp Recall: {overlap:.3f}")
                    print(f"    - Attribution Precision: {precision:.3f}")
                    print(f"    - Mean Attribution: {np.mean(attr_norm):.3f}")
    else:
        print("⚠️  No explanations could be generated")
    
    print(f"\n🎉 Demo completed! Generated visualization files:")
    print(f"   - validation_comparison_sample_*.png")
    print(f"   - explainability_comparison_sample_*.png")
    
    print(f"\n💡 To see this functionality in interactive form, run:")
    print(f"   streamlit run app.py")
    print(f"   Then go to '🔬 Validation Comparison' section")

if __name__ == "__main__":
    demo_validation_comparison()
