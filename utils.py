"""
Utility functions for the Colon Polyp Detection Streamlit App
Extracted and adapted from the Jupyter notebook pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Captum imports
from captum.attr import IntegratedGradients, GuidedBackprop, LayerGradCam

# Quantus imports
from quantus import Sparseness

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

# Note: We use Captum's LayerGradCam implementation instead of custom implementation
# for better integration and reliability

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image
    image_resized = cv2.resize(image, target_size)
    
    # Convert to RGB if needed
    if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_resized
    
    # Normalize to [0, 1]
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image_rgb

def predict_segmentation(model, image_tensor, device):
    """Generate segmentation prediction"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        
        # Convert to binary mask
        binary_mask = (prediction > 0.5).float()
        
        return prediction, binary_mask

class BatchedScalarModel(nn.Module):
    """
    Wrap a model so its forward returns a per-sample scalar tensor of shape (batch,).
    If the base model returns a map/tensor per sample (e.g. segmentation), we reduce
    per-sample by mean. If the base returns a 0-dim scalar, we unsqueeze to (1,).
    """
    def __init__(self, base_model, reduction='mean'):
        super().__init__()
        self.base = base_model
        self.reduction = reduction

    def forward(self, x):
        out = self.base(x)

        # If forward returns tuple/list/dict, try to extract the main tensor
        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            # common keys: 'out' or first value
            out = out.get('out', next(iter(out.values())))

        if not torch.is_tensor(out):
            # If it's a Python scalar convert to tensor on the same device
            out = torch.tensor(out, device=x.device)

        # If scalar 0-dim -> make it batch-shaped (1,)
        if out.dim() == 0:
            return out.unsqueeze(0)  # shape (1,)

        # If it already has a batch dimension and is a single value per sample: return as-is
        # (e.g., shape (batch,) or (batch,1))
        if out.dim() == 1:
            return out

        if out.dim() >= 2:
            # Reduce per-sample (flatten each sample then reduce)
            b = out.size(0)
            per_sample = out.view(b, -1)
            if self.reduction == 'mean':
                return per_sample.mean(dim=1)  # shape (batch,)
            elif self.reduction == 'sum':
                return per_sample.sum(dim=1)
            else:
                raise ValueError("Unknown reduction: " + str(self.reduction))

def tensor_to_numpy_img(tensor, normalize=True, eps=1e-9):
    """
    Convert a tensor to a numpy image-like array.
    - Accepts tensors shaped (1,C,H,W), (C,H,W), (1,H,W), (H,W) or (H,W,C).
    - Returns HxWxC for multi-channel tensors, HxW for single-channel tensors.
    - If normalize=True, scales array to [0,1].
    """
    t = tensor.detach().cpu()
    # remove batch dim if present
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)

    # (C,H,W) -> (H,W,C)
    if t.ndim == 3:
        c, h, w = t.shape
        if c == 1:
            arr = t.squeeze(0).numpy()         # H, W
        else:
            arr = t.permute(1, 2, 0).numpy()   # H, W, C
    elif t.ndim == 2:
        arr = t.numpy()                       # H, W
    else:
        # fallback: convert everything else to numpy
        arr = t.numpy()

    if normalize:
        # only normalize float-like arrays (prevents integer arrays from unexpected scaling)
        arr = arr.astype(np.float32)
        mn = arr.min()
        mx = arr.max()
        if mx - mn >= eps:
            arr = (arr - mn) / (mx - mn + eps)
        else:
            arr = arr * 0.0  # constant -> zeros to avoid NaNs

    return arr

def generate_explanations(model, input_tensor, device):
    """Generate multiple explanations for the model prediction"""
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)
    
    explanations = {}
    
    # Create a wrapped model that returns scalar outputs
    wrapped_model = BatchedScalarModel(model, reduction='mean')
    
    # Integrated Gradients
    try:
        ig = IntegratedGradients(wrapped_model)
        attr_ig = ig.attribute(input_tensor, n_steps=50)
        explanations['Integrated Gradients'] = attr_ig.detach().cpu().numpy()
    except Exception as e:
        print(f"Integrated Gradients failed: {e}")
    
    # Guided Backprop - use the wrapped model
    try:
        gbp = GuidedBackprop(wrapped_model)
        attr_gbp = gbp.attribute(input_tensor)
        explanations['Guided Backprop'] = attr_gbp.detach().cpu().numpy()
    except Exception as e:
        print(f"Guided Backprop failed: {e}")
    
    # Grad-CAM - Find a suitable convolutional layer
    target_layer = None
    
    # Try to find a specific layer first
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Conv2d)):
            if any(layer_name in name.lower() for layer_name in ['dconv_down4', 'conv4', 'layer3', 'decoder', 'up']):
                target_layer = module
                break
    
    # If no specific layer found, use the last conv layer
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
    
    if target_layer is not None:
        try:
            gradcam = LayerGradCam(wrapped_model, target_layer)
            attr_gradcam = gradcam.attribute(input_tensor, target=None)
            
            # Upsample to input size
            attr_gradcam_upsampled = torch.nn.functional.interpolate(
                attr_gradcam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
            )
            explanations['Grad-CAM'] = attr_gradcam_upsampled.detach().cpu().numpy()
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
    
    # Guided Grad-CAM (combination of Guided Backprop and Grad-CAM)
    if 'Guided Backprop' in explanations and 'Grad-CAM' in explanations:
        try:
            attr_gbp = torch.tensor(explanations['Guided Backprop'])
            attr_gradcam = torch.tensor(explanations['Grad-CAM'])
            
            # Ensure gradcam has same channels as guided backprop
            if attr_gradcam.shape[1] == 1 and attr_gbp.shape[1] > 1:
                attr_gradcam = attr_gradcam.repeat(1, attr_gbp.shape[1], 1, 1)
            
            guided_gradcam = attr_gbp * attr_gradcam
            explanations['Guided Grad-CAM'] = guided_gradcam.numpy()
        except Exception as e:
            print(f"Guided Grad-CAM combination failed: {e}")
    
    return explanations

def calculate_metrics(prediction, ground_truth=None):
    """Calculate various performance metrics"""
    metrics = {}
    
    # Basic prediction stats
    pred_np = prediction.detach().cpu().numpy()
    metrics['mean_confidence'] = np.mean(pred_np)
    metrics['max_confidence'] = np.max(pred_np)
    metrics['polyp_area_percentage'] = np.mean(pred_np > 0.5) * 100
    
    if ground_truth is not None:
        gt_np = ground_truth.detach().cpu().numpy() if isinstance(ground_truth, torch.Tensor) else ground_truth
        pred_binary = (pred_np > 0.5).astype(int)
        gt_binary = (gt_np > 0.5).astype(int)
        
        # Calculate Dice score
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary)
        dice = (2 * intersection) / (union + 1e-8)
        metrics['dice_score'] = dice
        
        # Calculate IoU
        intersection = np.sum(pred_binary * gt_binary)
        union_iou = np.sum((pred_binary + gt_binary) > 0)
        iou = intersection / (union_iou + 1e-8)
        metrics['iou'] = iou
        
        # Pixel-wise accuracy
        accuracy = np.mean(pred_binary == gt_binary)
        metrics['pixel_accuracy'] = accuracy
    
    return metrics

def evaluate_explanations(explanations, input_np, model, device):
    """Evaluate explanation quality using Quantus metrics"""
    results = {}
    
    # Convert to proper format
    y_target = np.array([1])  # Assume polyp is present
    
    # Initialize metrics
    sparseness_metric = Sparseness()
    
    for name, attr in explanations.items():
        try:
            # Calculate sparseness
            score = sparseness_metric(
                model=model, 
                x_batch=input_np, 
                y_batch=y_target, 
                a_batch=attr
            )
            results[name] = {
                'sparseness': score[0] if isinstance(score, list) else score,
                'mean_attribution': np.mean(np.abs(attr)),
                'max_attribution': np.max(np.abs(attr)),
                'non_zero_percentage': np.mean(np.abs(attr.flatten()) > 1e-6) * 100
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

def create_visualization(image_rgb, prediction, explanations, binary_mask=None):
    """Create  visualization of results"""
    n_methods = len(explanations)
    if n_methods == 0:
        return None
        
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    if n_methods == 1:
        axes = axes.reshape(2, -1)
    elif n_methods == 0:
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        axes.imshow(image_rgb)
        axes.set_title('Original Image', fontsize=12, fontweight='bold')
        axes.axis('off')
        return fig
    
    # Original image and prediction
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Show prediction
    pred_np = prediction.detach().cpu().numpy().squeeze()
    axes[1, 0].imshow(image_rgb)
    im = axes[1, 0].imshow(pred_np, alpha=0.6, cmap='jet')
    axes[1, 0].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Show explanations
    for i, (name, attr) in enumerate(explanations.items()):
        # Convert attribution to proper numpy format
        if isinstance(attr, np.ndarray):
            attr_tensor = torch.tensor(attr)
        else:
            attr_tensor = attr
            
        # Use the helper function to convert to proper format
        attr_map = tensor_to_numpy_img(attr_tensor, normalize=True)
        
        # If multi-channel, take mean across channels
        if attr_map.ndim == 3:
            attr_display = attr_map.mean(axis=-1)
        else:
            attr_display = attr_map
        
        # Show attribution overlay
        axes[0, i + 1].imshow(image_rgb)
        im1 = axes[0, i + 1].imshow(attr_display, alpha=0.7, cmap='hot')
        axes[0, i + 1].set_title(f'{name}\nAttribution', fontsize=12, fontweight='bold')
        axes[0, i + 1].axis('off')
        plt.colorbar(im1, ax=axes[0, i + 1], fraction=0.046, pad=0.04)
        
        # Show attribution only
        axes[1, i + 1].imshow(attr_display, cmap='hot')
        axes[1, i + 1].set_title(f'{name}\n(Isolated)', fontsize=12, fontweight='bold')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for Streamlit"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    return img_str

def load_validation_data():
    """Load validation dataset if available"""
    try:
        X_val = np.load("data/CVC-ClinicDB/processed_data/X_val.npy")
        y_val = np.load("data/CVC-ClinicDB/processed_data/y_val.npy")
        
        # Convert to tensors
        X_val_tensor = torch.from_numpy(X_val).permute(0, 3, 1, 2).float()
        y_val_tensor = torch.from_numpy(y_val).unsqueeze(1).float()
        
        return X_val_tensor, y_val_tensor, True
    except FileNotFoundError:
        return None, None, False

def create_comparison_visualization(image, ground_truth_mask, prediction_mask, prediction_prob=None):
    """
    Create a comparison visualization showing original, ground truth, and prediction
    
    Args:
        image: Original image (numpy array, HxWxC)
        ground_truth_mask: Ground truth mask (numpy array, HxW)
        prediction_mask: Binary prediction mask (numpy array, HxW)
        prediction_prob: Prediction probabilities (optional, numpy array, HxW)
    
    Returns:
        matplotlib figure
    """
    if prediction_prob is not None:
        # 4-panel view with probabilities
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ['Original Image', 'Ground Truth Mask', 'Predicted Mask', 'Prediction Probabilities']
    else:
        # 3-panel view
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Original Image', 'Ground Truth Mask', 'Predicted Mask']
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title(titles[1], fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(prediction_mask, cmap='gray')
    axes[2].set_title(titles[2], fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Prediction probabilities (if provided)
    if prediction_prob is not None:
        im = axes[3].imshow(prediction_prob, cmap='jet', vmin=0, vmax=1)
        axes[3].set_title(titles[3], fontsize=12, fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def create_explainability_comparison(image, ground_truth_mask, explanations):
    """
    Create explainability comparison with ground truth
    
    Args:
        image: Original image (numpy array, HxWxC)
        ground_truth_mask: Ground truth mask (numpy array, HxW)
        explanations: Dictionary of explanation methods and their attributions
    
    Returns:
        matplotlib figure
    """
    n_explanations = len(explanations)
    n_cols = n_explanations + 2  # +2 for original and ground truth
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Explanations
    for i, (method_name, attribution) in enumerate(explanations.items(), start=2):
        # Convert attribution to proper format
        if isinstance(attribution, np.ndarray):
            attr_tensor = torch.tensor(attribution)
        else:
            attr_tensor = attribution
            
        attr_map = tensor_to_numpy_img(attr_tensor, normalize=True)
        
        # If multi-channel, take mean across channels
        if attr_map.ndim == 3:
            attr_display = attr_map.mean(axis=-1)
        else:
            attr_display = attr_map
        
        # Show attribution overlay
        axes[i].imshow(image)
        im = axes[i].imshow(attr_display, alpha=0.7, cmap='hot')
        axes[i].set_title(f'{method_name}\nAttribution', fontsize=12, fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def calculate_validation_metrics(predictions, ground_truths):
    """
    Calculate metrics for validation dataset
    
    Args:
        predictions: Array of prediction probabilities
        ground_truths: Array of ground truth masks
    
    Returns:
        Dictionary of metrics
    """
    # Convert to binary predictions
    pred_binary = (predictions > 0.5).astype(int)
    gt_binary = (ground_truths > 0.5).astype(int)
    
    # Calculate per-sample metrics
    dice_scores = []
    iou_scores = []
    accuracies = []
    
    for pred, gt in zip(pred_binary, gt_binary):
        # Dice score
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        dice = (2 * intersection) / (union + 1e-8)
        dice_scores.append(dice)
        
        # IoU
        intersection = np.sum(pred * gt)
        union_iou = np.sum((pred + gt) > 0)
        iou = intersection / (union_iou + 1e-8)
        iou_scores.append(iou)
        
        # Pixel accuracy
        accuracy = np.mean(pred == gt)
        accuracies.append(accuracy)
    
    return {
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),
        'mean_iou': np.mean(iou_scores),
        'std_iou': np.std(iou_scores),
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'individual_dice': dice_scores,
        'individual_iou': iou_scores,
        'individual_accuracy': accuracies
    }

def sample_validation_images(X_val_tensor, y_val_tensor, n_samples=10, random_seed=42):
    """
    Sample random images from validation dataset
    
    Args:
        X_val_tensor: Validation images tensor
        y_val_tensor: Validation masks tensor
        n_samples: Number of samples to return
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (sampled_images, sampled_masks, sample_indices)
    """
    np.random.seed(random_seed)
    total_samples = len(X_val_tensor)
    sample_indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
    
    sampled_images = X_val_tensor[sample_indices]
    sampled_masks = y_val_tensor[sample_indices]
    
    return sampled_images, sampled_masks, sample_indices
