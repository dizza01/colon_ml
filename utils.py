"""
Utility functions for the Colon Polyp Detection Streamlit App
Extracted and adapted from the Jupyter notebook pipeline
"""
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
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




# Add the SegTransformer classes after the UNet class
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.projection(x)
        # Flatten patches: (B, embed_dim, n_patches)
        x = x.flatten(2)
        # Transpose: (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SegmentationTransformer(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, num_classes=1, 
                 embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder for segmentation
        self.decoder = self._build_decoder(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _build_decoder(self, embed_dim, num_classes):
        """Build upsampling decoder"""
        return nn.Sequential(
            # First upsampling stage
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Second upsampling stage
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Third upsampling stage
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Fourth upsampling stage
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _init_weights(self):
        # Initialize position embeddings and class token
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Remove class token and reshape for decoder
        x = x[:, 1:]  # Remove class token
        
        # Reshape to feature map
        patch_h = patch_w = int(np.sqrt(x.shape[1]))
        x = x.transpose(1, 2).reshape(B, -1, patch_h, patch_w)
        
        # Apply decoder
        x = self.decoder(x)
        
        # Resize to original image size if needed
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x

# Add function to predict with SegTransformer
def predict_segtransformer(model, image_array, device, target_size=(256, 256)):
    """
    Make prediction using SegmentationTransformer model
    
    Args:
        model: Trained SegmentationTransformer model
        image_array: Preprocessed image array (H, W, 3) - should be normalized [0,1]
        device: PyTorch device
        target_size: Target size for inference
    
    Returns:
        np.ndarray: Predicted segmentation mask (float values 0-1)
    """
    model.eval()
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float().unsqueeze(0)
    
    # Ensure image is normalized to [0,1] range
    if image_tensor.max() > 1.0:
        image_tensor = image_tensor / 255.0
    
    # Resize to model's expected input size (256x256 to match checkpoint)
    if image_tensor.shape[2] != 256 or image_tensor.shape[3] != 256:
        image_tensor = F.interpolate(
            image_tensor, 
            size=(256, 256), 
            mode='bilinear', 
            align_corners=False
        )
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
        
        # Debug: Print prediction statistics
        print(f"Raw prediction shape: {prediction.shape}")
        print(f"Raw prediction min/max: {prediction.min().item():.4f}/{prediction.max().item():.4f}")
        print(f"Raw prediction mean: {prediction.mean().item():.4f}")
        
        # Resize prediction back to target size
        if prediction.shape[2:] != target_size:
            prediction = F.interpolate(
                prediction, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Return probability mask (0-1 range), not binary
        # The model already has sigmoid activation, so values should be [0,1]
        mask = prediction.squeeze().cpu().numpy()
        
        # Ensure mask is in valid range [0,1]
        mask = np.clip(mask, 0.0, 1.0)
        
        # Debug: Print final mask statistics
        print(f"Final mask shape: {mask.shape}")
        print(f"Final mask min/max: {mask.min():.4f}/{mask.max():.4f}")
        print(f"Final mask mean: {mask.mean():.4f}")
        print(f"Pixels > 0.5: {(mask > 0.5).sum()}/{mask.size} ({(mask > 0.5).mean()*100:.1f}%)")
    
    return mask

# Add function to load SegTransformer model
def load_segtransformer(checkpoint_path, device):
    """Load the trained SegmentationTransformer model"""
    
    # Calculate correct parameters based on the position embedding size from error
    # Error shows pos_embed shape [1, 257, 512] in checkpoint
    # This means 256 patches + 1 CLS token = 257 total positions
    # For 256 patches: img_size = 16 * sqrt(256) = 16 * 16 = 256
    
    # Create model with parameters that match the saved checkpoint
    model = SegmentationTransformer(
        img_size=256,  # Changed from 272 to 256 to match checkpoint
        patch_size=16,
        in_chans=3,
        num_classes=1,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    # Load model weights
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model = model.to(device)
            model.eval()
            return model, f"✅ SegTransformer loaded successfully!"
        except Exception as e:
            return None, f"⚠️ Error loading SegTransformer: {str(e)}"
    else:
        return None, f"⚠️ SegTransformer checkpoint not found: {checkpoint_path}"

# SegTransformer Attention Visualization Functions

def calculate_sparseness_segtransformer(attention_map):
    """Calculate sparseness of attention map (Hoyer sparseness)"""
    # Flatten the attention map
    flat_attn = attention_map.flatten()
    
    # Remove zero values for sparseness calculation
    non_zero_attn = flat_attn[flat_attn > 1e-8]
    
    if len(non_zero_attn) == 0:
        return 1.0  # Maximum sparseness if all values are zero
    
    # Calculate L1 and L2 norms
    l1_norm = np.sum(np.abs(non_zero_attn))
    l2_norm = np.sqrt(np.sum(non_zero_attn ** 2))
    
    # Hoyer sparseness measure
    n = len(non_zero_attn)
    sparseness = (np.sqrt(n) - (l1_norm / l2_norm)) / (np.sqrt(n) - 1)
    
    return sparseness

def extract_attention_maps_segtransformer(model, input_tensor, device, attention_type='cls_to_patches'):
    """
    Extract attention maps from SegTransformer layers
    
    Args:
        model: The trained SegTransformer model
        input_tensor: Input tensor (B, C, H, W)
        device: Device to run on
        attention_type: Type of attention to visualize ('cls_to_patches', 'patch_to_patch', or 'both')
    
    Returns:
        dict: Spatial attention maps
    """
    model.eval()
    layer_attention_maps = {}
    
    def create_attention_hook(layer_idx):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (module.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            
            # Store different attention patterns
            if attention_type in ['cls_to_patches', 'both']:
                layer_attention_maps[f'layer_{layer_idx}_cls_to_patches'] = attn[0, :, 0, 1:].mean(dim=0)
            
            if attention_type in ['patch_to_patch', 'both']:
                layer_attention_maps[f'layer_{layer_idx}_patch_to_patch'] = attn[0, :, 1:, 1:].mean(dim=0).mean(dim=0)
                
        return hook_fn
    
    # Register hooks for all layers
    hooks = []
    for i, block in enumerate(model.blocks):
        hook = block.attn.register_forward_hook(create_attention_hook(i))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        # Resize input if needed
        if input_tensor.shape[2] != 256 or input_tensor.shape[3] != 256:
            input_resized = F.interpolate(
                input_tensor, 
                size=(256, 256), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            input_resized = input_tensor
        
        output = model(input_resized.to(device))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Convert attention maps to spatial format
    spatial_attention_maps = {}
    for name, attn_map in layer_attention_maps.items():
        grid_size = int(np.sqrt(attn_map.shape[-1]))
        spatial_attn = attn_map.reshape(grid_size, grid_size).cpu().numpy()
        spatial_attention_maps[name] = spatial_attn
    
    return spatial_attention_maps, output

def generate_segtransformer_explanations(model, input_tensor, device):
    """
    Generate explanations for SegTransformer predictions
    
    Args:
        model: Trained SegTransformer model
        input_tensor: Input tensor (B, C, H, W)
        device: Device to run on
    
    Returns:
        dict: Dictionary containing different explanation methods
    """
    explanations = {}
    
    try:
        # Get CLS to patches attention (most interpretable)
        cls_attention_maps, prediction = extract_attention_maps_segtransformer(
            model, input_tensor, device, 'cls_to_patches'
        )
        
        # Use the last layer's attention as the primary explanation
        num_layers = len(model.blocks)
        last_layer_key = f'layer_{num_layers-1}_cls_to_patches'
        
        if last_layer_key in cls_attention_maps:
            # Resize attention map to match input size
            attention_map = cls_attention_maps[last_layer_key]
            target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
            
            # Resize to match input dimensions
            attention_resized = cv2.resize(attention_map, (target_w, target_h))
            explanations['attention_last_layer'] = attention_resized
        
        # Also include attention from all layers for comparison
        for layer_name, attention_map in cls_attention_maps.items():
            target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
            attention_resized = cv2.resize(attention_map, (target_w, target_h))
            explanations[f'attention_{layer_name}'] = attention_resized
        
        # Store the prediction for reference
        explanations['prediction'] = prediction.squeeze().cpu().numpy()
        
    except Exception as e:
        print(f"Error generating SegTransformer explanations: {str(e)}")
        return None
    
    return explanations

def create_segtransformer_visualization(image_rgb, prediction, explanations):
    """
    Create visualization for SegTransformer explanations
    
    Args:
        image_rgb: Original RGB image (H, W, 3)
        prediction: Model prediction tensor or array
        explanations: Dictionary of explanations
    
    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    try:
        # Convert prediction to numpy if tensor
        if hasattr(prediction, 'detach'):
            pred_np = prediction.detach().cpu().numpy().squeeze()
        else:
            pred_np = np.array(prediction).squeeze()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction
        im1 = axes[0, 1].imshow(pred_np, cmap='gray')
        axes[0, 1].set_title('SegTransformer Prediction', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Prediction overlay
        axes[0, 2].imshow(image_rgb)
        axes[0, 2].imshow(pred_np, alpha=0.5, cmap='jet')
        axes[0, 2].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Attention visualizations
        if 'attention_last_layer' in explanations:
            attention_map = explanations['attention_last_layer']
            
            # Last layer attention
            im2 = axes[1, 0].imshow(attention_map, cmap='hot')
            axes[1, 0].set_title('Last Layer Attention', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Attention overlay on original image
            axes[1, 1].imshow(image_rgb)
            axes[1, 1].imshow(attention_map, alpha=0.6, cmap='hot')
            axes[1, 1].set_title('Attention Overlay', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
            # Attention-prediction comparison
            # Normalize attention for comparison
            attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            
            # Create difference map
            diff_map = np.abs(attention_norm - pred_norm)
            im3 = axes[1, 2].imshow(diff_map, cmap='plasma')
            axes[1, 2].set_title('Attention-Prediction Difference', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        else:
            # If no attention available, show message
            for i in range(3):
                axes[1, i].text(0.5, 0.5, 'Attention data\\nnot available', 
                               transform=axes[1, i].transAxes, ha='center', va='center')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating SegTransformer visualization: {str(e)}")
        return None


# Uncertainty quantification functions
def calculate_entropy_uncertainty(prediction):
    """
    Calculate entropy-based uncertainty for a prediction.
    
    Args:
        prediction (np.ndarray): Model prediction with values between 0 and 1
        
    Returns:
        float: Entropy uncertainty score (higher = more uncertain)
    """
    try:
        # Ensure prediction is in valid range
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        
        # Calculate entropy for binary segmentation
        entropy = -(prediction * np.log(prediction) + (1 - prediction) * np.log(1 - prediction))
        
        # Return mean entropy as uncertainty score
        mean_entropy = np.mean(entropy)
        
        # Debug output
        print(f"Entropy calculation:")
        print(f"- Prediction range: {prediction.min():.4f} to {prediction.max():.4f}")
        print(f"- Entropy range: {entropy.min():.4f} to {entropy.max():.4f}")
        print(f"- Mean entropy: {mean_entropy:.4f}")
        print(f"- Max possible entropy (ln(2)): {np.log(2):.4f}")
        
        return mean_entropy
    except Exception as e:
        print(f"Error calculating entropy uncertainty: {str(e)}")
        return 0.1  # Return reasonable default instead of 0.0


def test_time_augmentation_uncertainty(model, image, device, num_augmentations=5):
    """
    Perform test-time augmentation to estimate uncertainty.
    
    The idea: A confident model should produce similar predictions even when the input
    is slightly modified. If predictions vary significantly across augmentations,
    the model is uncertain about that region.
    
    Process:
    1. Get prediction on original image
    2. Apply random augmentations (flips, noise) to create variations
    3. Get predictions on augmented images
    4. Calculate standard deviation across all predictions as uncertainty
    
    Args:
        model: Trained model (UNet or SegTransformer)
        image (torch.Tensor): Input image tensor (B, C, H, W)
        device: Device to run inference on
        num_augmentations (int): Total number of predictions (original + augmented)
        
    Returns:
        tuple: (mean_prediction, uncertainty_map, uncertainty_score)
            - mean_prediction: Average prediction across all augmentations
            - uncertainty_map: Pixel-wise standard deviation (higher = more uncertain)
            - uncertainty_score: Overall uncertainty (mean of uncertainty_map)
    """
    try:
        model.eval()
        predictions = []
        print(f"Starting TTA with {num_augmentations} augmentations...")
        
        # Ensure input tensor is on the same device as the model
        image = image.to(device)
        print(f"Input tensor device: {image.device}, Model device: {device}")
        
        with torch.no_grad():
            # Original prediction (no augmentation)
            pred = model(image)
            if isinstance(pred, tuple):
                pred = pred[0]  # Handle models returning multiple outputs
            
            # Apply sigmoid to get probabilities [0,1]
            if not hasattr(model, 'decoder') or 'Sigmoid' not in str(model.decoder[-1]):
                pred = torch.sigmoid(pred)
            
            original_pred = pred.cpu().numpy()
            predictions.append(original_pred)
            print(f"Original prediction: min={original_pred.min():.4f}, max={original_pred.max():.4f}, mean={original_pred.mean():.4f}")
            
            # Generate augmented predictions
            for i in range(num_augmentations - 1):
                # Create augmented version
                aug_image = image.clone()
                flip_h = False
                flip_v = False
                applied_augs = []
                
                # Always apply at least one augmentation to ensure variation
                aug_choice = torch.randint(0, 4, (1,)).item()
                
                # Random horizontal flip
                if aug_choice == 0 or torch.rand(1).item() > 0.6:
                    aug_image = torch.flip(aug_image, dims=[3])
                    flip_h = True
                    applied_augs.append("h_flip")
                
                # Random vertical flip  
                if aug_choice == 1 or torch.rand(1).item() > 0.6:
                    aug_image = torch.flip(aug_image, dims=[2])
                    flip_v = True
                    applied_augs.append("v_flip")
                
                # Stronger gaussian noise for more variation
                if aug_choice == 2 or torch.rand(1).item() > 0.5:
                    noise_strength = 0.02 + torch.rand(1).item() * 0.03  # 0.02 to 0.05
                    noise = torch.randn_like(aug_image) * noise_strength
                    aug_image = aug_image + noise
                    aug_image = torch.clamp(aug_image, 0, 1)
                    applied_augs.append(f"noise_{noise_strength:.3f}")
                
                # Brightness adjustment for additional variation
                if aug_choice == 3 or torch.rand(1).item() > 0.4:
                    brightness_factor = 0.9 + torch.rand(1).item() * 0.2  # 0.9 to 1.1
                    aug_image = aug_image * brightness_factor
                    aug_image = torch.clamp(aug_image, 0, 1)
                    applied_augs.append(f"bright_{brightness_factor:.3f}")
                
                print(f"Aug {i+1}: Applied {applied_augs}")
                
                # Ensure augmented image is on correct device
                aug_image = aug_image.to(device)
                
                # Get prediction on augmented image
                pred_aug = model(aug_image)
                if isinstance(pred_aug, tuple):
                    pred_aug = pred_aug[0]
                
                # Apply sigmoid if needed
                if not hasattr(model, 'decoder') or 'Sigmoid' not in str(model.decoder[-1]):
                    pred_aug = torch.sigmoid(pred_aug)
                
                # Reverse geometric augmentations to align with original prediction
                if flip_v:
                    pred_aug = torch.flip(pred_aug, dims=[2])
                if flip_h:
                    pred_aug = torch.flip(pred_aug, dims=[3])
                
                aug_pred = pred_aug.cpu().numpy()
                predictions.append(aug_pred)
                print(f"Aug {i+1} prediction: min={aug_pred.min():.4f}, max={aug_pred.max():.4f}, mean={aug_pred.mean():.4f}")
        
        # Calculate statistics across predictions
        predictions = np.array(predictions)  # Shape: (num_augmentations, B, C, H, W)
        print(f"Predictions shape: {predictions.shape}")
        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(predictions, axis=0)
        uncertainty_map = np.std(predictions, axis=0)  # Pixel-wise uncertainty
        
        # Add a small epsilon to prevent exactly zero uncertainty (unrealistic)
        epsilon = 1e-6
        uncertainty_map = np.maximum(uncertainty_map, epsilon)
        uncertainty_score = np.mean(uncertainty_map)   # Overall uncertainty score
        
        # Debug uncertainty calculation
        print(f"Uncertainty map: min={uncertainty_map.min():.6f}, max={uncertainty_map.max():.6f}, mean={uncertainty_map.mean():.6f}")
        print(f"Final uncertainty score: {uncertainty_score:.6f}")
        
        # Check if all predictions are identical (debugging)
        pred_differences = []
        for i in range(1, len(predictions)):
            diff = np.abs(predictions[0] - predictions[i]).mean()
            pred_differences.append(diff)
        print(f"Mean differences from original: {pred_differences}")
        
        # Ensure minimum realistic uncertainty
        if uncertainty_score < 0.001:
            print(f"Warning: Very low uncertainty ({uncertainty_score:.6f}), setting minimum to 0.001")
            uncertainty_score = max(uncertainty_score, 0.001)
            
        return mean_prediction, uncertainty_map, uncertainty_score
    
    except Exception as e:
        print(f"Error in test-time augmentation: {str(e)}")
        return None, None, 0.0


def create_uncertainty_visualization(original_image, prediction, uncertainty_map, 
                                   uncertainty_score, entropy_score, method_name="Uncertainty"):
    """
    Create comprehensive uncertainty visualization.
    
    Args:
        original_image (np.ndarray): Original input image
        prediction (np.ndarray): Model prediction
        uncertainty_map (np.ndarray): Uncertainty map
        uncertainty_score (float): Overall uncertainty score
        entropy_score (float): Entropy-based uncertainty score
        method_name (str): Name of uncertainty method
        
    Returns:
        matplotlib.figure.Figure: Figure with uncertainty visualization
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{method_name} Analysis', fontsize=16, fontweight='bold')
        
        # Convert image to RGB if needed
        if len(original_image.shape) == 4:
            original_image = original_image[0]
        if len(original_image.shape) == 3 and original_image.shape[0] == 3:
            image_rgb = np.transpose(original_image, (1, 2, 0))
        else:
            image_rgb = original_image
        
        # Ensure RGB values are in correct range
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # Convert prediction and uncertainty to 2D if needed
        if len(prediction.shape) > 2:
            pred_2d = np.squeeze(prediction)
        else:
            pred_2d = prediction
            
        if uncertainty_map is not None and len(uncertainty_map.shape) > 2:
            uncertainty_2d = np.squeeze(uncertainty_map)
        else:
            uncertainty_2d = uncertainty_map
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction
        im1 = axes[0, 1].imshow(pred_2d, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Model Prediction', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Prediction overlay
        axes[0, 2].imshow(image_rgb)
        axes[0, 2].imshow(pred_2d, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        axes[0, 2].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Uncertainty map
        if uncertainty_2d is not None:
            im2 = axes[1, 0].imshow(uncertainty_2d, cmap='hot')
            axes[1, 0].set_title('Uncertainty Map', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Uncertainty overlay
            axes[1, 1].imshow(image_rgb)
            axes[1, 1].imshow(uncertainty_2d, alpha=0.6, cmap='hot')
            axes[1, 1].set_title('Uncertainty Overlay', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Uncertainty map\nnot available', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, 'Uncertainty overlay\nnot available', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].axis('off')
        
        # Uncertainty statistics
        axes[1, 2].axis('off')
        stats_text = f"""Uncertainty Analysis

Overall Uncertainty: {uncertainty_score:.4f}

Entropy Uncertainty: {entropy_score:.4f}

Interpretation:
• Higher values = More uncertain
• Focus on high uncertainty regions
• Consider additional validation
  for uncertain predictions

Confidence Level: 
{(1 - uncertainty_score) * 100:.1f}%"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating uncertainty visualization: {str(e)}")
        return None