
import sys
sys.path.append('./GeoSeg')

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from geoseg.datasets.loveda_dataset import LoveDATestDataset, CLASSES, PALETTE
from geoseg.models.UNetFormer import UNetFormer
from tools.cfg import py2cfg

print("="*70)
print("🚀 Starting UNetFormer Model Evaluation")
print("="*70)

# Helper Functions
def label2rgb(mask):
    """Convert label mask to RGB visualization"""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]  # background
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]      # building
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]    # road
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]      # water
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]  # barren
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]      # forest
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]  # agricultural
    return mask_rgb

def calculate_iou(pred, target, num_classes):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return ious

def calculate_dice(pred, target, num_classes):
    """Calculate Dice coefficient for each class"""
    dice_scores = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum()
        
        if pred_inds.sum() + target_inds.sum() == 0:
            dice_scores.append(float('nan'))
        else:
            dice_scores.append(2 * intersection / (pred_inds.sum() + target_inds.sum()))
    
    return dice_scores

def calculate_pixel_accuracy(pred, target):
    """Calculate overall pixel accuracy"""
    correct = (pred == target).sum()
    total = target.size
    return correct / total

# Validation Dataset
class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.images = []
        self.masks = []
        
        # Collect all image and mask pairs
        for split in ['Rural', 'Urban']:
            img_dir = os.path.join(data_root, split, 'images_png')
            mask_dir = os.path.join(data_root, split, 'masks_png')
            
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    mask_path = os.path.join(mask_dir, img_file)
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)
        
        print(f"✓ Found {len(self.images)} validation samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        
        # FIX: Convert label 7 to label 6 for agricultural class
        # The ground truth uses 7 for agricultural, but model uses 6
        mask[mask == 7] = 6
        # Also convert 0 to 0 (background) - some masks might use 255 for unlabeled
        mask[mask == 255] = 0
        
        # Normalize image
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        
        return img, mask, self.images[idx]

# Main execution
if __name__ == "__main__":
    # Setup
    print(f"\n📦 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Model checkpoint path (don't load config as it tries to initialize datasets)
    checkpoint_path = './GeoSeg/trained_models/archive/unetformer-archive-512crop-ms-epoch30/last.ckpt'
    
    print(f"\n📊 Classes: {CLASSES}")
    print(f"📊 Number of classes: {len(CLASSES)}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 Using device: {device}")
    
    print("⏳ Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = UNetFormer(num_classes=len(CLASSES))
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'net.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Load validation dataset
    print("\n📂 Loading validation dataset...")
    val_data_root = './archive/Val/Val'
    
    if not os.path.exists(val_data_root):
        print(f"❌ Warning: {val_data_root} not found!")
        sys.exit(1)
    
    val_dataset = ValidationDataset(val_data_root)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Run inference
    print("\n🔄 Running inference on validation set...")
    all_predictions = []
    all_targets = []
    all_images = []
    all_ious = []
    all_dice = []
    all_accuracies = []
    
    with torch.no_grad():
        for idx, (img, mask, img_path) in enumerate(tqdm(val_loader, desc="Processing")):
            img = img.to(device)
            mask = mask.to(device)
            
            # Forward pass
            output = model(img)
            
            # Get prediction
            pred = output.argmax(dim=1)
            
            # Move to CPU for metrics calculation
            pred_np = pred.cpu().numpy()[0]
            mask_np = mask.cpu().numpy()[0]
            img_np = img.cpu().numpy()[0]
            
            # Denormalize image for visualization
            img_np = img_np.transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Calculate metrics
            iou = calculate_iou(pred_np, mask_np, len(CLASSES))
            dice = calculate_dice(pred_np, mask_np, len(CLASSES))
            accuracy = calculate_pixel_accuracy(pred_np, mask_np)
            
            all_predictions.append(pred_np)
            all_targets.append(mask_np)
            all_images.append(img_np)
            all_ious.append(iou)
            all_dice.append(dice)
            all_accuracies.append(accuracy)
            
            # Process 50 images for evaluation
            if idx >= 49:
                break
    
    print(f"✓ Processed {len(all_predictions)} validation samples")
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    mean_dice_per_class = np.nanmean(all_dice, axis=0)
    mean_accuracy = np.mean(all_accuracies)
    mean_iou = np.nanmean(mean_iou_per_class)
    mean_dice = np.nanmean(mean_dice_per_class)
    
    # Print results
    print("\n" + "="*70)
    print(" "*20 + "OVERALL PERFORMANCE METRICS")
    print("="*70)
    print(f"\n🎯 Mean Pixel Accuracy: {mean_accuracy*100:.2f}%")
    print(f"🎯 Mean IoU (mIoU): {mean_iou*100:.2f}%")
    print(f"🎯 Mean Dice Score: {mean_dice*100:.2f}%")
    
    print("\n" + "="*70)
    print(" "*25 + "PER-CLASS METRICS")
    print("="*70)
    print(f"\n{'Class':<15} {'IoU (%)':<12} {'Dice (%)':<12}")
    print("-"*70)
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name:<15} {mean_iou_per_class[i]*100:>10.2f}  {mean_dice_per_class[i]*100:>10.2f}")
    print("="*70)
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    # 1. Per-class metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].bar(range(len(CLASSES)), mean_iou_per_class * 100, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Intersection over Union (IoU) per Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(CLASSES)))
    axes[0].set_xticklabels(CLASSES, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=mean_iou*100, color='r', linestyle='--', label=f'Mean IoU: {mean_iou*100:.2f}%')
    axes[0].legend()
    
    axes[1].bar(range(len(CLASSES)), mean_dice_per_class * 100, color='coral', alpha=0.8)
    axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Dice Score (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Dice Coefficient per Class', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(CLASSES)))
    axes[1].set_xticklabels(CLASSES, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=mean_dice*100, color='r', linestyle='--', label=f'Mean Dice: {mean_dice*100:.2f}%')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ per_class_metrics.png")
    
    # 2. Confusion Matrix
    all_preds_flat = np.concatenate([p.flatten() for p in all_predictions])
    all_targets_flat = np.concatenate([t.flatten() for t in all_targets])
    
    cm = confusion_matrix(all_targets_flat, all_preds_flat, labels=range(len(CLASSES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES, 
                cbar_kws={'label': 'Normalized Count'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12, fontweight='bold')
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ confusion_matrix.png")
    
    # 3. Predictions comparison
    num_samples = min(10, len(all_predictions))
    indices = np.linspace(0, len(all_predictions)-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    for i, idx in enumerate(indices):
        img = all_images[idx]
        pred = all_predictions[idx]
        target = all_targets[idx]
        
        error_map = (pred != target).astype(np.uint8)
        pred_rgb = label2rgb(pred)
        target_rgb = label2rgb(target)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {idx+1}: Original Image', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_rgb)
        axes[i, 1].set_title('Ground Truth', fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_rgb)
        iou_mean = np.nanmean(all_ious[idx])
        axes[i, 2].set_title(f'Prediction (IoU: {iou_mean*100:.2f}%)', fontweight='bold')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        error_pct = (error_map.sum() / error_map.size) * 100
        axes[i, 3].set_title(f'Error Map ({error_pct:.2f}% wrong)', fontweight='bold')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ predictions_comparison.png")
    
    # 4. Error Analysis
    error_rates = []
    for pred, target in zip(all_predictions, all_targets):
        error_rate = ((pred != target).sum() / target.size) * 100
        error_rates.append(error_rate)
    
    error_rates = np.array(error_rates)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].hist(error_rates, bins=30, color='indianred', alpha=0.7, edgecolor='black')
    axes[0].axvline(error_rates.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {error_rates.mean():.2f}%')
    axes[0].axvline(np.median(error_rates), color='blue', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(error_rates):.2f}%')
    axes[0].set_xlabel('Error Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Pixel Error Rates', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].boxplot(error_rates, vert=True)
    axes[1].set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Error Rate Statistics', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ error_analysis.png")
    
    # 5. Best and Worst Predictions
    sample_ious = [np.nanmean(iou) for iou in all_ious]
    best_idx = np.argmax(sample_ious)
    worst_idx = np.argmin(sample_ious)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for col, (img, title) in enumerate([
        (all_images[best_idx], 'Best: Original'),
        (label2rgb(all_targets[best_idx]), 'Ground Truth'),
        (label2rgb(all_predictions[best_idx]), f'Prediction (IoU: {sample_ious[best_idx]*100:.2f}%)'),
        ((all_predictions[best_idx] != all_targets[best_idx]).astype(np.uint8), 'Error Map')
    ]):
        if col == 3:
            axes[0, col].imshow(img, cmap='Reds', vmin=0, vmax=1)
        else:
            axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
    
    for col, (img, title) in enumerate([
        (all_images[worst_idx], 'Worst: Original'),
        (label2rgb(all_targets[worst_idx]), 'Ground Truth'),
        (label2rgb(all_predictions[worst_idx]), f'Prediction (IoU: {sample_ious[worst_idx]*100:.2f}%)'),
        ((all_predictions[worst_idx] != all_targets[worst_idx]).astype(np.uint8), 'Error Map')
    ]):
        if col == 3:
            axes[1, col].imshow(img, cmap='Reds', vmin=0, vmax=1)
        else:
            axes[1, col].imshow(img)
        axes[1, col].set_title(title, fontsize=12, fontweight='bold')
        axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('best_worst_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ best_worst_predictions.png")
    
    # 6. Class Legend
    colors = [
        [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
        [159, 129, 183], [0, 255, 0], [255, 195, 128]
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    from matplotlib.patches import Rectangle
    for i, (class_name, color) in enumerate(zip(CLASSES, colors)):
        rect = Rectangle((0, i*0.8), 1, 0.6, facecolor=np.array(color)/255.0, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(1.2, i*0.8 + 0.3, class_name, fontsize=14, fontweight='bold', va='center')
    
    ax.set_xlim(-0.2, 4)
    ax.set_ylim(-0.5, len(CLASSES)*0.8)
    ax.set_title('Class Color Legend', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('class_legend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ class_legend.png")
    
    # Final Summary
    print("\n" + "="*70)
    print(" "*20 + "📊 ERROR ANALYSIS SUMMARY")
    print("="*70)
    print(f"\n   Mean Error Rate: {error_rates.mean():.2f}%")
    print(f"   Median Error Rate: {np.median(error_rates):.2f}%")
    print(f"   Std Deviation: {error_rates.std():.2f}%")
    print(f"   Min Error: {error_rates.min():.2f}%")
    print(f"   Max Error: {error_rates.max():.2f}%")
    print(f"   Best Sample IoU: {max(sample_ious)*100:.2f}%")
    print(f"   Worst Sample IoU: {min(sample_ious)*100:.2f}%")
    
    print("\n" + "="*70)
    print(" "*15 + "✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\n💾 All visualization files saved in current directory")
