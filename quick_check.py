import sys
sys.path.append('./GeoSeg')

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from geoseg.datasets.loveda_dataset import CLASSES
from geoseg.models.UNetFormer import UNetFormer

print("="*70)
print("Quick Model Check")
print("="*70)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './GeoSeg/trained_models/archive/unetformer-archive-512crop-ms-epoch30/last.ckpt'

print(f"\nDevice: {device}")
print(f"Loading checkpoint from: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

if 'state_dict' in checkpoint:
    print(f"State dict keys (first 5): {list(checkpoint['state_dict'].keys())[:5]}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")

# Load one validation image
val_data_root = './archive/Val/Val'
img_path = None
mask_path = None

for split in ['Rural', 'Urban']:
    img_dir = os.path.join(val_data_root, split, 'images_png')
    mask_dir = os.path.join(val_data_root, split, 'masks_png')
    
    if os.path.exists(img_dir) and os.path.exists(mask_dir):
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        if img_files:
            img_path = os.path.join(img_dir, img_files[0])
            mask_path = os.path.join(mask_dir, img_files[0])
            break

if img_path and os.path.exists(mask_path):
    print(f"\nTesting with image: {img_path}")
    
    # Load image and mask
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"Mask value counts:")
    unique, counts = np.unique(mask, return_counts=True)
    for val, count in zip(unique, counts):
        if val < len(CLASSES):
            print(f"  {CLASSES[val]} ({val}): {count} pixels ({count/mask.size*100:.2f}%)")
        else:
            print(f"  Unknown ({val}): {count} pixels")
    
    # Initialize and load model
    model = UNetFormer(num_classes=len(CLASSES))
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    
    model = model.to(device)
    model.eval()
    
    # Prepare image
    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).cpu().numpy()[0]
    
    print(f"\nPrediction shape: {pred.shape}")
    print(f"Prediction unique values: {np.unique(pred)}")
    print(f"Prediction value counts:")
    unique_pred, counts_pred = np.unique(pred, return_counts=True)
    for val, count in zip(unique_pred, counts_pred):
        if val < len(CLASSES):
            print(f"  {CLASSES[val]} ({val}): {count} pixels ({count/pred.size*100:.2f}%)")
    
    # Calculate accuracy for this one image
    accuracy = (pred == mask).sum() / mask.size
    print(f"\nPixel Accuracy for this image: {accuracy*100:.2f}%")
    
    # Check if there's a consistent offset
    print(f"\nChecking for label offset issues...")
    for offset in range(-7, 8):
        if offset == 0:
            continue
        shifted_pred = np.clip(pred + offset, 0, len(CLASSES)-1)
        shifted_acc = (shifted_pred == mask).sum() / mask.size
        if shifted_acc > 0.5:
            print(f"  With offset {offset:+d}: accuracy = {shifted_acc*100:.2f}%")
    
    print("\n" + "="*70)
    print("Check complete!")
    print("="*70)
else:
    print("No validation images found!")
