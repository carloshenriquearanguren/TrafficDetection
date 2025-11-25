#!/usr/bin/env python3
"""
Updated train.py to use COCO preprocessing pipeline
Following the milestone report specifications for Waymo/KITTI obstacle detection
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import your COCO dataset and transforms
from coco_dataset import COCODataset
from your_transforms import get_transform, get_simple_transform

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Also add gradient checkpointing if using a model that supports it
torch.backends.cudnn.benchmark = True

# Training configuration
CONFIG = {
    # Data paths - adjust these to your COCO converted data
    'waymo_img': 'waymo_images',
    'waymo_ann': 'waymo_annotations.json',
    'kitti_img': 'kitti_images', 
    'kitti_ann': 'kitti_annotations.json',
    
    # Training parameters (from milestone report)
    'batch_size': 2,  # Reduced for stability
    'num_epochs': 12,
    'lr': 0.0001,  # Much lower learning rate to prevent NaN
    'weight_decay': 1e-4,
    'num_workers': 2,  # Reduced
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Model parameters
    'num_classes': 4,  # background + 3 classes (vehicle, pedestrian, cyclist)
    'pretrained': True,
    
    # Checkpointing
    'save_dir': './checkpoints',
    'save_every': 2,  # Save every N epochs
}


def collate_fn(batch):
    """Custom collate function for object detection"""
    return tuple(zip(*batch))


def validate_targets(targets):
    """Validate and fix invalid bounding boxes to prevent NaN losses"""
    for target in targets:
        boxes = target['boxes']
        # Check for invalid boxes
        if len(boxes) == 0:
            continue
            
        # Check box format [x1, y1, x2, y2] where x2 > x1, y2 > y1
        if torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1]):
            print(f"Invalid boxes found, fixing...")
            # Fix invalid boxes
            boxes[:, 2] = torch.clamp(boxes[:, 2], min=boxes[:, 0] + 1)
            boxes[:, 3] = torch.clamp(boxes[:, 3], min=boxes[:, 1] + 1)
            target['boxes'] = boxes
            
        # Check for boxes outside reasonable bounds
        boxes = torch.clamp(boxes, min=0)
        target['boxes'] = boxes
    
    return targets


def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def create_model(num_classes, pretrained=True):
    """
    Create Mask R-CNN model as specified in milestone report
    ResNet-50 + FPN backbone, pretrained on COCO
    """
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


def create_datasets():
    """Create training and validation datasets"""
    print("Creating datasets...")
    
    # Check if COCO data exists
    if not os.path.exists(CONFIG['waymo_ann']):
        print(f"Warning: Waymo annotations not found at {CONFIG['waymo_ann']}")
        print("Make sure to run coco_transform.py first!")
        return None, None
    
    if not os.path.exists(CONFIG['kitti_ann']):
        print(f"Warning: KITTI annotations not found at {CONFIG['kitti_ann']}")
        print("Make sure to run coco_transform.py first!")
        return None, None
    
    # Create transforms
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)
    
    # Create individual datasets
    print("Loading Waymo dataset...")
    waymo_ds = COCODataset(
        CONFIG['waymo_img'], 
        CONFIG['waymo_ann'], 
        transforms=train_transform
    )
    
    print("Loading KITTI dataset...")
    kitti_ds = COCODataset(
        CONFIG['kitti_img'], 
        CONFIG['kitti_ann'], 
        transforms=train_transform
    )
    
    # Combine datasets
    train_ds = ConcatDataset([waymo_ds, kitti_ds])
    
    # For validation, use a subset of the training data
    # In production, you'd have separate validation sets
    val_ds = ConcatDataset([
        COCODataset(CONFIG['waymo_img'], CONFIG['waymo_ann'], transforms=val_transform),
        COCODataset(CONFIG['kitti_img'], CONFIG['kitti_ann'], transforms=val_transform)
    ])
    
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    
    return train_ds, val_ds


def create_data_loaders(train_ds, val_ds):
    """Create data loaders"""
    print("Creating data loaders...")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],  # Same batch size for validation
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    return train_loader, val_loader


def debug_batch(images, targets, batch_idx):
    """Debug function to check batch data"""
    if batch_idx == 0:  # Only debug first batch
        print(f"Batch size: {len(images)}")
        for i, (img, target) in enumerate(zip(images, targets)):
            print(f"Image {i}: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}")
            print(f"  Boxes: {target['boxes'].shape}, Labels: {target['labels'].shape}")
            if len(target['boxes']) > 0:
                print(f"  Box range: {target['boxes'].min():.3f} to {target['boxes'].max():.3f}")
                print(f"  Label range: {target['labels'].min()} to {target['labels'].max()}")


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch with NaN protection"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
    print("-" * 50)
    
    for i, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Validate targets to prevent NaN
        targets = validate_targets(targets)
        
        # Debug first batch
        debug_batch(images, targets, i)
        
        try:
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Check for NaN losses
            if not torch.isfinite(losses):
                print(f"Non-finite loss detected at batch {i}, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            
        except RuntimeError as e:
            print(f"Runtime error at batch {i}: {e}")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
        
        # Print progress and clear cache periodically
        if (i + 1) % 10 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Batch {i + 1}/{len(data_loader)}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Current Loss: {losses.item():.4f}")
            
            # Print individual loss components
            loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            print(f"  Components: {loss_str}")
            
            # Clear cache every 10 batches
            torch.cuda.empty_cache()
        
        # Print GPU utilization every 50 batches
        if (i + 1) % 50 == 0:
            print_gpu_utilization()
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, data_loader, device):
    """Validation loop with fix for model output handling"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    print("\nValidating...")
    with torch.no_grad():
        for images, targets in data_loader:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Validate targets
            targets = validate_targets(targets)
            
            try:
                # Set model to training mode temporarily to get loss dict
                model.train()
                loss_dict = model(images, targets)
                model.eval()
                
                # Check if we got a dictionary (training mode) or list (eval mode)
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    
                    if torch.isfinite(losses):
                        total_loss += losses.item()
                        num_batches += 1
                else:
                    # Model returned predictions instead of losses
                    print(f"Model returned {type(loss_dict)} instead of loss dict, skipping loss calculation")
                    num_batches += 1  # Still count the batch as processed
                    
            except RuntimeError as e:
                print(f"Validation error: {e}")
                continue
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_loss:.4f} (computed from {num_batches} batches)")
    else:
        avg_loss = 0
        print("Validation completed but no losses computed")
    
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': CONFIG
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    """Main training function"""
    print("Starting training with COCO preprocessed data...")
    print(f"Device: {CONFIG['device']}")
    print(f"Configuration: {CONFIG}")
    
    # Create datasets
    train_ds, val_ds = create_datasets()
    if train_ds is None:
        print("Failed to create datasets. Exiting.")
        return
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_ds, val_ds)
    
    # Create model
    print("Creating model...")
    model = create_model(CONFIG['num_classes'], CONFIG['pretrained'])
    model.to(CONFIG['device'])
    
    # Create optimizer with lower learning rate
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, StepLR
    
    warmup_epochs = 1
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = StepLR(optimizer, step_size=4, gamma=0.1)
    
    lr_scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, CONFIG['device'], epoch)
        
        # Validate
        val_loss = validate(model, val_loader, CONFIG['device'])
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['save_every'] == 0:
            checkpoint_path = os.path.join(
                CONFIG['save_dir'], 
                f"maskrcnn_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CONFIG['save_dir'], "maskrcnn_best.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"New best model saved! Validation loss: {val_loss:.4f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 60)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # Check for required files
    required_files = ['coco_dataset.py', 'your_transforms.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        print("Make sure all files are in the same directory.")
        sys.exit(1)
    
    main()
