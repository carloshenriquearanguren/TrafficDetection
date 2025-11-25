import os
import torch
import torchvision
from torch.utils.data import DataLoader
from coco_dataset import COCODataset
from transforms import get_transform
import traceback
from torchvision import transforms

def collate_fn(batch):
    """Custom collate function to handle variable batch sizes"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))

# paths to your converted COCO data
COCO_ROOT = "./coco"
WAYMO_IMG = f"{COCO_ROOT}/waymo/images"
WAYMO_ANN = f"{COCO_ROOT}/waymo/annotations.json"
KITTI_IMG = f"{COCO_ROOT}/kitti/images"
KITTI_ANN = f"{COCO_ROOT}/kitti/annotations.json"

# transforms
train_tf  = get_transform(train=True)

# instantiate COCO datasets
waymo_ds = COCODataset(WAYMO_IMG, WAYMO_ANN, transforms=train_tf)
kitti_ds = COCODataset(KITTI_IMG, KITTI_ANN, transforms=train_tf)

# combine them
from torch.utils.data import ConcatDataset
train_ds = ConcatDataset([waymo_ds, kitti_ds])

# DataLoader
from torch.utils.data import DataLoader
def collate_fn(batch): return tuple(zip(*batch))
train_loader = DataLoader(
    train_ds, batch_size=2, shuffle=True,
    num_workers=4, collate_fn=collate_fn
)
# Model setup
model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
)
num_classes = 4  # Background + 3 classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
in_mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_mask_features, 256, num_classes
)

# Optimizer and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop with error handling
for epoch in range(10):
    model.train()
    total_loss = 0
    batch_count = 0
    max_batches = 100  # Limit batches per epoch for testing
    
    try:
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            try:
                # Skip empty batches
                if len(images) == 0:
                    continue
                    
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Check for NaN losses
                if torch.isnan(losses):
                    print(f"NaN loss detected, skipping batch {batch_idx}")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += losses.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {losses.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")
        traceback.print_exc()
        continue
    
    avg_loss = total_loss / max(batch_count, 1)
    print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")

# Save model
print("Saving model...")
torch.save(model.state_dict(), "maskrcnn_waymo_kitti.pth")
print("Training completed!")
