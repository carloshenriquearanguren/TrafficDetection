import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import wandb
import os
from typing import Optional, List, Tuple
import random

class SemanticSegmentationModel(nn.Module):
    """Semantic segmentation model with encoder-decoder architecture"""
    
    def __init__(self, num_classes: int, backbone: str = 'resnet50'):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.channels = [256, 512, 1024, 2048]  # ResNet-50 feature channels
        elif backbone == 'resnet101':
            self.backbone = torchvision.models.resnet101(pretrained=True)
            self.channels = [256, 512, 1024, 2048]  # ResNet-101 feature channels
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(self.channels[-1], self.channels[-2]),
            DecoderBlock(self.channels[-2], self.channels[-3]),
            DecoderBlock(self.channels[-3], self.channels[-4]),
            DecoderBlock(self.channels[-4], 64)
        ])
        
        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for decoder and final layer"""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.final.weight, mode='fan_out', nonlinearity='relu')
        if self.final.bias is not None:
            nn.init.constant_(self.final.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Encoder
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 5, 6, 7]:  # Store features from each ResNet block
                features.append(x)
        
        # Decoder
        x = features[-1]
        for i, decoder in enumerate(self.decoder):
            x = decoder(x, features[-(i+2)])
        
        # Final classification
        x = self.final(x)
        
        return x

class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    
    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection"""
        x = self.up(x)
        
        # Handle different sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x

class SemanticSegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform: Optional[A.Compose] = None):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.transform = transform
        
        print(f"[Dataset Debug] Initializing dataset for split: '{split}'")
        print(f"[Dataset Debug] Looking for data in: {self.split_dir}")
        
        self.samples = []
        if not self.split_dir.exists():
            print(f"[Dataset Debug] ERROR: Directory does not exist: {self.split_dir}")
            return
        if not self.split_dir.is_dir():
            print(f"[Dataset Debug] ERROR: Path is not a directory: {self.split_dir}")
            return
            
        sample_folders_found = 0
        manifests_loaded = 0
        for sample_folder in sorted(self.split_dir.iterdir()):
            if sample_folder.is_dir():
                sample_folders_found += 1
                manifest_path = sample_folder / 'manifest.json'
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            self.samples.append(json.load(f))
                            manifests_loaded += 1
                    except json.JSONDecodeError as e:
                        print(f"[Dataset Debug] ERROR: Could not decode JSON from {manifest_path}: {e}")
                # else: # Optional: print if manifest not found in a sample folder
                #     print(f"[Dataset Debug] Manifest not found in {sample_folder}")
        
        print(f"[Dataset Debug] Found {sample_folders_found} potential sample folders.")
        print(f"[Dataset Debug] Loaded {manifests_loaded} manifests.")
        print(f"[Dataset Debug] Total samples collected for split '{split}': {len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        sample_id = sample['sample_id']
        
        # Load image
        image_path = self.split_dir / sample_id / sample['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.split_dir / sample_id / sample['semantic_mask_path']
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask,
            'sample_id': sample_id
        }

class SemanticSegmentationTrainer:
    """Trainer for semantic segmentation model"""
    
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str = 'cuda', output_dir: str = './outputs'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f'Checkpoint saved: {path}')

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Semantic Segmentation Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set up transforms
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(1080, 1920), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = SemanticSegmentationDataset(
        args.data_dir,
        split='training',
        transform=train_transform
    )
    
    val_dataset = SemanticSegmentationDataset(
        args.data_dir,
        split='validation',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = SemanticSegmentationModel(num_classes=8).to(device)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize trainer
    trainer = SemanticSegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir
    )
    
    # Train model
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1)
    
    print("Training completed!")

if __name__ == "__main__":
    main()