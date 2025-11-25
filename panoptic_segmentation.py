import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import wandb
import os
import argparse

class WaymoMultiModalDataset(Dataset):
    """Dataset for multi-modal (camera + LiDAR) Waymo data"""
    
    def __init__(self, data_dir, split='train', transform=None, image_size=(1920, 1080)):
        self.data_root = Path(data_dir)
        self.split_dir = self.data_root / split
        self.transform = transform
        self.image_size = image_size # Default or expected image size for masks if no image loaded
        
        self.samples = []
        # Assuming each sample has a manifest.json in its own subdirectory
        # e.g., data_dir/train/sample_001/manifest.json, data_dir/train/sample_002/manifest.json
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        for sample_folder in sorted(self.split_dir.iterdir()):
            if sample_folder.is_dir():
                manifest_path = sample_folder / 'manifest.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        self.samples.append(json.load(f))
                else:
                    print(f"Warning: manifest.json not found in {sample_folder}")
        
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.split_dir}. Check data structure and manifest files.")

        print(f"Loaded Waymo {split} dataset from {self.split_dir}")
        print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_manifest = self.samples[idx]
        sample_id = sample_manifest['sample_id']
        sample_base_path = self.split_dir / sample_id # Path to the specific sample's folder

        # Load image
        image_path_rel = sample_manifest['image_path']
        image_path_abs = sample_base_path / image_path_rel
        if not image_path_abs.exists():
            raise FileNotFoundError(f"Image not found: {image_path_abs} for sample {sample_id}")
        image = cv2.imread(str(image_path_abs))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations expects RGB
        current_image_h, current_image_w = image.shape[:2]

        # Load LiDAR point cloud
        lidar_path_rel = sample_manifest['lidar_path']
        lidar_path_abs = sample_base_path / lidar_path_rel
        if not lidar_path_abs.exists():
            raise FileNotFoundError(f"LiDAR data not found: {lidar_path_abs} for sample {sample_id}")
        point_cloud = np.load(lidar_path_abs)  # Assuming preprocessed .npy format
        
        # Load calibration matrices
        calib = sample_manifest['calibration']
        camera_intrinsic = np.array(calib['camera_intrinsic'])
        lidar_to_camera = np.array(calib['lidar_to_camera'])
        
        # Load annotations (panoptic mask)
        panoptic_mask = None
        if 'panoptic_mask_path' in sample_manifest and sample_manifest['panoptic_mask_path']:
            mask_path_rel = sample_manifest['panoptic_mask_path']
            mask_path_abs = sample_base_path / mask_path_rel
            if mask_path_abs.exists():
                panoptic_mask = cv2.imread(str(mask_path_abs), cv2.IMREAD_UNCHANGED)
                if panoptic_mask is None:
                    print(f"Warning: Failed to load panoptic mask {mask_path_abs} for sample {sample_id}")
                    # Create a dummy mask if loading fails but path exists
                    panoptic_mask = np.zeros((current_image_h, current_image_w), dtype=np.uint8)
            else:
                print(f"Warning: Panoptic mask path specified but not found: {mask_path_abs} for sample {sample_id}")
                panoptic_mask = np.zeros((current_image_h, current_image_w), dtype=np.uint8)
        else:
            # Create a dummy mask if no path provided
            panoptic_mask = np.zeros((current_image_h, current_image_w), dtype=np.uint8)

        # Prepare target dictionary (can be expanded with boxes, labels if needed later)
        target = {
            'panoptic_mask': panoptic_mask, # Will be transformed by self.transform
            'sample_id': sample_id,
            # Store original image size for potential back-referencing or transforms that need it
            'orig_size': torch.tensor([current_image_h, current_image_w]) 
        }
        
        # Apply transforms (e.g., from albumentations)
        # Albumentations typically takes image and masks (plural)
        if self.transform:
            # The transform should handle both image and target (containing mask)
            # For albumentations, it might look like: transformed = self.transform(image=image, masks=[panoptic_mask])
            # Then you would update image = transformed['image'] and target['panoptic_mask'] = transformed['masks'][0]
            # This example assumes self.transform is a Compose that handles image and target dict
            
            # A more generic way if your self.transform expects (image, target_dict) and returns (image, target_dict)
            transformed_image, transformed_target = self.transform(image, target)
            image = transformed_image
            target = transformed_target 
            # Ensure mask is a tensor after transforms
            if not isinstance(target['panoptic_mask'], torch.Tensor):
                 target['panoptic_mask'] = torch.from_numpy(target['panoptic_mask']).long()
        else:
            # If no transforms, convert image to tensor manually if needed by model, and mask to tensor
            # This depends on what your model expects. For now, let's assume transforms will handle it.
            # If using ToTensorV2 from albumentations in self.transform, image will be tensor.
            # Mask should be converted to tensor.
            target['panoptic_mask'] = torch.from_numpy(target['panoptic_mask']).long()

        return {
            'image': image, # Should be a tensor after transforms
            'point_cloud': torch.from_numpy(point_cloud).float(),
            'camera_intrinsic': torch.from_numpy(camera_intrinsic).float(),
            'lidar_to_camera': torch.from_numpy(lidar_to_camera).float(),
            'panoptic_mask': target['panoptic_mask'], # Should be a tensor
            'sample_id': sample_id,
            'orig_size': target.get('orig_size', torch.tensor([current_image_h, current_image_w]))
        }

class LiDARProjection:
    """Utility class for LiDAR-camera projection and fusion"""
    
    @staticmethod
    def project_lidar_to_image(points, camera_intrinsic, lidar_to_camera):
        """Project LiDAR points to camera image plane"""
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Transform to camera coordinate system
        points_cam = (lidar_to_camera @ points_homo.T).T
        
        # Remove points behind camera
        mask = points_cam[:, 2] > 0
        points_cam = points_cam[mask]
        
        # Project to image plane
        points_2d = (camera_intrinsic @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        return points_2d, mask
    
    @staticmethod
    def create_depth_map(points_2d, depths, image_shape, max_depth=80.0):
        """Create depth map from projected LiDAR points"""
        h, w = image_shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        # Filter points within image bounds
        valid_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h) &
            (depths > 0) & (depths < max_depth)
        )
        
        valid_points = points_2d[valid_mask].astype(int)
        valid_depths = depths[valid_mask]
        
        # Fill depth map (handle multiple points per pixel by taking closest)
        for i, (x, y) in enumerate(valid_points):
            if depth_map[y, x] == 0 or valid_depths[i] < depth_map[y, x]:
                depth_map[y, x] = valid_depths[i]
        
        return depth_map

class PanopticFPN(nn.Module):
    """Panoptic FPN for unified instance and semantic segmentation"""
    
    def __init__(self, num_classes=3, num_stuff_classes=8):
        super().__init__()
        
        # Use Detectron2's Panoptic FPN as base
        from detectron2.modeling import build_model
        from detectron2.config import get_cfg
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
        ))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_stuff_classes
        
        self.model = build_model(cfg)
        self.num_classes = num_classes
        self.num_stuff_classes = num_stuff_classes
    
    def forward(self, images):
        """Forward pass returning both instance and semantic predictions"""
        if self.training:
            return self.model(images)
        else:
            with torch.no_grad():
                predictions = self.model(images)
                return self.postprocess_panoptic(predictions)
    
    def postprocess_panoptic(self, predictions):
        """Convert model outputs to panoptic format by combining instance and semantic predictions"""
        panoptic_results = []
        
        for pred in predictions:
            # Get instance predictions
            instances = pred["instances"]
            if len(instances) == 0:
                # If no instances, just use semantic segmentation
                panoptic_seg = pred["sem_seg"].argmax(dim=0)
                segments_info = []
            else:
                # Get instance masks and scores
                instance_masks = instances.pred_masks
                instance_scores = instances.scores
                instance_classes = instances.pred_classes
                
                # Get semantic segmentation
                sem_seg = pred["sem_seg"]
                
                # Initialize panoptic segmentation with semantic segmentation
                panoptic_seg = sem_seg.argmax(dim=0)
                
                # Create segments_info list
                segments_info = []
                
                # Sort instances by score for NMS
                sorted_indices = torch.argsort(instance_scores, descending=True)
                instance_masks = instance_masks[sorted_indices]
                instance_scores = instance_scores[sorted_indices]
                instance_classes = instance_classes[sorted_indices]
                
                # Process each instance
                for i in range(len(instances)):
                    mask = instance_masks[i]
                    score = instance_scores[i]
                    class_id = instance_classes[i]
                    
                    # Only keep high-confidence instances
                    if score > 0.5:  # Confidence threshold
                        # Convert instance mask to binary
                        binary_mask = mask > 0.5
                        
                        # Check for overlap with existing instances
                        overlap = False
                        for prev_mask in segments_info:
                            if (binary_mask & prev_mask["mask"]).any():
                                overlap = True
                                break
                        
                        if not overlap:
                            # Update panoptic segmentation
                            # Use a unique ID for each instance (class_id * 1000 + instance_id)
                            instance_id = len(segments_info) + 1
                            panoptic_id = int(class_id) * 1000 + instance_id
                            panoptic_seg[binary_mask] = panoptic_id
                            
                            # Add to segments_info
                            segments_info.append({
                                "id": panoptic_id,
                                "category_id": int(class_id),
                                "isthing": True,
                                "score": float(score),
                                "mask": binary_mask  # Store mask for overlap checking
                            })
            
            # Add stuff classes
            for class_id in range(self.num_stuff_classes):
                mask = (panoptic_seg == class_id)
                if mask.any():
                    stuff_id = (class_id + self.num_classes) * 1000
                    panoptic_seg[mask] = stuff_id
                    segments_info.append({
                        "id": stuff_id,
                        "category_id": class_id + self.num_classes,
                        "isthing": False,
                        "mask": mask  # Store mask for overlap checking
                    })
            
            panoptic_results.append({
                "panoptic_seg": panoptic_seg,
                "segments_info": segments_info
            })
        
        return panoptic_results

class MultiModalFusionNetwork(nn.Module):
    """Network for fusing camera and LiDAR features"""
    
    def __init__(self, camera_backbone, lidar_backbone):
        super().__init__()
        
        self.camera_backbone = camera_backbone
        self.lidar_backbone = lidar_backbone
        
        # Fusion modules
        self.early_fusion = EarlyFusion()
        self.late_fusion = LateFusion()
        
        # Panoptic head
        self.panoptic_head = PanopticHead(
            in_channels=512,
            num_thing_classes=3,
            num_stuff_classes=8
        )
    
    def forward(self, camera_input, lidar_input, depth_map):
        # Extract camera features
        camera_features = self.camera_backbone(camera_input)
        
        # Extract LiDAR features  
        lidar_features = self.lidar_backbone(lidar_input)
        
        # Early fusion with depth guidance
        fused_early = self.early_fusion(camera_features, depth_map)
        
        # Late fusion
        fused_features = self.late_fusion(fused_early, lidar_features)
        
        # Panoptic prediction
        panoptic_output = self.panoptic_head(fused_features)
        
        return panoptic_output

class EarlyFusion(nn.Module):
    """Early fusion module combining RGB and depth"""
    
    def __init__(self):
        super().__init__()
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(384 + 128, 512, 3, padding=1),  # RGB features + depth features
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    
    def forward(self, rgb_features, depth_map):
        # Encode depth information
        depth_features = self.depth_encoder(depth_map.unsqueeze(1))
        
        # Resize depth features to match RGB features
        depth_features = F.interpolate(
            depth_features, 
            size=rgb_features.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Concatenate and fuse
        combined = torch.cat([rgb_features, depth_features], dim=1)
        fused = self.fusion_conv(combined)
        
        return fused

class LateFusion(nn.Module):
    """Late fusion module combining camera and LiDAR features"""
    
    def __init__(self):
        super().__init__()
        
        self.attention = CrossModalAttention(512, 256)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512)
        )
    
    def forward(self, camera_features, lidar_features):
        # Apply cross-modal attention
        attended_features = self.attention(camera_features, lidar_features)
        
        # Global average pooling for feature vectors
        cam_global = F.adaptive_avg_pool2d(camera_features, 1).flatten(1)
        lidar_global = F.adaptive_avg_pool2d(lidar_features, 1).flatten(1)
        att_global = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)
        
        # Concatenate and fuse
        combined = torch.cat([cam_global, lidar_global, att_global], dim=1)
        fused_global = self.fusion_mlp(combined)
        
        # Broadcast back to spatial dimensions
        fused_spatial = fused_global.unsqueeze(-1).unsqueeze(-1)
        fused_spatial = fused_spatial.expand(-1, -1, *camera_features.shape[-2:])
        
        return fused_spatial

class CrossModalAttention(nn.Module):
    """Cross-modal attention between camera and LiDAR features"""
    
    def __init__(self, camera_dim, lidar_dim):
        super().__init__()
        
        self.camera_proj = nn.Conv2d(camera_dim, 256, 1)
        self.lidar_proj = nn.Conv2d(lidar_dim, 256, 1)
        
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
    def forward(self, camera_features, lidar_features):
        B, C, H, W = camera_features.shape
        
        # Project features
        cam_proj = self.camera_proj(camera_features)
        lidar_proj = self.lidar_proj(lidar_features)
        
        # Reshape to sequence format
        cam_seq = cam_proj.flatten(2).transpose(1, 2)  # B, HW, C
        lidar_seq = lidar_proj.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Apply attention (camera as query, lidar as key/value)
        attended, _ = self.attention(cam_seq, lidar_seq, lidar_seq)
        
        # Reshape back to spatial format
        attended = attended.transpose(1, 2).reshape(B, 256, H, W)
        
        return attended

class PanopticHead(nn.Module):
    """Head for panoptic segmentation combining instance and semantic segmentation"""
    
    def __init__(self, in_channels, num_thing_classes, num_stuff_classes):
        super().__init__()
        
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        
        # Feature pyramid network
        self.fpn = nn.ModuleList([
            nn.Conv2d(in_channels, 256, 1) for _ in range(4)  # P2, P3, P4, P5
        ])
        
        # Semantic segmentation head
        self.sem_seg_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_stuff_classes, 1)
        )
        
        # Instance segmentation head
        self.instance_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Instance classification
        self.instance_cls = nn.Conv2d(256, num_thing_classes, 1)
        
        # Instance mask prediction
        self.mask_pred = nn.Conv2d(256, 1, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [P2, P3, P4, P5]
        Returns:
            dict containing:
                - sem_seg: Semantic segmentation logits
                - instance_cls: Instance classification logits
                - mask_pred: Instance mask predictions
        """
        # Process features through FPN
        fpn_features = []
        for i, feat in enumerate(features):
            fpn_features.append(self.fpn[i](feat))
        
        # Upsample all features to P2 size
        p2_size = fpn_features[0].shape[-2:]
        for i in range(1, len(fpn_features)):
            fpn_features[i] = F.interpolate(
                fpn_features[i], size=p2_size, mode='bilinear', align_corners=False
            )
        
        # Combine FPN features
        combined_features = sum(fpn_features)
        
        # Semantic segmentation
        sem_seg = self.sem_seg_head(combined_features)
        
        # Instance segmentation
        instance_features = self.instance_head(combined_features)
        instance_cls = self.instance_cls(instance_features)
        mask_pred = self.mask_pred(instance_features)
        
        return {
            'sem_seg': sem_seg,
            'instance_cls': instance_cls,
            'mask_pred': mask_pred
        }

class PanopticLoss(nn.Module):
    """Combined loss for panoptic segmentation"""
    
    def __init__(self, num_thing_classes, num_stuff_classes, weights=None):
        super().__init__()
        
        self.instance_loss = nn.CrossEntropyLoss(
            weight=weights['instance'] if weights else None
        )
        self.semantic_loss = nn.CrossEntropyLoss(
            weight=weights['semantic'] if weights else None
        )
        self.center_loss = nn.MSELoss()
        
        self.lambda_instance = 1.0
        self.lambda_semantic = 1.0  
        self.lambda_center = 0.1
    
    def forward(self, predictions, targets):
        instance_loss = self.instance_loss(
            predictions['instance_cls'], 
            targets['instance_masks']
        )
        
        semantic_loss = self.semantic_loss(
            predictions['sem_seg'],
            targets['semantic_masks']
        )
        
        center_loss = self.center_loss(
            predictions['mask_pred'],
            targets['center_maps']
        )
        
        total_loss = (
            self.lambda_instance * instance_loss +
            self.lambda_semantic * semantic_loss +
            self.lambda_center * center_loss
        )
        
        return {
            'total_loss': total_loss,
            'instance_loss': instance_loss,
            'semantic_loss': semantic_loss,
            'center_loss': center_loss
        }

class PanopticTrainer:
    """Training pipeline for panoptic segmentation with sensor fusion"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MultiModalFusionNetwork(
            camera_backbone=self._build_camera_backbone(),
            lidar_backbone=self._build_lidar_backbone()
        ).to(self.device)
        
        # Loss function
        self.criterion = PanopticLoss(
            num_thing_classes=config['num_thing_classes'],
            num_stuff_classes=config['num_stuff_classes']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
    
    def _build_camera_backbone(self):
        """Build camera feature extractor"""
        import torchvision.models as models
        backbone = models.resnet50(pretrained=True)
        return nn.Sequential(*list(backbone.children())[:-2])
    
    def _build_lidar_backbone(self):
        """Build LiDAR feature extractor (simplified point net)"""
        return PointNetBackbone(in_channels=4, out_channels=256)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            point_clouds = batch['point_cloud'].to(self.device)
            
            # Create depth maps
            depth_maps = []
            for i in range(len(batch['point_cloud'])):
                pc = batch['point_cloud'][i].numpy()
                intrinsic = batch['camera_intrinsic'][i].numpy()
                extrinsic = batch['lidar_to_camera'][i].numpy()
                
                points_2d, mask = LiDARProjection.project_lidar_to_image(
                    pc[:, :3], intrinsic, extrinsic
                )
                depth_map = LiDARProjection.create_depth_map(
                    points_2d, pc[mask, 2], images.shape[-2:]
                )
                depth_maps.append(torch.from_numpy(depth_map))
            
            depth_maps = torch.stack(depth_maps).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images, point_clouds, depth_maps)
            
            # Prepare targets for loss computation
            targets = {
                'instance_masks': batch['panoptic_mask'].to(self.device),
                'semantic_masks': batch['panoptic_mask'].to(self.device),
                'center_maps': self._create_center_maps(batch['panoptic_mask'].to(self.device))
            }
            
            # Calculate loss
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Instance': f'{loss_dict["instance_loss"].item():.4f}',
                'Semantic': f'{loss_dict["semantic_loss"].item():.4f}',
                'Center': f'{loss_dict["center_loss"].item():.4f}'
            })
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_metrics = {
            'instance_loss': 0,
            'semantic_loss': 0,
            'center_loss': 0,
            'iou': 0
        }
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device)
                point_clouds = batch['point_cloud'].to(self.device)
                
                # Create depth maps
                depth_maps = []
                for i in range(len(batch['point_cloud'])):
                    pc = batch['point_cloud'][i].numpy()
                    intrinsic = batch['camera_intrinsic'][i].numpy()
                    extrinsic = batch['lidar_to_camera'][i].numpy()
                    
                    points_2d, mask = LiDARProjection.project_lidar_to_image(
                        pc[:, :3], intrinsic, extrinsic
                    )
                    depth_map = LiDARProjection.create_depth_map(
                        points_2d, pc[mask, 2], images.shape[-2:]
                    )
                    depth_maps.append(torch.from_numpy(depth_map))
                
                depth_maps = torch.stack(depth_maps).to(self.device)
                
                # Forward pass
                predictions = self.model(images, point_clouds, depth_maps)
                
                # Prepare targets
                targets = {
                    'instance_masks': batch['panoptic_mask'].to(self.device),
                    'semantic_masks': batch['panoptic_mask'].to(self.device),
                    'center_maps': self._create_center_maps(batch['panoptic_mask'].to(self.device))
                }
                
                # Calculate loss
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
                
                # Calculate IoU
                iou = self._calculate_iou(predictions, targets)
                
                # Update metrics
                total_loss += loss.item()
                total_metrics['instance_loss'] += loss_dict['instance_loss'].item()
                total_metrics['semantic_loss'] += loss_dict['semantic_loss'].item()
                total_metrics['center_loss'] += loss_dict['center_loss'].item()
                total_metrics['iou'] += iou
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{iou:.4f}'
                })
        
        # Average metrics
        num_batches = len(dataloader)
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_metrics
    
    def _create_center_maps(self, panoptic_masks):
        """Create center maps from panoptic masks"""
        batch_size = panoptic_masks.size(0)
        center_maps = torch.zeros_like(panoptic_masks, dtype=torch.float)
        
        for b in range(batch_size):
            unique_ids = torch.unique(panoptic_masks[b])
            for id in unique_ids:
                if id == 0:  # Skip background
                    continue
                mask = (panoptic_masks[b] == id)
                if mask.any():
                    # Calculate center of mass
                    y_indices, x_indices = torch.where(mask)
                    center_y = y_indices.float().mean()
                    center_x = x_indices.float().mean()
                    
                    # Create Gaussian heatmap
                    y_grid, x_grid = torch.meshgrid(
                        torch.arange(mask.size(0), device=mask.device),
                        torch.arange(mask.size(1), device=mask.device)
                    )
                    sigma = 2.0
                    heatmap = torch.exp(-((y_grid - center_y)**2 + (x_grid - center_x)**2) / (2 * sigma**2))
                    center_maps[b] = torch.maximum(center_maps[b], heatmap)
        
        return center_maps
    
    def _calculate_iou(self, predictions, targets):
        """Calculate IoU between predictions and targets"""
        pred_masks = predictions['instance_cls'].argmax(dim=1)
        target_masks = targets['instance_masks']
        
        intersection = (pred_masks & target_masks).float().sum((1, 2))
        union = (pred_masks | target_masks).float().sum((1, 2))
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config['output_dir'],
            f'panoptic_segmentation_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

class PointNetBackbone(nn.Module):
    """PointNet backbone for processing LiDAR point clouds"""
    
    def __init__(self, in_channels=4, out_channels=256):
        super().__init__()
        
        # Point feature embedding
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # Feature transform network
        self.transform_net = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Global feature network
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, out_channels, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, points):
        """
        Args:
            points: B x N x 4 (x, y, z, intensity)
        Returns:
            B x out_channels feature vector
        """
        # Transpose to B x 4 x N
        points = points.transpose(2, 1)
        
        # Point feature embedding
        x = self.mlp1(points)
        
        # Feature transform
        transform = self.transform_net(x)
        transform = torch.max(transform, 2, keepdim=True)[0]
        transform = transform.view(-1, 1024)
        
        # Global feature
        x = self.mlp2(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Combine features
        x = x + transform
        
        # Output projection
        x = x.unsqueeze(2)  # Add channel dimension
        x = self.mlp3(x)
        x = x.squeeze(2)  # Remove channel dimension
        
        return x

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Panoptic Segmentation Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set up transforms
    train_transform = A.Compose([
        A.RandomResizedCrop(height=1080, width=1920, scale=(0.8, 1.0)),
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
    train_dataset = WaymoMultiModalDataset(
        args.data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = WaymoMultiModalDataset(
        args.data_dir,
        split='val',
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
    model = MultiModalFusionNetwork(
        camera_backbone=PanopticFPN(num_classes=3, num_stuff_classes=8),
        lidar_backbone=PointNetBackbone()
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    
    # Initialize loss function
    criterion = PanopticLoss(
        num_thing_classes=3,
        num_stuff_classes=8
    ).to(device)
    
    # Initialize trainer
    trainer = PanopticTrainer({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'device': device,
        'output_dir': args.output_dir
    })
    
    # Train model
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss['total_loss']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1)
    
    print("Training completed!")

if __name__ == "__main__":
    main()