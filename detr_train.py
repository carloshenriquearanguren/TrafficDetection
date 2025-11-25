"""
#!/usr/bin/env python3
DETR (Detection Transformer) training script for Waymo/KITTI datasets
Following milestone report specifications for transformer-based detection
Fixed transformer implementation compatible with DETR architecture
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision.models import resnet50
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou
from typing import Optional

torch.autograd.set_detect_anomaly(True)

# Import your COCO dataset and transforms (same as Mask R-CNN)
from coco_dataset import COCODataset
from your_transforms import get_transform

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

# DETR Configuration
CONFIG = {
    # Data paths (same as Mask R-CNN)
    'waymo_img': 'waymo_images',
    'waymo_ann': 'waymo_annotations.json',
    'kitti_img': 'kitti_images', 
    'kitti_ann': 'kitti_annotations.json',
    
    # DETR-specific parameters
    'batch_size': 1,  # Increased from 1 for better gradient estimates
    'num_epochs': 100,  # Reduced from 150 with better early stopping
    'lr': 1e-4,  # Base learning rate
    'lr_backbone': 1e-5,  # Lower LR for backbone
    'weight_decay': 1e-4,
    'num_workers': 4,  # Increased for better data loading
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Model parameters
    'num_classes': 3,  # vehicle, pedestrian, cyclist (no background in DETR)
    'hidden_dim': 256,  # Transformer hidden dimension
    'nheads': 8,  # Multi-head attention heads
    'num_encoder_layers': 6,  # Transformer encoder layers
    'num_decoder_layers': 6,  # Transformer decoder layers
    'num_queries': 100,  # Number of object queries
    'pretrained_backbone': True,
    
    # Loss weights (DETR-specific) - FIXED for better balance
    'cls_loss_coef': 1,    # Reduced classification weight
    'bbox_loss_coef': 5,   # Keep same
    'giou_loss_coef': 2,   # Keep GIoU weight
    'eos_coef': 0.1,       # No-object class weight
    
    # Training improvements
    'patience': 15,        # Early stopping patience
    'min_lr': 1e-7,        # Minimum learning rate
    'warmup_epochs': 5,    # Learning rate warmup
    'dropout': 0.1,        # Dropout rate
    
    # Checkpointing
    'save_dir': './detr_checkpoints',
    'save_every': 10,
}


class PositionEmbeddingSine(nn.Module):
    """
    Positional encoding using sine/cosine functions
    Essential for transformer to understand spatial relationships
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with positional embedding support"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    """Custom transformer decoder layer with positional embedding support"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos: Optional[torch.Tensor] = None, 
                query_pos: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    """Custom transformer encoder with positional embedding support"""
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Custom transformer decoder with positional embedding support"""
    
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos, tgt_mask=tgt_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DETRTransformer(nn.Module):
    """DETR-compatible transformer with positional embedding support"""
    
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, query_embed, pos_embed):
        """
        Args:
            src: [H*W, B, d_model] - flattened image features
            query_embed: [num_queries, d_model] - learnable object queries
            pos_embed: [H*W, B, d_model] - positional embeddings
        """
        batch_size = src.shape[1]
        
        # Encoder
        memory = self.encoder(src, pos=pos_embed)
        
        # Decoder
        tgt = torch.zeros_like(query_embed).unsqueeze(1).repeat(1, batch_size, 1)
        query_pos = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_pos)
        
        return hs.transpose(0, 1)  # [B, num_queries, d_model]


class HungarianMatcher(nn.Module):
    """Improved Hungarian matcher for DETR with better error handling"""
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs cannot be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Perform the matching with improved error handling"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Skip if no targets
        if not any(len(v["labels"]) > 0 for v in targets):
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        # Flatten to compute the cost matrices
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # Concatenate targets - only non-empty ones
        valid_targets = [v for v in targets if len(v["labels"]) > 0]
        if not valid_targets:
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        tgt_ids = torch.cat([v["labels"] for v in valid_targets])
        tgt_bbox = torch.cat([v["boxes"] for v in valid_targets])
        
        # Ensure valid target indices
        tgt_ids = torch.clamp(tgt_ids, min=0, max=out_prob.shape[1]-1)
        
        # Classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # L1 cost of boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU cost with error handling
        try:
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            
            # Ensure valid box format
            out_bbox_xyxy = self._ensure_valid_boxes_xyxy(out_bbox_xyxy)
            tgt_bbox_xyxy = self._ensure_valid_boxes_xyxy(tgt_bbox_xyxy)
            
            cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
        except Exception as e:
            print(f"GIoU cost computation failed: {e}")
            cost_giou = torch.zeros_like(cost_bbox)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Handle variable number of targets per image
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        
        start_idx = 0
        for i, size in enumerate(sizes):
            if size == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
            else:
                cost_matrix = C[i, :, start_idx:start_idx + size]
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    indices.append((torch.as_tensor(row_ind, dtype=torch.int64), 
                                  torch.as_tensor(col_ind, dtype=torch.int64)))
                except Exception as e:
                    print(f"Assignment failed for image {i}: {e}")
                    indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                start_idx += size
        
        return indices
    
def _ensure_valid_boxes(self, boxes):
    min_size = 1e-4
    
    # Create a copy to avoid modifying the original
    valid_boxes = boxes.clone()
    
    # Instead of in-place operations, create new values
    valid_boxes[:, 2] = torch.maximum(valid_boxes[:, 0] + min_size, valid_boxes[:, 2])
    valid_boxes[:, 3] = torch.maximum(valid_boxes[:, 1] + min_size, valid_boxes[:, 3])
    
    return valid_boxes

def box_cxcywh_to_xyxy(x):
    """Convert boxes from center format to corner format"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from corner format to center format"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# Simplified linear sum assignment
def linear_sum_assignment(cost_matrix):
    """Simplified Hungarian algorithm"""
    cost_matrix = cost_matrix.numpy()
    try:
        from scipy.optimize import linear_sum_assignment as lsa
        row_ind, col_ind = lsa(cost_matrix)
        return row_ind, col_ind
    except ImportError:
        # Fallback to greedy assignment if scipy not available
        row_ind = []
        col_ind = []
        for i in range(min(cost_matrix.shape)):
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            row_ind.append(min_idx[0])
            col_ind.append(min_idx[1])
            cost_matrix[min_idx[0], :] = np.inf
            cost_matrix[:, min_idx[1]] = np.inf
        return np.array(row_ind), np.array(col_ind)


class DETR(nn.Module):
    """
    DETR: Detection Transformer
    End-to-end object detection with transformers
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, num_queries=100):
        super().__init__()
        
        # Backbone (ResNet-50)
        self.backbone = resnet50(pretrained=CONFIG['pretrained_backbone'])
        # Remove the final classification layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Reduce channel dimension from 2048 to hidden_dim
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Custom DETR Transformer
        self.transformer = DETRTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=CONFIG['dropout']
        )
        
        # Object queries (learnable)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 coordinates (x,y,w,h)
        
        # Matcher
        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        
        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self, images, targets=None):
        # Extract features from backbone
        features = self.backbone(images)  # [B, 2048, H, W]
        
        # Project to transformer dimension
        src = self.input_proj(features)  # [B, hidden_dim, H, W]
        
        # Generate positional embeddings
        pos = self.position_embedding(src)  # [B, hidden_dim, H, W]
        
        # Flatten spatial dimensions for transformer
        batch_size, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [H*W, B, hidden_dim]
        pos = pos.flatten(2).permute(2, 0, 1)  # [H*W, B, hidden_dim]
        
        # Object queries
        query_embed = self.query_embed.weight  # [num_queries, hidden_dim]
        
        # Transformer forward pass
        hs = self.transformer(src, query_embed, pos)  # [B, num_queries, hidden_dim]
        
        # Prediction heads
        outputs_class = self.class_embed(hs)  # [B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [B, num_queries, 4]
        
        out = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }
        
        if self.training and targets is not None:
            # Compute losses during training
            loss_dict = self.compute_loss(out, targets)
            return loss_dict
        else:
            return out

    def compute_loss(self, outputs, targets):
	    """Compute DETR losses with proper tensor handling to avoid boolean ambiguity"""
	    device = outputs['pred_logits'].device

	    # 1) Hungarian matching
	    indices = self.matcher(outputs, targets)

	    # 2) Classification loss
	    target_classes_o = [t["labels"][J] for t, (_, J) in zip(targets, indices)]
	    batch_size = outputs['pred_logits'].shape[0]
	    num_queries = outputs['pred_logits'].shape[1]
	    
	    # Initialize target classes with "no-object" class (background)
	    target_classes = torch.full(
	        (batch_size, num_queries),
	        fill_value=self.num_classes,
	        dtype=torch.int64,
	        device=device
	    )
	    
	    # Set matched predictions to their target classes
	    for i, target_class_indices in enumerate(target_classes_o):
	        if len(target_class_indices) > 0:
	            matched_query_indices = indices[i][0]
	            target_classes[i, matched_query_indices] = target_class_indices

	    # Create class weights (lower weight for no-object class)
	    class_weights = torch.ones(self.num_classes + 1, device=device)
	    class_weights[-1] = CONFIG['eos_coef']
	    
	    # Classification loss
	    loss_ce = F.cross_entropy(
	        outputs['pred_logits'].transpose(1, 2),
	        target_classes,
	        weight=class_weights,
	        reduction='mean'
	    )

	    # 3) Count total number of target boxes
	    num_boxes = sum(len(t["labels"]) for t in targets)
	    
	    # If no boxes, return only classification loss
	    if num_boxes == 0:
	        return {
	            'loss_ce': loss_ce,
	            'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
	            'loss_giou': torch.tensor(0.0, device=device, requires_grad=True)
	        }

	    # 4) Get matched predictions and targets for bbox loss
	    idx = self._get_src_permutation_idx(indices)
	    src_boxes = outputs['pred_boxes'][idx]
	    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

	    # Ensure boxes are valid and in [0,1] range
	    src_boxes = torch.clamp(src_boxes, min=0.0, max=1.0)
	    target_boxes = torch.clamp(target_boxes, min=0.0, max=1.0)

	    # L1 bounding box loss
	    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='sum') / num_boxes

	    # 5) GIoU loss - convert to corner format for GIoU calculation
	    src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
	    target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
	    
	    # Ensure boxes have valid dimensions (width and height > 0)
	    src_boxes_xyxy = self._ensure_valid_boxes(src_boxes_xyxy)
	    target_boxes_xyxy = self._ensure_valid_boxes(target_boxes_xyxy)
	    
	    # Compute GIoU
	    try:
	        giou_matrix = generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
	        # Take diagonal elements (matched pairs) and compute loss
	        loss_giou = (1.0 - torch.diag(giou_matrix)).mean()
	    except Exception as e:
	        print(f"GIoU computation failed: {e}")
	        loss_giou = torch.tensor(0.0, device=device, requires_grad=True)

	    # 6) Scale losses and return
	    loss_dict = {
	        'loss_ce': loss_ce * CONFIG['cls_loss_coef'],
	        'loss_bbox': loss_bbox * CONFIG['bbox_loss_coef'],
	        'loss_giou': loss_giou * CONFIG['giou_loss_coef']
	    }

	    return loss_dict

    def _ensure_valid_boxes(self, boxes):
        """Ensure boxes have valid dimensions to avoid GIoU computation issues"""
        # Clamp coordinates to valid range
        boxes = torch.clamp(boxes, min=0.0, max=1.0)
    
        # Ensure x1 >= x0 and y1 >= y0 (width and height >= 0)
        boxes[:, 2] = torch.maximum(boxes[:, 0], boxes[:, 2])  # x1 >= x0
        boxes[:, 3] = torch.maximum(boxes[:, 1], boxes[:, 3])  # y1 >= y0
    
        # Ensure minimum box size to avoid numerical issues
        min_size = 1e-6
        boxes[:, 2] = torch.maximum(boxes[:, 0] + min_size, boxes[:, 2])
        boxes[:, 3] = torch.maximum(boxes[:, 1] + min_size, boxes[:, 3])
    
        return boxes

    def _get_src_permutation_idx(self, indices):
        """Get indices for source (prediction) side"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class MLP(nn.Module):
    """Multi-layer perceptron for bounding box regression"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def collate_fn(batch):
    """
    Collate function that pads images to the largest size in the batch (with zeros),
    and returns the original sizes (without padding) for box normalization.
    """
    images, targets = zip(*batch)
    
    # Get original sizes (H, W) for each image (after transform, without padding)
    original_sizes = [(img.shape[1], img.shape[2]) for img in images]   # Note: images are (C, H, W)
    
    # Find max height and width
    max_h = max(size[0] for size in original_sizes)
    max_w = max(size[1] for size in original_sizes)
    
    # Pad images to [max_h, max_w]
    padded_images = []
    for img in images:
        c, h, w = img.shape
        # Create a new image tensor for the padded image
        padded_img = torch.zeros((c, max_h, max_w), dtype=img.dtype)
        padded_img[:, :h, :w] = img   # Place the original image in the top-left
        padded_images.append(padded_img)
    
    # Stack the padded images
    padded_images = torch.stack(padded_images, dim=0)   # (batch_size, c, max_h, max_w)
    
    return padded_images, targets, original_sizes


def normalize_boxes_for_detr(targets, image_sizes):
    """Normalize bounding boxes to [0,1] range for DETR with improved handling"""
    for i, target in enumerate(targets):
        if len(target['boxes']) == 0:
            continue
            
        h, w = image_sizes[i]
        boxes = target['boxes'].clone()
        
        # Ensure boxes are in valid format and not negative
        boxes = torch.clamp(boxes, min=0.0)
        
        # Ensure boxes don't exceed image boundaries
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], max=float(w))
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], max=float(h))
        
        # Ensure x2 > x1 and y2 > y1 (valid box geometry)
        boxes[:, 2] = torch.maximum(boxes[:, 0] + 1.0, boxes[:, 2])
        boxes[:, 3] = torch.maximum(boxes[:, 1] + 1.0, boxes[:, 3])
        
        # Convert from absolute to relative coordinates
        if w > 0 and h > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / float(w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / float(h)
        
        # Final clamp to [0, 1] range
        boxes = torch.clamp(boxes, min=0.0, max=1.0)
        
        # Convert to center format (cx, cy, w, h)
        boxes = box_xyxy_to_cxcywh(boxes)
        
        # Ensure width and height are positive and not too small
        boxes[:, 2:] = torch.clamp(boxes[:, 2:], min=1e-6, max=1.0)
        
        target['boxes'] = boxes
    return targets

def get_warmup_lr(epoch, warmup_epochs, base_lr):
    """Linear warmup learning rate"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None):
    """Train DETR for one epoch with improved error handling"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_bbox_loss = 0
    total_giou_loss = 0
    num_batches = 0

    print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
    print("-" * 60)

    for i, (images, targets, image_sizes) in enumerate(data_loader):
        try:
            # Move to device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Filter out empty targets
            valid_targets = []
            valid_image_sizes = []
            valid_images = []
            
            for j, (target, img_size, img) in enumerate(zip(targets, image_sizes, images)):
                if len(target['labels']) > 0:
                    valid_targets.append(target)
                    valid_image_sizes.append(img_size)
                    valid_images.append(img)
            
            if len(valid_targets) == 0:
                print(f"Skipping batch {i} - no valid targets")
                continue
            
            # Stack valid images
            if len(valid_images) < len(images):
                images = torch.stack(valid_images)
                targets = valid_targets
                image_sizes = valid_image_sizes

            # Normalize boxes for DETR
            targets = normalize_boxes_for_detr(targets, image_sizes)

            # Forward pass
            loss_dict = model(images, targets)
            
            # Check for valid losses
            losses = sum(loss_dict.values())
            
            # Validate all loss components
            if not torch.isfinite(losses):
                print(f"Non-finite total loss at batch {i}: {losses.item()}, skipping")
                continue
                
            invalid_loss = False
            for loss_name, loss_val in loss_dict.items():
                if not torch.isfinite(loss_val):
                    print(f"Non-finite {loss_name} at batch {i}: {loss_val.item()}, skipping")
                    invalid_loss = True
                    break
            
            if invalid_loss:
                continue

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            # Accumulate statistics
            total_loss += losses.item()
            total_ce_loss += loss_dict['loss_ce'].item()
            total_bbox_loss += loss_dict['loss_bbox'].item()
            total_giou_loss += loss_dict['loss_giou'].item()
            num_batches += 1

            # Progress logging
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                avg_ce = total_ce_loss / max(num_batches, 1)
                avg_bbox = total_bbox_loss / max(num_batches, 1)
                avg_giou = total_giou_loss / max(num_batches, 1)
                print(f"  Batch {i+1:4d}/{len(data_loader):4d} | "
                      f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, "
                      f"BBox: {avg_bbox:.4f}, GIoU: {avg_giou:.4f})")

        except Exception as e:
            print(f"Error in training batch {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return total_loss / max(num_batches, 1)
    
def evaluate(model, data_loader, device):
    """Evaluate DETR model"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_bbox_loss = 0
    total_giou_loss = 0
    num_batches = 0
    
    print("\nEvaluating...")
    
    with torch.no_grad():
        for i, (images, targets, image_sizes) in enumerate(data_loader):
            # Move to device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Skip batch if no targets
            if not any(len(t['labels']) > 0 for t in targets):
                continue
                
            # Normalize boxes for DETR
            targets = normalize_boxes_for_detr(targets, image_sizes)
            
            try:
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                
                # Check for valid losses
                if torch.isfinite(losses) and all(torch.isfinite(l) for l in loss_dict.values()):
                    total_loss += losses.item()
                    total_ce_loss += loss_dict['loss_ce'].item()
                    total_bbox_loss += loss_dict['loss_bbox'].item()
                    total_giou_loss += loss_dict['loss_giou'].item()
                    num_batches += 1
                    
            except Exception as e:
                print(f"Error in evaluation batch {i}: {e}")
                continue
    
    if num_batches == 0:
        return float('inf'), float('inf'), float('inf'), float('inf')
        
    avg_loss = total_loss / num_batches
    avg_ce = total_ce_loss / num_batches
    avg_bbox = total_bbox_loss / num_batches
    avg_giou = total_giou_loss / num_batches
    
    print(f"Validation - Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, "
          f"BBox: {avg_bbox:.4f}, GIoU: {avg_giou:.4f})")
    
    return avg_loss, avg_ce, avg_bbox, avg_giou


def main():
    """Main training function"""
    print("Starting DETR Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['lr']}")
    print(f"Number of epochs: {CONFIG['num_epochs']}")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Initialize datasets
    print("Loading datasets...")
    
    # Training datasets
    waymo_train = COCODataset(
        CONFIG['waymo_img'], 
        CONFIG['waymo_ann'], 
        get_transform(train=True)
    )
    kitti_train = COCODataset(
        CONFIG['kitti_img'], 
        CONFIG['kitti_ann'], 
        get_transform(train=True)
    )
    
    # Validation datasets (you might want to create separate val splits)
    waymo_val = COCODataset(
        CONFIG['waymo_img'], 
        CONFIG['waymo_ann'], 
        get_transform(train=False)
    )
    kitti_val = COCODataset(
        CONFIG['kitti_img'], 
        CONFIG['kitti_ann'], 
        get_transform(train=False)
    )
    
    # Combine datasets
    train_dataset = ConcatDataset([waymo_train, kitti_train])
    val_dataset = ConcatDataset([waymo_val, kitti_val])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if CONFIG['device'] == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing DETR model...")
    model = DETR(
        num_classes=CONFIG['num_classes'],
        hidden_dim=CONFIG['hidden_dim'],
        nheads=CONFIG['nheads'],
        num_encoder_layers=CONFIG['num_encoder_layers'],
        num_decoder_layers=CONFIG['num_decoder_layers'],
        num_queries=CONFIG['num_queries']
    )
    
    model.to(CONFIG['device'])
    
    # Initialize optimizer with different learning rates for backbone and transformer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": CONFIG['lr_backbone'],
        },
    ]
    
    optimizer = optim.AdamW(
        param_dicts, 
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=CONFIG['patience'] // 2,
        min_lr=CONFIG['min_lr'],
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
        
        # Learning rate warmup
        if epoch < CONFIG['warmup_epochs']:
            warmup_lr = get_warmup_lr(epoch, CONFIG['warmup_epochs'], CONFIG['lr'])
            for param_group in optimizer.param_groups:
                if 'lr_backbone' not in param_group:
                    param_group['lr'] = warmup_lr
        
        # Train one epoch
        try:
            train_loss = train_one_epoch(model, optimizer, train_loader, CONFIG['device'], epoch)
            train_losses.append(train_loss)
            
            # Validation
            val_loss, val_ce, val_bbox, val_giou = evaluate(model, val_loader, CONFIG['device'])
            val_losses.append(val_loss)
            
            # Learning rate scheduling (after warmup)
            if epoch >= CONFIG['warmup_epochs']:
                scheduler.step(val_loss)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                best_model_path = os.path.join(CONFIG['save_dir'], 'best_detr_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': CONFIG
                }, best_model_path)
                
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            # Save checkpoint every N epochs
            if (epoch + 1) % CONFIG['save_every'] == 0:
                checkpoint_path = os.path.join(CONFIG['save_dir'], f'detr_checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': CONFIG
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: epoch_{epoch+1}")
            
            # Early stopping
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"Error in epoch {epoch + 1}: {e}")
            continue
    
    # Save final model
    final_model_path = os.path.join(CONFIG['save_dir'], 'final_detr_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, final_model_path)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {CONFIG['save_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
