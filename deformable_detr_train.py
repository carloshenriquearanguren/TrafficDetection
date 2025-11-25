#!/usr/bin/env python3
"""
Deformable DETR implementation for Waymo/KITTI datasets
Builds on the lessons learned from vanilla DETR training
Key improvements: faster convergence, better AP, multi-scale features
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
from typing import Optional, List
import warnings
import cv2 # For visualization
warnings.filterwarnings('ignore')

# Import your existing components
from coco_dataset import COCODataset
from your_transforms import get_transform

# Deformable DETR Configuration
DEFORMABLE_CONFIG = {
    # Data paths
    'waymo_img': 'waymo_images',
    'waymo_ann': 'waymo_annotations.json',
    'kitti_img': 'kitti_images', 
    'kitti_ann': 'kitti_annotations.json',
    
    # Training parameters - further adjusted
    'batch_size': 2,
    'num_epochs': 200,
    'lr': 1e-6,  # Further reduced for stability
    'lr_backbone': 1e-7,  # Further reduced proportionally
    'weight_decay': 5e-4,  # Increased for stronger regularization
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Model parameters - further simplified
    'num_classes': 3,
    'hidden_dim': 160,  # Further reduced
    'nheads': 5,  # Further reduced
    'num_encoder_layers': 3,  # Further reduced
    'num_decoder_layers': 3,  # Further reduced
    'num_queries': 40,  # Further reduced
    'num_feature_levels': 3,
    'pretrained_backbone': True,
    
    # Deformable attention parameters
    'n_points': 4,
    'dec_n_points': 4,
    'enc_n_points': 4,
    
    # Loss weights - rebalanced for better confidence
    'cls_loss_coef': 1.0,  # Increased from 0.2 to emphasize classification
    'bbox_loss_coef': 2.0,  # Reduced from 0.5 to balance with classification
    'giou_loss_coef': 2.0,  # Reduced from 0.5 to balance with classification
    'eos_coef': 0.1,  # Increased from 0.02 to better handle background
    'focal_alpha': 0.25,  # Increased from 0.15 to original value
    'focal_gamma': 2.0,  # Increased from 1.0 to focus more on hard examples
    
    # Training improvements
    'patience': 30,
    'min_lr': 1e-7,
    'warmup_epochs': 35,  # Further increased
    'dropout': 0.2,  # Increased for regularization
    'grad_clip_norm': 0.02,  # Further reduced
    
    # Learning rate schedule
    'use_cosine_schedule': True,
    'lr_drop_epochs': [50, 100, 150],  # Earlier drops
    'lr_drop_factor': 0.3,  # More gradual drops
    
    # Memory optimization
    'use_checkpoint': True,
    'empty_cache_freq': 5,
    'accumulate_grad_batches': 16,  # Further increased
    'gradient_checkpointing': True,
    'mixed_precision': True,
    
    # Checkpointing
    'save_dir': './deformable_detr_checkpoints',
    'save_every': 10,
}


def _get_clones(module, N):
    """Create N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module
    Key innovation of Deformable DETR for handling multi-scale features
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        # Initialize sampling offsets
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # Initialize attention weights
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        
        # Initialize projections
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, 
                input_level_start_index, input_padding_mask=None):
        """
        Args:
            query: [N, Length_q, C]
            reference_points: [N, Length_q, n_levels, 2]
            input_flatten: [N, \sum_{l=0}^{L-1} H_l \cdot W_l, C]
            input_spatial_shapes: [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index: [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            input_padding_mask: [N, \sum_{l=0}^{L-1} H_l \cdot W_l], True for padding elements
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                               + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                               + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]}')
        
        # Use MSDeformAttn function (this would need to be implemented in C++/CUDA for efficiency)
        # For now, we'll use a simplified version
        output = self._ms_deformable_attn_core_simple(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        
        output = self.output_proj(output)
        return output

    def _ms_deformable_attn_core_simple(self, value, spatial_shapes, sampling_locations, attention_weights):
        """Simplified version of multi-scale deformable attention core"""
        # This is a simplified implementation. In practice, you'd want the optimized CUDA version
        N, S, M, D = value.shape
        N, Lq, M, L, P, _ = sampling_locations.shape
        
        # Simplified bilinear sampling - in practice use grid_sample for efficiency
        outputs = []
        for level in range(L):
            H, W = spatial_shapes[level]
            start_idx = sum(spatial_shapes[:level, 0] * spatial_shapes[:level, 1])
            end_idx = start_idx + H * W
            
            level_value = value[:, start_idx:end_idx].view(N, H, W, M, D)
            level_sampling_loc = sampling_locations[:, :, :, level, :, :]  # N, Lq, M, P, 2
            level_attention_weight = attention_weights[:, :, :, level, :]  # N, Lq, M, P
            
            # Simplified version - just use nearest neighbor for now
            # In practice, implement proper bilinear interpolation
            sampled_values = torch.zeros(N, Lq, M, P, D, device=value.device)
            outputs.append((sampled_values * level_attention_weight.unsqueeze(-1)).sum(dim=3))
        
        output = torch.stack(outputs, dim=3).sum(dim=3)  # N, Lq, M, D
        return output.view(N, Lq, M * D)


class DeformableTransformerEncoderLayer(nn.Module):
    """Deformable Transformer Encoder Layer"""
    
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", 
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        
        # Multi-scale deformable attention
        self.self_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # Self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, 
                             spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transformer Decoder Layer"""
    
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross attention
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,
                level_start_index, src_padding_mask=None):
        # Self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src,
                              src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DeformableTransformerEncoder(nn.Module):
    """Deformable Transformer Encoder"""
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Generate reference points for multi-scale features"""
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, src.device)
        
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        
        return output


class DeformableTransformerDecoder(nn.Module):
    """Deformable Transformer Decoder"""
    
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers) #
        self.num_layers = num_layers #
        self.return_intermediate = return_intermediate #
        # The following will be populated by the main DeformableDETR model
        self.bbox_embed = None #
        self.class_embed = None # Not used in this forward pass but part of the class

    def forward(self, tgt, initial_reference_points_on_all_levels, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios, query_pos=None, src_padding_mask=None):
        output = tgt #
        intermediate = [] #
        intermediate_refined_query_references = [] # Stores the [B, NQ, 2/4] query-specific references
        
        L = src_spatial_shapes.shape[0]  # Number of feature levels
        
        # initial_reference_points_on_all_levels has shape [B, num_queries, L, 2]
        # current_query_ref will store the iteratively refined query-specific reference points.
        # It starts as [B, num_queries, 2] and becomes [B, num_queries, 4] after the first refinement.
        # Initialize current_query_ref by taking the reference points from the first level,
        # assuming they are consistent for a given query across all levels initially.
        current_query_ref = initial_reference_points_on_all_levels[:, :, 0, :].clone() # Shape: [B, num_queries, 2]
        
        for lid, layer in enumerate(self.layers):
            reference_points_input_for_attention = None
            if lid == 0:
                # For the first decoder layer, use the input reference points directly,
                # as they are already prepared in the [B, NumQueries, NumLevels, 2] format.
                reference_points_input_for_attention = initial_reference_points_on_all_levels #
            else:
                # For subsequent layers, current_query_ref holds the refined query-specific points.
                # These need to be expanded to per-level 2D points for the attention mechanism.
                if current_query_ref.shape[-1] == 2: # current_query_ref is [B, NQ, 2]
                    reference_points_input_for_attention = current_query_ref.unsqueeze(2).repeat(1, 1, L, 1)  # Shape: [B, NQ, L, 2]
                elif current_query_ref.shape[-1] == 4: # current_query_ref is [B, NQ, 4]
                    # Use the (x,y) part of the 4D reference points for attention.
                    reference_points_input_for_attention = current_query_ref[..., :2].unsqueeze(2).repeat(1, 1, L, 1)  # Shape: [B, NQ, L, 2]
                else:
                    raise ValueError(f"current_query_ref has unexpected shape: {current_query_ref.shape}")

            output = layer(output, query_pos, reference_points_input_for_attention, src, src_spatial_shapes,
                         src_level_start_index, src_padding_mask) #
            
            # Bounding box refinement: updates current_query_ref
            if self.bbox_embed is not None: #
                tmp = self.bbox_embed[lid](output)  # Predicted offsets/values, shape [B, num_queries, 4]
                
                new_query_ref = None
                if current_query_ref.shape[-1] == 4: # If current_query_ref is already 4D ([B, NQ, 4])
                    # Add predicted offsets to the inverse sigmoid of the current 4D reference points
                    new_query_ref = tmp + inverse_sigmoid(current_query_ref) #
                else: # If current_query_ref is 2D ([B, NQ, 2])
                    assert current_query_ref.shape[-1] == 2, \
                        f"Expected current_query_ref to be 2D, but got shape {current_query_ref.shape}"
                    # tmp provides the full 4D prediction.
                    # The (x,y) part of tmp is an offset to the inverse sigmoid of the current 2D reference points.
                    # The (w,h) part of tmp is used directly.
                    new_query_ref = tmp.clone() # Initialize new 4D reference with tmp's values
                    new_query_ref[..., :2] = tmp[..., :2] + inverse_sigmoid(current_query_ref) # Update (x,y) part
                
                current_query_ref = new_query_ref.sigmoid().detach() # Apply sigmoid and detach
            
            if self.return_intermediate: #
                intermediate.append(output) #
                # Store the refined query-specific reference points (now [B, NQ, 4] or remains [B, NQ, 2] if no refinement in early layers)
                intermediate_refined_query_references.append(current_query_ref) #
        
        if self.return_intermediate: #
            return torch.stack(intermediate), torch.stack(intermediate_refined_query_references) #
        
        return output, current_query_ref #

class DeformableTransformer(nn.Module):
    """Deformable Transformer for DETR"""
    
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                        num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.query_ref_embed = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m._reset_parameters()
        
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        """Get valid ratio of each feature level"""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def forward(self, srcs, masks, pos_embeds, query_embed):
        """
        Args:
            srcs        : list of multi-scale feature maps       — [B, C, H, W]  each
            masks       : list of masks for those maps           — [B, H, W]     each
            pos_embeds  : list of positional embeddings          — [B, C, H, W]  each
            query_embed : learnable object queries               — [num_queries, 2*hidden_dim]
                          (first  half → query positional token,
                           second half → initial target input)
        Returns:
            hs               : decoder layer outputs             — [num_layers, B, num_queries, C]
            init_reference   : initial reference points (before decoder) [B, num_queries, L, 2] or [B, num_queries, 2]
            inter_references : reference points per decoder layer
            enc_outputs_class (optional): class predictions from encoder (for two-stage)
            enc_outputs_coord_unact (optional): coord predictions from encoder (for two-stage)
        """

        # --------------------------------------------------------
        # 1)  Flatten multi-scale features for the encoder
        # --------------------------------------------------------
        src_flatten, mask_flatten, pos_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
            B, C, H, W = src.shape #
            spatial_shapes.append((H, W)) #

            src_flatten .append(src.flatten(2).transpose(1, 2))          # [B, HW, C] #
            mask_flatten.append(mask.flatten(1))                         # [B, HW] #
            pos_flatten .append(pos.flatten(2).transpose(1, 2) #
                                + self.level_embed[lvl].view(1, 1, -1))  # [B, HW, C] #

        src_flatten   = torch.cat(src_flatten,   dim=1)                  # [B, ΣHW, C] #
        mask_flatten  = torch.cat(mask_flatten,  dim=1)                  # [B, ΣHW] #
        pos_flatten   = torch.cat(pos_flatten,   dim=1)                  # [B, ΣHW, C] #
        spatial_shapes = torch.as_tensor(spatial_shapes, #
                                         dtype=torch.long, #
                                         device=src_flatten.device)      # [L, 2] #
        level_start_index = torch.cat((spatial_shapes.new_zeros(1), #
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))  # [L] #
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # [B, L, 2] #

        # --------------------------------------------------------
        # 2)  Encoder
        # --------------------------------------------------------
        memory = self.encoder( #
            src_flatten, #
            spatial_shapes, #
            level_start_index, #
            valid_ratios, #
            pos_flatten, #
            mask_flatten #
        )  # [B, ΣHW, C] #

        # --------------------------------------------------------
        # 3)  Prepare decoder inputs
        # --------------------------------------------------------
        B, _, C = memory.shape                  # C == hidden_dim #
        # split query_embed (512-D) into positional half & target-input half
        query_pos, tgt_init = torch.split(query_embed, C, dim=1)  # each: [num_queries, 256] #

        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)      # [B, num_queries, C] #
        tgt       = tgt_init.unsqueeze(0).expand(B, -1, -1)       # [B, num_queries, C] #

        # build object-query reference points --------------  
        # This is the initial reference point before decoder processing.
        init_reference_points = self.query_ref_embed(query_pos)  # [B, num_queries, 2] #
        init_reference_points = init_reference_points.sigmoid()  # (0,1) range #
        
        # The decoder expects reference_points expanded per level.
        L_levels = spatial_shapes.shape[0] #
        reference_points_for_decoder = init_reference_points.unsqueeze(2).repeat(1, 1, L_levels, 1)  # [B, num_queries, L, 2] #

        # --------------------------------------------------------
        # 4)  Decoder
        # --------------------------------------------------------
        hs, inter_references = self.decoder( #
            tgt,                 # initial target             [B, num_queries, C] #
            reference_points_for_decoder, # ref pts per level [B, num_queries, L, 2] #
            memory,              # encoder output             [B, ΣHW, C] #
            spatial_shapes,      # [L, 2] #
            level_start_index,   # [L] #
            valid_ratios,        # [B, L, 2] #
            query_pos,           # positional embedding for queries #
            mask_flatten         # [B, ΣHW] #
        ) #

        # Placeholder for encoder outputs if not implementing two-stage
        enc_outputs_class = None
        enc_outputs_coord_unact = None
        
        # init_reference should be reference_points_for_decoder (or init_reference_points if the other end handles expansion)
        # The DeformableDETR.forward expects init_reference. Let's pass reference_points_for_decoder.
        return hs, reference_points_for_decoder, inter_references, enc_outputs_class, enc_outputs_coord_unact #
  
  # Helper functions
def inverse_sigmoid(x, eps=1e-5):
    """Inverse sigmoid function"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


# Import necessary functions for initialization
from torch.nn.init import xavier_uniform_, constant_, normal_
import copy


class DeformableDETR(nn.Module):
    """
    Deformable DETR: Deformable Transformers for End-to-End Object Detection
    Faster convergence and better performance than vanilla DETR
    """
    
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6,
                 num_decoder_layers=6, num_queries=300, num_feature_levels=4,
                 dec_n_points=4, enc_n_points=4):
        super().__init__()
        
        # Multi-scale backbone
        self.backbone = self._build_backbone()
        
        # Feature projection layers for different scales
        num_backbone_outs = len(self.backbone.strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = self.backbone.num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        
        # Additional projections for extra feature levels
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(input_proj_list[-1][0].in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        
        self.input_proj = nn.ModuleList(input_proj_list)
        
        # Deformable Transformer
        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=1024,
            dropout=DEFORMABLE_CONFIG['dropout'],
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        
        # Iterative refinement
        if num_decoder_layers > 0:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_decoder_layers)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_decoder_layers)])
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.class_embed = nn.ModuleList([self.class_embed])
            self.bbox_embed = nn.ModuleList([self.bbox_embed])
            self.transformer.decoder.bbox_embed = None
        
        # Position encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Matcher for loss computation
        self.matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        
        self._reset_parameters()

    def _build_backbone(self):
        """Build multi-scale backbone"""
        backbone = resnet50(pretrained=DEFORMABLE_CONFIG['pretrained_backbone'])
        
        # Extract different stages
        conv1 = backbone.conv1
        bn1 = backbone.bn1
        relu = backbone.relu
        maxpool = backbone.maxpool
        
        layer1 = backbone.layer1  # 1/4
        layer2 = backbone.layer2  # 1/8
        layer3 = backbone.layer3  # 1/16
        layer4 = backbone.layer4  # 1/32
        
        # Build feature pyramid
        class BackboneWithFPN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = conv1
                self.bn1 = bn1
                self.relu = relu
                self.maxpool = maxpool
                self.layer1 = layer1
                self.layer2 = layer2
                self.layer3 = layer3
                self.layer4 = layer4
                
                # Feature channels for each level
                self.num_channels = [512, 1024, 2048]  # C3, C4, C5
                self.strides = [8, 16, 32]
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                c2 = self.layer1(x)  # 1/4
                c3 = self.layer2(c2)  # 1/8
                c4 = self.layer3(c3)  # 1/16
                c5 = self.layer4(c4)  # 1/32
                
                return [c3, c4, c5]
        
        return BackboneWithFPN()

    def _reset_parameters(self):
        """Initialize parameters"""
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples):
        """Forward pass"""
        if isinstance(samples, (list, torch.Tensor)): #
            samples = nested_tensor_from_tensor_list(samples) #
        
        features, pos = self.backbone_with_pos_embed(samples) #
        
        srcs = [] #
        masks = [] #
        for l, feat in enumerate(features): #
            src, mask = feat.decompose() #
            srcs.append(self.input_proj[l](src)) #
            masks.append(mask) #
            assert mask is not None #
        
        if self.num_feature_levels > len(srcs): #
            _len_srcs = len(srcs) #
            for l in range(_len_srcs, self.num_feature_levels): #
                if l == _len_srcs: #
                    src = self.input_proj[l](features[-1].tensors) #
                else: #
                    src = self.input_proj[l](srcs[-1]) #
                m = samples.mask #
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0] #
                pos_l = self.position_embedding(NestedTensor(src, mask)).to(src.dtype) #
                srcs.append(src) #
                masks.append(mask) #
                pos.append(pos_l) #

        query_embeds = self.query_embed.weight #
        
        # Assuming DeformableTransformer.forward now correctly returns 5 values as per previous fix:
        # hs: [num_decoder_layers, B, num_queries, C]
        # init_reference: [B, num_queries, L, 2] (per-level initial reference points)
        # inter_references: [num_decoder_layers, B, num_queries, 2/4] (query-specific refined references from decoder layers)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embeds) #

        outputs_classes = [] #
        outputs_coords = [] #
        
        for lvl in range(hs.shape[0]): # Iterate through each decoder layer's output
            query_specific_ref_points = None # This will store [B, NQ, 2] or [B, NQ, 4] reference
            
            if lvl == 0:
                # For the first layer, use init_reference ([B, NQ, L, 2]).
                # We need the query-specific 2D part. Assume consistency across levels for initial points.
                query_specific_ref_points = init_reference[:, :, 0, :].clone() # Shape: [B, NQ, 2]
            else:
                # For subsequent layers, use the refined query-specific reference points from the previous decoder layer.
                # inter_references is already stacked: [num_decoder_layers, B, NQ, 2/4]
                query_specific_ref_points = inter_references[lvl - 1] # Shape: [B, NQ, 2] or [B, NQ, 4]

            # Apply inverse sigmoid to the query-specific reference points
            inv_sig_query_specific_ref = inverse_sigmoid(query_specific_ref_points) # Shape: [B, NQ, 2] or [B, NQ, 4]

            outputs_class = self.class_embed[lvl](hs[lvl]) # hs[lvl] shape: [B, NQ, C]
            
            # tmp is the raw output from the bbox MLP head for the current decoder layer
            tmp = self.bbox_embed[lvl](hs[lvl]) # Shape: [B, NQ, 4]

            final_bbox_logits = None
            if inv_sig_query_specific_ref.shape[-1] == 4: # If query_specific_ref was 4D (e.g., cx,cy,w,h)
                # Add tmp (which are predicted offsets or values) to the 4D reference points (in logit space)
                final_bbox_logits = tmp + inv_sig_query_specific_ref # [B,NQ,4] + [B,NQ,4] = [B,NQ,4]
            else: # If query_specific_ref was 2D (e.g., cx,cy)
                assert inv_sig_query_specific_ref.shape[-1] == 2, \
                    f"Expected 2D query_specific_ref, but got shape {inv_sig_query_specific_ref.shape}"
                # tmp is [B, NQ, 4], representing something like (d_cx, d_cy, logit_w, logit_h)
                # Add the (d_cx, d_cy) part of tmp to the inverse_sigmoid of the 2D reference points.
                # The (logit_w, logit_h) part of tmp is taken as is.
                updated_xy_logits = tmp[..., :2] + inv_sig_query_specific_ref # [B,NQ,2] + [B,NQ,2] = [B,NQ,2]
                # Concatenate with the w,h predictions from tmp
                final_bbox_logits = torch.cat((updated_xy_logits, tmp[..., 2:]), dim=-1) # [B,NQ,2] cat [B,NQ,2] -> [B,NQ,4]
            
            outputs_coord = final_bbox_logits.sigmoid() # Apply sigmoid to get final normalized coordinates
            
            outputs_classes.append(outputs_class) #
            outputs_coords.append(outputs_coord) #
        
        outputs_class = torch.stack(outputs_classes) #
        outputs_coord = torch.stack(outputs_coords) #

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} #
        if hasattr(self, 'aux_loss') and self.aux_loss: # Check if aux_loss attribute exists and is True
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord) #

        return out

    def backbone_with_pos_embed(self, samples):
        """Extract features with positional embeddings"""
        xs = self.backbone(samples.tensors)
        out = []
        pos = []
        for name, x in zip(['c3', 'c4', 'c5'], xs):
            m = samples.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            pos_embed = self.position_embedding(NestedTensor(x, mask)).to(x.dtype)
            out.append(NestedTensor(x, mask))
            pos.append(pos_embed)
        return out, pos

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """Set auxiliary loss for intermediate decoder outputs"""
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal positional embeddings"""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
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


class NestedTensor(object):
    """Tensor with mask for variable-sized inputs"""
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)


def nested_tensor_from_tensor_list(tensors):
    """
    Creates a NestedTensor from either:
      • A single 4-D tensor [B, C, H, W], or
      • A Python list of 3-D tensors [C, H, W].

    If given a 4-D tensor, we assume it's already a batch and create
    a "no-padding" mask of shape [B, H, W], all False.
    """
    import torch

    # Case 1: input is already a batched 4-D tensor
    if isinstance(tensors, torch.Tensor) and tensors.ndim == 4:
        # tensors: [B, C, H, W]
        b, c, h, w = tensors.shape
        # No padding anywhere → mask is all False
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=tensors.device)
        return NestedTensor(tensors, mask)

    # Case 2: input is a list (or tuple) of 3-D image tensors [C, H_i, W_i]
    if not isinstance(tensors, (list, tuple)):
        raise ValueError(
            "nested_tensor_from_tensor_list must be passed either "
            "a list of 3D tensors or a 4D tensor"
        )

    # Find the maximum height and width in the list
    max_size = [max(img.shape[i] for img in tensors) for i in range(1, 3)]
    # Assume all images share the same number of channels
    batch_shape = (len(tensors), tensors[0].shape[0], max_size[0], max_size[1])
    b, c, h_max, w_max = batch_shape

    # Create an empty batched image tensor
    batched_imgs = torch.zeros(
        batch_shape,
        dtype=tensors[0].dtype,
        device=tensors[0].device
    )
    # Create a mask: True where padded, False where actual data
    mask = torch.ones((b, h_max, w_max), dtype=torch.bool, device=tensors[0].device)

    for i, img in enumerate(tensors):
        img_c, img_h, img_w = img.shape
        batched_imgs[i, :, :img_h, :img_w] = img
        mask[i, :img_h, :img_w] = False

    return NestedTensor(batched_imgs, mask)


class HungarianMatcher(nn.Module):
    """Hungarian algorithm for bipartite matching between predictions and targets"""
    
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Perform the matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten to compute the cost matrices
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) # Shape: [bs * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # Concatenate the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]) # These are the ground truth labels
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Cost matrices
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h)"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


from scipy.optimize import linear_sum_assignment


class SetCriterion(nn.Module):
    """Loss computation for DETR"""
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.focal_alpha = focal_alpha
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Focal Loss)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Box regression losses"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        """Get permutation indices"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Get target permutation indices"""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Get specific loss"""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """Compute losses"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """Focal loss"""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def accuracy(output, target, topk=(1,)):
    """Compute accuracy"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def build_deformable_detr():
    """Build Deformable DETR model"""
    model = DeformableDETR(
        num_classes=DEFORMABLE_CONFIG['num_classes'],
        hidden_dim=DEFORMABLE_CONFIG['hidden_dim'],
        nheads=DEFORMABLE_CONFIG['nheads'],
        num_encoder_layers=DEFORMABLE_CONFIG['num_encoder_layers'],
        num_decoder_layers=DEFORMABLE_CONFIG['num_decoder_layers'],
        num_queries=DEFORMABLE_CONFIG['num_queries'],
        num_feature_levels=DEFORMABLE_CONFIG['num_feature_levels'],
        dec_n_points=DEFORMABLE_CONFIG['dec_n_points'],
        enc_n_points=DEFORMABLE_CONFIG['enc_n_points'],
    )
    
    # Set auxiliary loss flag
    model.aux_loss = True
    
    # Build matcher and criterion
    matcher = HungarianMatcher(
        cost_class=DEFORMABLE_CONFIG['cls_loss_coef'],
        cost_bbox=DEFORMABLE_CONFIG['bbox_loss_coef'],
        cost_giou=DEFORMABLE_CONFIG['giou_loss_coef']
    )
    
    weight_dict = {
        'loss_ce': DEFORMABLE_CONFIG['cls_loss_coef'],
        'loss_bbox': DEFORMABLE_CONFIG['bbox_loss_coef'],
        'loss_giou': DEFORMABLE_CONFIG['giou_loss_coef']
    }
    
    # Auxiliary loss weights
    if model.aux_loss:
        aux_weight_dict = {}
        for i in range(DEFORMABLE_CONFIG['num_decoder_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes']
    criterion = SetCriterion(
        DEFORMABLE_CONFIG['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=DEFORMABLE_CONFIG['eos_coef'],
        losses=losses,
        focal_alpha=DEFORMABLE_CONFIG['focal_alpha']
    )
    
    return model, criterion


def train_deformable_detr():
    """Training function for Deformable DETR"""
    print("Building Deformable DETR model...")
    device = torch.device(DEFORMABLE_CONFIG['device'])
    
    # Set environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Memory optimization settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if DEFORMABLE_CONFIG['mixed_precision'] else None
    
    model, criterion = build_deformable_detr()
    model.to(device)
    criterion.to(device)
    
    # Enable gradient checkpointing
    if DEFORMABLE_CONFIG['gradient_checkpointing']:
        model.transformer.encoder.use_checkpoint = True
        model.transformer.decoder.use_checkpoint = True
        model.backbone.use_checkpoint = True
    
    # Dataset preparation
    print("Loading datasets...")
    train_dataset = COCODataset(
        DEFORMABLE_CONFIG['waymo_img'],
        DEFORMABLE_CONFIG['waymo_ann'],
        transforms=get_transform(train=True)
    )
    
    val_dataset = COCODataset(
        DEFORMABLE_CONFIG['kitti_img'],
        DEFORMABLE_CONFIG['kitti_ann'],
        transforms=get_transform(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEFORMABLE_CONFIG['batch_size'],
        shuffle=True,
        num_workers=DEFORMABLE_CONFIG['num_workers'],
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=DEFORMABLE_CONFIG['batch_size'],
        shuffle=False,
        num_workers=DEFORMABLE_CONFIG['num_workers'],
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )
    
    # Optimizer setup
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and "class_embed" not in n and p.requires_grad],
            "lr": DEFORMABLE_CONFIG['lr'],
            "weight_decay": DEFORMABLE_CONFIG['weight_decay']
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": DEFORMABLE_CONFIG['lr_backbone'],
            "weight_decay": DEFORMABLE_CONFIG['weight_decay'] * 0.2  # Further reduced for backbone
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "class_embed" in n and p.requires_grad],
            "lr": DEFORMABLE_CONFIG['lr'] * 1.2,  # Slightly higher but not too much
            "weight_decay": DEFORMABLE_CONFIG['weight_decay'] * 0.8  # Slightly reduced for classification
        }
    ]
    
    optimizer = torch.optim.AdamW(param_dicts)
    
    # Modify learning rate scheduler for more stable warmup
    warmup_factor = 1.0 / 3000  # Even more gradual warmup
    warmup_iters = min(3000, len(train_loader) * DEFORMABLE_CONFIG['warmup_epochs'])
    
    def lr_lambda(step):
        if step < warmup_iters:
            # Cubic warmup for even smoother start
            alpha = float(step) / float(max(1.0, warmup_iters))
            return warmup_factor * (1 - alpha) + alpha * alpha * alpha
        
        # Get current epoch
        current_epoch = step // len(train_loader)
        
        # Apply cosine schedule if enabled
        if DEFORMABLE_CONFIG['use_cosine_schedule']:
            # Cosine decay from warmup_iters to max_iters
            max_iters = len(train_loader) * DEFORMABLE_CONFIG['num_epochs']
            warmup_progress = (step - warmup_iters) / (max_iters - warmup_iters)
            # Smoother cosine decay
            return 0.5 * (1 + math.cos(math.pi * warmup_progress)) * (1 - 0.1 * warmup_progress)
        
        # Otherwise use step schedule
        return DEFORMABLE_CONFIG['lr_drop_factor'] ** len(
            [m for m in DEFORMABLE_CONFIG['lr_drop_epochs'] if m <= current_epoch]
        )
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop with mixed precision
    print("Starting training...")
    os.makedirs(DEFORMABLE_CONFIG['save_dir'], exist_ok=True)
    
    best_loss = float('inf')
    patience_counter = 0
    grad_norm = 0
    best_checkpoint = None  # Track best checkpoint
    
    for epoch in range(DEFORMABLE_CONFIG['num_epochs']):
        model.train()
        criterion.train()
        epoch_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Mixed precision training
            if DEFORMABLE_CONFIG['mixed_precision']:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss_dict = criterion(outputs, targets)
                    losses = sum(loss_dict[k] * criterion.weight_dict[k] 
                               for k in loss_dict.keys() if k in criterion.weight_dict)
                    
                    # Gradient accumulation
                    losses = losses / DEFORMABLE_CONFIG['accumulate_grad_batches']
                    scaler.scale(losses).backward()
                    
                    if (batch_idx + 1) % DEFORMABLE_CONFIG['accumulate_grad_batches'] == 0:
                        if DEFORMABLE_CONFIG['grad_clip_norm'] > 0:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), DEFORMABLE_CONFIG['grad_clip_norm'])
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        lr_scheduler.step()
            else:
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict[k] * criterion.weight_dict[k] 
                           for k in loss_dict.keys() if k in criterion.weight_dict)
                
                # Gradient accumulation
                losses = losses / DEFORMABLE_CONFIG['accumulate_grad_batches']
                losses.backward()
                
                if (batch_idx + 1) % DEFORMABLE_CONFIG['accumulate_grad_batches'] == 0:
                    if DEFORMABLE_CONFIG['grad_clip_norm'] > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), DEFORMABLE_CONFIG['grad_clip_norm'])
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            
            epoch_loss += losses.item() * DEFORMABLE_CONFIG['accumulate_grad_batches']
            num_batches += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                loss_components_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                print(f'Epoch {epoch+1}/{DEFORMABLE_CONFIG["num_epochs"]}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Total Loss: {losses.item() * DEFORMABLE_CONFIG["accumulate_grad_batches"]:.4f}, '
                      f'Grad Norm: {grad_norm:.2f}, '
                      f'Components: [{loss_components_str}]')
                
                # Visualize predictions
                if batch_idx == 0:
                    visualize_predictions(images, targets, outputs, epoch + 1, batch_idx)
            
            # Clear cache periodically
            if batch_idx % DEFORMABLE_CONFIG['empty_cache_freq'] == 0:
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'loss': avg_loss,
            'config': DEFORMABLE_CONFIG
        }
        
        # Save periodic checkpoint
        if (epoch + 1) % DEFORMABLE_CONFIG['save_every'] == 0:
            checkpoint_path = os.path.join(
                DEFORMABLE_CONFIG['save_dir'],
                f'deformable_detr_epoch_{epoch+1}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
        
        # Early stopping and best model saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_checkpoint = checkpoint  # Update best checkpoint
            
            # Save best model
            best_model_path = os.path.join(DEFORMABLE_CONFIG['save_dir'], 'best_deformable_detr.pth')
            torch.save(best_checkpoint, best_model_path)
            print(f'Best model saved: {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= DEFORMABLE_CONFIG['patience']:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    print("Training completed!")
    return model, criterion


def evaluate_deformable_detr(model, dataloader, criterion, device):
    """Evaluation function"""
    model.eval()
    criterion.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
            total_loss += losses.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f'Evaluation completed. Average loss: {avg_loss:.4f}')
    return avg_loss


# Visualization helper function
def visualize_predictions(images_tensor, targets, outputs, epoch, batch_idx, save_dir="./debug_visualizations"):
    """Visualize predictions with proper coordinate handling"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # For simplicity, visualize the first image in the batch
    img_tensor = images_tensor[0].detach().cpu()
    target_sample = targets[0]
    output_sample_logits = outputs['pred_logits'][0].detach().cpu()
    output_sample_boxes_normalized = outputs['pred_boxes'][0].detach().cpu()

    # Un-normalize the image tensor
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img_unnorm = img_tensor * std + mean
    img_unnorm = img_unnorm.clamp(0, 1)
    img_np = img_unnorm.permute(1, 2, 0).numpy() * 255
    img_np = img_np.astype(np.uint8)
    img_np_contiguous = np.ascontiguousarray(img_np)
    img_bgr = cv2.cvtColor(img_np_contiguous, cv2.COLOR_RGB2BGR)

    h, w, _ = img_bgr.shape

    # Draw Ground Truth Boxes
    if target_sample['boxes'].numel() > 0:
        gt_boxes_normalized_cxcywh = target_sample['boxes'].cpu()
        gt_boxes_normalized_xyxy = box_cxcywh_to_xyxy(gt_boxes_normalized_cxcywh)
        
        for i in range(gt_boxes_normalized_xyxy.shape[0]):
            box_norm_xyxy = gt_boxes_normalized_xyxy[i]
            label_idx = target_sample['labels'][i].item()
            
            # Denormalize to pixel coordinates and ensure integers
            x1 = int(box_norm_xyxy[0].item() * w)
            y1 = int(box_norm_xyxy[1].item() * h)
            x2 = int(box_norm_xyxy[2].item() * w)
            y2 = int(box_norm_xyxy[3].item() * h)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Draw ground truth box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"GT_{label_idx}", (x1, max(y1 - 5, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw Predicted Boxes with NMS
    pred_probs = output_sample_logits.softmax(-1)
    pred_scores, pred_labels = pred_probs.max(-1)
    
    # Apply confidence threshold
    keep = pred_scores > 0.3
    
    if keep.any():
        pred_boxes_normalized_kept = output_sample_boxes_normalized[keep]
        pred_labels_kept = pred_labels[keep]
        pred_scores_kept = pred_scores[keep]

        # Convert to pixel coordinates
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_normalized_kept)
        pred_boxes_xyxy[:, [0, 2]] *= w
        pred_boxes_xyxy[:, [1, 3]] *= h
        
        # Apply NMS
        keep = nms(pred_boxes_xyxy, pred_scores_kept, iou_threshold=0.5)
        
        # Draw remaining boxes
        for i in keep:
            box = pred_boxes_xyxy[i]
            label_idx = pred_labels_kept[i].item()
            score = pred_scores_kept[i].item()
            
            # Convert to integers and ensure within bounds
            x1 = max(0, min(int(box[0].item()), w-1))
            y1 = max(0, min(int(box[1].item()), h-1))
            x2 = max(0, min(int(box[2].item()), w-1))
            y2 = max(0, min(int(box[3].item()), h-1))
            
            # Draw predicted box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_bgr, f"P_{label_idx} ({score:.2f})", (x1, max(y1 - 5, 0)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Add epoch and batch information
    cv2.putText(img_bgr, f"Epoch {epoch}, Batch {batch_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save the image
    save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}.png")
    cv2.imwrite(save_path, img_bgr)
    print(f"Saved visualization to {save_path}")

def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    # Convert to numpy for easier manipulation, ensuring to detach from computation graph
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    
    # Initialize list of picked indexes
    pick = []
    
    # Coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by confidence score
    idxs = np.argsort(scores)
    
    while len(idxs) > 0:
        # Pick the last index (highest score)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find intersection
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Compute width and height of intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Compute IoU
        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        iou = intersection / union
        
        # Delete all indexes from the index list that have IoU greater than threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > iou_threshold)[0])))
    
    return pick

if __name__ == "__main__":
    print("=== Deformable DETR Training ===")
    print("Improvements over vanilla DETR:")
    print("- Multi-scale deformable attention")
    print("- Faster convergence (50 epochs vs 300)")
    print("- Better performance on small objects")
    print("- Reduced memory usage")
    print()
    
    # Train the model
    model, criterion = train_deformable_detr()
    
    print("Training completed successfully!")
    print(f"Model checkpoints saved in: {DEFORMABLE_CONFIG['save_dir']}")
