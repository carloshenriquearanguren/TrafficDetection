import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2 import model_zoo
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from yacs.config import CfgNode as CN
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.data import build_detection_train_loader, DatasetMapper
import logging
import os
from tqdm import tqdm
import wandb
from typing import Optional, Tuple, List, Dict
import math
from detectron2.modeling.backbone import FPN

@BACKBONE_REGISTRY.register()
def build_swin_backbone(cfg, input_shape):
    """
    Build a Swin Transformer + FPN backbone for Detectron2.

    - The raw SwinTransformer produces features named ["res2","res3","res4","res5"].
    - We feed those four stages into a Detectron2 FPN to produce ["p2","p3","p4","p5"],
      each with the same channel dimension = cfg.MODEL.FPN.OUT_CHANNELS.
    """
    model_type = cfg.MODEL.SWIN.MODEL_TYPE

    # 1) Instantiate the raw Swin backbone
    if model_type == "swin_tiny":
        bottom_up = swin_tiny(img_size=cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE)
    elif model_type == "swin_small":
        bottom_up = swin_small(img_size=cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE)
    elif model_type == "swin_base":
        bottom_up = swin_base(img_size=cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE)
    elif model_type == "swin_large":
        bottom_up = swin_large(img_size=cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE)
    else:
        raise ValueError(f"Unknown Swin model type: {model_type}")

    # 2) The four Swin feature maps we want to wrap with FPN:
    in_features = cfg.MODEL.SWIN.OUT_FEATURES    # e.g. ["res2","res3","res4","res5"]

    # 3) All FPN levels will be this many channels:
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS    # typically 256

    # 4) We only need p2→p5, so no top_block (no p6):
    top_block = None

    # 5) Construct the Detectron2 FPN:
    fpn = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm="",          # no extra BN in FPN
        top_block=top_block,
        fuse_type="sum",  # standard FPN sum
    )
    return fpn

# Setup logging
setup_logger()
logging.getLogger("detectron2").setLevel(logging.WARNING)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp(nn.Module):
    """MLP module for Swin Transformer"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition input into windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): window size
        H (int): Height of the feature map (this should be the padded height)
        W (int): Width of the feature map (this should be the padded width)

    Returns:
        x: (B, H, W, C) - tensor with padded dimensions
    """
    nWb, win_h, win_w, C = windows.shape
    B = int(nWb / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention"""
    
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int,
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x is (B, L, C) where L = H * W
        H, W = self.input_resolution # Original height and width of the feature map
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to be multiples of window_size
        # Calculate padding needed for width (W) and height (H)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        # Apply padding. F.pad takes (padding_left, padding_right, padding_top, padding_bottom)
        # for the last two dimensions of the input tensor (W, H).
        # We need to pad the H and W dimensions.
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b)) # (0,0) for C dim, (pad_l, pad_r) for W, (pad_t, pad_b) for H
        _, Hp, Wp, _ = x.shape # Get padded dimensions (Hp = H + pad_b, Wp = W + pad_r)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            # Pass the padded dimensions (Hp, Wp) to window_reverse
            x = window_reverse(attn_windows, self.window_size, Hp, Wp) 
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            # Pass the padded dimensions (Hp, Wp) to window_reverse
            x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Unpad feature maps (slice back to original H, W)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous() # H and W are the original (unpadded) dimensions
        
        x = x.view(B, H * W, C)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer"""
    
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape [B, L, C], where L == H*W exactly
               and (H, W) = self.input_resolution.
        Returns:
            Tensor of shape [B, (H_padded/2)*(W_padded/2), 2*C]
            after 2×2 patch merging.
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, (
            f"PatchMerging expected L=H*W={H*W}, but got L={L}. "
            "Check that input was padded so H and W match."
        )

        # 1) Pad H, W to make them even (only if odd)
        pad_h = H % 2
        pad_w = W % 2

        # reshape into [B, H, W, C]
        x = x.view(B, H, W, C)
        if pad_h or pad_w:
            # F.pad: (left, right, top, bottom) for the last two dims
            # Here, no channel pad, so (0, 0, 0, pad_w, 0, pad_h)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        H_padded = H + pad_h
        W_padded = W + pad_w

        # 2×2 merging: take (0,0),(0,1),(1,0),(1,1) patches and concatenate channels
        x = x.view(
            B,
            H_padded // 2,
            2,
            W_padded // 2,
            2,
            C
        )  # [B, H_padded/2, 2, W_padded/2, 2, C]
        x = (
            x.permute(0, 1, 3, 2, 4, 5)  # [B, H_padded/2, W_padded/2, 2, 2, C]
            .contiguous()
            .view(B, -1, 4 * C)
        )  # [B, (H_padded/2)*(W_padded/2), 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # now [B, (H_padded/2)*(W_padded/2), 2*C]
        return x

class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int,
                 num_heads: int, window_size: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Remove the strict size assertion for flexibility
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformer(Backbone):
    """Swin Transformer backbone for object detection (with Detectron2)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm

        # Determine final "num_features" (used by Detectron2)
        if self.num_layers == 1:
            self.num_features = embed_dim
        else:
            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        # absolute position embedding (optional)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # build the 4 Swin "stages"
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        # final layer norm
        self.norm = norm_layer(self.num_features)

        # FIXED: Use res2, res3, res4, res5 as feature names (standard Detectron2 naming)
        self._out_features = ["res2", "res3", "res4", "res5"]

        # Stride mapping: patch‐embed stride = patch_size, each subsequent stage doubles stride
        patch_s = patch_size
        self._out_feature_strides = {
            "res2": patch_s,      # 4
            "res3": patch_s * 2,  # 8
            "res4": patch_s * 4,  # 16
            "res5": patch_s * 8,  # 32
        }

        self._out_feature_channels = {
            "res2": embed_dim,           # 96
            "res3": embed_dim * 2,       # 192
            "res4": embed_dim * 4,       # 384
            "res5": embed_dim * 8,       # 768
        }

        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Swin‐style trunc_normal init for Linear & zero‐init for LayerNorm bias."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run Swin‐Transformer and return a dict of feature‐maps with standard Detectron2 names.
        We first pad the input so that after `patch_embed` and repeated 2× downsampling,
        every patch grid remains an integer size.
        """
        B, C, H_orig, W_orig = x.shape

        # 1) Compute total_stride = patch_size × 2^(num_layers−1)
        patch_size = self.patch_embed.patch_size[0]
        num_down = self.num_layers - 1
        total_stride = patch_size * (2 ** num_down)

        # 2) Pad the input on the right/bottom so that H_orig, W_orig are divisible by total_stride
        pad_H = (total_stride - (H_orig % total_stride)) % total_stride
        pad_W = (total_stride - (W_orig % total_stride)) % total_stride
        if pad_H > 0 or pad_W > 0:
            # F.pad: (left, right, top, bottom)
            x = F.pad(x, (0, pad_W, 0, pad_H))

        H_padded, W_padded = x.shape[2], x.shape[3]
        # 3) Compute the initial patch grid size (floor‐division by patch_size)
        self.patches_resolution = [
            H_padded // patch_size,
            W_padded // patch_size,
        ]
        current_H, current_W = self.patches_resolution

        # 4) Apply patch embedding + optional absolute‐pos embedding
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outputs: Dict[str, torch.Tensor] = {}
        for i, layer_module in enumerate(self.layers):
            # Tell each block what (H, W) to expect
            layer_module.input_resolution = (current_H, current_W)
            for block in layer_module.blocks:
                block.input_resolution = (current_H, current_W)

            x = layer_module(x)

            # If we just did a downsample (PatchMerging), halve via ceil; else keep same
            if layer_module.downsample is not None:
                out_H = math.ceil(current_H / 2)
                out_W = math.ceil(current_W / 2)
            else:
                out_H, out_W = current_H, current_W

            feat_name = self._out_features[i]
            C_out = self._out_feature_channels[feat_name]
            # Reshape to (B, C_out, H_out, W_out)
            x_reshaped = x.transpose(1, 2).reshape(B, C_out, out_H, out_W)
            outputs[feat_name] = x_reshaped

            current_H, current_W = out_H, out_W

        # Final layer‐norm on the last stage's output (if any)
        if self._out_features:
            last_feature = self._out_features[-1]
            if last_feature in outputs:
                norm_in = outputs[last_feature]
                Bn, Cn, Hn, Wn = norm_in.shape
                norm_flat = norm_in.flatten(2).transpose(1, 2)
                norm_out_flat = self.norm(norm_flat)
                outputs[last_feature] = norm_out_flat.transpose(1, 2).reshape(Bn, Cn, Hn, Wn)

        return outputs


    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

def swin_tiny(img_size: int = 224, **kwargs) -> SwinTransformer:
    """Constructs a Swin-T model"""
    model = SwinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        **kwargs)
    return model

def swin_small(img_size: int = 224, **kwargs) -> SwinTransformer:
    """Constructs a Swin-S model"""
    model = SwinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        **kwargs)
    return model

def swin_base(img_size: int = 224, **kwargs) -> SwinTransformer:
    """Constructs a Swin-B model"""
    model = SwinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        **kwargs)
    return model

def swin_large(img_size: int = 224, **kwargs) -> SwinTransformer:
    """Constructs a Swin-L model"""
    model = SwinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        **kwargs)
    return model


class WaymoDetectionDataset:
    """Waymo dataset registration for Detectron2"""
    
    def __init__(self, root):
        #root = "~/waymo_dataset"
        self.root = Path(root)
        self.register_datasets()

    def register_datasets(self):
        # assume your JSON is named "waymo_annotations.json" under root/annotations/
        json_file = self.root / "annotations" / "waymo_annotations.json"
        image_dir = self.root / "images"
        # Because your JSON likely contains entries for BOTH train & val in one file,
        # here we register it as a single split:
        register_coco_instances(
                "waymo_train",
                {},
                "/home/ubuntu/waymo_dataset/annotations/waymo_annotations.json",
                "/home/ubuntu/waymo_images/"
        )
        MetadataCatalog.get("waymo_train").set(
            thing_classes=["vehicle","pedestrian","cyclist"]
        )

class SwinTransformerConfig:
    """
    Detectron-2 helper that wires a Swin-Transformer backbone into
    Cascade Mask R-CNN + FPN.
    """
    def __init__(self, model_type: str = "swin_tiny"):
        self.cfg = get_cfg()

        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        self.cfg.MODEL.SWIN = CN()
        self.cfg.MODEL.SWIN.MODEL_TYPE      = model_type
        self.cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
        # We will expose exactly these four Swin stages to FPN:
        self.cfg.MODEL.SWIN.OUT_FEATURES    = ["res2", "res3", "res4", "res5"]

        # 3) Tell Detectron2 to call our custom backbone builder
        self.cfg.MODEL.BACKBONE.NAME = "build_swin_backbone"
        self.cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        self.cfg.MODEL.FPN.NUM_LEVELS   = 4
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256   # all res2–res5 will have 256 channels
        # --------------------------------------------------------------------
        # 4) Override RPN + ROI heads so they consume exactly 4 levels (res2…res5)
        #    instead of the default p2…p6 (which would be 5 levels).
        # --------------------------------------------------------------------
        #   RPN will now look for features named "res2","res3","res4","res5"
        self.cfg.MODEL.RPN.IN_FEATURES       = ["p2", "p3", "p4", "p5"]
        self.cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

        # 5) Override the anchor‐generator sizes so there are exactly 4 lists:
        #    (Mask R-CNN template uses 5 levels by default; that causes the “sizes of length 5 but num_features is 4” error.)
        #    Here we supply 4 levels: [[32],[64],[128],[256]]
        self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256]]
        self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] * 4

        # 7) Finally, set your classes (e.g., 3 classes: vehicle, pedestrian, cyclist).
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        self.cfg.MODEL.RETINANET.NUM_CLASSES = 3


    def get_config(self):
        return self.cfg.clone()

class CustomTrainer(DefaultTrainer):
    """
    Custom trainer for Swin Transformer + object detection
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

def setup_config(model_type="swin_tiny", 
                 num_classes=3,
                 data_dir="./data",
                 output_dir="./output",
                 learning_rate=0.001,
                 batch_size=4,
                 max_iter=40000):
    """Setup configuration for training"""
    
    config_helper = SwinTransformerConfig(model_type)
    cfg = config_helper.get_config()
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("waymo_train",)
    cfg.DATASETS.TEST  = ()
    
    # Training configuration
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    
    # Input configuration
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Output configuration
    cfg.OUTPUT_DIR = output_dir
    
    # Evaluation configuration
    cfg.TEST.EVAL_PERIOD = 5000
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    return cfg

def load_pretrained_swin_weights(model, pretrained_path):
    """Load pretrained Swin Transformer weights"""
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Filter out unnecessary keys
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                backbone_state_dict[new_key] = v
            elif not any(x in k for x in ['head', 'fc', 'classifier']):
                backbone_state_dict[k] = v
        
        # Load weights with strict=False to handle missing/extra keys
        missing_keys, unexpected_keys = model.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    else:
        print("No pretrained weights provided or file not found")

def train_model(cfg, pretrained_swin_path=None, resume=False):
    """Train the Swin Transformer object detection model"""
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Initialize wandb logging (optional)
    if wandb.run is None:
        wandb.init(
            project="swin-transformer-detection",
            config={
                "model_type": cfg.MODEL.SWIN.MODEL_TYPE,
                "learning_rate": cfg.SOLVER.BASE_LR,
                "batch_size": cfg.SOLVER.IMS_PER_BATCH,
                "max_iter": cfg.SOLVER.MAX_ITER,
            }
        )
    
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    
    # Load pretrained Swin weights if provided
    if pretrained_swin_path:
        load_pretrained_swin_weights(trainer.model, pretrained_swin_path)
    
    trainer.train()
    
    return trainer

def evaluate_model(cfg, model_path):
    """Evaluate the trained model"""
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("waymo_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    
    # You would typically run evaluation on validation set here
    # This is a placeholder for the evaluation loop
    print("Evaluation completed")
    
    return evaluator

def inference_on_image(cfg, model_path, image_path, output_path):
    """Run inference on a single image"""
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    
    # Load and process image
    image = cv2.imread(image_path)
    outputs = predictor(image)
    
    # Visualize results
    from detectron2.utils.visualizer import Visualizer, ColorMode
    
    metadata = MetadataCatalog.get("waymo_val")
    v = Visualizer(
        image[:, :, ::-1], 
        metadata=metadata, 
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW
    )
    
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = vis.get_image()[:, :, ::-1]
    
    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Inference result saved to {output_path}")
    
    return outputs

def main():
    """Main training and evaluation pipeline"""
    
    # Setup dataset
    data_dir = "/home/ubuntu/waymo_images"  # Update with your data directory
    dataset = WaymoDetectionDataset(data_dir)
    
    # Setup configuration
    cfg = setup_config(
        model_type="swin_tiny",
        num_classes=3,
        data_dir=data_dir,
        output_dir="./output",
        learning_rate=0.001,
        batch_size=4,
        max_iter=40000
    )
    
    # Train model
    print("Starting training...")
    trainer = train_model(
        cfg, 
        pretrained_swin_path="./pretrained/swin_tiny_patch4_window7_224.pth",  # Optional
        resume=False
    )
    
    # Evaluate model
    print("Starting evaluation...")
    model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluate_model(cfg, model_path)
    
    # Run inference on sample image
    print("Running inference...")
    inference_on_image(
        cfg, 
        model_path, 
        "./sample_image.jpg", 
        "./inference_result.jpg"
    )
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()

# Additional utility functions for data preparation and analysis

def convert_waymo_to_coco(waymo_data_dir, output_dir):
    """Convert Waymo dataset format to COCO format"""
    # This is a placeholder - implement based on your specific Waymo data format
    print("Converting Waymo dataset to COCO format...")
    # Implementation depends on your specific Waymo data structure
    pass

def analyze_dataset_statistics(dataset_name):
    """Analyze dataset statistics"""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    stats = {
        "total_images": len(dataset_dicts),
        "total_annotations": 0,
        "class_counts": {},
        "avg_instances_per_image": 0
    }
    
    for record in dataset_dicts:
        stats["total_annotations"] += len(record["annotations"])
        for ann in record["annotations"]:
            class_id = ann["category_id"]
            stats["class_counts"][class_id] = stats["class_counts"].get(class_id, 0) + 1
    
    stats["avg_instances_per_image"] = stats["total_annotations"] / stats["total_images"]
    
    print(f"Dataset Statistics for {dataset_name}:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Annotations: {stats['total_annotations']}")
    print(f"Average Instances per Image: {stats['avg_instances_per_image']:.2f}")
    print(f"Class Distribution: {stats['class_counts']}")
    
    return stats

def benchmark_model_speed(cfg, model_path, num_images=100):
    """Benchmark model inference speed"""
    cfg.MODEL.WEIGHTS = model_path
    predictor = DefaultPredictor(cfg)
    
    # Create dummy images for benchmarking
    dummy_images = []
    for _ in range(num_images):
        dummy_image = np.random.randint(0, 255, (800, 1333, 3), dtype=np.uint8)
        dummy_images.append(dummy_image)
    
    # Warm up
    for _ in range(10):
        predictor(dummy_images[0])
    
    # Benchmark
    import time
    start_time = time.time()
    
    for img in dummy_images:
        predictor(img)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_images / total_time
    
    print(f"Benchmark Results:")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average Time per Image: {total_time/num_images*1000:.2f} ms")
    
    return fps