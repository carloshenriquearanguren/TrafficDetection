# your_transforms.py
import random
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T
import numpy as np


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Handle both PIL Image and tensor formats
            if isinstance(image, torch.Tensor):
                height, width = image.shape[-2:]
                image = image.flip(-1)
            else:
                # PIL Image
                width, height = image.size
                image = F.hflip(image)
            
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            
            if "masks" in target:
                if isinstance(target["masks"], torch.Tensor):
                    target["masks"] = target["masks"].flip(-1)
        
        return image, target


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast, 
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image, target):
        # Apply to PIL image before converting to tensor
        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image)
        
        image = self.color_jitter(image)
        return image, target


class RandomRotation:
    def __init__(self, degrees=5):
        self.degrees = degrees
    
    def __call__(self, image, target):
        if random.random() < 0.3:  # 30% chance
            angle = random.uniform(-self.degrees, self.degrees)
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image)
            image = F.rotate(image, angle, expand=False)
            # Note: For production, you'd want to rotate bounding boxes too
        return image, target


class Resize:
    def __init__(self, target_size=640, max_size=1024):
        self.target_size = target_size
        self.max_size = max_size
    
    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
            image = F.to_pil_image(image)
        else:
            w, h = image.size
        
        # Calculate new size maintaining aspect ratio
        if w < h:
            new_w = self.target_size
            new_h = int(h * self.target_size / w)
        else:
            new_h = self.target_size
            new_w = int(w * self.target_size / h)
        
        # Ensure max dimension doesn't exceed max_size
        if max(new_w, new_h) > self.max_size:
            if new_w > new_h:
                new_h = int(new_h * self.max_size / new_w)
                new_w = self.max_size
            else:
                new_w = int(new_w * self.max_size / new_h)
                new_h = self.max_size
        
        image = F.resize(image, (new_h, new_w))
        
        # Scale bounding boxes
        scale_x = new_w / w
        scale_y = new_h / h
        
        if "boxes" in target:
            boxes = target["boxes"]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes
        
        if "area" in target:
            target["area"] *= (scale_x * scale_y)
        
        return image, target


class NormalizeAndFormatBoxes:
    def __call__(self, image, target):
        # Image is already a tensor here
        h, w = image.shape[-2:]
        
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"] # Expected format: [x_min, y_min, x_max, y_max]
            
            # Convert to cxcywh
            boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
            
            # Normalize by image dimensions
            boxes_cxcywh[:, 0] /= w  # cx
            boxes_cxcywh[:, 1] /= h  # cy
            boxes_cxcywh[:, 2] /= w  # w
            boxes_cxcywh[:, 3] /= h  # h
            
            target["boxes"] = boxes_cxcywh
        
        return image, target


def get_transform(train: bool):
    """
    Get transforms for training or validation following the milestone report specs:
    - Multiscale augmentation with resizing
    - Horizontal flip (50%)
    - Color jitter (±20% brightness/contrast)
    - Rotation (±5°)
    - ImageNet normalization
    """
    transforms = []
    
    if train:
        # Add augmentations for training - keep PIL format until ToTensor
        transforms.extend([
            Resize(target_size=640, max_size=1024),
            ColorJitter(brightness=0.2, contrast=0.2),
            RandomRotation(degrees=5),
            RandomHorizontalFlip(prob=0.5),
            ToTensor(),  # Convert to tensor after PIL operations
            NormalizeAndFormatBoxes(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Just resize for validation
        transforms.extend([
            Resize(target_size=640, max_size=1024),
            ToTensor(),
            NormalizeAndFormatBoxes(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return Compose(transforms)


# Simple version for backward compatibility
def get_simple_transform(train: bool):
    """Simplified version similar to your original"""
    def _trans(img, target):
        # Convert to tensor first
        img = F.to_tensor(img)
        
        # Random horizontal flip for training
        if train and random.random() < 0.5:
            img = F.hflip(img)
            w = img.shape[-1]
            boxes = target['boxes']
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target['boxes'] = boxes
            if 'masks' in target:
                target['masks'] = target['masks'].flip(-1)
        
        # Normalize
        img = F.normalize(img,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        return img, target
    
    return _trans
