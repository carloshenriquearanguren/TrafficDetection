# coco_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import numpy as np


class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms
        
        # Print dataset statistics
        print(f"Loaded COCO dataset from {ann_file}")
        print(f"Images: {len(self.ids)}")
        print(f"Categories: {len(self.coco.cats)}")
        
        # Print category statistics
        for cat_id, cat_info in self.coco.cats.items():
            ann_count = len(self.coco.getAnnIds(catIds=[cat_id]))
            print(f"  {cat_info['name']}: {ann_count} annotations")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = info['file_name']
        
        # Load image
        img_path = os.path.join(self.img_dir, path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        
        for ann in anns:
            # Skip invalid annotations
            if ann.get('iscrowd', 0) and len(ann.get('segmentation', [])) == 0:
                continue
                
            x, y, w, h = ann['bbox']
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
                
            # Convert to [x1, y1, x2, y2] format
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] - 1)
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
            
            # Handle masks (if available)
            if 'segmentation' in ann and ann['segmentation']:
                try:
                    mask = self.coco.annToMask(ann)
                    masks.append(torch.as_tensor(mask, dtype=torch.uint8))
                except:
                    # If mask conversion fails, create a dummy mask
                    h_img, w_img = info['height'], info['width']
                    masks.append(torch.zeros((h_img, w_img), dtype=torch.uint8))
            else:
                # Create dummy mask if no segmentation available
                h_img, w_img = info['height'], info['width']
                masks.append(torch.zeros((h_img, w_img), dtype=torch.uint8))
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, info['height'], info['width']), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            if masks:
                masks = torch.stack(masks)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
        }
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target
