import os
import torch
import numpy as np
from torch.utils.data import IterableDataset
import random
from torchvision import transforms
from PIL import Image
from torchvision.ops import box_convert

class KITTIDataset(IterableDataset):  # Now inherits from IterableDataset
    def __init__(self, root_dir, split='training', transform=None, shuffle=True):
        """
        Args:
            root_dir (string): Directory with all the KITTI data
            split (string): 'training' or 'testing'
            transform (callable, optional): Optional transform to be applied
            shuffle (bool): Whether to shuffle the dataset
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.image_dir = os.path.join(root_dir, split, 'image_2')
        self.label_dir = os.path.join(root_dir, split, 'label_2')
        
        # Get all image files
        if os.path.exists(self.image_dir):
            self.image_files = [f for f in os.listdir(self.image_dir) 
                               if f.endswith('.png')]
        else:
            print(f"Warning: KITTI image directory not found: {self.image_dir}")
            self.image_files = []
        
        # Class mapping
        self.class_map = {
            'Car': 1,
            'Van': 1,
            'Truck': 1,
            'Pedestrian': 2,
            'Person_sitting': 2,
            'Cyclist': 3,
            'Tram': 1,
            'Misc': 0,  # Ignore class
            'DontCare': 0  # Ignore class
        }
        
        # Valid classes
        self.valid_classes = ['Car', 'Pedestrian', 'Cyclist']
        print(f"Loaded KITTI dataset with {len(self.image_files)} images")
               
        # Add default transform to convert to tensor
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __iter__(self):
        # Create indices and shuffle if needed
        indices = list(range(len(self.image_files)))
        if self.shuffle:
            random.shuffle(indices)
        
        for idx in indices:
            try:
                item = self._get_item(idx)
                if item is not None:
                    yield item
            except Exception as e:
                print(f"Error processing KITTI item {idx}: {e}")
                continue

    def _get_item(self, idx):
        """Get a single item from the dataset"""
        try:
            # Load image
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                return None
                
            image = Image.open(img_path).convert('RGB')
            width, height = image.size
            
            # Apply transform to image
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Initialize target structure
            target = {
                'boxes': [],
                'labels': [],
                'image_id': torch.tensor([idx]),
                'area': [],
                'iscrowd': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, height, width), dtype=torch.uint8)
            }
            
            # Load annotations if in training mode
            if self.split == 'training':
                label_name = img_name.replace('.png', '.txt')
                label_path = os.path.join(self.label_dir, label_name)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        data = line.strip().split()
                        if len(data) < 15:  # Skip invalid lines
                            continue
                        class_name = data[0]
                        
                        # Skip invalid classes
                        if class_name not in self.valid_classes:
                            continue
                        
                        # Get bounding box coordinates
                        try:
                            x1 = float(data[4])
                            y1 = float(data[5])
                            x2 = float(data[6])
                            y2 = float(data[7])
                        except (ValueError, IndexError):
                            continue
                        
                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                            continue
                        
                        # Clip coordinates to image boundaries
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        
                        # Skip boxes that became invalid after clipping
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Add to target
                        target['boxes'].append([x1, y1, x2, y2])
                        target['labels'].append(self.class_map[class_name])
                        target['area'].append(area)
            
            # Convert lists to tensors
            if target['boxes']:
                target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
                target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
                target['area'] = torch.tensor(target['area'], dtype=torch.float32)
                target['iscrowd'] = torch.zeros(len(target['labels']), dtype=torch.int64)
                
                # Create simple rectangular masks for each bounding box
                masks = []
                for box in target['boxes']:
                    mask = torch.zeros((height, width), dtype=torch.uint8)
                    x1, y1, x2, y2 = box.int()
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1.item(), width-1))
                    y1 = max(0, min(y1.item(), height-1))
                    x2 = max(0, min(x2.item(), width-1))
                    y2 = max(0, min(y2.item(), height-1))
                    
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = 1
                    masks.append(mask)
                
                if masks:
                    target['masks'] = torch.stack(masks, dim=0)
                else:
                    target['masks'] = torch.zeros((0, height, width), dtype=torch.uint8)
            else:
                # No valid boxes found
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
                target['area'] = torch.zeros((0,), dtype=torch.float32)
                target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
                target['masks'] = torch.zeros((0, height, width), dtype=torch.uint8)
            
            return image_tensor, target
            
        except Exception as e:
            print(f"Error processing KITTI item {idx}: {e}")
            return None

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.image_files)
