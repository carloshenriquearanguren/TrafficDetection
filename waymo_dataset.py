import torch
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
import numpy as np
from torch.utils.data import IterableDataset
from torchvision import transforms
import cv2
from PIL import Image

class WaymoDataset(IterableDataset):
    def __init__(self, gcs_path="gs://waymo_open_dataset_v_1_4_3/individual_files/training/*.tfrecord", transform=None):
        self.files = tf.io.gfile.glob(gcs_path)
        print(f"Found {len(self.files)} Waymo files")
        self.class_map = {
            1: 1,  # Vehicle
            2: 2,  # Pedestrian
            4: 3   # Cyclist
        }
        # Don't set default transform - handle conversion manually
        self.transform = transform
        
    def decode_image(self, image_proto):
        """Decode Waymo image from protobuf format"""
        # Get the image bytes
        image_bytes = image_proto.image
        
        # Decode using OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            # Fallback: try PIL
            from io import BytesIO
            pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return np.array(pil_image)
    
    def parse_frame(self, data):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(data.numpy())
        
        # Convert images and labels
        for img_idx, (img, camera_labels) in enumerate(zip(frame.images, frame.camera_labels)):
            try:
                # Decode image properly
                image_np = self.decode_image(img)
                height, width = image_np.shape[:2]
                
                # Apply transforms to PIL image if provided, otherwise convert directly
                if self.transform:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(image_np)
                    image_tensor = self.transform(pil_image)
                else:
                    # Convert numpy array directly to tensor
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                
                # Extract bounding boxes and classes
                boxes = []
                classes = []
                
                for label in camera_labels.labels:
                    if label.type not in self.class_map: 
                        continue
                    
                    # Get 2D bounding box from camera labels
                    box = [
                        label.box.center_x - label.box.length/2,
                        label.box.center_y - label.box.width/2,
                        label.box.center_x + label.box.length/2,
                        label.box.center_y + label.box.width/2
                    ]
                    
                    # Clamp to image boundaries
                    box[0] = max(0, min(box[0], width-1))
                    box[1] = max(0, min(box[1], height-1))
                    box[2] = max(0, min(box[2], width-1))
                    box[3] = max(0, min(box[3], height-1))
                    
                    # Skip invalid boxes
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                        
                    boxes.append(box)
                    classes.append(self.class_map[label.type])
                
                # Create masks
                masks = []
                for box in boxes:
                    mask = torch.zeros((height, width), dtype=torch.uint8)
                    x1, y1, x2, y2 = map(int, box)
                    # Clamp to boundaries
                    x1 = max(0, min(x1, width-1))
                    y1 = max(0, min(y1, height-1))
                    x2 = max(0, min(x2, width-1))
                    y2 = max(0, min(y2, height-1))
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = 1
                    masks.append(mask)
                
                if masks:
                    masks = torch.stack(masks, dim=0)
                else:
                    masks = torch.zeros((0, height, width), dtype=torch.uint8)
                
                if boxes:
                    target = {
                        "boxes": torch.tensor(boxes, dtype=torch.float32),
                        "labels": torch.tensor(classes, dtype=torch.int64),
                        "image_id": torch.tensor([hash(str(img.name) + str(img_idx))]),
                        "area": (torch.tensor(boxes)[:, 3] - torch.tensor(boxes)[:, 1]) * \
                                (torch.tensor(boxes)[:, 2] - torch.tensor(boxes)[:, 0]),
                        "iscrowd": torch.zeros(len(classes), dtype=torch.int64),
                        "masks": masks
                    }
                    yield image_tensor, target
                    
            except Exception as e:
                print(f"Error processing Waymo frame: {e}")
                continue

    def __iter__(self):
        # Limit to first few files for testing
        files_to_use = self.files[:2]  # Use only first 2 files for testing
        dataset = tf.data.TFRecordDataset(files_to_use, compression_type='')
        for record in dataset:
            yield from self.parse_frame(record)
