import os
import json
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from waymo_dataset import WaymoDataset
from kitti_dataset import KITTIDataset


def create_categories():
    """
    Defines COCO categories mapping for vehicle, pedestrian, cyclist.
    """
    return [
        {"id": 1, "name": "vehicle",    "supercategory": "object"},
        {"id": 2, "name": "pedestrian", "supercategory": "object"},
        {"id": 3, "name": "cyclist",   "supercategory": "object"},
    ]


def convert_to_coco(dataset, out_json_path, out_img_dir=None, max_items=None):
    """
    Iterate over `dataset` (an IterableDataset yielding (img_tensor, target) pairs).
    Builds COCO-format JSON and optionally saves each image to `out_img_dir`.

    Args:
        dataset: Iterable yielding (torch.Tensor [C,H,W], target dict) pairs.
        out_json_path: Path to write annotations JSON.
        out_img_dir: Directory to save images (optional).
        max_items: If provided, stop after this many samples.
    """
    coco = {
        "info": {
            "description": "Waymo/KITTI to COCO conversion",
            "version": "1.0",
            "year": 2024,
            "contributor": "Stanford CS231N",
            "date_created": "2024-01-01"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": create_categories()
    }

    ann_id = 1
    img_transform = T.ToPILImage()
    
    if out_img_dir:
        os.makedirs(out_img_dir, exist_ok=True)

    try:
        for idx, (img_t, tgt) in enumerate(dataset):
            if max_items and idx >= max_items:
                break

            # Handle different tensor formats
            if img_t.dim() == 4:  # Batch dimension present
                img_t = img_t.squeeze(0)
            
            # Ensure tensor is in [C,H,W] format
            if img_t.dim() == 3:
                C, H, W = img_t.shape
            else:
                raise ValueError(f"Unexpected image tensor shape: {img_t.shape}")

            # Prepare image metadata
            file_name = f"{idx:012d}.jpg"
            coco["images"].append({
                "id": idx,
                "file_name": file_name,
                "width": W,
                "height": H,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            # Save image if requested
            if out_img_dir:
                try:
                    # Ensure tensor is in [0,1] range for PIL conversion
                    if img_t.max() > 1.0:
                        img_t = img_t / 255.0
                    
                    img_pil = img_transform(img_t.cpu())
                    img_pil.save(os.path.join(out_img_dir, file_name))
                except Exception as e:
                    print(f"Warning: Could not save image {idx}: {e}")
                    continue

            # Process annotations
            if "boxes" not in tgt or "labels" not in tgt:
                print(f"Warning: Missing boxes or labels in sample {idx}")
                continue

            boxes = tgt["boxes"]
            labels = tgt["labels"]
            
            # Convert to numpy if needed
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            # Handle areas
            if "area" in tgt:
                areas = tgt["area"]
                if isinstance(areas, torch.Tensor):
                    areas = areas.cpu().numpy()
            else:
                # Calculate areas from boxes [x1,y1,x2,y2]
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                areas = np.array(areas)

            # Handle iscrowd
            if "iscrowd" in tgt:
                iscrowd = tgt["iscrowd"]
                if isinstance(iscrowd, torch.Tensor):
                    iscrowd = iscrowd.cpu().numpy()
            else:
                iscrowd = np.zeros(len(boxes), dtype=int)

            # Process each bounding box
            for i, (box, cat_id, area) in enumerate(zip(boxes, labels, areas)):
                # Skip invalid boxes
                if len(box) != 4:
                    continue
                
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Skip boxes that are too small (as mentioned in report)
                if w <= 0 or h <= 0 or area < 100:
                    continue
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, W-1))
                y1 = max(0, min(y1, H-1))
                w = min(w, W - x1)
                h = min(h, H - y1)
                
                ann = {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO format: [x,y,width,height]
                    "area": float(area),
                    "iscrowd": int(iscrowd[i] if i < len(iscrowd) else 0),
                    "segmentation": []  # Empty for bounding box only
                }
                coco["annotations"].append(ann)
                ann_id += 1

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images...")

    except Exception as e:
        print(f"Error during conversion: {e}")
        raise

    # Write JSON
    print(f"Writing annotations to {out_json_path}...")
    with open(out_json_path, "w") as f:
        json.dump(coco, f, indent=2)
    
    print(f"Conversion complete! Processed {len(coco['images'])} images with {len(coco['annotations'])} annotations.")


def main():
    """
    Example conversion for first 500 samples each as specified in milestone report.
    """
    print("Converting Waymo to COCO...")
    try:
        waymo_ds = WaymoDataset(
            gcs_path="gs://waymo_open_dataset_v_1_4_3/individual_files/training/*.tfrecord",
            transform=T.Compose([
                T.ToTensor(),
                # Add normalization as specified in report
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        convert_to_coco(
            waymo_ds,
            out_json_path="waymo_annotations.json",
            out_img_dir="waymo_images",
            max_items=500
        )
    except Exception as e:
        print(f"Error processing Waymo dataset: {e}")

    print("\nConverting KITTI to COCO...")
    try:
        home = Path.home()
        kitti_ds = KITTIDataset(
            root_dir=str(home / "datasets/kitti"),
            split='training',
            transform=T.Compose([
                T.ToTensor(),
                # Add normalization as specified in report
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            shuffle=False
        )
        convert_to_coco(
            kitti_ds,
            out_json_path="kitti_annotations.json",
            out_img_dir="kitti_images",
            max_items=500
        )
    except Exception as e:
        print(f"Error processing KITTI dataset: {e}")

    print("\nCOCO conversion complete!")


if __name__ == "__main__":
    main()
