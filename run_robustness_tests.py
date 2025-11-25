import torch
from torch.utils.data import DataLoader
from deformable_detr_train import build_deformable_detr, COCODataset, get_transform, box_cxcywh_to_xyxy, generalized_box_iou
from detr_train import DETR
from train2 import create_model as build_mask_rcnn
from robustness import RobustnessEvaluator
import os
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou

def _decode_outputs(outputs, score_thresh: float = 0.1):
    """
    Convert raw Deformable-DETR / DETR outputs into the format the IoU/accuracy
    code expects.  Works for a single Tensor or for a list of per-batch Tensors.

    Parameters
    ----------
    outputs : dict with keys
        - 'pred_logits': [B, Q, C+1]  (last class = background)
        - 'pred_boxes' : [B, Q, 4]    (cx,cy,w,h) in **normalised [0,1] coords**
    score_thresh : float
        Only predictions with prob >= score_thresh are kept.

    Returns
    -------
    list[dict]  (length == batch size B)
        Each dict contains:
            'boxes'  – tensor [N,4]  (xmin, ymin, xmax, ymax)  in [0,1]
            'labels' – tensor [N]    class-ids 0…C-1
            'scores' – tensor [N]    confidences
    """
    print("\nDEBUG: Decoding outputs")
    print("DEBUG: Outputs type:", type(outputs))
    print("DEBUG: Outputs keys:", outputs.keys() if isinstance(outputs, dict) else "Not a dict")
    
    # softmax over classes once for the whole batch
    probas = outputs["pred_logits"].softmax(-1)          # [B,Q,C+1]
    boxes  = outputs["pred_boxes"]                       # [B,Q,4]
    
    print(f"DEBUG: probas shape: {probas.shape}")
    print(f"DEBUG: boxes shape: {boxes.shape}")

    results = []
    for probs_i, boxes_i in zip(probas, boxes):          # iterate over B
        # drop background column and take the best foreground class
        scores_i, labels_i = probs_i[:, :-1].max(-1)     # each [Q]
        
        print(f"DEBUG: scores_i shape: {scores_i.shape}")
        print(f"DEBUG: labels_i shape: {labels_i.shape}")
        print(f"DEBUG: max score: {scores_i.max().item()}")
        print(f"DEBUG: min score: {scores_i.min().item()}")
        print(f"DEBUG: mean score: {scores_i.mean().item()}")
        print(f"DEBUG: score threshold: {score_thresh}")

        keep = scores_i >= score_thresh
        if keep.any():
            # (cx,cy,w,h) → (xmin,ymin,xmax,ymax)
            cx, cy, w, h = boxes_i[keep].unbind(-1)
            xmin = (cx - 0.5 * w).clamp(0, 1)
            ymin = (cy - 0.5 * h).clamp(0, 1)
            xmax = (cx + 0.5 * w).clamp(0, 1)
            ymax = (cy + 0.5 * h).clamp(0, 1)

            results.append({
                "boxes":  torch.stack([xmin, ymin, xmax, ymax], dim=-1),
                "labels": labels_i[keep],
                "scores": scores_i[keep],
            })
        else:  # nothing above threshold
            device = boxes_i.device
            results.append({
                "boxes":  torch.empty(0, 4, device=device),
                "labels": torch.empty(0, dtype=torch.long, device=device),
                "scores": torch.empty(0, device=device),
            })

    return results

def _xywh_to_norm_xyxy(boxes_xywh: torch.Tensor,
                       img_w: int,
                       img_h: int) -> torch.Tensor:
    """
    Convert absolute-pixel (x, y, w, h) boxes to **normalised**
    (xmin, ymin, xmax, ymax) ∈ [0,1] – the format used by _decode_outputs().
    """
    x, y, w, h = boxes_xywh.unbind(-1)
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h

    return torch.stack(
        [xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h], dim=-1
    ).clamp(0, 1)

def prepare_targets(targets):
    """Convert COCO format targets to tensor format"""
    if not targets:
        return {'labels': torch.zeros(0, dtype=torch.long), 'boxes': torch.zeros(0, 4)}
    
    if isinstance(targets, (list, tuple)):
        # If targets is already a list/tuple of dicts, return the first one
        if len(targets) > 0 and isinstance(targets[0], dict):
            target = targets[0]
            # Convert boxes to normalized xyxy format
            if 'boxes' in target and 'image_size' in target:
                boxes_xywh = target['boxes']
                img_w, img_h = target['image_size']
                # Convert from (x,y,w,h) to (x1,y1,x2,y2) format
                x, y, w, h = boxes_xywh.unbind(-1)
                x1 = x / img_w
                y1 = y / img_h
                x2 = (x + w) / img_w
                y2 = (y + h) / img_h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
                return {'labels': target['labels'], 'boxes': boxes_xyxy}
            return target
    
    # If targets is a dict, convert boxes if needed
    if isinstance(targets, dict):
        if 'boxes' in targets and 'image_size' in targets:
            boxes_xywh = targets['boxes']
            img_w, img_h = targets['image_size']
            # Convert from (x,y,w,h) to (x1,y1,x2,y2) format
            x, y, w, h = boxes_xywh.unbind(-1)
            x1 = x / img_w
            y1 = y / img_h
            x2 = (x + w) / img_w
            y2 = (y + h) / img_h
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
            return {'labels': targets['labels'], 'boxes': boxes_xyxy}
        return targets
    
    # If we get here, something is wrong with the format
    print(f"Warning: Unexpected target format: {type(targets)}")
    return {'labels': torch.zeros(0, dtype=torch.long), 'boxes': torch.zeros(0, 4)}

def load_model(model_type, checkpoint_path):
    """Load model based on type"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    if model_type == 'deformable_detr':
        model, criterion = build_deformable_detr()
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif model_type == 'detr':
        # Create DETR model with default parameters
        model = DETR(
            num_classes=3,  # vehicle, pedestrian, cyclist
            hidden_dim=256,
            nheads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_queries=100
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif model_type == 'mask_rcnn':
        model = build_mask_rcnn()
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model

def calculate_detection_accuracy(outputs, targets, iou_threshold=0.5, score_threshold=0.5):
    """Calculate detection accuracy for DETR-like model outputs"""
    total_correct = 0
    total_targets = 0
    
    print("\nDEBUG: Model outputs type:", type(outputs))
    print("DEBUG: Model outputs keys:", outputs.keys() if isinstance(outputs, dict) else "Not a dict")
    
    # Convert model outputs to the expected format
    predictions = _decode_outputs(outputs, score_threshold)
    
    # Process each image in the batch
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"\nDEBUG: Processing image {i}")
        # Get predictions and ground truth
        pred_boxes = pred['boxes']    # [N,4] in (xmin,ymin,xmax,ymax) format
        pred_labels = pred['labels']  # [N] class IDs
        pred_scores = pred['scores']  # [N] confidence scores
        gt_boxes = target['boxes']    # [M,4] in (xmin,ymin,xmax,ymax) format
        gt_labels = target['labels']  # [M] class IDs
        
        print(f"DEBUG: Num predictions: {len(pred_boxes)}")
        print(f"DEBUG: Num ground truth: {len(gt_boxes)}")
        
        if len(gt_boxes) == 0:
            print("DEBUG: No ground truth boxes, skipping")
            continue
            
        if len(pred_boxes) == 0:
            print("DEBUG: No predictions after threshold, skipping")
            continue
        
        print(f"DEBUG: Prediction boxes (first 2):\n{pred_boxes[:2]}")
        print(f"DEBUG: Prediction labels (first 2): {pred_labels[:2]}")
        print(f"DEBUG: Prediction scores (first 2): {pred_scores[:2]}")
        print(f"DEBUG: Ground truth boxes (first 2):\n{gt_boxes[:2]}")
        print(f"DEBUG: Ground truth labels (first 2): {gt_labels[:2]}")
            
        # Calculate IoU matrix
        ious = box_iou(pred_boxes, gt_boxes)  # [N,M]
        print(f"DEBUG: IoU matrix shape: {ious.shape}")
        if ious.numel() > 0:
            print(f"DEBUG: Sample IoUs (first row): {ious[0, :5] if ious.shape[0] > 0 else 'N/A'}")
        
        # For each ground truth box, find the best matching prediction
        max_ious_per_gt, matched_pred_indices = ious.max(0)
        print(f"DEBUG: Max IoUs per GT: {max_ious_per_gt}")
        print(f"DEBUG: Matched pred indices: {matched_pred_indices}")
        
        # Count correct detections (IoU > threshold and correct class)
        correct = (max_ious_per_gt > iou_threshold) & (pred_labels[matched_pred_indices] == gt_labels)
        correct_for_this_image = correct.sum().item()
        print(f"DEBUG: Correct detections for this image: {correct_for_this_image}")
        
        total_correct += correct_for_this_image
        total_targets += len(gt_boxes)
    
    print(f"\nDEBUG: Total correct: {total_correct}, Total targets: {total_targets}")
    return total_correct, total_targets

def run_robustness_tests(model, model_name, val_loader, device):
    """Run all robustness tests for a single model"""
    print(f"\nTesting {model_name}...")
    evaluator = RobustnessEvaluator(model, device)
    results = {}
    
    # Create a dataset for robustness testing
    class RobustnessDataset(torch.utils.data.Dataset):
        def __init__(self, images, targets):
            self.images = images
            self.targets = targets
            
        def __len__(self):
            return len(self.images)
            
        def __getitem__(self, idx):
            return self.images[idx], self.targets[idx]
    
    # Weather robustness
    print(f"\nTesting Weather Robustness for {model_name}...")
    for images, targets in val_loader:
        try:
            # Convert targets to appropriate format
            processed_targets = []
            for t in targets:
                processed_target = prepare_targets(t)
                if processed_target['labels'].numel() > 0:  # Only include if there are valid labels
                    processed_targets.append(processed_target)
            
            if not processed_targets:  # Skip if no valid targets
                print("Warning: No valid targets found in batch")
                continue
            
            # Move tensors to device
            images = [img.to(device) for img in images]
            processed_targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                                for k, v in t.items()} for t in processed_targets]
            
            # Create dataset and dataloader for robustness testing
            robustness_dataset = RobustnessDataset(images, processed_targets)
            robustness_loader = torch.utils.data.DataLoader(
                robustness_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: tuple(zip(*x))
            )
            
            def evaluate_weather_robustness_wrapper(dataloader):
                print("Evaluating weather robustness...")
                model.eval()
                
                conditions = [
                    ('clear', lambda x: x),  # No augmentation
                    ('light_rain', lambda x: evaluator.weather_aug.add_rain(x, 0.2)),
                    ('heavy_rain', lambda x: evaluator.weather_aug.add_rain(x, 0.5)),
                    ('light_fog', lambda x: evaluator.weather_aug.add_fog(x, 0.3)),
                    ('heavy_fog', lambda x: evaluator.weather_aug.add_fog(x, 0.6)),
                    ('low_light', lambda x: evaluator.weather_aug.adjust_lighting(x, 0.3, 1.0)),
                    ('bright_light', lambda x: evaluator.weather_aug.adjust_lighting(x, 1.7, 1.0)),
                    ('motion_blur', lambda x: evaluator.weather_aug.add_motion_blur(x, 15, 45))
                ]
                
                results = {}
                for condition_name, augmentation_fn in conditions:
                    print(f"Testing {condition_name}...")
                    total_correct = 0
                    total_targets = 0
                    
                    with torch.no_grad():
                        for images, targets in tqdm(dataloader, desc=f"Testing {condition_name}"):
                            try:
                                # Apply weather condition
                                augmented_images = []
                                for img in images:
                                    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                    aug_img = augmentation_fn(img_np)
                                    aug_img = torch.from_numpy(aug_img.astype(np.float32) / 255.0).permute(2, 0, 1)
                                    augmented_images.append(aug_img)
                                
                                augmented_batch = torch.stack(augmented_images).to(device)
                                
                                # Forward pass
                                outputs = model(augmented_batch)
                                
                                # Calculate accuracy
                                correct, total = calculate_detection_accuracy(outputs, targets)
                                total_correct += correct
                                total_targets += total
                                
                            except Exception as e:
                                print(f"Error processing batch: {str(e)}")
                                continue
                    
                    if total_targets > 0:
                        accuracy = 100 * total_correct / total_targets
                    else:
                        accuracy = 0
                    
                    results[condition_name] = {
                        'accuracy': accuracy,
                        'total_targets': total_targets,
                        'correct_detections': total_correct
                    }
                    print(f"{condition_name}: {accuracy:.2f}% accuracy")
                
                return results
            
            results['weather'] = evaluate_weather_robustness_wrapper(robustness_loader)
            break
        except Exception as e:
            print(f"Error in weather robustness test: {str(e)}")
            continue
    
    # Adversarial robustness
    print(f"\nTesting Adversarial Robustness for {model_name}...")
    for images, targets in val_loader:
        try:
            # Convert targets to appropriate format
            processed_targets = []
            for t in targets:
                processed_target = prepare_targets(t)
                if processed_target['labels'].numel() > 0:  # Only include if there are valid labels
                    processed_targets.append(processed_target)
            
            if not processed_targets:  # Skip if no valid targets
                print("Warning: No valid targets found in batch")
                continue
            
            # Move tensors to device
            images = [img.to(device) for img in images]
            processed_targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                                for k, v in t.items()} for t in processed_targets]
            
            # Create dataset and dataloader for robustness testing
            robustness_dataset = RobustnessDataset(images, processed_targets)
            robustness_loader = torch.utils.data.DataLoader(
                robustness_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: tuple(zip(*x))
            )
            
            def evaluate_adversarial_robustness_wrapper(dataloader):
                print("Evaluating adversarial robustness...")
                model.eval()
                
                attacks = [
                    ('clean', lambda m, x, y: x),
                    ('fgsm_0.01', lambda m, x, y: evaluator.adversarial.fgsm_attack(m, x, y, 0.01)),
                    ('fgsm_0.03', lambda m, x, y: evaluator.adversarial.fgsm_attack(m, x, y, 0.03)),
                    ('pgd_0.01', lambda m, x, y: evaluator.adversarial.pgd_attack(m, x, y, 0.01)),
                    ('pgd_0.03', lambda m, x, y: evaluator.adversarial.pgd_attack(m, x, y, 0.03))
                ]
                
                results = {}
                for attack_name, attack_fn in attacks:
                    print(f"Testing {attack_name}...")
                    total_correct = 0
                    total_targets = 0
                    
                    # Loop for adversarial example generation and evaluation
                    for images, targets_tuple in tqdm(dataloader, desc=f"Testing {attack_name}"):
                        try:
                            # Ensure images and targets are on the correct device
                            # images is a tuple of tensors, targets_tuple is a tuple of dicts
                            image_batch = torch.stack(images).to(device)
                            # Adversarial attacks now expect a single target dict, and robustness_loader has batch_size=1
                            single_target_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets_tuple[0].items()}

                            # Generate adversarial examples (gradient calculation happens here)
                            # Model should be in eval mode for consistency with how attacks are typically performed
                            model.eval() # Ensure model is in eval mode before attack
                            adversarial_batch = attack_fn(model, image_batch, single_target_dict)
                            
                            # Evaluate the adversarial examples
                            with torch.no_grad():
                                model.eval() # Ensure model is in eval mode for inference
                                outputs = model(adversarial_batch)
                                
                                # Calculate accuracy (targets_tuple is used here as calculate_detection_accuracy expects a list/tuple of target dicts)
                                correct, total = calculate_detection_accuracy(outputs, targets_tuple)
                                total_correct += correct
                                total_targets += total
                                
                        except Exception as e:
                            print(f"Error processing batch: {str(e)}")
                            continue
                    
                    if total_targets > 0:
                        accuracy = 100 * total_correct / total_targets
                    else:
                        accuracy = 0
                    
                    results[attack_name] = {
                        'accuracy': accuracy,
                        'total_targets': total_targets,
                        'correct_detections': total_correct
                    }
                    print(f"{attack_name}: {accuracy:.2f}% accuracy")
                
                return results
            
            results['adversarial'] = evaluate_adversarial_robustness_wrapper(robustness_loader)
            break
        except Exception as e:
            print(f"Error in adversarial robustness test: {str(e)}")
            continue
    
    # Occlusion robustness
    print(f"\nTesting Occlusion Robustness for {model_name}...")
    for images, targets in val_loader:
        try:
            # Convert targets to appropriate format
            processed_targets = []
            for t in targets:
                processed_target = prepare_targets(t)
                if processed_target['labels'].numel() > 0:  # Only include if there are valid labels
                    processed_targets.append(processed_target)
            
            if not processed_targets:  # Skip if no valid targets
                print("Warning: No valid targets found in batch")
                continue
            
            # Move tensors to device
            images = [img.to(device) for img in images]
            processed_targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                                for k, v in t.items()} for t in processed_targets]
            
            # Create dataset and dataloader for robustness testing
            robustness_dataset = RobustnessDataset(images, processed_targets)
            robustness_loader = torch.utils.data.DataLoader(
                robustness_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: tuple(zip(*x))
            )
            
            def evaluate_occlusion_robustness_wrapper(dataloader):
                print("Evaluating occlusion robustness...")
                model.eval()
                
                occlusion_types = [
                    ('no_occlusion', lambda x: x),
                    ('light_occlusion', lambda x: evaluator.occlusion.random_occlusion(x, 2, 30)),
                    ('medium_occlusion', lambda x: evaluator.occlusion.random_occlusion(x, 3, 50)),
                    ('heavy_occlusion', lambda x: evaluator.occlusion.random_occlusion(x, 5, 70))
                ]
                
                results = {}
                for occlusion_name, occlusion_fn in occlusion_types:
                    print(f"Testing {occlusion_name}...")
                    total_correct = 0
                    total_targets = 0
                    
                    with torch.no_grad():
                        for images, targets in tqdm(dataloader, desc=f"Testing {occlusion_name}"):
                            try:
                                # Apply occlusion
                                occluded_images = []
                                for img in images:
                                    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                    occ_img = occlusion_fn(img_np)
                                    occ_img = torch.from_numpy(occ_img.astype(np.float32) / 255.0).permute(2, 0, 1)
                                    occluded_images.append(occ_img)
                                
                                occluded_batch = torch.stack(occluded_images).to(device)
                                
                                # Forward pass
                                outputs = model(occluded_batch)
                                
                                # Calculate accuracy
                                correct, total = calculate_detection_accuracy(outputs, targets)
                                total_correct += correct
                                total_targets += total
                                
                            except Exception as e:
                                print(f"Error processing batch: {str(e)}")
                                continue
                    
                    if total_targets > 0:
                        accuracy = 100 * total_correct / total_targets
                    else:
                        accuracy = 0
                    
                    results[occlusion_name] = {
                        'accuracy': accuracy,
                        'total_targets': total_targets,
                        'correct_detections': total_correct
                    }
                    print(f"{occlusion_name}: {accuracy:.2f}% accuracy")
                
                return results
            
            results['occlusion'] = evaluate_occlusion_robustness_wrapper(robustness_loader)
            break
        except Exception as e:
            print(f"Error in occlusion robustness test: {str(e)}")
            continue
    
    # Noise robustness
    print(f"\nTesting Noise Robustness for {model_name}...")
    for images, targets in val_loader:
        try:
            # Convert targets to appropriate format
            processed_targets = []
            for t in targets:
                processed_target = prepare_targets(t)
                if processed_target['labels'].numel() > 0:  # Only include if there are valid labels
                    processed_targets.append(processed_target)
            
            if not processed_targets:  # Skip if no valid targets
                print("Warning: No valid targets found in batch")
                continue
            
            # Move tensors to device
            images = [img.to(device) for img in images]
            processed_targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                                for k, v in t.items()} for t in processed_targets]
            
            # Create dataset and dataloader for robustness testing
            robustness_dataset = RobustnessDataset(images, processed_targets)
            robustness_loader = torch.utils.data.DataLoader(
                robustness_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: tuple(zip(*x))
            )
            
            def evaluate_noise_robustness_wrapper(dataloader):
                print("Evaluating noise robustness...")
                model.eval()
                
                noise_types = [
                    ('clean', 0.0, 'gaussian'),
                    ('gaussian_0.01', 0.01, 'gaussian'),
                    ('gaussian_0.05', 0.05, 'gaussian'),
                    ('gaussian_0.1', 0.1, 'gaussian'),
                    ('salt_pepper_0.01', 0.01, 'salt_pepper'),
                    ('salt_pepper_0.05', 0.05, 'salt_pepper')
                ]
                
                results = {}
                for noise_name, noise_level, noise_type in noise_types:
                    print(f"Testing {noise_name}...")
                    total_correct = 0
                    total_targets = 0
                    
                    with torch.no_grad():
                        for images, targets in tqdm(dataloader, desc=f"Testing {noise_name}"):
                            try:
                                # Apply noise
                                noisy_images = []
                                for img in images:
                                    if noise_level > 0:
                                        if noise_type == 'gaussian':
                                            noise = torch.randn_like(img) * noise_level
                                            noisy_img = torch.clamp(img + noise, 0, 1)
                                        elif noise_type == 'salt_pepper':
                                            noisy_img = img.clone()
                                            salt_pepper_mask = torch.rand_like(img) < noise_level
                                            salt_mask = torch.rand_like(img) < 0.5
                                            noisy_img[salt_pepper_mask & salt_mask] = 1.0
                                            noisy_img[salt_pepper_mask & ~salt_mask] = 0.0
                                    else:
                                        noisy_img = img
                                    noisy_images.append(noisy_img)
                                
                                noisy_batch = torch.stack(noisy_images).to(device)
                                
                                # Forward pass
                                outputs = model(noisy_batch)
                                
                                # Calculate accuracy
                                correct, total = calculate_detection_accuracy(outputs, targets)
                                total_correct += correct
                                total_targets += total
                                
                            except Exception as e:
                                print(f"Error processing batch: {str(e)}")
                                continue
                    
                    if total_targets > 0:
                        accuracy = 100 * total_correct / total_targets
                    else:
                        accuracy = 0
                    
                    results[noise_name] = {
                        'accuracy': accuracy,
                        'total_targets': total_targets,
                        'correct_detections': total_correct
                    }
                    print(f"{noise_name}: {accuracy:.2f}% accuracy")
                
                return results
            
            results['noise'] = evaluate_noise_robustness_wrapper(robustness_loader)
            break
        except Exception as e:
            print(f"Error in noise robustness test: {str(e)}")
            continue
    
    # Generate visualizations
    print(f"\nCreating visualizations for {model_name}...")
    try:
        os.makedirs(f"robustness_plots_{model_name}", exist_ok=True)
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
    
    return results

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create validation dataset and dataloader
    val_dataset = COCODataset(
        'kitti_images',
        'kitti_annotations.json',
        transforms=get_transform(train=False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create checkpoint directories if they don't exist
    checkpoint_dirs = [
        './deformable_detr_checkpoints',
        './detr_checkpoints',
        './mask_rcnn_checkpoints'
    ]
    for dir_path in checkpoint_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Define models to test with their checkpoint paths
    models = {
        'deformable_detr': {
            'path': './deformable_detr_checkpoints/best_deformable_detr.pth',
            'required': True  # This is the main model we want to test
        },
        'detr': {
            'path': './detr_checkpoints/best_detr_model.pth',
            'required': False  # Optional comparison model
        },
        'mask_rcnn': {
            'path': './mask_rcnn_checkpoints/maskrcnn_best.pth',
            'required': False  # Optional comparison model
        }
    }
    
    # Run tests for each model
    all_results = {}
    for model_name, model_info in models.items():
        try:
            print(f"\nLoading {model_name}...")
            checkpoint_path = model_info['path']
            
            if not os.path.exists(checkpoint_path):
                if model_info['required']:
                    print(f"Error: Required model checkpoint not found at {checkpoint_path}")
                    return
                else:
                    print(f"Warning: Optional model checkpoint not found at {checkpoint_path}")
                    continue
                
            model = load_model(model_name, checkpoint_path)
            model = model.to(device)
            
            results = run_robustness_tests(model, model_name, val_loader, device)
            if results:  # Only add results if tests were successful
                all_results[model_name] = results
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            if model_info['required']:
                return
            continue
    
    # Generate comparison report only if we have results
    if all_results:
        print("\nGenerating comparison report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"robustness_comparison_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nRobustness evaluation completed!")
        print(f"Results saved in {report_path}")
        print("Visualizations saved in respective model directories")
    else:
        print("\nNo successful test results to report.")

if __name__ == "__main__":
    main() 