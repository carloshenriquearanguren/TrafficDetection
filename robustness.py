import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import json
import pandas as pd
from tqdm import tqdm
import wandb
from collections import defaultdict
import random
from typing import Optional, Tuple, List
import os
from deformable_detr_train import box_cxcywh_to_xyxy, generalized_box_iou

class WeatherAugmentation:
    """Weather and lighting condition augmentations for robustness testing"""
    
    @staticmethod
    def add_rain(image, intensity=0.3):
        """Add rain effect to image"""
        h, w, c = image.shape
        
        # Create rain drops
        num_drops = int(intensity * 1000)
        rain_drops = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(num_drops):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            length = random.randint(10, 30)
            
            # Draw rain line
            end_y = min(y + length, h-1)
            cv2.line(rain_drops, (x, y), (x, end_y), 255, 1)
        
        # Apply motion blur to rain
        kernel = np.zeros((15, 15))
        kernel[:, 7] = 1/15
        rain_drops = cv2.filter2D(rain_drops, -1, kernel)
        
        # Overlay rain on image
        rain_mask = rain_drops[..., np.newaxis] / 255.0
        rain_color = np.array([200, 200, 255])  # Slightly blue-white
        
        image_with_rain = image.copy().astype(np.float32)
        image_with_rain = (1 - rain_mask) * image_with_rain + rain_mask * rain_color
        
        return np.clip(image_with_rain, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_fog(image, intensity=0.5):
        """Add fog effect to image"""
        h, w = image.shape[:2]
        
        # Create fog mask with gradient
        fog_mask = np.ones((h, w), dtype=np.float32)
        
        # Add some randomness to fog density
        for _ in range(5):
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius = random.randint(50, 200)
            
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            fog_mask[mask] *= (0.5 + random.random() * 0.3)
        
        # Apply fog
        fog_color = np.array([240, 240, 240])  # Light gray
        fog_overlay = intensity * fog_mask[..., np.newaxis]
        
        image_with_fog = image.astype(np.float32)
        image_with_fog = (1 - fog_overlay) * image_with_fog + fog_overlay * fog_color
        
        return np.clip(image_with_fog, 0, 255).astype(np.uint8)
    
    @staticmethod
    def adjust_lighting(image, brightness_factor=1.0, contrast_factor=1.0):
        """Adjust brightness and contrast"""
        image = image.astype(np.float32)
        
        # Adjust brightness and contrast
        image = image * contrast_factor + (brightness_factor - 1) * 127.5
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_motion_blur(image, kernel_size=15, angle=0):
        """Add motion blur to simulate camera movement"""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate kernel for different angles
        if angle != 0:
            center = (kernel_size // 2, kernel_size // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred

class AdversarialAttacks:
    """Adversarial attack implementations for robustness testing"""
    
    @staticmethod
    def fgsm_attack(model, image: torch.Tensor, target: dict, epsilon=0.03):
        """Fast Gradient Sign Method attack"""
        image.requires_grad_(True)
        
        # Forward pass
        output = model(image)
        
        pred_logits = output['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = output['pred_boxes'][0]    # [num_queries, 4]
        
        target_labels = target['labels'] # [num_gt_boxes]
        target_boxes = target['boxes']   # [num_gt_boxes, 4]

        if target_labels.numel() == 0: # No ground truth boxes
            loss = torch.tensor(0.0, device=image.device, requires_grad=True)
        else:
            cost_class = -pred_logits.softmax(-1)[:, target_labels]
            cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes),
                                           box_cxcywh_to_xyxy(target_boxes))
            C = cost_bbox + cost_class + cost_giou 
            matched_query_indices = torch.argmin(C, dim=0) 
            cls_loss = F.cross_entropy(pred_logits[matched_query_indices], target_labels)
            box_loss = F.l1_loss(pred_boxes[matched_query_indices], target_boxes)
            loss = cls_loss + box_loss
        
        # Zero any existing gradients
        if image.grad is not None:
            image.grad.zero_()
            
        loss.backward()
        
        perturbed_image = image.clone() # Start with original image
        if image.grad is not None: # Check if gradient was computed
            data_grad = image.grad.data
            perturbed_image = image + epsilon * data_grad.sign()
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image.detach()
    
    @staticmethod
    def pgd_attack(model, image: torch.Tensor, target: dict, epsilon=0.03, alpha=0.01, num_steps=10):
        """Projected Gradient Descent attack"""
        original_image = image.clone().detach() 
        adv_image = image.clone().detach()
        
        for _ in range(num_steps):
            adv_image.requires_grad_(True)
            
            output = model(adv_image)
            
            pred_logits = output['pred_logits'][0]  
            pred_boxes = output['pred_boxes'][0]    
            
            target_labels = target['labels'] 
            target_boxes = target['boxes']   

            if target_labels.numel() == 0:
                loss = torch.tensor(0.0, device=adv_image.device, requires_grad=True)
            else:
                cost_class = -pred_logits.softmax(-1)[:, target_labels]
                cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes),
                                               box_cxcywh_to_xyxy(target_boxes))
                C = cost_bbox + cost_class + cost_giou
                matched_query_indices = torch.argmin(C, dim=0)
                
                cls_loss = F.cross_entropy(pred_logits[matched_query_indices], target_labels)
                box_loss = F.l1_loss(pred_boxes[matched_query_indices], target_boxes)
                loss = cls_loss + box_loss
            
            # Zero any existing gradients for adv_image
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            loss.backward()
            
            with torch.no_grad():
                if adv_image.grad is not None: # Check if gradient was computed
                    data_grad = adv_image.grad.data
                    adv_image = adv_image + alpha * data_grad.sign()
                    perturbation = torch.clamp(adv_image - original_image, -epsilon, epsilon)
                    adv_image = torch.clamp(original_image + perturbation, 0, 1)
                # If grad is None, adv_image remains unchanged for this step but is still clamped to original image +/- epsilon ball
                else:
                    perturbation = torch.clamp(adv_image - original_image, -epsilon, epsilon) # Ensure it stays within epsilon ball of original
                    adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image = adv_image.detach()
        
        return adv_image

class OcclusionTesting:
    """Test model robustness to occlusions"""
    
    @staticmethod
    def random_occlusion(image, num_patches=5, patch_size=50):
        """Add random rectangular occlusions"""
        h, w = image.shape[:2]
        occluded_image = image.copy()
        
        for _ in range(num_patches):
            # Random position and size
            x = random.randint(0, max(0, w - patch_size))
            y = random.randint(0, max(0, h - patch_size))
            
            # Random patch size variation
            patch_w = random.randint(patch_size//2, patch_size)
            patch_h = random.randint(patch_size//2, patch_size)
            
            # Occlude with random color or black
            if random.random() > 0.5:
                color = [random.randint(0, 255) for _ in range(3)]
            else:
                color = [0, 0, 0]  # Black
            
            occluded_image[y:y+patch_h, x:x+patch_w] = color
        
        return occluded_image
    
    @staticmethod
    def object_occlusion(image, masks, occlusion_ratio=0.3):
        """Occlude detected objects partially"""
        occluded_image = image.copy()
        
        for mask in masks:
            if random.random() < 0.5:  # 50% chance to occlude each object
                # Get object bounding box
                coords = np.where(mask > 0)
                if len(coords[0]) == 0:
                    continue
                
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Occlude part of the object
                occlusion_h = int((y_max - y_min) * occlusion_ratio)
                occlusion_w = int((x_max - x_min) * occlusion_ratio)
                
                start_y = random.randint(y_min, max(y_min, y_max - occlusion_h))
                start_x = random.randint(x_min, max(x_min, x_max - occlusion_w))
                
                occluded_image[start_y:start_y+occlusion_h, 
                              start_x:start_x+occlusion_w] = [128, 128, 128]
        
        return occluded_image

class AdversarialTrainer:
    """Adversarial training for improving model robustness"""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03, alpha: float = 0.01,
                 num_steps: int = 10, device: str = 'cuda'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = device
    
    def generate_adversarial_example(self, images: torch.Tensor, targets: dict) -> torch.Tensor:
        """Generate adversarial examples using PGD attack"""
        images = images.clone().detach().to(self.device)
        images.requires_grad = True
        
        # Initialize perturbation
        delta = torch.zeros_like(images).to(self.device)
        delta.uniform_(-self.epsilon, self.epsilon)
        
        for _ in range(self.num_steps):
            # Forward pass
            outputs = self.model(images + delta)
            loss = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            delta = delta + self.alpha * images.grad.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            
            # Zero gradients
            images.grad.zero_()
        
        return images + delta
    
    def compute_loss(self, outputs: dict, targets: dict) -> torch.Tensor:
        """Compute loss for adversarial training"""
        # Classification loss
        cls_loss = F.cross_entropy(outputs['pred_logits'], targets['labels'])
        
        # Box regression loss
        box_loss = F.smooth_l1_loss(outputs['pred_boxes'], targets['boxes'])
        
        return cls_loss + box_loss

class RobustnessEvaluator:
    """Comprehensive robustness evaluation suite"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.results = defaultdict(dict)
        
        # Weather augmentation
        self.weather_aug = WeatherAugmentation()
        self.adversarial = AdversarialAttacks()
        self.occlusion = OcclusionTesting()
    
    def evaluate_weather_robustness(self, dataloader, conditions=None):
        """Evaluate model performance under different weather conditions"""
        if conditions is None:
            conditions = [
                ('clear', lambda x: x),  # No augmentation
                ('light_rain', lambda x: self.weather_aug.add_rain(x, 0.2)),
                ('heavy_rain', lambda x: self.weather_aug.add_rain(x, 0.5)),
                ('light_fog', lambda x: self.weather_aug.add_fog(x, 0.3)),
                ('heavy_fog', lambda x: self.weather_aug.add_fog(x, 0.6)),
                ('low_light', lambda x: self.weather_aug.adjust_lighting(x, 0.3, 1.0)),
                ('bright_light', lambda x: self.weather_aug.adjust_lighting(x, 1.7, 1.0)),
                ('motion_blur', lambda x: self.weather_aug.add_motion_blur(x, 15, 45))
            ]
        
        print("Evaluating weather robustness...")
        self.model.eval()
        
        for condition_name, augmentation_fn in conditions:
            correct = 0
            total = 0
            condition_predictions = []
            condition_targets = []
            
            with torch.no_grad():
                for images, targets in tqdm(dataloader, desc=f"Testing {condition_name}"):
                    # Apply weather condition
                    augmented_images = []
                    for img in images:
                        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        aug_img = augmentation_fn(img_np)
                        aug_img = torch.from_numpy(aug_img.astype(np.float32) / 255.0).permute(2, 0, 1)
                        augmented_images.append(aug_img)
                    
                    augmented_batch = torch.stack(augmented_images).to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(augmented_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    condition_predictions.extend(predicted.cpu().numpy())
                    condition_targets.extend(targets.cpu().numpy())
            
            accuracy = 100 * correct / total
            self.results['weather'][condition_name] = {
                'accuracy': accuracy,
                'predictions': condition_predictions,
                'targets': condition_targets
            }
            print(f"{condition_name}: {accuracy:.2f}% accuracy")
    
    def evaluate_adversarial_robustness(self, dataloader, attacks=None):
        """Evaluate model robustness against adversarial attacks"""
        if attacks is None:
            attacks = [
                ('clean', lambda m, x, y: x),
                ('fgsm_0.01', lambda m, x, y: self.adversarial.fgsm_attack(m, x, y, 0.01)),
                ('fgsm_0.03', lambda m, x, y: self.adversarial.fgsm_attack(m, x, y, 0.03)),
                ('pgd_0.01', lambda m, x, y: self.adversarial.pgd_attack(m, x, y, 0.01)),
                ('pgd_0.03', lambda m, x, y: self.adversarial.pgd_attack(m, x, y, 0.03))
            ]
        
        print("Evaluating adversarial robustness...")
        
        for attack_name, attack_fn in attacks:
            correct = 0
            total = 0
            attack_predictions = []
            attack_targets = []
            
            for images, targets in tqdm(dataloader, desc=f"Testing {attack_name}"):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Generate adversarial examples
                adversarial_images = attack_fn(self.model, images, targets)
                
                with torch.no_grad():
                    outputs = self.model(adversarial_images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    attack_predictions.extend(predicted.cpu().numpy())
                    attack_targets.extend(targets.cpu().numpy())
            
            accuracy = 100 * correct / total
            self.results['adversarial'][attack_name] = {
                'accuracy': accuracy,
                'predictions': attack_predictions,
                'targets': attack_targets
            }
            print(f"{attack_name}: {accuracy:.2f}% accuracy")
    
    def evaluate_occlusion_robustness(self, dataloader, occlusion_types=None):
        """Evaluate model robustness to occlusions"""
        if occlusion_types is None:
            occlusion_types = [
                ('no_occlusion', lambda x: x),
                ('light_occlusion', lambda x: self.occlusion.random_occlusion(x, 2, 30)),
                ('medium_occlusion', lambda x: self.occlusion.random_occlusion(x, 3, 50)),
                ('heavy_occlusion', lambda x: self.occlusion.random_occlusion(x, 5, 70))
            ]
        
        print("Evaluating occlusion robustness...")
        self.model.eval()
        
        for occlusion_name, occlusion_fn in occlusion_types:
            correct = 0
            total = 0
            occlusion_predictions = []
            occlusion_targets = []
            
            with torch.no_grad():
                for images, targets in tqdm(dataloader, desc=f"Testing {occlusion_name}"):
                    # Apply occlusion
                    occluded_images = []
                    for img in images:
                        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        occ_img = occlusion_fn(img_np)
                        occ_img = torch.from_numpy(occ_img.astype(np.float32) / 255.0).permute(2, 0, 1)
                        occluded_images.append(occ_img)
                    
                    occluded_batch = torch.stack(occluded_images).to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(occluded_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    occlusion_predictions.extend(predicted.cpu().numpy())
                    occlusion_targets.extend(targets.cpu().numpy())
            
            accuracy = 100 * correct / total
            self.results['occlusion'][occlusion_name] = {
                'accuracy': accuracy,
                'predictions': occlusion_predictions,
                'targets': occlusion_targets
            }
            print(f"{occlusion_name}: {accuracy:.2f}% accuracy")
    
    def evaluate_noise_robustness(self, dataloader, noise_types=None):
        """Evaluate model robustness to different types of noise"""
        if noise_types is None:
            noise_types = [
                ('clean', 0.0, 'gaussian'),
                ('gaussian_0.01', 0.01, 'gaussian'),
                ('gaussian_0.05', 0.05, 'gaussian'),
                ('gaussian_0.1', 0.1, 'gaussian'),
                ('salt_pepper_0.01', 0.01, 'salt_pepper'),
                ('salt_pepper_0.05', 0.05, 'salt_pepper')
            ]
        
        print("Evaluating noise robustness...")
        self.model.eval()
        
        for noise_name, noise_level, noise_type in noise_types:
            correct = 0
            total = 0
            noise_predictions = []
            noise_targets = []
            
            with torch.no_grad():
                for images, targets in tqdm(dataloader, desc=f"Testing {noise_name}"):
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                    if noise_level > 0:
                        if noise_type == 'gaussian':
                            noise = torch.randn_like(images) * noise_level
                            noisy_images = torch.clamp(images + noise, 0, 1)
                        elif noise_type == 'salt_pepper':
                            noisy_images = images.clone()
                            salt_pepper_mask = torch.rand_like(images) < noise_level
                            salt_mask = torch.rand_like(images) < 0.5
                            noisy_images[salt_pepper_mask & salt_mask] = 1.0
                            noisy_images[salt_pepper_mask & ~salt_mask] = 0.0
                    else:
                        noisy_images = images
                    
                    outputs = self.model(noisy_images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    noise_predictions.extend(predicted.cpu().numpy())
                    noise_targets.extend(targets.cpu().numpy())
            
            accuracy = 100 * correct / total
            self.results['noise'][noise_name] = {
                'accuracy': accuracy,
                'predictions': noise_predictions,
                'targets': noise_targets
            }
            print(f"{noise_name}: {accuracy:.2f}% accuracy")
    
    def comprehensive_evaluation(self, dataloader):
        """Run all robustness tests"""
        print("Starting comprehensive robustness evaluation...")
        
        self.evaluate_weather_robustness(dataloader)
        self.evaluate_adversarial_robustness(dataloader)
        self.evaluate_occlusion_robustness(dataloader)
        self.evaluate_noise_robustness(dataloader)
        
        print("Comprehensive evaluation completed!")
    
    def generate_report(self, class_names=None, save_path='robustness_report.json'):
        """Generate comprehensive robustness report"""
        report = {
            'summary': {},
            'detailed_results': self.results,
            'recommendations': []
        }
        
        # Calculate summary statistics
        for test_type, test_results in self.results.items():
            accuracies = [result['accuracy'] for result in test_results.values()]
            report['summary'][test_type] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'num_conditions': len(accuracies)
            }
        
        # Generate recommendations
        if 'weather' in self.results:
            weather_accs = [r['accuracy'] for r in self.results['weather'].values()]
            if min(weather_accs) < 70:
                report['recommendations'].append(
                    "Weather robustness is low. Consider data augmentation with weather effects."
                )
        
        if 'adversarial' in self.results:
            adv_accs = [r['accuracy'] for r in self.results['adversarial'].values() 
                       if 'clean' not in r]
            if adv_accs and min(adv_accs) < 50:
                report['recommendations'].append(
                    "Adversarial robustness is low. Consider adversarial training."
                )
        
        if 'occlusion' in self.results:
            occ_accs = [r['accuracy'] for r in self.results['occlusion'].values()]
            if min(occ_accs) < 60:
                report['recommendations'].append(
                    "Occlusion robustness is low. Consider training with occlusion augmentation."
                )
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Robustness report saved to {save_path}")
        return report
    
    def visualize_results(self, save_dir='robustness_plots'):
        """Create visualizations of robustness results"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Plot accuracy comparison across all tests
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        test_types = list(self.results.keys())
        for i, test_type in enumerate(test_types):
            if i < len(axes):
                conditions = list(self.results[test_type].keys())
                accuracies = [self.results[test_type][cond]['accuracy'] for cond in conditions]
                
                axes[i].bar(range(len(conditions)), accuracies)
                axes[i].set_title(f'{test_type.title()} Robustness')
                axes[i].set_ylabel('Accuracy (%)')
                axes[i].set_xticks(range(len(conditions)))
                axes[i].set_xticklabels(conditions, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(test_types), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/robustness_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confusion matrices for each test type
        for test_type, test_results in self.results.items():
            for condition, results in test_results.items():
                if 'predictions' in results and 'targets' in results:
                    cm = confusion_matrix(results['targets'], results['predictions'])
                    
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {test_type.title()} - {condition}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.savefig(f'{save_dir}/cm_{test_type}_{condition}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"Visualizations saved to {save_dir}/")

# Example usage and utility functions
def setup_robustness_evaluation(model, test_dataloader, device='cuda'):
    """Setup and run comprehensive robustness evaluation"""
    evaluator = RobustnessEvaluator(model, device)
    
    # Run comprehensive evaluation
    evaluator.comprehensive_evaluation(test_dataloader)
    
    # Generate report and visualizations
    report = evaluator.generate_report()
    evaluator.visualize_results()
    
    return evaluator, report

def compare_models_robustness(models, model_names, test_dataloader, device='cuda'):
    """Compare robustness of multiple models"""
    all_results = {}
    
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        evaluator = RobustnessEvaluator(model, device)
        evaluator.comprehensive_evaluation(test_dataloader)
        all_results[name] = evaluator.results
    
    # Create comparison visualization
    comparison_data = []
    for model_name, results in all_results.items():
        for test_type, test_results in results.items():
            for condition, result in test_results.items():
                comparison_data.append({
                    'model': model_name,
                    'test_type': test_type,
                    'condition': condition,
                    'accuracy': result['accuracy']
                })
    
    df = pd.DataFrame(comparison_data)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    test_types = df['test_type'].unique()
    for i, test_type in enumerate(test_types):
        if i < len(axes):
            test_data = df[df['test_type'] == test_type]
            pivot_data = test_data.pivot(index='condition', columns='model', values='accuracy')
            
            pivot_data.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{test_type.title()} Robustness Comparison')
            axes[i].set_ylabel('Accuracy (%)')
            axes[i].legend(title='Model')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_results, df

# WandB integration for experiment tracking
def log_robustness_to_wandb(evaluator, run_name="robustness_evaluation"):
    """Log robustness results to Weights & Biases"""
    if not wandb.run:
        wandb.init(project="model_robustness", name=run_name)
    
    # Log summary metrics
    for test_type, test_results in evaluator.results.items():
        accuracies = [result['accuracy'] for result in test_results.values()]
        
        wandb.log({
            f"{test_type}_mean_accuracy": np.mean(accuracies),
            f"{test_type}_std_accuracy": np.std(accuracies),
            f"{test_type}_min_accuracy": np.min(accuracies),
            f"{test_type}_max_accuracy": np.max(accuracies)
        })
        
        # Log individual condition results
        for condition, result in test_results.items():
            wandb.log({f"{test_type}_{condition}_accuracy": result['accuracy']})
    
    print("Results logged to Weights & Biases")

def main():
    """Main function for robustness evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Model Robustness')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./robustness_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = torch.load(args.model_path)
    model.eval()
    
    # Create dataloader
    dataset = WaymoMultiModalDataset(args.data_dir, split='val')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize evaluator
    evaluator = RobustnessEvaluator(model, device=args.device)
    
    # Evaluate different types of robustness
    noise_results = evaluator.evaluate_noise_robustness(
        dataloader,
        noise_levels=[0.01, 0.02, 0.03, 0.04, 0.05]
    )
    
    blur_results = evaluator.evaluate_blur_robustness(
        dataloader,
        kernel_sizes=[3, 5, 7, 9, 11]
    )
    
    brightness_results = evaluator.evaluate_brightness_robustness(
        dataloader,
        brightness_levels=[0.5, 0.75, 1.25, 1.5]
    )
    
    # Combine results
    results = {
        'noise': noise_results,
        'blur': blur_results,
        'brightness': brightness_results
    }
    
    # Save results
    with open(os.path.join(args.output_dir, 'robustness_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Robustness evaluation completed!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()