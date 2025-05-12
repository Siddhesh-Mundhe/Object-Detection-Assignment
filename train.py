import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2  # Using MobileNet for faster training
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import os
from torchvision.datasets import VOCDetection
from torch.optim import SGD
import time
import random
import numpy as np

# VOC class names
VOC_CLASSES = [
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Step 1: Choose a lightweight backbone for faster training
def get_backbone():
    """Using MobileNetV2 as a faster lightweight backbone"""
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    return backbone

# Step 2: Create a simplified detection model
def create_detection_model(num_classes=21):
    """Create a simpler, faster object detection model"""
    backbone = get_backbone()
    
    # Freeze all backbone layers to speed up training
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Simplified anchor generator with fewer anchors
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),  # Reduced number of sizes
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create model with smaller input size for faster processing
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=300,  # Smaller image size
        max_size=400   # Smaller image size
    )
    
    return model

# Step 3: Create a tiny dataset for ultra-fast training
def get_tiny_dataset(root_dir, max_samples=50):
    """Load a very small subset of VOC for quick training"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((300, 300)),  # Resize to smaller dimensions
    ])
    
    # Try to load the dataset
    try:
        dataset = VOCDetection(
            root=root_dir, 
            year='2007', 
            image_set='train',
            download=True, 
            transform=transform
        )
        
        # Create a very small subset for quick training
        if len(dataset) > max_samples:
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = Subset(dataset, indices)
            
        # Split into even smaller train/val sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

# Step 4: Transform targets to the format expected by Faster R-CNN
def transform_target(target):
    """Convert VOC annotations to FasterRCNN format"""
    boxes = []
    labels = []
    
    objects = target['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]
    
    for obj in objects:
        bbox = obj['bndbox']
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])
        
        # Skip invalid boxes
        if xmin >= xmax or ymin >= ymax:
            continue
        
        boxes.append([xmin, ymin, xmax, ymax])
        
        # Get class name and index
        class_name = obj['name']
        class_idx = VOC_CLASSES.index(class_name) if class_name in VOC_CLASSES else 0
        labels.append(class_idx)
    
    # If no valid boxes, create a dummy box
    if len(boxes) == 0:
        boxes = [[0, 0, 1, 1]]
        labels = [0]  # Background class
    
    return {
        'boxes': torch.as_tensor(boxes, dtype=torch.float32),
        'labels': torch.as_tensor(labels, dtype=torch.int64),
        'image_id': torch.tensor([0])
    }

# Step 5: Custom collate function
def custom_collate_fn(batch):
    """Collate function for the dataloader"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(transform_target(target))
    
    return images, targets

# Step 6: Quick training function (1 epoch only)
def quick_train(model, dataloader, device):
    """Ultra-fast training for demonstration purposes"""
    model.train()
    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, 
        momentum=0.9
    )
    
    print(f"Starting quick training on {device}")
    start_time = time.time()
    
    # Just train for a set number of batches
    max_batches = 10
    for i, (images, targets) in enumerate(dataloader):
        if i >= max_batches:
            break
            
        try:
            # Process batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            optimizer.step()
            
            print(f"  Batch {i+1}/{max_batches}, Loss: {losses.item():.4f}")
            
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            continue
    
    train_time = time.time() - start_time
    print(f"Quick training completed in {train_time:.2f} seconds")
    
    return model

# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating simplified model with MobileNetV2 backbone")
    model = create_detection_model(num_classes=len(VOC_CLASSES))
    model.to(device)
    
    # Load tiny dataset
    print("Loading a very small subset of Pascal VOC dataset")
    root_dir = './VOCdevkit'
    train_dataset, val_dataset = get_tiny_dataset(root_dir, max_samples=20)  # Only 20 samples total
    
    if train_dataset:
        dataloader = DataLoader(
            train_dataset, 
            batch_size=2,  # Smaller batch size
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0  # No parallel loading to avoid issues
        )
        
        # Quick train the model (just a few batches)
        model = quick_train(model, dataloader, device)
        
        # Save the model
        model_save_path = 'quick_rcnn_model.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    else:
        print("Failed to load dataset. Using a pre-trained model instead.")
        # Load a pre-trained model from torchvision
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        model.to(device)
        torch.save(model.state_dict(), 'pretrained_rcnn_model.pth')
        print("Pre-trained model saved to pretrained_rcnn_model.pth")