import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# VOC class names
VOC_CLASSES = [
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Use distinct colors for visualization
COLORS = np.random.uniform(0, 255, size=(len(VOC_CLASSES), 3))

# Function to create the same model architecture as the quick training script
def create_mobilenet_model():
    # Create the MobileNetV2 backbone
    backbone = mobilenet_v2(pretrained=False).features
    backbone.out_channels = 1280
    
    # Create anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create the model
    model = FasterRCNN(
        backbone,
        num_classes=len(VOC_CLASSES),
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=300,
        max_size=400
    )
    
    return model

def detect_objects(model, image_path, device, confidence_threshold=0.5):
    """Run object detection on an image"""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
        
        # Transform the image
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img_tensor = transform(image).to(device)
        
        # Run inference
        with torch.no_grad():
            prediction = model([img_tensor])
        
        # Extract predictions
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        keep = scores >= confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        print(f"Detected {len(boxes)} objects with confidence threshold {confidence_threshold}")
        
        return image, boxes, scores, labels
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, [], [], []

def draw_boxes(image, boxes, scores, labels):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
        box = [int(i) for i in box]
        x1, y1, x2, y2 = box

        # Use RED box and WHITE label for high contrast
        class_name = VOC_CLASSES[label_idx]
        label_text = f"{class_name}: {score:.2f}"

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)

        if hasattr(font, 'getbbox'):
            text_bbox = font.getbbox(label_text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        else:
            text_width, text_height = 100, 20

        draw.rectangle(
            [(x1, y1 - text_height - 4), (x1 + text_width, y1)],
            fill="red"
        )
        draw.text((x1, y1 - text_height - 4), label_text, fill="white", font=font)

        print(f"  Detection {i+1}: {class_name} (Score: {score:.2f}, Box: {box})")

    return image


def main():
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python simple_test.py <image_path> <model_path> [confidence_threshold]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    print(f"Running detection on: {image_path}")
    print(f"Using model: {model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_mobilenet_model()
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to a pre-trained model")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    
    model.to(device)
    model.eval()
    
    # Run detection
    image, boxes, scores, labels = detect_objects(model, image_path, device, confidence_threshold)
    
    if image is None:
        print("Failed to process the image.")
        sys.exit(1)
    
    # Draw boxes
    result_image = draw_boxes(image.copy(), boxes, scores, labels)
    
    # Save result
    base_name = os.path.basename(image_path)
    result_path = f"result_{base_name}"
    result_image.save(result_path)
    print(f"Result saved to {result_path}")
    
    # Display result
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(result_image))
    plt.axis('off')
    plt.title(f"Object Detection Results")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()