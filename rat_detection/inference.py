import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os
import json
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import transform

def draw_boxes(image, boxes, scores):
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(np.int32)  # 정수형으로 변환
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{score:.2f}", fill="red")
    return image

def load_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2  # 배경 클래스 + 객체 클래스
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def inference(model, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    for image_path in image_files:
        image_file = os.path.basename(image_path)
        
        # 이미지 열기 및 변환
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)[0]
        
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        image_with_boxes = draw_boxes(image, boxes, scores)
        
        relative_path = os.path.relpath(image_path, input_dir)
        output_image_path = os.path.join(output_dir, f"result_{relative_path}")
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # image_with_boxes.save(output_image_path)
        
        results = []
        for box, score in zip(boxes, scores):
            result = {
                'box': box.tolist(),
                'score': score.item()
            }
            results.append(result)
        
        output_json_path = os.path.join(output_dir, f"{os.path.splitext(relative_path)[0]}_bbox.json")
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)



input_dir =  ""
output_dir =   ""

model_path = 'saved_models/model_epoch_19.pth'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = load_model(model_path, device)
inference(model, input_dir, output_dir, device)
