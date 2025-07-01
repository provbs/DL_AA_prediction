import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import CustomDataset, transform
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json

import os

def collate_fn(batch):
    return tuple(zip(*batch))

# 경로 설정
images_path = '../../dataset/train/detection/imagesTr'
annotations_path = '../../dataset/train/detection/labelsTr'
output_dir = './validation_results'
os.makedirs(output_dir, exist_ok=True)

# 모델 저장 경로 설정
model_save_path = './saved_models'
os.makedirs(model_save_path, exist_ok=True)

# 데이터셋 및 데이터로더 생성
dataset = CustomDataset(images_path, annotations_path, transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 모델 정의
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 배경 클래스 + 객체 클래스
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 20

train_losses = []
val_scores = []

def draw_boxes(image, boxes, scores):
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(np.int32)  # 정수형으로 변환
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    i = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # print("training")
        # print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        train_loss += losses.item()
        
        if i % 10 == 0:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item()}")
        i += 1
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 모델 평가
    model.eval()
    total_score = 0
    num_boxes = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)

            # 첫 번째 배치의 첫 번째 이미지에 대해 예측 결과를 시각화
            if len(outputs) > 0 : 
                image_np = images[0].cpu().numpy().transpose(1, 2, 0)
                image_np = (image_np * 255).astype(np.uint8)
                image_pil = Image.fromarray(image_np)
                boxes = outputs[0]['boxes'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy()
                image_with_boxes = draw_boxes(image_pil, boxes, scores)
                image_with_boxes.save(os.path.join(output_dir, f'val_epoch{epoch}.png'))

                # 예측 결과를 JSON 파일로 저장
                results = []
                for box, score in zip(boxes, scores):
                    result = {
                        'box': box.tolist(),
                        'score': score.item()
                    }
                    results.append(result)
                
                with open(os.path.join(output_dir, f'val_epoch{epoch}.json'), 'w') as f:
                    json.dump(results, f, indent=4)

            
            # 각 이미지에 대해 모든 예측의 scores의 평균을 구합니다.
            for output in outputs:
                scores = output['scores']
                total_score += scores.sum().item()
                num_boxes += scores.size(0)
    
    avg_score = total_score / num_boxes if num_boxes > 0 else 0
    val_scores.append(avg_score)
    
    print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Score: {avg_score}")

    # 모델 저장
    if epoch == num_epochs-1 : 
        torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pth'))
    
    # 그래프 업데이트
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 2), val_scores, label='Validation Score')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_plot.png')
    plt.close()
