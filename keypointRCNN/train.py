import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from dataloader import get_dataloader
import torchvision.models as models
import torch.nn as nn

import torch.optim as optim

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetClassifier, self).__init__()
        # Pre-trained resnet18 모델 로드
        self.model = models.resnet50(pretrained=True)
        
        # 입력 이미지 크기 조정을 위해 첫 번째 레이어를 업데이트
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Fully connected layer 조정 (출력 클래스 수를 4로 설정)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

import numpy as np
import cv2

def crop_image_by_keypoints(image, keypoints, padding=50):

    keypoints = keypoints['keypoints'].cpu().numpy()[0]  # 텐서를 numpy로 변환하고 CPU로 이동

    # x, y 좌표를 정수형으로 변환
    x_coords = [int(pt[0]) for pt in keypoints]
    y_coords = [int(pt[1]) for pt in keypoints]

    # 최소와 최대 좌표 계산
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # 패딩을 추가한 좌표 설정 (여유 공간 추가)
    x_min_with_padding = x_min - padding
    x_max_with_padding = x_max + padding
    y_min_with_padding = y_min - padding
    y_max_with_padding = y_max + padding

    # 이미지 크기 확인
    image_h, image_w, _ = image.permute(1, 2, 0).cpu().numpy().shape

    # 좌표가 이미지 경계를 벗어나는지 확인
    need_padding = (x_min_with_padding < 0 or y_min_with_padding < 0 or 
                    x_max_with_padding >= image_w or y_max_with_padding >= image_h)

    # 텐서를 numpy 배열로 변환 (GPU에서 CPU로 이동하고 배열 변환)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    if need_padding:
        # 패딩을 추가해야 하는 경우
        pad_top = abs(min(y_min_with_padding, 0))
        pad_left = abs(min(x_min_with_padding, 0))
        pad_bottom = max(y_max_with_padding - image_h + 1, 0)
        pad_right = max(x_max_with_padding - image_w + 1, 0)

        # 이미지를 제로 패딩하여 확장
        image_np = np.pad(image_np, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

        # 패딩된 이미지에서 새로운 크롭 좌표 계산
        x_min_with_padding = max(x_min_with_padding, 0)
        y_min_with_padding = max(y_min_with_padding, 0)
        x_max_with_padding = min(x_max_with_padding, image_np.shape[1] - 1)
        y_max_with_padding = min(y_max_with_padding, image_np.shape[0] - 1)

    # 이미지 크롭
    cropped_image = image_np[y_min_with_padding:y_max_with_padding + 1, x_min_with_padding:x_max_with_padding + 1]

    # 이미지 저장 (디버깅용)
    cropped_image_png = (cropped_image * 255).astype(np.uint8)
    cv2.imwrite("./crop.png", cropped_image_png)

    # 이미지 크기 조정 및 반환
    output_size = (512, 512)
    resized_cropped_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

    resized_cropped_image_png = (resized_cropped_image * 255).astype(np.uint8)
    cv2.imwrite("./crop_resize.png", resized_cropped_image_png)

    return resized_cropped_image





def get_model(num_keypoints):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(in_features, num_keypoints)
    return model


# 원 그리기 함수
def draw_points_on_image(image, points, output_path):
    points = points[:, :2]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    
    for (x, y) in points:
        center_coordinates = (int(x), int(y))
        radius = 5
        color = (0, 255, 0)  # Green color in BGR
        thickness = -1  # Solid circle
        image = cv2.circle(image, center_coordinates, radius, color, thickness)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

def train_one_epoch(model, classifier_model, optimizer, optimizer_classifier, dataloader, device, epoch, phase='train', log_file=None):
    model.train() if phase == 'train' else model.eval()
    classifier_model.train() if phase == 'train' else classifier_model.eval()
    total_loss_rcnn = 0.0
    total_loss_cls = 0.0
    data_loader = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}")
    val_total_correct = 0
    val_total_samples = 0

    # 특정 클래스에 weight를 주기위함
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)  # 생성 시 GPU로 바로 이동


    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        keypointrcnn_targets = [{k: v.to(device) for k, v in t.items() if k != 'status'} for t in targets]
        classifier_targets = [{k: v.to(device) for k, v in t.items() if k == 'status'} for t in targets]
        keypoint_only_targets = [{k: v.to(device) for k, v in t.items() if k == 'keypoints'} for t in targets]

        if images is None or len(images) == 0:
            continue

        if phase == 'train':
            # if i > -1: continue #validation 확인용
            with torch.set_grad_enabled(True):
                loss_dict = model(images, keypointrcnn_targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss_rcnn += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                images_classifier_input = []

                images = torch.stack(images)

                for c in range(len(images)):
                    cropped_image = crop_image_by_keypoints(images[c], keypoint_only_targets[c])
                    cropped_image = cropped_image.transpose(2, 0, 1)
                    images_classifier_input.append(cropped_image)
                    
                images_classifier_input = torch.stack([torch.tensor(img) for img in images_classifier_input], dim=0).to(device)

                # Classifier로 keypoint와 이미지 전달 후 학습
                classifier_output = classifier_model(images_classifier_input)

                classifier_targets = torch.tensor([t['status'].item() for t in targets], dtype=torch.long).to(device)

                classifier_loss = nn.CrossEntropyLoss(weight = class_weights)(classifier_output, classifier_targets)

                print("====== train classifier output / target ======")
                print(classifier_output)
                print(classifier_targets)
                print("==================================")

                total_loss_cls += classifier_loss.item()
                
                # Classifier 손실 역전파
                optimizer_classifier.zero_grad() 
                classifier_loss.backward()
                optimizer_classifier.step()

        else:
            model.eval()
            classifier_model.eval()

            val_total_loss_rcnn = 0.0
            val_total_loss_cls = 0.0
            val_total_accuracy= 0.0

            with torch.no_grad():
                # train사용으로 강제적으로 ㅣoss 추출.
                images = list(image.to(device) for image in images)
                keypointrcnn_targets = [{k: v.to(device) for k, v in t.items() if k != 'status'} for t in targets]
                classifier_targets = torch.tensor([t['status'].item() for t in targets], dtype=torch.long).to(device)

                val_images_classifier_input = []
                for c in range(len(images)):
                    val_cropped_image = crop_image_by_keypoints(images[c], keypoint_only_targets[c])
                    val_cropped_image = val_cropped_image.transpose(2, 0, 1)
                    val_images_classifier_input.append(val_cropped_image)
                    
                val_images_classifier_input = torch.stack([torch.tensor(img) for img in val_images_classifier_input], dim=0).to(device)

                # Forward pass through the classifier model
                # 클래스별 가중치 설정 (예: 클래스 0에 2배 가중치, 클래스 1과 2는 기본 가중치 1)
                # class 1 = toe_off
                classifier_output = classifier_model(val_images_classifier_input)
                print("====== val classifier output / target ======")
                print(classifier_output)
                print(classifier_targets)
                print("==================================")

                classifier_loss = nn.CrossEntropyLoss(weight = class_weights)(classifier_output, classifier_targets)
                val_total_loss_cls += classifier_loss.item()

                # Accuracy calculation
                _, predicted_labels = torch.max(classifier_output, 1)  # 가장 높은 확률을 갖는 클래스 선택
                correct_predictions = (predicted_labels == classifier_targets).sum().item()  # 맞춘 개수 계산
                total_predictions = classifier_targets.size(0)  # 전체 개수
                # Accuracy 계산
                
                val_total_correct += correct_predictions
                val_total_samples += total_predictions



    if phase == 'train':
        avg_loss_rcnn= total_loss_rcnn / len(dataloader)
        avg_loss_cls = total_loss_cls / len(dataloader)
        print(f"{phase.capitalize()} Epoch: {epoch}, Average keypoint Loss: {avg_loss_rcnn}, Average classifier Loss: {avg_loss_cls}")

        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{phase.capitalize()} Epoch: {epoch}, TRAIN :Average keypoint Loss: {avg_loss_rcnn}, Average classifier Loss: {avg_loss_cls}\n")

        avg_loss = avg_loss_rcnn + avg_loss_cls
        return avg_loss
    else:
        # Validation 손실 계산
        avg_val_loss_rcnn = val_total_loss_rcnn / len(dataloader)
        avg_val_loss = avg_val_loss_rcnn
        avg_acc = val_total_correct / val_total_samples
        print("========= validation summary ============")
        print("val total correct :", val_total_correct)
        print("val total samples :", val_total_samples)

        print(f"{phase.capitalize()} Epoch: {epoch},  Average keypoint Loss: {avg_val_loss_rcnn},Classification Accuracy : {avg_acc}")
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{phase.capitalize()} Epoch: {epoch},  Average keypoint Loss: {avg_val_loss_rcnn}, Classification Accuracy : {avg_acc}\n")

        avg_val_loss = avg_val_loss_rcnn
        return avg_val_loss, avg_acc

def plot_losses(train_losses, val_losses, epoch):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(f'loss_plot_epoch.png')
    plt.close()

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image_dir = "/workspace/ds_Toe_off_angle/data/trainset/imagesTr"
    label_dir = "/workspace/ds_Toe_off_angle/data/trainset/labelsTr"
    bbox_dir ="/workspace/ds_Toe_off_angle/data/trainset/imagesTr_bbox"
    transform = T.Compose([T.ToTensor()])

    train_loader, valid_loader = get_dataloader(image_dir, label_dir, bbox_dir, transform, batch_size=4, shuffle=True, valid_split=0.1)

    num_keypoints = 3
    model = get_model(num_keypoints)
    classifier_model = ResNetClassifier(num_classes=4)

    model.to(device)
    classifier_model.to(device)

    params = [p for p in model.parameters() if p.requires_grad] + [p for p in classifier_model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Classifier optimizer
    optimizer_classifier = optim.Adam(classifier_model.parameters(), lr=1e-3)


    global num_epochs
    num_epochs = 50  # Assuming a total of 100 epochs
    log_file = 'training_log.txt'
    if not os.path.exists('pth_files'):
        os.makedirs('pth_files')
    with open(log_file, 'w') as f:
        f.write("Training and Validation Loss Log\n")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, classifier_model, optimizer, optimizer_classifier, train_loader, device, epoch, phase='train', log_file=log_file)
        val_loss, val_acc = train_one_epoch(model, classifier_model, optimizer,optimizer_classifier,  valid_loader, device, epoch, phase='val', log_file=log_file)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        plot_losses(train_losses, val_losses, epoch)


        torch.save(model.state_dict(), f'pth_files/keypoint_rcnn_epoch_{epoch+1}_{train_loss}.pth')
        torch.save(classifier_model.state_dict(), f'pth_files/classifier_epoch_{epoch+1}_{val_acc}.pth')
            
            
    torch.save(model.state_dict(), 'keypoint_rcnn_final.pth')
    torch.save(classifier_model.state_dict(), f'pth_files/classifier_final.pth')
if __name__ == "__main__":
    main()
