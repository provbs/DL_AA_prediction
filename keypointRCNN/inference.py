import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
from PIL import Image

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_file

def get_model(num_keypoints):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(in_features, num_keypoints)
    return model

# Bounding box와 keypoint를 그리는 함수
def draw_points_and_boxes_on_image(image, points, bboxes, output_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Draw keypoints
    points = points[:, :2]
    for (x, y) in points:
        center_coordinates = (int(x), int(y))
        radius = 5
        color = (0, 255, 0)  # Green color in BGR
        thickness = -1  # Solid circle
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
    
    # Draw bounding boxes
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # red color in BGR

    cv2.imwrite(output_path, image)  # Save the image with keypoints and bboxes directly, keeping the original size

def draw_no_points_just_save(image, output_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    cv2.imwrite(output_path, image)  # Save the image with keypoints directly, keeping the original size

def load_bbox(json_path):
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        bbox_data = json.load(f)
    
    if not isinstance(bbox_data, list) or len(bbox_data) == 0:
        return None
    
    # bbox가 score와 함께 저장되어 있는 것을 가정하고 score 0.9 이상인 것만 필터링
    filtered_bboxes = [item['box'] for item in bbox_data if item['score'] >= 0.989]

    if len(filtered_bboxes) == 0 : return None
    
    return filtered_bboxes

def inference(model, dataloader, device, output_dir, bbox_dir):
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('len of dataloader is', len(dataloader))
    
    with torch.no_grad():
        for i, (images, image_files) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            predictions = model(images)
            
            for idx, prediction in enumerate(predictions):

                # BBox JSON 파일 경로 설정 (bbox_dir 사용)
                bbox_json_path = os.path.join(bbox_dir, f"{os.path.splitext(image_files[idx])[0]}_bbox.json")
                bboxes = load_bbox(bbox_json_path)

                if bboxes is None:
                    img_np = images[idx].permute(1, 2, 0).cpu().numpy() * 255
                    img_np = img_np.astype(np.uint8)
                    output_image_path = os.path.join(output_dir, f"output_{image_files[idx]}")
                    draw_no_points_just_save(img_np, output_image_path)
                    continue

                img_np = images[idx].permute(1, 2, 0).cpu().numpy() * 255
                img_np = img_np.astype(np.uint8)

                # Keypoints 처리
                scores = prediction['scores'].cpu().numpy()
                keypoints = prediction['keypoints'].cpu().numpy()

                # score가 가장 높은 keypoint 선택
                best_index = np.argmax(scores)
                output_points = keypoints[best_index]

                output_image_path = os.path.join(output_dir, f"output_{image_files[idx]}")
                draw_points_and_boxes_on_image(img_np, output_points, bboxes, output_image_path)

                output_json_path = os.path.join(output_dir, f"output_{image_files[idx]}.json")
                keypoints_list = output_points.tolist()
                with open(output_json_path, 'w') as f:
                    json.dump({"keypoints": keypoints_list, "boxes": bboxes}, f)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    input_dir = '../../dataset/testset/240809/Female_Naive_#1_to_png'
    output_dir = '../../dataset/testset/240809/inference/Female_Naive_#1_to_png'
    bbox_dir = '../../dataset/testset/240809/Female_Naive_#1_to_png_bbox'  # bbox JSON 파일이 위치한 디렉토리
    model_path = 'pth_files/keypoint_rcnn_epoch_30.pth'
    
    transform = T.Compose([T.ToTensor()])
    dataset = InferenceDataset(input_dir, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_keypoints = 3
    model = get_model(num_keypoints)
    model.to(device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    
    inference(model, dataloader, device, output_dir, bbox_dir)

if __name__ == "__main__":
    main()
