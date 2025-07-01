import os
import json
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, images_path, annotations_path, transform=None):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        annotation_path = os.path.join(self.annotations_path, img_name.replace('.png', '.json'))
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        bndbox = annotation['bndbox']
        boxes = torch.tensor([bndbox], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  # 모든 객체를 동일한 클래스로 설정
        
        if self.transform:
            image = self.transform(image)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return image, target

transform = transforms.Compose([
    transforms.ToTensor()
])
