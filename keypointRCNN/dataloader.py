import os
import json
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image

def collate_fn(batch):
    # None 값 제거 (continue와 동일한 효과)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        print("Warning: Collate function encountered an empty batch.")
        return [], []  # 빈 배치가 있을 경우 None으로 처리하지 않고 빈 튜플 반환

    return tuple(zip(*batch))


class KeypointDataset(Dataset):
    def __init__(self, image_dir, label_dir, bbox_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_file = self.image_files[idx]
            image_path = os.path.join(self.image_dir, image_file)
            
            # Extract the number part from the image file name
            label_number = image_file.split('_')[0]
            label_path = os.path.join(self.label_dir, f"{label_number}.json")
            bbox_path = os.path.join(self.bbox_dir, f"{os.path.splitext(image_file)[0]}_bbox.json")

            image = Image.open(image_path).convert("RGB")
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            with open(bbox_path, 'r') as f:
                bbox_data = json.load(f)

            keypoints = [
                label_data["point_foot"] + [1],
                label_data["point_ankle"] + [1],
                label_data["point_thigh"] + [1]
            ]

            # Fetch status from the JSON label file
            status = label_data["status"]
            if status == "initialcontact" : 
                status = 0
            elif status == "toeoff" : 
                status =1
            elif status == "midswing" : 
                status =2
            elif status == "midstance" : 
                status =3
            else : 
                return None

            # Select the box with the highest score
            best_bbox = max(bbox_data, key=lambda x: x["score"])["box"]

            # Ensure coordinates are within bounds
            for i in range(len(best_bbox)):
                if i % 2 == 0:  # x coordinate
                    best_bbox[i] = max(0, min(best_bbox[i], 1024))
                else:  # y coordinate
                    best_bbox[i] = max(0, min(best_bbox[i], 1024))

            for keypoint in keypoints:
                for i in range(len(keypoint) - 1):
                    if i % 2 == 0:  # x coordinate
                        keypoint[i] = max(0, min(keypoint[i], 1024))
                    else:  # y coordinate
                        keypoint[i] = max(0, min(keypoint[i], 1024))

            target = {}
            target["boxes"] = torch.as_tensor([best_bbox], dtype=torch.float32)
            target["labels"] = torch.ones((1,), dtype=torch.int64)
            target["keypoints"] = torch.as_tensor([keypoints], dtype=torch.float32)
            target["status"] = torch.as_tensor([status], dtype=torch.float32) 

            if self.transform:
                image = self.transform(image)

            return image, target
        
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {e}")
            return None

def get_dataloader(image_dir, label_dir, bbox_dir, transform=None, batch_size=4, shuffle=True, valid_split=0.1):
    dataset = KeypointDataset(image_dir, label_dir, bbox_dir, transform)
    dataset_size = len(dataset)
    valid_size = int(valid_split * dataset_size)
    train_size = dataset_size - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Save validation file names
    with open('validation_files.txt', 'w') as f:
        for idx in valid_dataset.indices:
            f.write(f"{dataset.image_files[idx]}\n")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader
