import os
from PIL import Image
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

class kimchi_dataset(Dataset):
    def __init__(self, dir='dataset/train'):
        self.dir = dir
        self.classes = sorted(os.listdir(self.dir))
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.all_data = self._load_data()
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.all_data)
    
    def _load_data(self):
        all_data = []
        for class_dir in self.classes:
            class_path = os.path.join(self.dir, class_dir)
            class_id = self.class_to_idx[class_dir]
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                all_data.append((image_path, class_id))
        return all_data

    def __getitem__(self, idx):
        image_path, class_idx = self.all_data[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        # class_idx to one-hot vector
        target_vector = torch.zeros(self.num_classes)
        target_vector[class_idx] = 1
    
        return image, target_vector



# Check DataLoader
if __name__ == "__main__":
    dataset = kimchi_dataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for epoch in range(10):
        for batch_idx, (image, target) in tqdm(enumerate(train_loader, start=1)):
            print(f'\nbatch : {batch_idx}')
            image_shape = image.shape
            target_shape = target.shape
            print(f'image_shape : {image_shape}')
            print(f'target_shape : {target_shape}')
            if batch_idx == 3:
                break

        print(len(dataset))
        print(len(train_loader))
        break