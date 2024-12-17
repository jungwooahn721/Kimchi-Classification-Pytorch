from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader

#from data_loader import kimchi_dataset # error??

class KimchiDataLoader(BaseDataLoader):
    def __init__(self, root_dir='dataset', mode='train', batch_size=256, shuffle=True, validation_split=0.0, num_workers=1):
        self.root_dir = root_dir
        self.mode = mode
        self.dir = os.path.join(self.root_dir, self.mode)
        self.dataset = kimchi_dataset(dir=self.dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
##########
#kimchi_dataset (custom dataset)

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
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),

        ])

    def __len__(self):
        #total_img = 0
        #for i in range(len(self.classes)):
        #    class_path = os.path.join(self.dir, self.classes[i])
        #    total_img += len(os.listdir(class_path))
        #return total_img
        return len(self.all_data)
    
    def _load_data(self):
        all_data = []
        for class_dir in self.classes:
            class_path = os.path.join(self.dir, class_dir)
            class_id = self.class_to_idx[class_dir]
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = Image.open(image_path)
                all_data.append((image, class_id))
        return all_data

    def __getitem__(self, idx):
        image, class_idx = self.all_data[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # class_idx to one-hot vector
        target_vector = torch.zeros(self.num_classes)
        target_vector[class_idx] = 1
    
        return image, target_vector
        
        