import os
import zipfile
import sys
import torch
import argparse
from PIL import Image
import torch.nn as nn
from torchvision import transforms

sys.path.append("./src/")

from utils import config

class Loader():
    def __init__(self, dataset=None, image_size: int = 128, batch_size: int = 1, split_size: float = 0.20):
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size
        
    def unzip_folder(self):
        os.makedirs(config()["path"]["processed_path"], exist_ok=True)
        
        with zipfile.ZipFile(self.dataset, mode='r') as zip_ref:
            zip_ref.extractall(path=config()["path"]["processed_path"])
            
        print(f"""Unzip folder {config()["path"]["processed_path"]}""".capitalize())
    
    def split_dataset(self):
        pass
    
    def transforms(self, type: str = "image"):
        if type == "image":
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_siz), Image.BICUBIC),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
    
    def features_extractor(self):
        pass
    
    def create_dataloader(self):
        pass
    
    @staticmethod
    def dataset_details():
        pass
    
if __name__ == "__main__":
    loader = Loader(dataset="./data/raw/dataset.zip")
    loader.unzip_folder()
