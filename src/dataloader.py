import os
impirt sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

class Loader():
    def __int__(self, dataset: str = None, image_size: int = 128, batch_size: int = 1, split_size: float = 0.20):
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size
        
    def unzip_folder(self):
        pass
    
    def split_dataset(self):
        pass
    
    def transforms(self):
        pass
    
    def features_extractor(self):
        pass
    
    def create_dataloader(self):
        pass
    
    @staticmethod
    def dataset_details():
        pass
    
    
if __name__ == "__main__":
    pass