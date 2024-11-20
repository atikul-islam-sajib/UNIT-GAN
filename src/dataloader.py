import os
import zipfile
import sys
import torch
import argparse
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split

sys.path.append("./src/")

from utils import config


class Loader:
    def __init__(
        self,
        dataset=None,
        image_size: int = 128,
        batch_size: int = 1,
        split_size: float = 0.20,
    ):
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.imageA = []
        self.imageB = []

    def unzip_folder(self):
        os.makedirs(config()["path"]["processed_path"], exist_ok=True)

        with zipfile.ZipFile(self.dataset, mode="r") as zip_ref:
            zip_ref.extractall(path=config()["path"]["processed_path"])

        print(f"""Unzip folder {config()["path"]["processed_path"]}""".capitalize())

    def split_dataset(self, X: list, y: list):
        if isinstance(X, list) and isinstance(y, list):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size, random_state=42
            )
            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

    def transforms(self, type: str = "image"):
        if type == "image":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((self.image_size, self.image_siz), Image.BICUBIC),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def features_extractor(self):
        dataset_path = os.path.join(config()["path"]["processed_path"], "dataset")

        images_path = os.path.join(dataset_path, "image")
        masks_path = os.path.join(dataset_path, "mask")

        for image in os.listdir(images_path):
            if image.endswith((".jpg", ".jpeg", ".png")) and (
                image in os.path.join(masks_path, image)
            ):
                image_path = os.path.join(images_path, image)
                mask_path = os.path.join(masks_path, image)

                self.imageA.append(image_path)
                self.imageB.append(mask_path)

        assert len(self.imageA) == len(self.imageB)

    def create_dataloader(self):
        pass

    @staticmethod
    def dataset_details():
        pass


if __name__ == "__main__":
    loader = Loader(dataset="./data/raw/dataset.zip")
    # loader.unzip_folder()
    loader.features_extractor()
