import os
import zipfile
import sys
import cv2
import torch
import argparse
import traceback
import warnings
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.append("./src/")

from utils import config, dump

warnings.filterwarnings("ignore")


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
                    transforms.Resize(
                        (self.image_size, self.image_size), Image.BICUBIC
                    ),
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

        images_path = os.path.join(dataset_path, "X")
        masks_path = os.path.join(dataset_path, "y")

        for image in os.listdir(images_path):
            if image.endswith((".jpg", ".jpeg", ".png")) and (
                image in os.path.join(masks_path, image)
            ):
                image_path = os.path.join(images_path, image)
                mask_path = os.path.join(masks_path, image)

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Mask not found: {mask_path}")
                    continue

                X = cv2.imread(image_path)
                y = cv2.imread(mask_path)

                X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
                y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

                X = self.transforms(type="image")(Image.fromarray(X))
                y = self.transforms()(Image.fromarray(y))

                self.imageA.append(X)
                self.imageB.append(y)

        assert len(self.imageA) == len(self.imageB)

        try:
            return self.split_dataset(X=self.imageA, y=self.imageB)
        except AssertionError as e:
            print(f"Assertion error: {e}")
            sys.exit(1)

    def create_dataloader(self):
        try:
            dataset = self.features_extractor()

            train_dataloader = DataLoader(
                dataset=list(zip(dataset["X_train"], dataset["y_train"])),
                batch_size=self.batch_size,
                shuffle=True,
            )
            valid_dataloader = DataLoader(
                dataset=list(zip(dataset["X_test"], dataset["y_test"])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            for filename, dataloader in [
                ("train_dataloader", train_dataloader),
                ("valid_dataloader", valid_dataloader),
            ]:
                dump(
                    filename=os.path.join(
                        config()["path"]["processed_path"], filename + ".pkl"
                    ),
                    value=dataloader,
                )

            print(
                "Train and valid dataloader saved successfully in the folder {}".format(
                    config()["path"]["processed_path"]
                ).capitalize()
            )

        except AssertionError as e:
            print(f"Assertion error: {e}")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            sys.exit(1)

    @staticmethod
    def dataset_details():
        pass


if __name__ == "__main__":
    loader = Loader(dataset="./data/raw/dataset.zip")
    # loader.unzip_folder()
    loader.create_dataloader()
