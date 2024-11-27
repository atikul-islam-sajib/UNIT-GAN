import os
import zipfile
import sys
import cv2
import torch
import argparse
import traceback
import warnings
import pandas as pd
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.append("./src/")

from utils import config, dump, load

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
                batch_size=self.batch_size * 16,
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
    def display_images():
        processed_path = os.path.join(config()["path"]["processed_path"])
        if os.path.exists(processed_path):
            train_dataloder = os.path.join(processed_path, "train_dataloader.pkl")
            valid_dataloder = os.path.join(processed_path, "valid_dataloader.pkl")

            train_dataloder = load(filename=train_dataloder)
            valid_dataloder = load(filename=valid_dataloder)

            valid_X, valid_Y = next(iter(valid_dataloder))

            num_of_rows = valid_X.size(0) // 2
            num_of_cols = valid_X.size(0) // num_of_rows

            plt.figure(figsize=(10, 20))

            for index, X in enumerate(valid_X):
                X = X.squeeze().permute(2, 1, 0).detach().cpu().numpy()
                y = valid_Y[index].squeeze().permute(2, 1, 0).detach().cpu().numpy()

                X = (X - X.min()) / (X.max() - X.min())
                y = (y - y.min()) / (y.max() - y.min())

                plt.subplot(2 * num_of_rows, 2 * num_of_cols, 2 * index + 1)
                plt.imshow(X)
                plt.title("X")
                plt.axis("off")

                plt.subplot(2 * num_of_rows, 2 * num_of_cols, 2 * index + 2)
                plt.imshow(y)
                plt.title("Y")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(config()["path"]["files_path"], "images.png"))
            plt.show()

    @staticmethod
    def dataset_details():
        processed_path = os.path.join(config()["path"]["processed_path"])
        if os.path.exists(processed_path):
            train_dataloder = os.path.join(processed_path, "train_dataloader.pkl")
            valid_dataloder = os.path.join(processed_path, "valid_dataloader.pkl")

            train_dataloder = load(filename=train_dataloder)
            valid_dataloder = load(filename=valid_dataloder)

            train_X, train_Y = next(iter(train_dataloder))
            valid_X, valid_Y = next(iter(valid_dataloder))

            pd.DataFrame(
                {
                    "Train X Shape": str(train_X.size()),
                    "Train Y Shape": str(train_Y.size()),
                    "Valid X Shape": str(valid_X.size()),
                    "Valid Y Shape": str(valid_Y.size()),
                    "total_train_dataset": sum(X.size(0) for X, _ in train_dataloder),
                    "total_valid_dataset": sum(X.size(0) for X, _ in valid_dataloder),
                    "total_dataset": (sum(X.size(0) for X, _ in train_dataloder))
                    + sum(Y.size(0) for Y, _ in valid_dataloder),
                },
                index=["Dataset Details"],
            ).T.to_csv(
                os.path.join(config()["path"]["files_path"], "dataset_details.csv")
            )

        else:
            print(f"Folder {processed_path} does not exist".capitalize())
            sys.exit(1)


if __name__ == "__main__":
    loader = Loader(dataset="./data/raw/dataset.zip")
    # loader.unzip_folder()
    loader.create_dataloader()

    Loader.dataset_details()
    Loader.display_images()
