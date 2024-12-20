import os
import sys
import yaml
import torch
import joblib
from tqdm import tqdm
import torch.nn as nn

sys.path.append("./src/")


def dump(filename=None, value=None):
    if (filename is not None) and (value is not None):
        joblib.dump(value=value, filename=filename)
    else:
        raise ValueError("Could not dump file".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)
    else:
        raise ValueError("Could not load file".capitalize())


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


def device_init(device: str = "cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")


def weight_init(m):
    classname = m.__class__.__name
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def clean_folders():
    train_models = config()["path"]["train_models"]
    best_model = config()["path"]["best_model"]
    metrics_path = config()["path"]["metrics_path"]
    train_images = config()["path"]["train_images"]
    test_image = config()["path"]["test_image"]

    for folder in tqdm(
        [train_images, test_image, train_models, best_model, metrics_path]
    ):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error occurred while cleaning folder: {folder}")
                print(f"Error: {e}")

        print("All files have been deleted.".capitalize())
