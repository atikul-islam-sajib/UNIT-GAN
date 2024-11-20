import os
import sys
import yaml
import torch
import torch.nn as nn

sys.path.append("./src/")

def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)