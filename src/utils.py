import os
import sys
import yaml
import torch
import joblib
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
