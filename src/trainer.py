import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from helper import helper
from utils import config, dump, load, device_init, weight_init


class Trainer:
    def __init__(
        self,
        epochs: int = 500,
        lr: float = 2e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        momentum: float = 0.95,
        adam: bool = True,
        SGD: bool = False,
        device: str = "cuda",
        l1_regularization: bool = False,
        l2_regularization: float = False,
        elasticNet_regularization: bool = False,
        verbose: bool = True,
        mlFlow: bool = True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.adam = adam
        self.SGD = SGD
        self.device = device
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticNet_regularization = elasticNet_regularization
        self.verbose = verbose
        self.mlFlow = mlFlow

        self.init = helper(
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
        )
