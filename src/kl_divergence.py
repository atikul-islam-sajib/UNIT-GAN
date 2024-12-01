import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()
        self.name = "KL Divergence".title()
        
    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return torch.mean(torch.pow(x, 2))
        else:
            raise ValueError("X should be the type of torch.Tensor".capitalize())
        
if __name__ == "__main__":
    loss = KLDivergence()
    
    predicted = torch.randn((1, 64))
    
    print(loss(predicted))