"""
Simple implementation of Pytorch MLP for inference.
Code from:
https://github.com/itsmealves/mlp-pytorch/blob/master/main.py
"""

import torch
import numpy as np

from torch import optim, nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


    
if __name__ == '__main__':
    model = Classifier()
    sample = torch.ones((4, 4), dtype=torch.float32)
    with torch.no_grad():
        pred = model(sample)
