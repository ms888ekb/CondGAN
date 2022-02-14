import torch
import torch.nn as nn
import numpy as np


class FCDiscriminator(nn.Module):
    def __init__(self, in_feat):
        super(FCDiscriminator, self).__init__()
        self.fc1 = nn.Linear(in_feat, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x