import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# define the model using pytorch https://github.com/harryjdavies/Python1D_CNNs/blob/master/CCN1D_pytorch_activity.py

n_features=1 # 1 dimension
n_classes = 3
batch_size = 4

class ConvNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels = n_features, out_channels = 16, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(32,100),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(100,n_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out