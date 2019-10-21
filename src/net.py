import torch
from torch import nn


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=9, stride=1,
                                     padding=0, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=4)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16,
                                     kernel_size=6, stride=1,
                                     padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=4)
        self.fc1 = torch.nn.Linear(16 * 6 * 18, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16 * 6 * 18)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
