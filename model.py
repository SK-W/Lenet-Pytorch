# -*- coding: utf-8 -*-
# Time : 2022/5/10 18:05
# Author : sk-w
# Email : 15734082105@163.com
# File : net
# Project : LeNet_Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as f


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        # corresponding input size
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # torch.Size([4, 3, 32, 32])
        x = self.conv1(x)
        # torch.Size([4, 16, 28, 28])
        x = f.relu(x)
        x = self.pool1(x)
        # torch.Size([4, 16, 14, 14])

        x = self.conv2(x)
        # torch.Size([4, 32, 10, 10])
        x = f.relu(x)
        x = self.pool2(x)
        # torch.Size([4, 32, 5, 5])

        x = x.view(-1, 32 * 5 * 5)
        x = f.relu(x)
        # torch.Size([4, 800])

        x = self.fc1(x)
        x = f.relu(x)
        # torch.Size([4, 120])

        x = self.fc2(x)
        x = f.relu(x)
        # torch.Size([4, 84])

        x = self.fc3(x)
        # torch.Size([4, 10])
        return x


if __name__ == '__main__':
    inputTensor = torch.rand(4, 3, 32, 32)
    net = LeNet()
    print(net)
    output = net.forward(inputTensor)
