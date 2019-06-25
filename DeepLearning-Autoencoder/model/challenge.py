'''
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        #first dimension of input, number of filters, filter size, stride, padding
        self.conv1 = nn.Conv2d(3, 16, 5, 1, 2)
        self.conv2 = nn.Conv2d(16, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 32, 5, 1, 2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2)

        self.fc1 = nn.Linear(1568, 512)
        self.fc2 = nn.Linear(512, 64) 
        self.fc3 = nn.Linear(64, 5)

        self.dropout = nn.Dropout(p=0.2)

        self.init_weights()

    def init_weights(self):
        # TODO:
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1/sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.00)

        for fc in [self.fc1, self.fc2, self.fc3]:
            F_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1/sqrt(F_in))
            nn.init.constant_(fc.bias, 0.00)

        #
        
    def forward(self, x):

        # TODO:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.pool(self.conv2_bn(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, C*H*W))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
