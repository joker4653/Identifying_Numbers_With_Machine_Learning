import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, MaxPool2d, Linear

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        """Basic Structure of a Convolutional Neural Network, 2 Convolutional Layers, 2 Max Pooling Layers, and 2 Fully Connected Layers"""
        self.conv1 = Conv2d(28*28, 1024, kernel_size=(3,3))
        self.maxPool1 = MaxPool2d(kernel_size=(2,2))
        self.conv2 = Conv2d(512, 256, kernel_size=(3,3))
        self.maxPool2 = MaxPool2d(kernel_size=(2,2))

        self.fc1 = Linear(256, 256-32)
        self.fc2 = Linear(256-32, 192)
        self.fc3 = Linear(192, 10)

    def forward(self, x):
        """Defines the activation of any Neurons in the network"""
        
        # convolution + pooling / relu
        x = F.relu(self.conv1(x))
        x = self.maxPool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxPool2(x)

        # Flatten (view) the data so it can be fed in the FCs
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)