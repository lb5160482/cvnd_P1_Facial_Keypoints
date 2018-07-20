## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(32,64,5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=0.1)

        self.conv3 = nn.Conv2d(64,128,3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(p=0.2)

        self.conv4 = nn.Conv2d(128,256,3)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(11*11*256,2048)
        self.dropout5 = nn.Dropout(p=0.4)
        self.fc1_bn = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048,68*2)
        self.dropout6 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        # x = self.conv1_bn(self.conv1(x))
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(x))
        # x = self.dropout1(x)
        

        # x = self.conv2_bn(self.conv2(x))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(x))
        # x = self.dropout2(x)
        

        # x = self.conv3_bn(self.conv3(x))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(x))
#         x = self.dropout3(x)

        # x = self.conv4_bn(self.conv4(x))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(x))
        x = self.dropout4(x)
        
        # x = self.conv5_bn(self.conv5(x))
        # x = self.conv5(x)
        # x = self.pool(F.relu(x))
        # x = self.dropout5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
#         x = self.fc1_bn(x)
        x = F.leaky_relu(x)
        x = self.dropout5(x)

        x = self.fc2(x)
        # x = F.relu(self.dropout7(x))
        x = F.leaky_relu(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
