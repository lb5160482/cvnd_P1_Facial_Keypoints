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
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        # self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0.01)  #zero_()
        self.conv1_bn = nn.BatchNorm2d(32)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32,64,3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        # self.conv2.weight.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(0.01)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64,128,3)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        # self.conv3.weight.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0.01)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(128,256,3)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        # self.conv4.weight.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(0.01)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(256,512,3)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        # self.conv5.weight.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(0.01)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.dropout5 = nn.Dropout(p=0.2)


        self.fc1 = nn.Linear(5*5*512,2000)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.dropout6 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(2000,136)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout7 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # x = self.conv1_bn(self.conv1(x))
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        # x=x.type(torch.cuda.FloatTensor)
        # x = self.dropout1(x)

        # x = self.conv2_bn(self.conv2(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        # x = self.dropout2(x)
        

        # x = self.conv3_bn(self.conv3(x))
        x = self.conv3(x)
        x = self.pool(F.relu(x))
        x = self.dropout3(x)

        # x = self.conv4_bn(self.conv4(x))
        x = self.conv4(x)
        x = self.pool(F.relu(x))
        x = self.dropout4(x)
        
        # x = self.conv5_bn(self.conv5(x))
        x = self.conv5(x)
        x = self.pool(F.relu(x))
        x = self.dropout5(x)
        

        x = x.view(-1, 5*5*512)
        x = self.fc1(x)
        x = F.relu(self.dropout6(x))
        # x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(self.dropout7(x))
        # x = F.relu(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
