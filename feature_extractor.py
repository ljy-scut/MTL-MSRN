##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/aliasvishnu/EEGNet
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Feature Extractor """
import math
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, input_nc, outchannel,kernel_size,stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, outchannel, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            #nn.Dropout(.2),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(outchannel)
        )
        if stride!=1 or input_nc*2==outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_nc, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            self.shortcut=None

    def forward(self, x):
        out = self.conv(x)
        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        out = F.relu(out,inplace=True)
        return out

class Route(nn.Module):
    def __init__(self, kernel_size):
        super(Route, self).__init__()
        self.block1 = ResidualBlock(64, 64, kernel_size, stride=2)
        self.block2 = ResidualBlock(64, 128, kernel_size,stride=2)
        self.block3 = ResidualBlock(128, 256, kernel_size,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, input_nc=4, num_classes=2):
        super(FeatureExtractor, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.Route1 = Route(3)
        self.Route2 = Route(5)
        self.Route3 = Route(7)
        self.fc = nn.Linear(256*3, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pre_conv(x)
        x1 = self.Route1(x)
        x2 = self.Route2(x)
        x3 = self.Route3(x)
        x = torch.cat((x1,x2,x3), 1)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    x= torch.randn(1,4,32,32)
    net=FeatureExtractor()
    y=net(x)
    summary(net,(4,32,32))
    print(y.shape)

        
