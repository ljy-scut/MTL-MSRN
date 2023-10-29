import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch

def variable_conv(in_planes, out_planes, kernel_size,stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)#全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _= x.size()#shape:(batch,C,feature)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)#将y扩展成x的形状


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes,kernel_size,stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = variable_conv(inplanes, planes,kernel_size,stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = variable_conv(planes, planes,kernel_size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MSResNet(nn.Module):
    def __init__(self, layers,input_channel=32,  num_classes=2):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock, 64, layers[0], kernel_size=3,stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock, 128, layers[1], kernel_size=3,stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock, 256, layers[2], kernel_size=3,stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock, 512, layers[3], kernel_size=3,stride=2)


        self.layer5x5_1 = self._make_layer5(BasicBlock, 64, layers[0], kernel_size=5,stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock, 128, layers[1], kernel_size=5,stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock, 256, layers[2], kernel_size=5,stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock, 512, layers[3],  kernel_size=5,stride=2)

        self.layer7x7_1 = self._make_layer7(BasicBlock, 64, layers[0], kernel_size=7,stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock, 128, layers[1], kernel_size=7,stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock, 256, layers[2], kernel_size=7,stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock, 512, layers[3], kernel_size=7,stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512*3, num_classes)

        # todo: modify the initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, kernel_size,stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes,kernel_size,stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes,kernel_size))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, kernel_size,stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes,kernel_size, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes,kernel_size))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, kernel_size,stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, kernel_size,stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes,kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.layer3x3_4(x)
        x=self.avgpool(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        y = self.layer5x5_4(y)
        y=self.avgpool(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        z = self.layer7x7_4(z)
        z=self.avgpool(z)

        out = torch.cat([x, y, z], dim=1)
        out = out.squeeze()
        # out = self.drop(out)
        #out1 = self.fc(out)

        return out

def ms_resnet_34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MSResNet([3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    x= torch.randn(2,32,128)
    net= ms_resnet_34()
    y=net(x)
    summary(net,(32,128))
    print(y.shape)







