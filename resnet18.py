import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride,1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample_conv = nn.Conv2d(in_channels,out_channels,1,2,0)
        if stride == 2:
            self.downsample = True
        else:
            self.downsample = False
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            x = self.downsample_conv(x)
        out = self.relu2(out+x)
        return out

blocks = [[64,64,1],
         [64,64,1],
         [64,128,2],
         [128,128,1],
         [128,256,2],
         [256,256,1],
         [256,512,2],
         [512,512,1]]

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3,2,1)
        self.residual_blocks = nn.Sequential()
        self.residual_blocks = nn.Sequential()
        for block in blocks:
            in_channels,out_channels,stride = block
            self.residual_blocks.append(BasicBlock(in_channels,out_channels,stride))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.residual_blocks(x)
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x,1))
        return x