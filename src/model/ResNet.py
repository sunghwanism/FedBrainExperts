import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def Conv3D(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv3D(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3D(out_ch, out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
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
    

class Bottleneck(nn.Module):
    
        def __init__(self, in_ch, out_ch, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = Conv3D(in_ch, out_ch, kernel_size=1)
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.conv2 = Conv3D(out_ch, out_ch, kernel_size=3, stride=stride, padding=1)
            self.bn2 = nn.BatchNorm3d(out_ch)
            self.conv3 = Conv3D(out_ch, out_ch*4, kernel_size=1)
            self.bn3 = nn.BatchNorm3d(out_ch*4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
    
    
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
    
            out = self.conv3(out)
            out = self.bn3(out)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = self.relu(out)
    
            return out



class ResNet(nn.Module):


    def __init__(self, block, layers, out_dim=1):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = Conv3D(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, out_dim)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                Conv3D(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model