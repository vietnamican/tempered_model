import torch
from torch import nn

from ..base import Base, ConvBatchNormRelu


class Bottleneck(Base):
    bottle_ratio = 4
    def __init__(
        self,
        inplanes,
        outplanes,
        stride=1,
        downsample=False
    ):
        super().__init__()
        self.downsample = downsample
        bottleneck_plane = outplanes // 4
        self.cbr1 = ConvBatchNormRelu(inplanes, bottleneck_plane, kernel_size=1, padding=0, bias=False)
        self.cbr2 = ConvBatchNormRelu(bottleneck_plane, bottleneck_plane, kernel_size=3, padding=1, stride=stride, bias=False)
        self.cbr3 = ConvBatchNormRelu(bottleneck_plane, outplanes, kernel_size=1, padding=0, with_relu=False, bias=False)
        if downsample:
            self.identity_layer = ConvBatchNormRelu(inplanes, outplanes, kernel_size=1, padding=0, stride=stride, with_relu=False, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        if self.downsample:
            identity = self.identity_layer(identity)
        x += identity
        return self.relu(x)

class Resnet50(Base):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 256, downsample=True),
            Bottleneck(256, 256),
            Bottleneck(256, 256)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 512, downsample=True, stride=2),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 1024, downsample=True, stride=2),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 2048, downsample=True, stride=2),
            Bottleneck(2048, 2048),
            Bottleneck(2048, 2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        y = self.maxpool(self.conv1(x))
        y1 = self.layer1(y)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y = self.avgpool(y4)
        y = self.fc(torch.flatten(y, 1))
        return y, y1, y2, y3, y4
    
    def migrate_from_torchvision(self, torch_vision_state_dict):
        def remove_num_batches_tracked(state_dict):
            new_state_dict = {}
            for name, p in state_dict.items():
                if not 'num_batches_tracked' in name:
                    new_state_dict[name] = p
            return new_state_dict

        self_state_dict = remove_num_batches_tracked(self.state_dict())
        source_state_dict = remove_num_batches_tracked(torch_vision_state_dict)

        with torch.no_grad():
            for i, ((name, p), (_name, _p)) in enumerate(zip(self_state_dict.items(), source_state_dict.items())):
                if p.shape == _p.shape:
                    print(i, 'copy to {} from {}'.format(name, _name))
                    p.copy_(_p)