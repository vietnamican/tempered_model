import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from ..base import Base, ConvBatchNormRelu
from ..tempered_model import TemperedModel


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
        self.cbr1 = ConvBatchNormRelu(
            inplanes, bottleneck_plane, kernel_size=1, padding=0, bias=False)
        self.cbr2 = ConvBatchNormRelu(
            bottleneck_plane, bottleneck_plane, kernel_size=3, padding=1, stride=stride, bias=False)
        self.cbr3 = ConvBatchNormRelu(
            bottleneck_plane, outplanes, kernel_size=1, padding=0, with_relu=False, bias=False)
        if downsample:
            self.identity_layer = ConvBatchNormRelu(
                inplanes, outplanes, kernel_size=1, padding=0, stride=stride, with_relu=False, bias=False)
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


class Resnet50Orig(Base):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
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
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(2048, 10)


class Resnet50Temper(Base):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
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
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 2048, downsample=True, stride=2),
            Bottleneck(2048, 2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(2048, 10)


class Resnet50(TemperedModel):
    def __init__(self, orig, tempered, mode, orig_module_names, tempered_module_names, is_trains):
        super().__init__(orig, tempered, mode, orig_module_names,
                         tempered_module_names, is_trains)

    def configure_optimizers(self):
        print("---------------------------------------------------------")
        print("Load configure from son, not father")
        print("---------------------------------------------------------")
        if self.mode == 'training':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        elif self.mode == 'temper':
            params = []
            for tempered_modules in self.tempered_modules:
                if isinstance(tempered_modules, list):
                    for tempered_module in tempered_modules:
                        params.extend(tempered_module.parameters())
                else:
                    params.extend(tempered_modules.parameters())
            optimizer = torch.optim.SGD(params, lr=0.01,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        elif self.mode == 'tuning':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            print("Not in one of modes ['trianing', 'temper', 'tuning']")

    def migrate_from_torchvision(self, state_dict):
        self_state_dict = self.filter_state_dict_except_prefix(
            self.state_dict(), 'tempered')
        self.migrate(self_state_dict, state_dict, force=True)
