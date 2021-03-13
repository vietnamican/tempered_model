from model.resnet.resnet50 import Resnet50
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from ..base import Base, ConvBatchNormRelu
from .basic import BasicBlock, BasicBlockTruncate
from ..tempered_model import TemperedModel


class Resnet34Orig(Base):
    def __init__(self, with_crelu=False):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlock(128, 128, with_crelu=with_crelu),
            BasicBlock(128, 128, with_crelu=with_crelu),
            BasicBlock(128, 128, with_crelu=with_crelu)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlock(512, 512, with_crelu=with_crelu),
            BasicBlock(512, 512, with_crelu=with_crelu)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()


class Resnet34Temper(Base):
    def __init__(self, with_crelu=False):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlockTruncate(64, 64, with_crelu=with_crelu),
            BasicBlockTruncate(64, 64, with_crelu=with_crelu),
            BasicBlockTruncate(64, 64, with_crelu=with_crelu)
        )
        self.layer2 = nn.Sequential(
            BasicBlockTruncate(64, 128, downsample=True,
                               stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu)
        )
        self.layer3 = nn.Sequential(
            BasicBlockTruncate(128, 256, downsample=True,
                               stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu)
        )
        self.layer4 = nn.Sequential(
            BasicBlockTruncate(256, 512, downsample=True,
                               stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()


class Resnet34PrunOrig(Base):
    def __init__(self, with_crelu=False):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu)
        )
        self.layer4 = nn.Sequential(
            BasicBlockTruncate(256, 512, downsample=True,
                               stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, 10)


class Resnet34Prun(Base):
    def __init__(self, with_crelu=False):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(256, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu)
        )
        self.layer4 = nn.Sequential(
            BasicBlockTruncate(256, 512, downsample=True,
                               stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(512, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 512, with_crelu=with_crelu)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, 10)


class Resnet34PrunTemper(Base):
    def __init__(self, with_crelu=False):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu)
        )
        self.layer4 = nn.Sequential(
            BasicBlockTruncate(256, 512, downsample=True,
                               stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, 10)


class Resnet34PrunTemperTemper(Base):
    def __init__(self, with_crelu=False):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True,
                       stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
        )
        self.layer4 = nn.Sequential(
            BasicBlockTruncate(256, 512, downsample=True,
                               stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, 10)


class Resnet34(TemperedModel):

    def __init__(self, orig, tempered, mode, orig_module_names, tempered_module_names, is_trains, with_crelu=False):
        super().__init__(orig, tempered, mode, orig_module_names,
                         tempered_module_names, is_trains)
        self.save_hyperparameters(
            'mode', 'orig_module_names', 'tempered_module_names', 'is_trains', 'with_crelu')

    def configure_optimizers(self):
        print("---------------------------------------------------------")
        print("Load configure from son, not father")
        print("---------------------------------------------------------")
        if self.mode == 'training':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100)
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
                optimizer, T_max=100)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        elif self.mode == 'tuning':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            print("Not in one of modes ['training', 'temper', 'tuning']")

    def migrate_from_torchvision(self, state_dict):
        self_state_dict = self.filter_state_dict_except_prefix(
            self.state_dict(), 'tempered')
        self.migrate(self_state_dict, state_dict, force=True)

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()
