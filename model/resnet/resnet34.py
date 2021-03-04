from model.resnet.resnet50 import Resnet50
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from ..base import Base, ConvBatchNormRelu
from .basic import BasicBlock, BasicBlockTruncate
from ..tempered_model import TemperedModel


class Resnet34(TemperedModel):

    def __init__(self, mode, orig_module_names, tempered_module_names, is_trains, with_crelu=False):
        super().__init__(mode, orig_module_names, tempered_module_names, is_trains)
        self.with_crelu = with_crelu
        with_crelu = False
        self.orig = nn.Module()
        self.orig.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.orig.layer1 = nn.Sequential(
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu),
            BasicBlock(64, 64, with_crelu=with_crelu)
        )
        self.orig.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True, stride=2, with_crelu=with_crelu),
            BasicBlock(128, 128, with_crelu=with_crelu),
            BasicBlock(128, 128, with_crelu=with_crelu),
            BasicBlock(128, 128, with_crelu=with_crelu)
        )
        self.orig.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True, stride=2, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu),
            BasicBlock(256, 256, with_crelu=with_crelu)
        )
        self.orig.layer4 = nn.Sequential(
            BasicBlock(256, 512, downsample=True, stride=2, with_crelu=with_crelu),
            BasicBlock(512, 512, with_crelu=with_crelu),
            BasicBlock(512, 512, with_crelu=with_crelu)
        )
        self.orig.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.orig.flatten = nn.Flatten(1)
        self.orig.fc = nn.Linear(512, 10)
        with_crelu = self.with_crelu
        self.tempered = nn.Module()
        self.tempered.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.tempered.layer1 = nn.Sequential(
            BasicBlockTruncate(64, 64, with_crelu=with_crelu),
            BasicBlockTruncate(64, 64, with_crelu=with_crelu),
            BasicBlockTruncate(64, 64, with_crelu=with_crelu)
        )
        self.tempered.layer2 = nn.Sequential(
            BasicBlockTruncate(64, 128, downsample=True, stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu),
            BasicBlockTruncate(128, 128, with_crelu=with_crelu)
        )
        self.tempered.layer3 = nn.Sequential(
            BasicBlockTruncate(128, 256, downsample=True, stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu),
            BasicBlockTruncate(256, 256, with_crelu=with_crelu)
        )
        self.tempered.layer4 = nn.Sequential(
            BasicBlockTruncate(256, 512, downsample=True, stride=2, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu),
            BasicBlockTruncate(512, 512, with_crelu=with_crelu)
        )
        self.tempered.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tempered.flatten = nn.Flatten(1)
        self.tempered.fc = nn.Linear(512, 10)

        self._setup_init(mode, orig_module_names, tempered_module_names, is_trains)

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
