from model.resnet.resnet50 import Resnet50
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from .base import Base, ConvBatchNormRelu
from .tempered_model import TemperedModel


class VGG16(TemperedModel):
    def __init__(self, mode, orig_module_names, tempered_module_names, is_trains):
        super().__init__(mode, orig_module_names, tempered_module_names, is_trains)
        base_depth = 64
        self.orig = nn.Module()
        self.orig.block1 = nn.Sequential(
            ConvBatchNormRelu(3, base_depth, kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth, base_depth,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.orig.block2 = nn.Sequential(
            ConvBatchNormRelu(base_depth, base_depth*2,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*2, base_depth*2,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.orig.block3 = nn.Sequential(
            ConvBatchNormRelu(base_depth*2, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.orig.block4 = nn.Sequential(
            ConvBatchNormRelu(base_depth*4, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.orig.block5 = nn.Sequential(
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.orig.flatten = nn.Flatten(1)
        self.orig.fc = nn.Sequential(
            nn.Linear(base_depth*8, 10)
        )
        self.tempered = nn.Module()
        self.tempered.block1 = nn.Sequential(
            ConvBatchNormRelu(3, base_depth, kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth, base_depth,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.tempered.block2 = nn.Sequential(
            ConvBatchNormRelu(base_depth, base_depth*2,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*2, base_depth*2,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.tempered.block3 = nn.Sequential(
            ConvBatchNormRelu(base_depth*2, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.tempered.block4 = nn.Sequential(
            ConvBatchNormRelu(base_depth*4, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.tempered.block5 = nn.Sequential(
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.tempered.flatten = nn.Flatten(1)
        self.tempered.fc = nn.Sequential(
            nn.Linear(base_depth*8, 10)
        )
        self._setup_init(mode, orig_module_names,
                         tempered_module_names, is_trains)

    def configure_optimizers(self):
        if self.mode == 'training':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01,
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
            optimizer = torch.optim.SGD(params, lr=0.001,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        elif self.mode == 'tuning':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.0001,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            print("Not in one of modes ['trianing', 'temper', 'tuning']")