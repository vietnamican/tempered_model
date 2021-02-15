from torch._C import ThroughputBenchmark
from model.resnet.resnet50 import Resnet50
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from ..base import Base, ConvBatchNormRelu

from .resnet50 import Resnet50


class BasicBlock(Base):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=False
    ) -> None:
        super().__init__()
        self.downsample = downsample
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvBatchNormRelu(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride)
        self.conv2 = ConvBatchNormRelu(
            planes, planes, kernel_size=3, padding=1, bias=False, with_relu=False)
        if downsample:
            self.identity_layer = ConvBatchNormRelu(
                inplanes, planes, kernel_size=1, padding=0, bias=False, with_relu=False, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            identity = self.identity_layer(x)

        out += identity
        out = self.relu(out)

        return out

    def migrate_from_torchvision(self, state_dict):
        def remove_num_batches_tracked(state_dict):
            new_state_dict = {}
            for name, p in state_dict.items():
                if not 'num_batches_tracked' in name:
                    new_state_dict[name] = p
            return new_state_dict
        self_state_dict = remove_num_batches_tracked(self.state_dict())
        source_state_dict = remove_num_batches_tracked(state_dict)

        with torch.no_grad():
            for i, ((name, p), (_name, _p)) in enumerate(zip(self_state_dict.items(), source_state_dict.items())):
                if p.shape == _p.shape:
                    print(i, 'copy to {} from {}'.format(name, _name))
                    p.copy_(_p)


class BasicBlockTruncate(Base):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=False
    ) -> None:
        super().__init__()
        self.downsample = downsample
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvBatchNormRelu(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride, with_relu=False)
        self.identity_layer = ConvBatchNormRelu(
            inplanes, planes, kernel_size=1, padding=0, bias=False, stride=stride, with_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if inplanes == planes and stride == 1:
            self.skip_layer = nn.BatchNorm2d(num_features=inplanes)
        else:
            self.skip_layer = None
        # self.skip_layer = nn.BatchNorm2d(
        #     num_features=inplanes) if inplanes == planes and stride == 1 else None

    def forward(self, x):
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x
        conv3 = self.conv1(x)
        identity = self.identity_layer(x)

        x = conv3 + identity + skip

        return self.relu(x)


class Resnet34(Base):
    orig_module_names = [
        'layer3.3',
        'layer3.4',
        'layer3.5',
        'layer4.1',
        'layer4.2'
    ]
    truncate_module_names = [
        'truncate.layer3.3',
        'truncate.layer3.4',
        'truncate.layer3.5',
        'truncate.layer4.1',
        'truncate.layer4.2'
    ]

    def __init__(
        self,
        mode='tuning'
    ):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, downsample=True, stride=2),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
        self.truncate = nn.Module()
        self.truncate.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlockTruncate(256, 256),
            BasicBlockTruncate(256, 256),
            BasicBlockTruncate(256, 256)
        )
        self.truncate.layer4 = nn.Sequential(
            BasicBlock(256, 512, downsample=True, stride=2),
            BasicBlockTruncate(512, 512),
            BasicBlockTruncate(512, 512)
        )
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        self.mode = mode
        if mode == 'truncating':
            self.criterion = nn.MSELoss()
            self.register_modules()
            self.register_truncate()
            self.freeze_except_prefix('truncate')
            self.freeze_with_prefix('truncate.layer3.0')
            self.freeze_with_prefix('truncate.layer3.1')
            self.freeze_with_prefix('truncate.layer3.2')
            self.freeze_with_prefix('truncate.layer4.0')
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.freeze_with_prefix('truncate')

    def forward(
            self,
            x,
            mode='tuning'
    ):
        if mode == 'inference':
            x = self.conv1(x)
            x = self.layer2(self.layer1(x))
            x = self.truncate.layer3(x)
            x = self.layer4[0](x)
            x = self.layer4[1](x)
            x = self.truncate.layer4[2](x)
            x = self.avgpool(x)
            return self.fc(torch.flatten(x, 1))
        else:
            x = self.conv1(x)
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            return self.fc(torch.flatten(x, 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'tuning':
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(dim=1)
            self.log('train_loss', loss)
            self.log('train_acc_step', self.accuracy(pred, y))
            return loss
        elif self.mode == 'truncating':
            self.forward(x)
            for n, l in self.loss.items():
                self.log('train_loss_{}'.format(n), l)
            loss = torch.sum(torch.stack(list(self.loss.values())))
            self.log('train_loss', loss)
            return loss

    def training_epoch_end(self, outs):
        if not self.mode == 'truncating':
            self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'tuning':
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(dim=1)
            self.log('val_loss', loss)
            self.log('val_acc_step', self.val_accuracy(pred, y))
            return loss
        elif self.mode == 'truncating':
            self.forward(x)
            for n, l in self.loss.items():
                self.log('val_loss_{}'.format(n), l)
            loss = torch.sum(torch.stack(list(self.loss.values())))
            self.log('val_loss', loss)
            return loss

    def validation_epoch_end(self, outs):
        if not self.mode == 'truncating':
            self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x, mode='inference')
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('test_loss', loss)
        self.log('test_acc_step', self.test_accuracy(pred, y))
        return loss
    
    def test_epoch_end(self, outputs):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        if not self.mode == 'truncating':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            # optimizer = torch.optim.SGD(
            #     self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer, step_size=30, gamma=0.1)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            params = []
            for truncate_module in self.truncate_modules:
                params.extend(truncate_module.parameters())
            optimizer = torch.optim.SGD(params, lr=0.01,
                    momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            # optimizer = torch.optim.Adam(params)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def register_modules(self):
        self.orig_modules = []
        self.truncate_modules = []
        self.loss = {}
        self.out = {}
        for name, module in self.named_modules():
            if name in self.orig_module_names:
                self.orig_modules.append(module)
                self.out[name] = None
                self.loss[name] = None
            if name in self.truncate_module_names:
                self.truncate_modules.append(module)

    def register_truncate(self):
        for orig_name, orig_module, truncate_module in zip(self.orig_module_names, self.orig_modules, self.truncate_modules):
            orig_module.register_forward_hook(
                self._training_truncate(orig_name, orig_module, truncate_module))

    def _training_truncate(self, name, orig_module, truncate_module):
        def hook(model, input, output):
            _output = truncate_module(input[0])
            self.out[name] = output
            self.criterion(_output, output)
            self.loss[name] = self.criterion(_output, output)
        return hook

    def migrate(self, state_dict):
        super().migrate(state_dict)
        if self.mode == 'truncating':
            self.truncate.layer3[0].migrate(
                self.layer3[0].state_dict(), force=True)
            self.truncate.layer3[1].migrate(
                self.layer3[1].state_dict(), force=True)
            self.truncate.layer3[2].migrate(
                self.layer3[2].state_dict(), force=True)
            self.truncate.layer4[0].migrate(
                self.layer4[0].state_dict(), force=True)

    def migrate_from_torchvision(self, state_dict):
        self_state_dict = self.filter_state_dict_except_prefix(
            self.state_dict(), 'truncate')
        super().migrate(self_state_dict, state_dict, force=True)
        self.truncate.layer3[0].migrate(
            self.filter_state_dict_with_prefix(state_dict, 'layer3.0'), force=True)
        self.truncate.layer3[1].migrate(
            self.filter_state_dict_with_prefix(state_dict, 'layer3.1'), force=True)
        self.truncate.layer3[2].migrate(
            self.filter_state_dict_with_prefix(state_dict, 'layer3.2'), force=True)
        self.truncate.layer4[0].migrate(
            self.filter_state_dict_with_prefix(state_dict, 'layer4.0'), force=True)
