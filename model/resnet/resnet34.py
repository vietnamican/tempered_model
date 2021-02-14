from torch._C import ThroughputBenchmark
from model.resnet.resnet50 import Resnet50
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional, Literal

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
    def __init__(
        self,
        mode: Literal['tuning', 'truncating', 'inference'] = 'tuning'
    ):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        self.module_names = [
            'layer3.3',
            'layer3.4',
            'layer3.5',
            'layer4.1',
            'layer4.2'
        ]
        self.orig_module_names = [
            self.layer3[3],
            self.layer3[4],
            self.layer3[5],
            self.layer4[1],
            self.layer4[2],
        ]
        self.truncate_module_names = [
            self.truncate.layer3[3],
            self.truncate.layer3[4],
            self.truncate.layer3[5],
            self.truncate.layer4[1],
            self.truncate.layer4[2],
        ]
        if mode == 'truncating':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.mode = mode
        if mode == 'truncating':
            self.register_truncate()
            self.loss = {}
            self.out = {}
            self.freeze_except_prefix('truncate')
        else:
            self.freeze_with_prefix('truncate')

    def forward(
            self,
            x,
            mode: Literal['tuning', 'truncating', 'inference'] = 'truncating'
    ):
        if mode == 'inference':
            x = self.maxpool(self.conv1(x))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            x = self.fc(torch.flatten(x, 1))
            return x
        elif mode == 'tuning':
            x = self.maxpool(self.conv1(x))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            return self.fc(torch.flatten(x, 1))
        elif mode == 'truncating':
            x = self.maxpool(self.conv1(x))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            x =  self.fc(torch.flatten(x, 1))
            return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'tuning':
            logit = self.forward(x, mode='tuning')
            loss = self.criterion(logit, y)
            pred = logit.argmax(dim=1)
            self.log('train_loss', loss)
            self.log('train_acc_step', self.accuracy(pred, y))
            return loss
        elif self.mode == 'truncating':
            self.forward(x, mode='truncating')
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
            logit = self.forward(x, mode='tuning')
            loss = self.criterion(logit, y)
            pred = logit.argmax(dim=1)
            self.log('val_loss', loss)
            self.log('val_acc_step', self.val_accuracy(pred, y))
            return loss
        elif self.mode == 'truncating':
            self.forward(x, mode='truncating')
            for n, l in self.loss.items():
                self.log('val_loss_{}'.format(n), l)
            loss = torch.sum(torch.stack(list(self.loss.values())))
            self.log('val_loss', loss)
            return loss

    def validation_epoch_end(self, outs):
        if not self.mode == 'truncating':
            self.log('val_acc_epoch', self.val_accuracy.compute())

    def configure_optimizers(self):
        if not self.mode == 'truncating':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            params = []
            for truncate_module in self.truncate_module_names:
                params.extend(truncate_module.parameters())
            optimizer = torch.optim.Adam(params)
            return optimizer

    def register_truncate(self):
        for name, module, truncate_module in zip(self.module_names, self.orig_module_names, self.truncate_module_names):
            module.register_forward_hook(
                self._training_truncate(name, module, truncate_module))

    def _training_truncate(self, name, module, truncate_module):
        truncate_module = truncate_module
        def hook(model, input, output):
            _output = truncate_module(input[0])
            self.out[name] = output
            self.criterion(_output, output)
            self.loss[name] = self.criterion(_output, output)
        return hook

    def migrate(self, state_dict):
        super().migrate(state_dict)
        def filter_state_dict_with_prefix(state_dict, prefix, is_remove_prefix=False):
            new_self_state_dict = {}
            if is_remove_prefix:
                prefix_length = len(prefix)
                for name, p in state_dict.items():
                    if name.startswith(prefix):
                        new_self_state_dict[name[prefix_length+1:]] = p
            else:
                for name, p in state_dict.items():
                    if name.startswith(prefix):
                        new_self_state_dict[name] = p
            return new_self_state_dict
        if self.mode == 'truncating':
            print('migrating layer3.0')
            self.truncate.layer3[0].migrate(filter_state_dict_with_prefix(state_dict, 'layer3.0', True))
            print('migrating layer3.1')
            self.truncate.layer3[1].migrate(filter_state_dict_with_prefix(state_dict, 'layer3.1', True))
            print('migrating layer3.2')
            self.truncate.layer3[2].migrate(filter_state_dict_with_prefix(state_dict, 'layer3.2', True))
            print('migrating layer4.0')
            self.truncate.layer4[0].migrate(filter_state_dict_with_prefix(state_dict, 'layer4.0', True))

    def migrate_from_torchvision(self, state_dict):
        def remove_num_batches_tracked(state_dict):
            new_state_dict = {}
            for name, p in state_dict.items():
                if not 'num_batches_tracked' in name:
                    new_state_dict[name] = p
            return new_state_dict
        def filter_state_dict_with_prefix(state_dict, prefix):
            new_self_state_dict = {}
            for name, p in state_dict.items():
                if name.startswith(prefix):
                    new_self_state_dict[name] = p
            return new_self_state_dict
        self_state_dict = remove_num_batches_tracked(self.state_dict())
        source_state_dict = remove_num_batches_tracked(state_dict)
        new_self_state_dict = {}
        for name, p in self_state_dict.items():
            if not name.startswith('truncate'):
                new_self_state_dict[name] = p

        with torch.no_grad():
            for i, ((name, p), (_name, _p)) in enumerate(zip(new_self_state_dict.items(), source_state_dict.items())):
                if p.shape == _p.shape:
                    print(i, 'copy to {} from {}'.format(name, _name))
                    p.copy_(_p)
        
        self.truncate.layer3[0].migrate_from_torchvision(filter_state_dict_with_prefix(source_state_dict, 'layer3.0'))
        self.truncate.layer3[1].migrate_from_torchvision(filter_state_dict_with_prefix(source_state_dict, 'layer3.1'))
        self.truncate.layer3[2].migrate_from_torchvision(filter_state_dict_with_prefix(source_state_dict, 'layer3.2'))
        self.truncate.layer4[0].migrate_from_torchvision(filter_state_dict_with_prefix(source_state_dict, 'layer4.0'))