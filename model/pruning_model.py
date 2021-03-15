import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from .base import Base, ConvBatchNormRelu


def _reassign_module(self, name, module=None):
    temp, first_name =self._get_reference_to_module(name)
    setattr(temp, first_name, module)

def _get_reference_to_module(self, name):
    trace = name.split('.')
    first_name = trace[-1]
    temp = self
    for node in trace[:-1]:
        if node.isnumeric():
            temp = temp[int(node)]
        else:
            temp = getattr(temp, node)
    return temp, first_name

def _get_module(self, name):
    trace = name.split('.')
    temp = self
    for node in trace:
        if node.isnumeric():
            temp = temp[int(node)]
        else:
            temp = getattr(temp, node)
    return temp

def _reassign_conv(self, name, weight):
    padding = (weight.shape[2] // 2, weight.shape[3] // 2)
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], (
        weight.shape[2], weight.shape[3]), padding=padding, bias=False, stride=self.stride)
    with torch.no_grad():
        conv.weight.copy_(weight)
    self._reassign_module(name, conv)

def _reassign_batchnorm(self, name, batchnorm, index):
    running_mean = batchnorm.running_mean[index]
    running_var = batchnorm.running_var[index]
    weight = batchnorm.weight[index]
    bias = batchnorm.bias[index]
    eps = batchnorm.eps
    res = nn.BatchNorm2d(index.shape[0])
    with torch.no_grad():
        res.running_mean.copy_(running_mean)
        res.running_var.copy_(running_var)
        res.weight.copy_(weight)
        res.bias.copy_(bias)
        res.eps = eps
    self._reassign_module(name, res)

class PruningModel(Base):
    def __init__(self, model, block_names, config_prun):
        super().__init__()
        self.model = model
        self.block_names = block_names
        setattr(PruningModel, '_reassign_module', _reassign_module)
        setattr(PruningModel, '_get_reference_to_module', _get_reference_to_module)
        setattr(PruningModel, '_get_module', _get_module)
        setattr(PruningModel, '_reassign_conv', _reassign_conv)
        setattr(PruningModel, '_reassign_batchnorm', _reassign_batchnorm)
        self._set_config_prun(config_prun)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1,
                                    momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()

    def _set_config_prun(self, config_prun):
        for item in config_prun:
            setattr(item['block_name'], 'prun', item['prun_method'])
            setattr(item['block_name'], '_prun', item['prun_algorithm'])
            setattr(item['block_name'], '_reassign_module', _reassign_module)
            setattr(item['block_name'], '_get_reference_to_module', _get_reference_to_module)
            setattr(item['block_name'], '_get_module', _get_module)
            setattr(item['block_name'], '_reassign_conv', _reassign_conv)
            setattr(item['block_name'], '_reassign_batchnorm', _reassign_batchnorm)

    

    def prun(self):
        # print(self)
        current_index = None
        for i in range(len(self.block_names)-1):
            print(self.block_names[i])
            current_block = self._get_module(self.block_names[i])
            current_index = current_block.prun(current_index)
            print(current_index.shape)
            self._reassign_module(self.block_names[i], current_block)
        current_block = self._get_module(self.block_names[-1])
        current_block.prun(current_index, is_take_prun=False)
        self._reassign_module(self.block_names[-1], current_block)
