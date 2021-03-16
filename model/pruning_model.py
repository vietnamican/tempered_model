import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from .base import Base, ConvBatchNormRelu


def _reassign_module(self, name, module=None):
    temp, first_name = _get_reference_to_module(self, name)
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


class PruningModel(object):
    def __init__(self, config_prun, model=None, block_names=None):
        super().__init__()
        self._set_config_prun(config_prun)
        self.model = model
        self.block_names = block_names

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
            setattr(item['block_name'], '_get_reference_to_module',
                    _get_reference_to_module)
            setattr(item['block_name'], '_get_module', _get_module)
            setattr(item['block_name'], '_reassign_conv', _reassign_conv)
            setattr(item['block_name'], '_reassign_batchnorm',
                    _reassign_batchnorm)

    def _prun(self, module, block_names, index):
        if isinstance(block_names[index], list):
            with torch.no_grad():
                current_block_names = block_names[index]
                current_index = None
                length = len(block_names[index])
                for i in range(length-1):
                    current_block = _get_module(module, current_block_names[i])
                    current_index = current_block.prun(current_index)
                    _reassign_module(module,
                        current_block_names[i], current_block)
                current_block = _get_module(module, current_block_names[-1])
                current_block.prun(current_index, is_take_prun=False)
                _reassign_module(module, current_block_names[-1], current_block)
        elif isinstance(module, block_names[index], str):
            with torch.no_grad():
                current_block = _get_module(module, block_names[index])
                current_block.prun()

    def prun(self, module, block_names):
        for i in range(len(block_names)):
            self._prun(module, block_names, i)
