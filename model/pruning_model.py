import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from .base import Base, ConvBatchNormRelu


class PruningModel(Base):
    def __init__(self, model, block_names):
        super().__init__()
        self.model = model
        self.block_names = block_names

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

    def _re_assign_module(self, name, module=None):
        trace = name.split('.')
        first_name = trace[-1]
        temp = self.model
        for node in trace[:-1]:
            if node.isnumeric():
                temp = temp[int(node)]
            else:
                temp = getattr(temp, node)
        setattr(temp, first_name, module)

    def _get_module(self, name):
        trace = name.split('.')
        temp = self.model
        for node in trace:
            if node.isnumeric():
                temp = temp[int(node)]
            else:
                temp = getattr(temp, node)
        return temp

    def prun(self):
        current_index = None
        for i in range(len(self.block_names)-1):
            current_block = self._get_module(self.block_names[i])
            current_index = current_block.prun(current_index)
            print(current_index.shape)
            self._re_assign_module(self.block_names[i], current_block)
        current_block = self._get_module(self.block_names[-1])
        current_block.prun(current_index, is_take_prun=False)
        self._re_assign_module(self.block_names[-1], current_block)
