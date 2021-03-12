import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from .base import Base, ConvBatchNormRelu


class PruningModel(Base):
    def __init__(self, model):
        super().__init__()
        self.model = model

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

    def _prun(self, current_name, current_layer, next_name, next_layer):
        # self._re_assign_module(current_name, nn.Conv2d(10, 10, 3, padding=1))
        weight = None
        if hasattr(current_layer, 'weight'):
            weight = current_layer.weight
        bias = None
        if hasattr(current_layer, 'bias'):
            bias = current_layer.bias
        try:
            print(weight.shape)
            print(bias.shape)
        except:
            pass
        dimensions = list(weight.shape)
        epsilon = 0.3
        sum = (weight**2).sum(dim=(1,2,3))
        print(sum)
        index = (sum > epsilon).nonzero(as_tuple=True)[0]
        print(index)
        print(index.shape)
        weight = weight[index, ...]
        print(weight.shape)
        dimensions[0] = weight.shape[0]
        self._re_assign_module(current_name, nn.Conv2d(dimensions[0], dimensions[1], (dimensions[2], dimensions[3]), padding=1))
        # self._re_assign_module(next_name, nn.Conv2d(10, 10, 3, padding=1))

    def prun(self):
        current_name = None
        current_layer = None
        next_name = None
        next_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if next_layer is None:
                    next_name = name
                    next_layer = module
                else:
                    current_name = next_name
                    current_layer = next_layer
                    next_layer = module
                    next_name = name
                    self._prun(current_name, current_layer, next_name, next_layer)
                    break
