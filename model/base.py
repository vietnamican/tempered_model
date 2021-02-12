from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvBatchNormRelu(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'with_relu' not in kwargs:
            self.with_relu = True
        else:
            self.with_relu = kwargs['with_relu']
            kwargs.pop('with_relu', None)
        if 'with_bn' not in kwargs:
            self.with_bn = True
        else:
            self.with_bn = kwargs['with_bn']
            kwargs.pop('with_bn', None)
        self.cbr = nn.Sequential(nn.Conv2d(*args, **kwargs))
        if self.with_bn:
            outplanes = args[1]
            self.cbr.add_module('bacthnorm', nn.BatchNorm2d(int(outplanes)))
        if self.with_relu:
            self.cbr.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.cbr(x)


class BaseException(Exception):
    def __init__(
            self,
            parameter,
            types: List):
        message = '{} type must be one of {}'.format(parameter, types)
        super().__init__(message)


class Base(pl.LightningModule):
    def __init__(self):
        super(Base, self).__init__()

    def migrate(
            self,
            state_dict: Dict
    ):
        def remove_num_batches_tracked(state_dict):
            new_state_dict = {}
            for name, p in state_dict.items():
                if not 'num_batches_tracked' in name:
                    new_state_dict[name] = p
            return new_state_dict
            
        self_state_dict = remove_num_batches_tracked(self.state_dict())
        state_dict = remove_num_batches_tracked(state_dict)

        state_dict_keys = state_dict.keys()
        with torch.no_grad():
            for i, (name, p) in enumerate(self_state_dict.items()):
                if name in state_dict_keys:
                    _p = state_dict[name]
                    if p.data.shape == _p.shape:
                        print(i, name)
                        p.copy_(_p)

    def remove_prefix_state_dict(
            self,
            state_dict: Dict,
            prefix: Union[str, int]
    ):
        result_state_dict = {}
        if isinstance(prefix, int):
            # TODO
            return state_dict
        elif isinstance(prefix, str):
            len_prefix_remove = len(prefix) + 1
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    result_state_dict[key[len_prefix_remove:]
                                      ] = state_dict[key]
                else:
                    result_state_dict[key] = state_dict[key]
            return result_state_dict
        else:
            raise BaseException('prefix', [str, int])

    def filter_prefix_state_dict(
        self,
        state_dict: Dict,
        prefix: str
    ):
        if not isinstance(prefix, str):
            raise BaseException('prefix', [str])


class BaseSequential(nn.Sequential, Base):
    pass
