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

    def remove_num_batches_tracked(self, state_dict):
            new_state_dict = {}
            for name, p in state_dict.items():
                if not 'num_batches_tracked' in name:
                    new_state_dict[name] = p
            return new_state_dict

    def migrate(
            self,
            state_dict: Dict,
            other_state_dict = None,
            force = False
    ): 
        if other_state_dict is None:
            des_state_dict = self.state_dict()
            source_state_dict = state_dict
        else:
            des_state_dict = state_dict
            source_state_dict = other_state_dict

        des_state_dict = self.remove_num_batches_tracked(des_state_dict)
        source_state_dict = self.remove_num_batches_tracked(source_state_dict)

        if not force:
            state_dict_keys = source_state_dict.keys()
            with torch.no_grad():
                for i, (name, p) in enumerate(des_state_dict.items()):
                    if name in state_dict_keys:
                        _p = source_state_dict[name]
                        if p.data.shape == _p.shape:
                            print(i, name)
                            p.copy_(_p)
        else:
            print('Force migrating...')
            with torch.no_grad():
                for i, ((name, p), (_name, _p)) in enumerate(zip(des_state_dict.items(), source_state_dict.items())):
                    if p.shape == _p.shape:
                        print(i, 'copy to {} from {}'.format(name, _name))
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

    def filter_state_dict_with_prefix(
        self,
        state_dict: Dict,
        prefix: str,
        is_remove_prefix = False
    ):
        if not isinstance(prefix, str):
            raise BaseException('prefix', [str])
        new_state_dict = {}
        if is_remove_prefix:
            prefix_length = len(prefix)
            for name, p in state_dict.items():
                if name.startswith(prefix):
                    new_state_dict[name[prefix_length+1:]] = p
        else:
            for name, p in state_dict.items():
                if name.startswith(prefix):
                    new_state_dict[name] = p
        return new_state_dict

    def filter_state_dict_except_prefix(
        self,
        state_dict: Dict,
        prefix: str,
    ):
        if not isinstance(prefix, str):
            raise BaseException('prefix', [str])
        new_state_dict = {}
        for name, p in state_dict.items():
            if not name.startswith(prefix):
                new_state_dict[name] = p
        return new_state_dict
    
    def freeze_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = False

    def freeze_with_prefix(self, prefix):
        for name, p in self.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = False

    def defrost_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = True

    def defrost_with_prefix(self, prefix):
        for name, p in self.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = True

class BaseSequential(nn.Sequential, Base):
    pass
