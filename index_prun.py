from functools import partial

from model.base import ConvBatchNormRelu
from model.resnet.basic import BasicBlock, BasicBlockTruncate
import os

import torch
from torch import nn as nn
import torchvision
from torchvision import transforms
from torchsummary import summary
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

try:
    import torch_xla.core.xla_model as xm
except:
    pass

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model.resnet.resnet34 import Resnet34, Resnet34Orig, Resnet34Prun, Resnet34PrunTemper, Resnet34PrunTemperTemper, Resnet34Temper
from model.pruning_model import PruningModel
# from model.resnet.resnet50 import Resnet50, Resnet50Orig, Resnet50Temper
from model.tempered_model import LogitTuneModel

device = 'cpu'


def release(model):
    is_self = True
    for module in model.modules():
        if is_self:
            is_self = False
            continue
        if hasattr(module, 'release'):
            module.release()


block_names = [
    'model.conv1',
    'model.layer1.0',
    'model.layer1.1',
    'model.layer1.2',
    'model.layer2.0',
    'model.layer2.1',
    'model.layer2.2',
    'model.layer2.3',
    'model.layer3.0',
    'model.layer3.1',
    'model.layer3.2',
    'model.layer3.3',
    'model.layer3.4',
    'model.layer3.5',
    'model.layer4.0',
    'model.layer4.1',
    'model.layer4.2',
]

def basic_block_prun(self, in_channels, is_take_prun=True):
    weight1 = self.conv1.cbr.conv.weight
    weight1 = weight1[:, in_channels, ...]
    index1, weight1 = self._prun(weight1)
    conv1 = nn.Conv2d(weight1.shape[1], weight1.shape[0],
                        (weight1.shape[2], weight1.shape[3]), padding=1)
    with torch.no_grad():
        conv1.weight.copy_(weight1)
        if not self.is_released:
            self._reassign_batchnorm('conv1.cbr.bn', self.conv1.cbr.bn, index1)
    self.conv1.cbr.conv = conv1

    weight2 = self.conv2.cbr.conv.weight
    weight2 = weight2[:, index1, ...]
    if is_take_prun:
        index2, weight2 = self._prun(weight2)
        if not self.is_released:
            self._reassign_batchnorm('conv2.cbr.bn', self.conv2.cbr.bn, index2)
    conv2 = nn.Conv2d(weight2.shape[1], weight2.shape[0],
                        (weight2.shape[2], weight2.shape[3]), padding=1)
    with torch.no_grad():
        conv2.weight.copy_(weight2)
    self.conv2.cbr.conv = conv2

    if self.downsample:
        weight_identity = self.identity_layer.cbr.conv.weight
        weight_identity = weight_identity[:, in_channels, ...]
        if is_take_prun:
            weight_identity = weight_identity[index2, ...]
            if not self.is_released:
                self._reassign_batchnorm('identity_layer.cbr.bn', self.identity_layer.cbr.bn, index2)
        identity_layer = nn.Conv2d(weight_identity.shape[1], weight_identity.shape[0], (
            weight_identity.shape[2], weight_identity.shape[3]), padding=0)
        self.identity_layer.cbr.conv = identity_layer
    if is_take_prun:
        return index2
    else:
        return None

def basic_block_truncate_prun(self, in_channels, is_take_prun=True):
    if self.is_released:
        weight = self.forward_path.weight
        weight = weight[:, in_channels, ...]
        if is_take_prun:
            index, weight = self._prun(weight)
        self._reassign_conv('forward_path', weight)
        if is_take_prun:
            return index
    else:
        weight1 = self.conv1.cbr.conv.weight
        weight1 = weight1[:, in_channels, ...]
        if is_take_prun:
            index1, weight1 = self._prun(weight1)
            self.conv1.cbr.bn = self._reassign_batchnorm(
                'conv1.cbr.bn',
                self.conv1.cbr.bn, index1)
        self._reassign_conv('conv1.cbr.conv', weight1)

        weight_identity = self.identity_layer.cbr.conv.weight
        weight_identity = weight_identity[:, in_channels, ...]
        if is_take_prun:
            weight_identity = weight_identity[index1, ...]
            self._reassign_batchnorm('identity_layer.cbr.bn', self.identity_layer.cbr.bn, index1)
        self._reassign_conv('identity_layer.cbr.conv', weight_identity)

        def _forward(self, x):
            print(self)
            conv3 = self.conv1(x)
            identity = self.identity_layer(x)
            return self.relu(conv3 + identity)
        # print(self)
        self._forward = partial(_forward, self)

        if is_take_prun:
            return index1


def conv_batchnorm_relu_prun(self, in_channels=None):
    weight = self.cbr.conv.weight
    if in_channels is not None:
        weight = weight[:, in_channels, ...]

    index, weight = self._prun(weight)
    print(weight.sum(dim=(1,2,3)))

    self.args = list(self.args)
    # reassign inplanes
    self.args[0] = weight.shape[1]
    # reassign outplanes
    self.args[1] = weight.shape[0]

    self.cbr.conv = nn.Conv2d(*(self.args), **(self.kwargs))
    with torch.no_grad():
        self.cbr.conv.weight.copy_(weight)
        if not self.is_released:
            self._reassign_batchnorm('cbr.bn', self.cbr.bn, index)
    return index


def lasso_group_prun(self, weight, epsilon=1e-5):
    sum = (weight**2).sum(dim=(1, 2, 3))
    epsilon = sum.mean()
    # epsilon = 1e-5
    index = (sum > epsilon).nonzero(as_tuple=True)[0]
    weight = weight[index, ...]
    return index, weight


config_prun = [
    {'block_name': BasicBlock, 'prun_method': basic_block_prun,
        'prun_algorithm': lasso_group_prun},
    {'block_name': BasicBlockTruncate, 'prun_method': basic_block_truncate_prun,
        'prun_algorithm': lasso_group_prun},
    {'block_name': ConvBatchNormRelu, 'prun_method': conv_batchnorm_relu_prun,
        'prun_algorithm': lasso_group_prun}
]
if __name__ == "__main__":
    pl.seed_everything(42)
    model = Resnet34Orig()
    checkpoint_path = 'checkpoint-epoch=199-val_acc_epoch=0.9254.ckpt'
    # checkpoint_path = 'export-checkpoint-epoch=72-val_acc_epoch=0.9218.ckpt'
    if device == 'cpu' or device == 'tpu':
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    # state_dict = checkpoint
    model.migrate(state_dict, force=True)
    # # print(model)
    # model.release()
    # print(model)
    prun_model = PruningModel(model, block_names, config_prun)
    # release(model.model)
    prun_model.prun()
    print(model)
    x = torch.Tensor(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
