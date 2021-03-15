from functools import partial

import torch
from torch import nn

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
    # print(weight.sum(dim=(1,2,3)))

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