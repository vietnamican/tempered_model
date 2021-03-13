from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from ..base import Base, CReLU, ConvBatchNormRelu


class BasicBlock(Base):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=False,
        with_crelu=False
    ) -> None:
        super().__init__()
        self.downsample = downsample
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvBatchNormRelu(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride, with_crelu=with_crelu)
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
        downsample=False,
        with_crelu=True
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.with_crelu = with_crelu
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if with_crelu:
            planes = planes // 2
        self.conv1 = ConvBatchNormRelu(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride, with_relu=False)
        self.identity_layer = ConvBatchNormRelu(
            inplanes, planes, kernel_size=1, padding=0, bias=False, stride=stride, with_relu=False)
        if with_crelu:
            self.relu = CReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if inplanes == planes and stride == 1:
            self.skip_layer = nn.BatchNorm2d(num_features=inplanes)
        else:
            self.skip_layer = None
        if self.skip_layer is not None:
            print(self.skip_layer)
            def _forward(self, x):
                conv3 = self.conv1(x)
                identity = self.identity_layer(x)
                skip = self.skip_layer(x)
                return self.relu(conv3 + identity + skip)
        else:
            print(self.skip_layer)
            def _forward(self, x):
                conv3 = self.conv1(x)
                identity = self.identity_layer(x)
                return self.relu(conv3 + identity)
        self._forward = partial(_forward, self)
    
    def forward(self, x):
        return self._forward(x)

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch.bacthnorm.running_mean
            running_var = branch.bacthnorm.running_var
            gamma = branch.bacthnorm.weight
            beta = branch.bacthnorm.bias
            eps = branch.bacthnorm.eps
        elif isinstance(branch, nn.BatchNorm2d):
            kernel = torch.zeros(self.inplanes, self.inplanes, 3, 3)
            for i in range(self.inplanes):
                kernel[i, i, 1, 1] = 1
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        return kernel * (gamma / (running_var + eps).sqrt()).reshape(-1, 1, 1, 1), beta - gamma / (running_var + eps).sqrt() * running_mean

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self.conv1.cbr[0].weight, self.conv1.cbr[0].bias
        kernel1x1, bias1x1 = self.identity_layer.cbr[0].weight, self.identity_layer.cbr[0].bias
        if self.skip_layer is not None:
            kernelskip, biasskip = self._fuse_bn_tensor(self.skip_layer)
        else:
            kernelskip, biasskip = 0, 0

        kernel = kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1]) + kernelskip
        bias = bias3x3 + bias1x1 + biasskip
        conv = nn.Conv2d(
            self.inplanes, self.planes, 3, stride=self.stride, padding=1)
        with torch.no_grad():
            conv.weight.copy_(kernel)
            conv.bias.copy_(bias)
        self.forward_path = conv

    def _release(self):
        self.get_equivalent_kernel_bias()
        def _forward(self, x):
            return self.relu(self.forward_path(x))
        self._forward = partial(_forward, self)
        
        delattr(self, 'conv1')
        delattr(self, 'identity_layer')
        if hasattr(self, 'skip_layer'):
            delattr(self, 'skip_layer')
    
    def release(self):
        if not self.is_released:
            self.is_released = True
            is_self = True
            for module in self.modules():
                if is_self:
                    is_self = False
                    continue
                if hasattr(module, 'release'):
                    module.release()
            self._release()

        