import torch
from torch import nn
import pytorch_lightning as pl

from ..base import Base, ConvBatchNormRelu


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
            skip = 0
        conv3 = self.conv1(x)
        identity = self.identity_layer(x)

        x = conv3 + identity + skip

        return self.relu(x)
