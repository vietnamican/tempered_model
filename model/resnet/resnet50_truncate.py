from model.resnet.resnet50 import Resnet50
import torch
from torch import nn
import pytorch_lightning as pl

from ..base import Base, ConvBatchNormRelu

from .resnet50 import Resnet50


class Bottleneck(Base):
    bottle_ratio = 4

    def __init__(
        self,
        inplanes,
        outplanes,
        stride=1,
        downsample=False
    ):
        super().__init__()
        self.downsample = downsample
        bottleneck_plane = outplanes // 4
        self.cbr1 = ConvBatchNormRelu(
            inplanes, bottleneck_plane, kernel_size=1, padding=0, bias=False)
        self.cbr2 = ConvBatchNormRelu(
            bottleneck_plane, bottleneck_plane, kernel_size=3, padding=1, stride=stride, bias=False)
        self.cbr3 = ConvBatchNormRelu(
            bottleneck_plane, outplanes, kernel_size=1, padding=0, with_relu=False, bias=False)
        if downsample:
            self.identity_layer = ConvBatchNormRelu(
                inplanes, outplanes, kernel_size=1, padding=0, stride=stride, with_relu=False, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        if self.downsample:
            identity = self.identity_layer(identity)
        x += identity
        return self.relu(x)


class Resnet50Truncate(Base):
    start_layer = 3

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 256, downsample=True),
            Bottleneck(256, 256),
            Bottleneck(256, 256)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 512, downsample=True, stride=2),
            Bottleneck(512, 512),
            Bottleneck(512, 512),
            Bottleneck(512, 512)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 1024, downsample=True, stride=2),
            Bottleneck(1024, 1024),
            Bottleneck(1024, 1024),
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 2048, downsample=True, stride=2),
            Bottleneck(2048, 2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
        self.orig_model = Resnet50()
        self.criterion = nn.MSELoss()
        self.test_criterion = nn.CrossEntropyLoss()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x, mode='eval', layer_index=-1):
        if mode == 'training':
            if layer_index == -1:
                y, y1, y2, y3, y4 = self.orig_model(x)
                _y3 = self.layer3(y2)
                _y4 = self.layer4(y3)
                return _y3, _y4, y3, y4
            else:
                ys = self.orig_model(x)
                x = ys[layer_index-1]
                y = ys[layer_index]
                layer_name = 'layer{}'.format(layer_index)
                _y = getattr(self, layer_name)(x)
                return _y, y
        else:
            y = self.maxpool(self.conv1(x))
            y = self.layer4(self.layer3(self.layer2(self.layer1(y))))
            y = self.avgpool(y)
            y = self.fc(torch.flatten(y, 1))
            return y

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        layer_index = optimizer_idx + self.start_layer
        _y, y = self.forward(x, 'training', layer_index)
        loss = self.criterion(_y, y)
        self.log('train_loss{}'.format(layer_index), loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _y3, _y4, y3, y4 = self.forward(x, 'training')
        loss3 = self.criterion(_y3, y3)
        loss4 = self.criterion(_y4, y4)
        self.log('val_loss3', loss3)
        self.log('val_loss4', loss4)
        loss = loss3 + loss4
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.test_criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('test_loss', loss)
        self.log('test_acc_step', self.test_accuracy(pred, y))

    def test_epoch_end(self, outputs):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.layer3.parameters()),
            torch.optim.Adam(self.layer4.parameters()),
        ]

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
            
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        # items['accuracy'] = self.accuracy.compute().item()
        return items

    def migrate_from_torchvision(self, torch_vision_state_dict):
        def remove_num_batches_tracked(state_dict):
            new_state_dict = {}
            for name, p in state_dict.items():
                if not 'num_batches_tracked' in name:
                    new_state_dict[name] = p
            return new_state_dict

        self_state_dict = remove_num_batches_tracked(self.state_dict())
        source_state_dict = remove_num_batches_tracked(torch_vision_state_dict)

        with torch.no_grad():
            for i, ((name, p), (_name, _p)) in enumerate(zip(self_state_dict.items(), source_state_dict.items())):
                if p.shape == _p.shape:
                    print(i, 'copy to {} from {}'.format(name, _name))
                    p.copy_(_p)
