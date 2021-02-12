import torch
from torch import nn
import pytorch_lightning as pl

from .base import Base, ConvBatchNormRelu


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(Base):
    def __init__(self, vgg_name='VGG16'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(pred, y))
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('val_loss', loss)
        self.log('val_acc_step', self.val_accuracy(pred, y))
        return loss

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        items['accuracy'] = self.accuracy.compute().item()
        return items
    
class OriginalModel(Base):
    def __init__(self):
        super().__init__()
        base_depth = 64
        self.block1 = nn.Sequential(
            ConvBatchNormRelu(3, base_depth, kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth, base_depth,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            ConvBatchNormRelu(base_depth, base_depth*2,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*2, base_depth*2,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            ConvBatchNormRelu(base_depth*2, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            ConvBatchNormRelu(base_depth*4, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.block5 = nn.Sequential(
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(base_depth*8, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(pred, y))
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('val_loss', loss)
        self.log('val_acc_step', self.val_accuracy(pred, y))
        return loss

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        items['accuracy'] = self.accuracy.compute().item()
        return items
