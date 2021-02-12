import torch
from torch import nn
import pytorch_lightning as pl

from .base import Base, ConvBatchNormRelu

class Original2(Base):
    def __init__(self):
        super().__init__()
        base_depth = 64
        self.block1 = nn.Sequential(
            ConvBatchNormRelu(3, base_depth, kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth, base_depth,
                              kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            ConvBatchNormRelu(base_depth, base_depth*2,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*2, base_depth*2,
                              kernel_size=3, padding=1),
        )
        self.block3 = nn.Sequential(
            ConvBatchNormRelu(base_depth*2, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
        )
        self.block4 = nn.Sequential(
            ConvBatchNormRelu(base_depth*4, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
        )
        self.block5 = nn.Sequential(
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(base_depth*8, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(self.maxpool(y1))
        y3 = self.block3(self.maxpool(y2))
        y4 = self.block4(self.maxpool(y3))
        y5 = self.block5(self.maxpool(y4))
        y = self.maxpool(y5)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        return y, y1, y2, y3, y4, y5

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, x1, _, _, _, _, logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(pred, y))
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, x1, _, _, _, _, logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('val_loss', loss)
        self.log('val_acc_step', self.val_accuracy(pred, y))
        return loss

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, x1, _, _, _, _, logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('test_loss', loss)
        self.log('test_acc_step', self.test_accuracy(pred, y))
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

class Fusion2(Base):
    def __init__(self):
        super().__init__()
        base_depth = 64
        self.block1 = nn.Sequential(
            ConvBatchNormRelu(3, base_depth, kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth, base_depth,
                              kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            ConvBatchNormRelu(base_depth, base_depth*2,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*2, base_depth*2,
                              kernel_size=3, padding=1),
        )
        self.block3 = nn.Sequential(
            ConvBatchNormRelu(base_depth*2, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=3, padding=1),
        )
        self.block4 = nn.Sequential(
            ConvBatchNormRelu(base_depth*4, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
        )
        self.block5 = nn.Sequential(
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=3, padding=1),
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(base_depth*8, 10)
        )
        self.orig_model = Original2()
        self.criterion = nn.MSELoss()
        self.test_criterion = nn.CrossEntropyLoss()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x, mode='eval', block_index=-1):
        if mode == 'training':
            if block_index == -1:
                y, y1, y2, y3, y4, y5 = self.orig_model(x)
                _y4 = self.block4(self.maxpool(y3))
                _y5 = self.block5(self.maxpool(y4))
                return _y4, _y5, y4, y5
            else:
                ys = self.orig_model(x)
                x = ys[block_index-1]
                y = ys[block_index]
                block_name = 'block{}'.format(block_index)
                _y = getattr(self, block_name)(self.maxpool(x))
                return _y, y

        else:
            x = self.maxpool(self.block1(x))
            x = self.maxpool(self.block2(x))
            x = self.maxpool(self.block3(x))
            x = self.maxpool(self.block4(x))
            x = self.maxpool(self.block5(x))
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
            return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        _y, y = self.forward(x, 'training', optimizer_idx+4)
        loss = self.criterion(_y, y)
        self.log('train_loss{}'.format(optimizer_idx+4), loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _y4, _y5, y4, y5 = self.forward(x, 'training', -1)
        # loss3 = self.criterion(_y3, y3)
        loss4 = self.criterion(_y4, y4)
        loss5 = self.criterion(_y5, y5)
        # self.log('val_loss3', loss3)
        self.log('val_loss4', loss4)
        self.log('val_loss5', loss5)
        loss = loss4 + loss5
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
            # torch.optim.Adam(self.block3.parameters()),
            torch.optim.Adam(self.block4.parameters()),
            torch.optim.Adam(self.block5.parameters()),
        ]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        # items['accuracy'] = self.accuracy.compute().item()
        return items

    def freeze_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = False

    def freeze_with_prefix(self, prefix):
        for name, p in self.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = False