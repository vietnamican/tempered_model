import torch
from torch import nn
import pytorch_lightning as pl

from .base import Base, ConvBatchNormRelu


class Original1(Base):
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
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items

    def freeze_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = False


class Fusion1(Base):
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
        )
        self.original_version = Original1()
        self.original_version.freeze_except_prefix("noexist")
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        orig_logit = self.original_version(x)
        loss = self.criterion(logit, orig_logit)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        orig_logit = self.original_version(x)
        loss = self.criterion(logit, orig_logit)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items

    def freeze_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = False


class Fusion1Full(Base):
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
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            ConvBatchNormRelu(base_depth*4, base_depth*8,
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
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(base_depth*8, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(dim=1)
        self.log('test_loss', loss)
        self.log('test_acc_step', self.test_accuracy(pred, y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        items['accuracy'] = self.accuracy.compute().item()
        return items

    def freeze_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = False


class Original1Full(Base):
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
        self.fc = nn.Sequential(
            nn.Linear(base_depth*8, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        y1 = self.block1(x)
        y1_ = nn.MaxPool2d(2, 2)(y1)
        y2 = self.block2(y1_)
        y2_ = nn.MaxPool2d(2, 2)(y2)
        y3 = self.block3(y2_)
        y3_ = nn.MaxPool2d(2, 2)(y3)
        y4 = self.block4(y3_)
        y4_ = nn.MaxPool2d(2, 2)(y4)
        y5 = self.block5(y4_)
        y5_ = nn.MaxPool2d(2, 2)(y5)
        y = y5_.view(y5_.shape[0], -1)
        y = self.fc(y)
        return y1, y2, y3, y4, y5, y

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
