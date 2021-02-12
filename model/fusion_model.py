import torch
from torch import nn
import pytorch_lightning as pl

from .base import Base, ConvBatchNormRelu

class FusionlModel(Base):
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
        optimizer = torch.optim.Adam(self.parameters())
        # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        #                                         optimizer, 
        #                                         lr_lambda=lambda epoch: 0.9
        # )
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        return {'optimizer': optimizer}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        items['accuracy'] = self.accuracy.compute().item()
        return items

    def freeze_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = False
        