import os

import torch
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import OriginalModel, VGG


transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=36)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=36)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    pl.seed_everything(42)
    # model = VGG('VGG19')
    model = OriginalModel()
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='original2_logs',
        log_graph=True,
    )
    loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
    )
    acc_callback = ModelCheckpoint(
        monitor='val_acc_epoch',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_acc_epoch:.4f}',
        save_top_k=-1,
        mode='max',
    )
    trainer = pl.Trainer(max_epochs=20,
                         logger=logger,
                         callbacks=[loss_callback, acc_callback])
    trainer.fit(model, trainloader, testloader)
