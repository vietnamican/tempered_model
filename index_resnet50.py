import os

import torch
import torchvision
from torchvision import transforms
from torchsummary import summary
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model.resnet import Resnet50


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

device = 'cpu'

def remove_module_with_prefix(state_dict, prefix='block1'):
    new_state_dict = {}
    for name, p in state_dict.items():
        if not name.startswith(prefix):
            new_state_dict[name] = p
    return new_state_dict

if __name__ == '__main__':
    pl.seed_everything(42)
    resnet50 = torchvision.models.resnet.resnet50(pretrained=True)
    model = Resnet50()
    model.migrate_from_torchvision(resnet50.state_dict())   
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='resnet50_logs',
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
    trainer = pl.Trainer(
        max_epochs=30,
        logger = logger,
        callbacks=[loss_callback, acc_callback]
    )
    trainer.fit(model, trainloader, testloader)