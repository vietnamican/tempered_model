import os

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import OriginalModel, VGG, Fusion1, Original1Full, Original1


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
    original_checkpoint_path = "./original_logs/version_1/checkpoints/checkpoint-epoch=19-val_acc_epoch=0.8789.ckpt"
    if device == 'cpu':
        original_checkpoint = torch.load(original_checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        original_checkpoint = torch.load(original_checkpoint_path)
    original_state_dict = original_checkpoint['state_dict']
    original1model = Original1Full()
    original1model.migrate(original_state_dict) 

    # truncate_checkpoint_path = "./truncate_logs/version_0/checkpoints/checkpoint-epoch=19-val_loss=0.0740.ckpt"
    # if device == 'cpu':
    #     truncate_checkpoint = torch.load(truncate_checkpoint_path, map_location=lambda storage, loc: storage)
    # else:
    #     truncate_checkpoint = torch.load(truncate_checkpoint_path)
    # truncate_state_dict = truncate_checkpoint['state_dict']
    # truncate_model = Fusion1()
    # truncate_model.migrate(truncate_state_dict)
    # truncate_model.original_version.migrate(original_state_dict)

    # orig1 = Original1()
    # orig1.migrate(original_state_dict)
    # criterion = nn.MSELoss()
    # for x, y in trainloader:
    #     _x0 = truncate_model(x)
    #     # _, x1, _, _, _, _ = original1model(x)
    #     _x1 = truncate_model.original_version(x)
    #     # _x2 = orig1(x)
    #     print(criterion(_x0, _x1))
    #     print(torch.allclose(_x0, _x1))
        # print(torch.allclose(x1, _x2))

    trainer = pl.Trainer(max_epochs=20,
                        #  logger=logger,
                        #  callbacks=[loss_callback, acc_callback]
                        )
    trainer.test(original1model, testloader)