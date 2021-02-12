import os
import pickle
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import Original1Full


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
    trainset, batch_size=1, shuffle=False, num_workers=1)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=1)

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
    # original_checkpoint_path = "./fusion_logs/version_18/checkpoints/checkpoint-epoch=18-val_acc_epoch=0.8938.ckpt"
    if device == 'cpu':
        original_checkpoint = torch.load(original_checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        original_checkpoint = torch.load(original_checkpoint_path)
    original_state_dict = original_checkpoint['state_dict']
    original1model = Original1Full()
    original1model.migrate(original_state_dict)

    traindataset = []
    for x, y in tqdm(trainloader):
        x, y1, y2, y3, y4, y5, y = original1model(x)
        x = x.squeeze(0)
        y1 = y1.squeeze(0)
        y2 = y2.squeeze(0)
        y3 = y3.squeeze(0)
        y4 = y4.squeeze(0)
        y5 = y5.squeeze(0)
        y = y.squeeze(0)
        traindataset.append(
            {
                'x': x, 
                'y1': y1,
                'y2': y2,
                'y3': y3,
                'y4': y4,
                'y5': y5,
                'y': y
            }
        )
    with open('trainset.ckpt', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    testdataset = []
    for x, y in tqdm(testloader):
        x, y1, y2, y3, y4, y5, y = original1model(x)
        x = x.squeeze(0)
        y1 = y1.squeeze(0)
        y2 = y2.squeeze(0)
        y3 = y3.squeeze(0)
        y4 = y4.squeeze(0)
        y5 = y5.squeeze(0)
        y = y.squeeze(0)
        testdataset.append(
            {
                'x': x, 
                'y1': y1,
                'y2': y2,
                'y3': y3,
                'y4': y4,
                'y5': y5,
                'y': y
            }
        )
    with open('testset.ckpt', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)