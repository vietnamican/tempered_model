import os

import torch
import torchvision
from torchvision import transforms
from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import OriginalModel, VGG, FusionlModel, Fusion1, Fusion1Full


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

def remove_module_except_prefix(state_dict, prefix='block1'):
    new_state_dict = {}
    for name, p in state_dict.items():
        if name.startswith(prefix):
            new_state_dict[name] = p
    return new_state_dict

if __name__ == '__main__':
    pl.seed_everything(42)

    # original_checkpoint_path = "./original_logs/version_1/checkpoints/checkpoint-epoch=19-val_acc_epoch=0.8789.ckpt"
    # if device == 'cpu':
    #     original_checkpoint = torch.load(original_checkpoint_path, map_location=lambda storage, loc: storage)
    # else:
    #     original_checkpoint = torch.load(original_checkpoint_path)
    # original_state_dict = original_checkpoint['state_dict']
    # model = Fusion1()
    # model.freeze_except_prefix('block4')
    # model.original_version.migrate(original_state_dict)
    # original_state_dict = remove_module_with_prefix(original_state_dict, 'block4')
    # model.migrate(original_state_dict)

    # logger = TensorBoardLogger(
    #     save_dir=os.getcwd(),
    #     name='truncate_logs',
    #     log_graph=True
    # )
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath='',
    #     filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
    #     save_top_k=-1,
    #     mode='min',
    # )
    # trainer = pl.Trainer(max_epochs=20,
    #                      logger=logger,
    #                      callbacks=[checkpoint_callback])
    # trainer.fit(model, trainloader, testloader)


    full_model = Fusion1Full()
    original_checkpoint_path = "./original_logs/version_1/checkpoints/checkpoint-epoch=19-val_acc_epoch=0.8789.ckpt"
    if device == 'cpu':
        original_checkpoint = torch.load(original_checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        original_checkpoint = torch.load(original_checkpoint_path)
    original_state_dict = original_checkpoint['state_dict']
    state_dict = remove_module_with_prefix(original_state_dict, 'block4')
    full_model.migrate(state_dict)

    fusion_checkpoint_path = "./truncate_logs/version_0/checkpoints/checkpoint-epoch=19-val_loss=0.0023.ckpt"
    if device == 'cpu':
        fusion_checkpoint = torch.load(fusion_checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        fusion_checkpoint = torch.load(fusion_checkpoint_path)
    fusion_state_dict = fusion_checkpoint['state_dict']
    fusion_state_dict = remove_module_except_prefix(fusion_state_dict, 'block4')
    full_model.migrate(fusion_state_dict)


    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='merge_logs',
        log_graph=True
    )
    loss_callback = ModelCheckpoint(
        monitor='test_loss',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
    )
    acc_callback = ModelCheckpoint(
        monitor='test_acc_epoch',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_acc_epoch:.4f}',
        save_top_k=-1,
        mode='max',
    )
    trainer = pl.Trainer(max_epochs=20,
                         logger=logger,
                         callbacks=[loss_callback, acc_callback])
    trainer.test(full_model, testloader)
