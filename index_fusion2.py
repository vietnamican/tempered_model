import os

import torch
import torchvision
from torchvision import transforms
from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import OriginalModel, Fusion2


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
    # model = Fusion2()
    
    # original_checkpoint_path = "./original_logs/version_1/checkpoints/checkpoint-epoch=19-val_acc_epoch=0.8789.ckpt"
    # if device == 'cpu':
    #     original_checkpoint = torch.load(original_checkpoint_path, map_location=lambda storage, loc: storage)
    # else:
    #     original_checkpoint = torch.load(original_checkpoint_path)
    # original_state_dict = original_checkpoint['state_dict']
    # model.orig_model.migrate(original_state_dict)
    # original_state_dict = remove_module_with_prefix(original_state_dict, 'block4')
    # original_state_dict = remove_module_with_prefix(original_state_dict, 'block5')
    # model.migrate(original_state_dict)
    # model.freeze_with_prefix('block1')
    # model.freeze_with_prefix('block2')
    # model.freeze_with_prefix('block3')
    # model.freeze_with_prefix('orig_model')

    # logger = TensorBoardLogger(
    #     save_dir=os.getcwd(),
    #     name='all_in_one_logs',
    #     log_graph=True
    # )
    # loss_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath='',
    #     filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
    #     save_top_k=-1,
    #     mode='min',
    # )
    # trainer = pl.Trainer(max_epochs=20,
    #                      logger=logger,
    #                      callbacks=[loss_callback])
    # trainer.fit(model, trainloader, testloader)

    all_in_one_checkpoint_path = "./all_in_one_logs/version_1/checkpoints/checkpoint-epoch=19-val_loss=0.0025.ckpt"
    if device == 'cpu':
        all_in_one_checkpoint = torch.load(all_in_one_checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        all_in_one_checkpoint = torch.load(all_in_one_checkpoint_path)
    all_in_one_state_dict = all_in_one_checkpoint['state_dict']
    model = Fusion2()
    model.migrate(all_in_one_state_dict)
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='all_in_one_test_logs',
        log_graph=True
    )
    loss_callback = ModelCheckpoint(
        monitor='tess_loss',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
    )
    acc_callback = ModelCheckpoint(
        monitor='test_acc_epoch',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{test_acc_epoch:.4f}',
        save_top_k=-1,
        mode='min',
    )
    trainer = pl.Trainer(max_epochs=20,
                         logger=logger,
                         callbacks=[loss_callback, acc_callback])
    trainer.test(model, testloader)