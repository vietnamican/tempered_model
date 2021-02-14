import os

import torch
import torchvision
from torchvision import transforms
from torchsummary import summary
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

try:
    import torch_xla.core.xla_model as xm
except:
    pass

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model.resnet import Resnet34


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

device = 'cpu'

if device == 'tpu':
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, num_workers=4, sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, num_workers=4, sampler=test_sampler)
else:
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


def remove_module_with_prefix(state_dict, prefix='block1'):
    new_state_dict = {}
    for name, p in state_dict.items():
        if not name.startswith(prefix):
            new_state_dict[name] = p
    return new_state_dict


if __name__ == '__main__':
    pl.seed_everything(42)

    ####################################
    ##     Training original          ##
    ####################################
    # resnet34 = torchvision.models.resnet.resnet34(pretrained=True)
    model = Resnet34('tuning')
    # model.migrate_from_torchvision(resnet34.state_dict())
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='resnet34_logs',
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

    ####################################
    ##       Tuning truncate          ##
    ####################################
    # model = Resnet50Truncate()
    # original_model_checkpoint_path = './resnet50_logs/version_1/checkpoints/checkpoint-epoch=24-val_acc_epoch=0.8545.ckpt'
    # if device == 'cpu' or device == 'tpu':
    #     original_checkpoint = torch.load(
    #         original_model_checkpoint_path, map_location=lambda storage, loc: storage)
    # else:
    #     original_checkpoint = torch.load(original_model_checkpoint_path)
    # original_state_dict = original_checkpoint['state_dict']
    # model.orig_model.migrate(original_state_dict)
    # # original_state_dict = remove_module_with_prefix(original_state_dict, 'layer3')
    # original_state_dict = remove_module_with_prefix(
    #     original_state_dict, 'layer4')
    # model.migrate(original_state_dict)
    # model.freeze_except_prefix('layer4')
    # # model.defrost_with_prefix('layer4')
    # logger = TensorBoardLogger(
    #     save_dir=os.getcwd(),
    #     name='resnet50_all_in_one_logs',
    #     log_graph=True
    # )
    # loss_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath='',
    #     filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
    #     save_top_k=-1,
    #     mode='min',
    # )
    # if device == 'tpu':
    #     trainer = pl.Trainer(max_epochs=30,
    #                          tpu_cores=8,
    #                          logger=logger,
    #                          callbacks=[loss_callback])
    # else:
    #     trainer = pl.Trainer(max_epochs=30,
    #                          logger=logger,
    #                          callbacks=[loss_callback])
    # trainer.fit(model, trainloader, testloader)

    ####################################
    ##       Testing truncate         ##
    ####################################
    # model = Resnet50Truncate()
    # truncate_model_checkpoint_path = './resnet50_logs/version_1/checkpoints/checkpoint-epoch=24-val_acc_epoch=0.8545.ckpt'
    # if device == 'cpu' or device == 'tpu':
    #     truncate_checkpoint = torch.load(truncate_model_checkpoint_path, map_location=lambda storage, loc: storage)
    # else:
    #     truncate_checkpoint = torch.load(truncate_model_checkpoint_path)
    # truncate_state_dict = truncate_checkpoint['state_dict']
    # model.migrate(truncate_state_dict)
    # logger = TensorBoardLogger(
    #     save_dir=os.getcwd(),
    #     name='resnet50_all_in_one_test_logs',
    #     log_graph=True
    # )
    # loss_callback = ModelCheckpoint(
    #     monitor='tess_loss',
    #     dirpath='',
    #     filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
    #     save_top_k=-1,
    #     mode='min',
    # )
    # acc_callback = ModelCheckpoint(
    #     monitor='test_acc_epoch',
    #     dirpath='',
    #     filename='checkpoint-{epoch:02d}-{test_acc_epoch:.4f}',
    #     save_top_k=-1,
    #     mode='min',
    # )
    # trainer = pl.Trainer(max_epochs=20,
    #                      logger=logger,
    #                      callbacks=[loss_callback, acc_callback])
    # trainer.test(model, testloader)
