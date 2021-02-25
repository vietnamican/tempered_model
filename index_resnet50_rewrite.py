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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model.resnet.resnet50_rewrite import Resnet50
from model.tempered_model import LogitTuneModel


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

orig_module_names = [
    'orig.conv1',
    'orig.layer1',
    'orig.layer2',
    'orig.layer3.0',
    'orig.layer3.1',
    ['orig.layer3.2', 'orig.layer3.3'],
    ['orig.layer3.4', 'orig.layer3.5'],
    'orig.layer4.0',
    ['orig.layer4.1', 'orig.layer4.2'],
    'orig.avgpool',
    'orig.flatten',
    'orig.fc'
]

tempered_module_names = [
    'tempered.conv1',
    'tempered.layer1',
    'tempered.layer2',
    'tempered.layer3.0',
    'tempered.layer3.1',
    'tempered.layer3.2',
    'tempered.layer3.3',
    'tempered.layer4.0',
    'tempered.layer4.1',
    'tempered.avgpool',
    'tempered.flatten',
    'tempered.fc'
]

is_trains = [
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    True,
    False,
    False,
    False,
]

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
    mode = 'logittuning'
    if mode == 'training':
        ####################################
        ##     Training original          ##
        ####################################
        resnet50 = torchvision.models.resnet.resnet50(pretrained=True)
        model = Resnet50('training', orig_module_names, tempered_module_names, is_trains)
        model.migrate_from_torchvision(resnet50.state_dict())
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name='resnet50_rewrite_training_logs',
            log_graph=True,
            version=0
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
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        if device == 'tpu':
            trainer = pl.Trainer(
                progress_bar_refresh_rate=20,
                tpu_cores=8,
                max_epochs=200,
                logger = logger,
                callbacks=[loss_callback, acc_callback, lr_monitor]
            )
        else:
            trainer = pl.Trainer(
                max_epochs=200,
                logger = logger,
                callbacks=[loss_callback, acc_callback, lr_monitor]
            )
        trainer.fit(model, trainloader, testloader)
    elif mode == 'temper':
        ###################################
        #             Temper             ##
        ###################################
        model = Resnet50('temper', orig_module_names, tempered_module_names, is_trains)
        # checkpoint_path = './resnet18_logs/version_0/checkpoints/checkpoint-epoch=20-val_acc_epoch=0.7936.ckpt'
        # if device == 'cpu' or device == 'tpu':
        #     checkpoint = torch.load(
        #         checkpoint_path, map_location=lambda storage, loc: storage)
        # else:
        #     checkpoint = torch.load(checkpoint_path)
        # state_dict = checkpoint['state_dict']
        # model.migrate_from_torchvision(state_dict)
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name='resnet50_rewrite_temper_logs',
            log_graph=True
        )
        loss_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='',
            filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
            save_top_k=-1,
            mode='min',
        )
        if device == 'tpu':
            trainer = pl.Trainer(
                progress_bar_refresh_rate=20,
                tpu_cores=8,
                max_epochs=200,
                logger = logger,
                callbacks=[loss_callback]
            )
        else:
            trainer = pl.Trainer(
                max_epochs=200,
                logger = logger,
                callbacks=[loss_callback]
            )
        trainer.fit(model, trainloader, testloader)
    elif mode == 'tuning':
        ###################################
        #             Tuning             ##
        ###################################
        model = Resnet50('tuning', orig_module_names, tempered_module_names, is_trains)
        # truncate_model_checkpoint_path = './resnet50_logs/version_1/checkpoints/checkpoint-epoch=24-val_acc_epoch=0.8545.ckpt'
        # if device == 'cpu' or device == 'tpu':
        #     truncate_checkpoint = torch.load(truncate_model_checkpoint_path, map_location=lambda storage, loc: storage)
        # else:
        #     truncate_checkpoint = torch.load(truncate_model_checkpoint_path)
        # truncate_state_dict = truncate_checkpoint['state_dict']
        # model.migrate(truncate_state_dict)
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name='resnet50_rewrite_tuning_logs',
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
        if device == 'tpu':
            trainer = pl.Trainer(
                progress_bar_refresh_rate=20,
                tpu_cores=8,
                max_epochs=100,
                logger = logger,
                callbacks=[loss_callback, acc_callback]
            )
        else:
            trainer = pl.Trainer(
                max_epochs=100,
                logger = logger,
                callbacks=[loss_callback, acc_callback]
            )
        trainer.fit(model, trainloader, testloader)
    elif mode == 'inference':
        ###################################
        #            Testing             ##
        ###################################
        model = Resnet50('inference', orig_module_names, tempered_module_names, is_trains)
        # truncate_model_checkpoint_path = './resnet50_logs/version_1/checkpoints/checkpoint-epoch=24-val_acc_epoch=0.8545.ckpt'
        # if device == 'cpu' or device == 'tpu':
        #     truncate_checkpoint = torch.load(truncate_model_checkpoint_path, map_location=lambda storage, loc: storage)
        # else:
        #     truncate_checkpoint = torch.load(truncate_model_checkpoint_path)
        # truncate_state_dict = truncate_checkpoint['state_dict']
        # model.migrate(truncate_state_dict)
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name='resnet50_rewrite_inference_logs',
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
        if device == 'tpu':
            trainer = pl.Trainer(
                progress_bar_refresh_rate=20,
                tpu_cores=8,
                max_epochs=30,
                logger = logger,
                callbacks=[loss_callback, acc_callback]
            )
        else:
            trainer = pl.Trainer(
                max_epochs=30,
                logger = logger,
                callbacks=[loss_callback, acc_callback]
            )
        trainer.test(model, testloader)
    elif mode == 'logittuning':
        model = LogitTuneModel(Resnet50, orig_module_names, tempered_module_names, is_trains, device=device, checkpoint_path="")
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name='resnet50_rewrite_logittuning_logs',
            log_graph=True
        )
        loss_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='',
            filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
            save_top_k=-1,
            mode='min',
        )
        if device == 'tpu':
            trainer = pl.Trainer(
                progress_bar_refresh_rate=20,
                tpu_cores=8,
                max_epochs=200,
                logger = logger,
                callbacks=[loss_callback]
            )
        else:
            trainer = pl.Trainer(
                max_epochs=200,
                logger = logger,
                callbacks=[loss_callback]
            )
        trainer.fit(model, trainloader, testloader)