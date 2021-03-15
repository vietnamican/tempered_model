import os

import torch
from torch import nn as nn
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

from model.resnet.resnet34 import Resnet34, Resnet34Orig, Resnet34Prun, Resnet34PrunTemper, Resnet34PrunTemperTemper, Resnet34Temper
from model.pruning_model import PruningModel
# from model.resnet.resnet50 import Resnet50, Resnet50Orig, Resnet50Temper
from model.tempered_model import LogitTuneModel

device = 'cpu'

def release(model):
    is_self = True
    for module in model.modules():
        if is_self:
            is_self = False
            continue
        if hasattr(module, 'release'):
            module.release()

block_names = [
    'conv1',
    'layer1.0',
    'layer1.1',
    'layer1.2',
    'layer2.0',
    'layer2.1',
    'layer2.2',
    'layer2.3',
    'layer3.0',
    'layer3.1',
    'layer3.2',
    'layer3.3',
    'layer3.4',
    'layer3.5',
    'layer4.0',
    'layer4.1',
    'layer4.2',
]

if __name__ == "__main__":
    pl.seed_everything(42)
    model = Resnet34Temper()
    # checkpoint_path = 'checkpoint-epoch=199-val_acc_epoch=0.9254.ckpt'
    # # checkpoint_path = 'export-checkpoint-epoch=72-val_acc_epoch=0.9218.ckpt'
    # if device == 'cpu' or device == 'tpu':
    #     checkpoint = torch.load(
    #         checkpoint_path, map_location=lambda storage, loc: storage)
    # else:
    #     checkpoint = torch.load(checkpoint_path)
    # state_dict = checkpoint['state_dict']
    # # state_dict = checkpoint
    # model.migrate(state_dict, force=True)
    # # print(model)
    # model.release()
    # print(model)
    prun_model = PruningModel(model, block_names)
    # release(model.model)
    prun_model.prun()
    print(model)
    x = torch.Tensor(4, 3, 32, 32)
    y = model(x)
    print(y.shape)