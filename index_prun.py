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

def release(model):
    is_self = True
    for module in model.modules():
        if is_self:
            is_self = False
            continue
        if hasattr(module, 'release'):
            module.release()

if __name__ == "__main__":
    model = PruningModel(Resnet34Temper())
    release(model.model)
    model.prun()
    print(model)