from functools import partial

from model.base import ConvBatchNormRelu
from model.resnet.basic import BasicBlock, BasicBlockTruncate
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
from model.utils import basic_block_prun, basic_block_truncate_prun, conv_batchnorm_relu_prun, lasso_group_prun

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
    'model.conv1',
    'model.layer1.0',
    'model.layer1.1',
    'model.layer1.2',
    'model.layer2.0',
    'model.layer2.1',
    'model.layer2.2',
    'model.layer2.3',
    'model.layer3.0',
    'model.layer3.1',
    'model.layer3.2',
    'model.layer3.3',
    'model.layer3.4',
    'model.layer3.5',
    'model.layer4.0',
    'model.layer4.1',
    'model.layer4.2',
]


config_prun = [
    {'block_name': BasicBlock, 'prun_method': basic_block_prun,
        'prun_algorithm': lasso_group_prun},
    {'block_name': BasicBlockTruncate, 'prun_method': basic_block_truncate_prun,
        'prun_algorithm': lasso_group_prun},
    {'block_name': ConvBatchNormRelu, 'prun_method': conv_batchnorm_relu_prun,
        'prun_algorithm': lasso_group_prun}
]
if __name__ == "__main__":
    pl.seed_everything(42)
    model = Resnet34Orig()
    checkpoint_path = 'checkpoint-epoch=199-val_acc_epoch=0.9254.ckpt'
    # checkpoint_path = 'export-checkpoint-epoch=72-val_acc_epoch=0.9218.ckpt'
    if device == 'cpu' or device == 'tpu':
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    # state_dict = checkpoint
    model.migrate(state_dict, force=True)
    # # print(model)
    # model.release()
    # print(model)
    prun_model = PruningModel(model, block_names, config_prun)
    # release(model.model)
    prun_model.prun()
    print(model)
    x = torch.Tensor(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
