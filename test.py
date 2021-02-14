import torch
import torchvision
from torchsummary import summary
from torch import nn
from torch.nn import functional as F

from model import OriginalModel, VGG, FusionlModel, Fusion2
from model.resnet import Bottleneck, Resnet50, Resnet50Truncate, Resnet34
from model.resnet.resnet34 import BasicBlock


if __name__ == '__main__':
#     # model = Resnet50()
#     # model = Resnet50Truncate()
    # resnet34 = torchvision.models.resnet.resnet34(pretrained=True)
    model = Resnet34('truncating')
    for name, p in model.named_modules():
        print(p)
        break

    # checkpoint_path = './resnet34_logs/version_0/checkpoints/checkpoint-epoch=07-val_acc_epoch=0.7394.ckpt'
    # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    # state_dict = checkpoint['state_dict']
    # model.migrate(state_dict)

    # model.freeze_except_prefix('truncate')


    # model = BasicBlock(128, 128)
    # for name, p in model.state_dict().items():
    #     print(name)
    # model = torchvision.models.resnet.resnet50()
    # summary(
    #     model,
    #     (3, 32, 32),
    #     col_names=[
    #         "input_size",
    #         "output_size",
    #         "num_params",
    #         "kernel_size",
    #         "mult_adds"
    #     ],
    #     depth=5)
# From 14728266 parameters to 9416010 parameters, reduce 5312256 parameters, accuracy reduce 0.65% (without refinement)
# From 14728266 parameters to 10006602 parameters, reduce 4721664 parameters, accuracy reduce 0.04% (without refinement)
