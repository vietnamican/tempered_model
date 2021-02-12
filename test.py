import torch
import torchvision
from torchsummary import summary

from model import OriginalModel, VGG, FusionlModel, Fusion2
from model.resnet import Bottleneck, Resnet50

if __name__ == '__main__':
    model = Resnet50()
    # model = torchvision.models.resnet.resnet50()
    summary(
        model,
        (3, 32, 32),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds"
        ],
        depth=5)

# From 14728266 parameters to 9416010 parameters, reduce 5312256 parameters, accuracy reduce 0.65% (without refinement)
# From 14728266 parameters to 10006602 parameters, reduce 4721664 parameters, accuracy reduce 0.04% (without refinement)
