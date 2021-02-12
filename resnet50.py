import torch
from torch import nn
import torchvision
from torchsummary import summary

if __name__ == '__main__':
    resnet50 = torchvision.models.resnet.resnet50(pretrained=True)
    # for name, p in resnet50.state_dict().items():
    #     print(name)
    # print(resnet50.layer1[0].bn1.running_mean.shape)
    summary(
        resnet50,
        (3, 32, 32),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            # "kernel_size",
            # "mult_adds"
        ],
        depth=5
    )
