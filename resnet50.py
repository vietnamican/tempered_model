import torch
from torch import nn
import torchvision
from torchsummary import summary

from model.resnet import Resnet50

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

resnet50_state_dict_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

if __name__ == '__main__':
    resnet50 = torchvision.models.resnet.resnet50(pretrained=True)
    resnet_state_dict = load_state_dict_from_url(resnet50_state_dict_url)
    model = Resnet50()
    model.migrate_from_torchvision(resnet50.state_dict())
    resnet50.eval()
    model.eval()
    inp = torch.Tensor(1, 3, 32, 32)
    out1 = resnet50(inp)
    out2 = model(inp)
    # if true, the rewrite code is right
    print(torch.allclose(out1, out2))
