# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_policy_backbone():
    """Prefer the bundled ImageNet encoder; otherwise use torchvision's cache."""
    from pathlib import Path
    from torchvision import models
    weights = Path(__file__).resolve().parent / 'weights/resnet101-cd907fc2.pth'
    if weights.is_file():
        model = models.resnet101(weights=None)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        return model
    return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

class myResnet(nn.Module):

    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=6):
        x = img
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        fc = x.mean(3).mean(2)
        if fc.dim() == 2 and fc.size(0) == 1:
            fc = fc.squeeze(0)
        att = F.adaptive_avg_pool2d(x, [att_size, att_size])
        if att.dim() == 4:
            att = att.permute(0, 2, 3, 1)
            if att.size(0) == 1:
                att = att.squeeze(0)
        return (fc, att)
