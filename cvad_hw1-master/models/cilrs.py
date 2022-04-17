import torch.nn as nn


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CarlaNet,self).__init__()

    def forward(self, img, command):
        pass
