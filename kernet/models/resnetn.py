"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.nn as nn

from kernet.models import Normalize
from kernet.models.resnet import ResNet


class ResNetN(ResNet):
    """
    ResNetN(ormalized).

    Have the last fc layer of ResNet normalize its input to unit norm.
    """

    def __init__(self, *args, **kwargs):
        super(ResNetN, self).__init__(*args, **kwargs)
        self.layer5 = nn.Sequential(
            self.layer5,
            Normalize()
        )
        self.print_network(self)
