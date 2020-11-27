"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from kernet.models.resnet import Bottleneck, ResNet


class ResNet101(ResNet):
    def __init__(self, opt):
        super(ResNet101, self).__init__(Bottleneck, [
            3, 4, 23, 3], in_channels=opt.in_channels, num_classes=opt.n_classes)
        self.print_network(self)
