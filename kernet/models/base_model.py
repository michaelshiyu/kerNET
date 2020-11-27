"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel


logger = logging.getLogger()


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def get_actual_model(self, net):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def print_network(self, net):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_actual_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)
