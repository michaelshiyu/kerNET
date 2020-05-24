"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import kernet_future.utils as utils
from kernet_future.layers.klinear import _kLayer, kLinear
from kernet_future.models.resnet import BasicBlock, Bottleneck, ResNet


logger = logging.getLogger()


class BasicBlockNoOutputReLU(BasicBlock):
  """
  The BasicBlock with the output ReLU nonlinearity stripped off.
  """
  def __init__(self, *args, **kwargs):
    super(BasicBlockNoOutputReLU, self).__init__(*args, **kwargs)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    return out


class BottleneckNoOutputReLU(Bottleneck):
  """
  The Bottleneck with the output ReLU nonlinearity stripped off.
  """
  def __init__(self, *args, **kwargs):
    super(BottleneckNoOutputReLU, self).__init__(*args, **kwargs)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    return out


class kResNet(ResNet):
  def __init__(self, opt, centers, block, num_blocks, num_classes=10):
    super(kResNet, self).__init__(block, num_blocks, num_classes, skip_layer=['layer5', 'fc'])
    self.opt = opt
    if opt.activation == 'tanh':
      self.kernel = 'nn_tanh'
      self.evaluation = 'direct'
    elif opt.activation == 'sigmoid':
      self.kernel = 'nn_sigmoid'
      self.evaluation = 'direct'
    elif opt.activation == 'relu':
      self.kernel = 'nn_relu'
      self.evaluation = 'direct'
    elif opt.activation == 'reapen':
      self.kernel = 'nn_reapen'
      self.evaluation = 'direct'
    elif opt.activation == 'gaussian':
      self.kernel = 'gaussian'
      self.evaluation = 'indirect'
    else:
      raise NotImplementedError()

    self.layer5 = self._make_layer_no_output_relu(block, 512, num_blocks[3], stride=2)

    if centers is not None:
      # centers is a tuple of (input, target)
      centers = utils.supervised_sample(centers[0], centers[1], opt.n_centers).clone().detach()
    else:
      centers = None
    fc = kLinear(in_features=512*block.expansion, out_features=num_classes,
                 kernel=self.kernel, evaluation=self.evaluation, centers=centers, sigma=opt.sigma)
    if opt.memory_efficient:
      fc = utils.to_committee(fc, opt.expert_size)
    self.fc = fc

  def _make_layer_no_output_relu(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides[:-1]:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion

    # last block does not have the output relu
    stride = strides[-1]
    last_block = BasicBlockNoOutputReLU if block is BasicBlock else BottleneckNoOutputReLU
    layers.append(last_block(self.in_planes, planes, stride))
    return nn.Sequential(*layers)

  def forward(self, input, update_centers=True):
    if update_centers:
      self.update_centers()
    return super().forward(input)

  def update_centers(self):
    # update centers
    if self.fc.evaluation == 'indirect':
      logger.debug('Updating centers for {} fc...'.format(self.__class__.__name__))
      update_fn = lambda x: self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
      _kLayer.update(self.fc, update_fn)

  @staticmethod
  def modify_commandline_options(parser, **kwargs):
    parser = _kLayer.modify_commandline_options(parser, **kwargs)
    parser.add_argument('--sigma', type=float, default=1.,
      help='The optional sigma hyperparameter for layer fc.')
    parser.add_argument('--n_centers', type=int, default=1000,
                        help='The number of centers for the kernelized fc layer. Note that kernels evaluated directly do not need centers. For them, this param has no effect.')
    return parser

  def split(self, n_parts, mode=1):
    """
    Split the model into n_parts parts for modular training.
    The last part is always the (kernelized) output layer.

    Args:
      n_parts (int): The number of parts to split.
      mode (1 or 2): How to split the model. Default: 1.

    Returns:
      models (tuple): A tuple of n_parts models.
      trainables (tuple): A tuple of n_parts sets of trainable params.
    """
    if n_parts < 1:
      raise ValueError('n_parts should be at least 1, got {} instead'.format(n_parts))
    if n_parts > 6:
      logger.warning('{} can be split into at most 6 parts. Splitting it into ' +
                    '6 parts instead of the requested {} parts...'.format(self.__class__.__name__, n_parts))
      n_parts = 6

    # in modular training (assumed training mode given split has been called),
    # update_centers do not need to be performed at each forward call. Define
    # new forward to bypass the forward function defined for the entire model
    self.forward = functools.partial(self.forward, update_centers=False)

    if mode == 1:
      models, trainables = self._split1(n_parts)
    elif mode == 2:
      models, trainables = self._split2(n_parts)
    else:
      raise ValueError('Invalid split mode: {}'.format(mode))

    logger.debug('Splitting {} into:'.format(self.__class__.__name__))
    for i, t in enumerate(trainables):
      logger.debug('part {}:\n'.format(i + 1) + str(t))

    return models, [_.parameters() for _ in trainables]

  def _split1(self, n_parts):
    """
    Split mode 1.

    Args:
      n_parts (int): The number of parts to split the model into.

    Returns:
      models (tuple): A tuple of modules.
      trainables (tuple): A tuple of submodules, with each being the submodule to be trained in the
        corresponding module.
    """
    output_layer = list(self.children())[-1]
    if n_parts == 1:
      models, trainables = (self,), (self,)
    elif n_parts == 2:
      hidden_layers = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
      t1 = hidden_layers
      models, trainables = (hidden_layers, self), (t1, output_layer)
    elif n_parts == 3:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-2]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[-1]
      models, trainables = (hidden_layers1, hidden_layers2, self), \
                           (t1, t2, output_layer)
    elif n_parts == 4:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-3]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-2]), self.opt)
      hidden_layers3 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[-1]
      t3 = hidden_layers3[-1]
      models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, self), \
                           (t1, t2, t3, output_layer)
    elif n_parts == 5:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-4]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-3]), self.opt)
      hidden_layers3 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-2]), self.opt)
      hidden_layers4 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[-1]
      t3 = hidden_layers3[-1]
      t4 = hidden_layers4[-1]
      models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, hidden_layers4, self), \
                           (t1, t2, t3, t4, output_layer)
    elif n_parts == 6:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-5]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-4]), self.opt)
      hidden_layers3 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-3]), self.opt)
      hidden_layers4 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-2]), self.opt)
      hidden_layers5 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[-1]
      t3 = hidden_layers3[-1]
      t4 = hidden_layers4[-1]
      t5 = hidden_layers5[-1]
      models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, hidden_layers4, hidden_layers5, self), \
                           (t1, t2, t3, t4, t5, output_layer)
    else:
      raise ValueError('Invalid n_parts: {}'.format(n_parts))

    return models, trainables

  def _split2(self, n_parts):
    """
    Split mode 2.

    Args:
      n_parts (int): The number of parts to split the model into.

    Returns:
      models (tuple): A tuple of modules.
      trainables (tuple): A tuple of submodules, with each being the submodule to be trained in the
        corresponding module.
    """
    output_layer = list(self.children())[-1]
    if n_parts == 1:
      models, trainables = (self,), (self,)
    elif n_parts == 2:
      hidden_layers = utils.attach_head(torch.nn.Sequential(*list(self.children())[:5]), self.opt)
      t1 = hidden_layers
      models, trainables = (hidden_layers, self), (t1, output_layer)
    elif n_parts == 3:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:3]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:5]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[3:]
      models, trainables = (hidden_layers1, hidden_layers2, self), \
                           (t1, t2, output_layer)
    elif n_parts == 4:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:2]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:4]), self.opt)
      hidden_layers3 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:5]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[2:]
      t3 = hidden_layers3[-1]
      models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, self), \
                           (t1, t2, t3, output_layer)
    elif n_parts == 5:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:2]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:3]), self.opt)
      hidden_layers3 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:4]), self.opt)
      hidden_layers4 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:5]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[-1]
      t3 = hidden_layers3[-1]
      t4 = hidden_layers4[-1]
      models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, hidden_layers4, self), \
                           (t1, t2, t3, t4, output_layer)
    elif n_parts == 6:
      hidden_layers1 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:1]), self.opt)
      hidden_layers2 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:2]), self.opt)
      hidden_layers3 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:3]), self.opt)
      hidden_layers4 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:4]), self.opt)
      hidden_layers5 = utils.attach_head(torch.nn.Sequential(*list(self.children())[:5]), self.opt)
      t1 = hidden_layers1
      t2 = hidden_layers2[-1]
      t3 = hidden_layers3[-1]
      t4 = hidden_layers4[-1]
      t5 = hidden_layers5[-1]
      models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, hidden_layers4, hidden_layers5, self), \
                           (t1, t2, t3, t4, t5, output_layer)
    else:
      raise ValueError('Invalid n_parts: {}'.format(n_parts))

    return models, trainables
