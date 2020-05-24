"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import kernet.utils as utils
from kernet.models.klenet5 import  kLeNet5
from kernet.layers.klinear import _kLayer, kLinear


logger = logging.getLogger()

class k3LeNet5(kLeNet5):
  def __init__(self, opt, centers=None, *args, **kwargs):
    super(k3LeNet5, self).__init__(opt, *args, **kwargs)

    logger.warning('k3LeNet5 does not have a purely linear layer between the ' +
                  'kernelized components and the rest of the model, which may ' +
                  'worsen performance')
    if centers is not None:
      # centers is a tuple of (input, target)
      centers1 = utils.supervised_sample(centers[0], centers[1], opt.n_centers1).clone().detach()
      centers2 = utils.supervised_sample(centers[0], centers[1], opt.n_centers2).clone().detach()
      centers3 = utils.supervised_sample(centers[0], centers[1], opt.n_centers3).clone().detach()
    else:
      centers1, centers2, centers3 = None, None, None

    self.fc1 = kLinear(in_features=self.feat_len, out_features=120,
        kernel=self.kernel, evaluation=self.evaluation, centers=centers1, sigma=opt.sigma1)
    self.fc2 = kLinear(in_features=120, out_features=84,
        kernel=self.kernel, evaluation=self.evaluation, centers=centers2, sigma=opt.sigma2)
    self.fc3 = kLinear(in_features=84, out_features=10,
        kernel=self.kernel, evaluation=self.evaluation, centers=centers3, sigma=opt.sigma3)

    if opt.memory_efficient:
      self.fc1 = utils.to_committee(self.fc1, opt.expert_size)
      self.fc2 = utils.to_committee(self.fc2, opt.expert_size)
      self.fc3 = utils.to_committee(self.fc3, opt.expert_size)


  def update_centers(self):
    # update centers
    if self.fc1.evaluation == 'indirect':
      # not using actual class name here for potential inheritance
      logger.debug('Updating centers for {} fc1...'.format(self.__class__.__name__))
      update_fn = lambda x: self.conv2(self.conv1(x))
      _kLayer.update(self.fc1, update_fn)
    if self.fc2.evaluation == 'indirect':
      logger.debug('Updating centers for {} fc2...'.format(self.__class__.__name__))
      update_fn = lambda x: self.fc1(self.conv2(self.conv1(x)))
      _kLayer.update(self.fc2, update_fn)
    if self.fc3.evaluation == 'indirect':
      logger.debug('Updating centers for {} fc3...'.format(self.__class__.__name__))
      update_fn = lambda x: self.fc2(self.fc1(self.conv2(self.conv1(x))))
      _kLayer.update(self.fc3, update_fn)


  @staticmethod
  def modify_commandline_options(parser, **kwargs):
    parser = kLeNet5.modify_commandline_options(parser, **kwargs)

    parser.add_argument('--n_centers1', type=int, default=1000,
      help='The number of centers for the kernelized fc layer 1. Note that kernels evaluated directly do not need centers. For them, this param has no effect.')
    parser.add_argument('--n_centers2', type=int, default=1000,
      help='The number of centers for the kernelized fc layer 2. Note that kernels evaluated directly do not need centers. For them, this param has no effect.')
    parser.add_argument('--n_centers3', type=int, default=1000,
      help='The number of centers for the kernelized fc layer 3. Note that kernels evaluated directly do not need centers. For them, this param has no effect.')
    parser.add_argument('--sigma1', type=float, default=1.,
      help='The optional sigma hyperparameter for layer fc1.')
    parser.add_argument('--sigma2', type=float, default=5.,
      help='The optional sigma hyperparameter for layer fc2.')
    parser.add_argument('--sigma3', type=float, default=9.,
      help='The optional sigma hyperparameter for layer fc3.')
    return parser
