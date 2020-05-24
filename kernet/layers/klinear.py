"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch

import kernet.utils as utils
from kernet.layers.kcore import Phi


class _kLayer(torch.nn.Module):
  """
  Base class for all kernelized layers.
  """
  def __init__(self, *args, **kwargs):
    super(_kLayer, self).__init__(*args, **kwargs)


  @staticmethod
  def modify_commandline_options(parser, **kwargs):
    parser.add_argument('--memory_efficient', type=utils.str2bool,
                        nargs='?', const=True, default=False,
                        help='Whether to convert kLinear layers to committees to save memory. Results in slower runtime if set to True.')
    parser.add_argument('--expert_size', type=int, default=300,
                        help='The expert_size param for the kLinear committees.')
    return parser


  @staticmethod
  def update(layer, update_fn):
    if not isinstance(layer, kLinearCommittee):
      layer.centers = update_fn(layer.centers_init)
    else:
      for i in range(layer.n_experts):
        expert = getattr(layer, 'expert' + str(i))
        expert.centers = update_fn(expert.centers_init)


class kLinear(_kLayer): 
  def __init__(self, out_features, in_features=None, kernel='gaussian', bias=True, evaluation='direct', centers=None, trainable_centers=False, sigma=1., *args, **kwargs):
    """
    A kernelized linear layer. With input x, this layer computes:
    w.T @ \phi(x) + bias, where 
      1) when evaluation is 'direct',
      \phi(x).T = (k(c_1, x), ..., k(c_m, x)), k is the kernel function, 
      and the c_i's are the centers;

      2) when evaluation is 'indirect',
      \phi(x).T is the actual image of x under the corresponding feature map.
      In this case, there is no need to specify the centers.

    Direct evaluation only works on kernels whose feature maps \phi's are 
    implementable. 

    Args:
      out_features (int): The output dimension. 
      in_features (int): The input dimension. Not needed for certain
      kernels. Default: None. 
      kernel (str): Name of the kernel. Default: 'gaussian'.
      bias (bool): If True, add a bias term to the output. Default: True. 
      evaluation (str): Whether to evaluate the kernel machines directly 
      as the inner products between weight vectors and \phi(input), which
      requires explicitly writing out \phi, or indirectly 
      via an approximation using the reproducing property, which works 
      for all kernels but can be less accurate and less efficient. 
      Default: 'direct'. Choices: 'direct' | 'indirect'.
      centers (tensor): A set of torch tensors. Needed only when evaluation
      is set to 'indirect'. Default: None.
      trainable_centers (bool): Whether to treat the centers as trainable
        parameters. Default: False.
      sigma (float): The sigma hyperparameter for the kernel. See the
        kernel definitions for details. Default: 1..
    """
    super(kLinear, self).__init__(*args, **kwargs)
    self.evaluation = evaluation.lower()
    if self.evaluation not in ['direct', 'indirect']:
      raise ValueError(
        'Evaluation method can be "direct" or "indirect", got {}'.format(evaluation)
        )
    
    # prepare the centers for indirect evaluation
    if self.evaluation == 'indirect':
      if trainable_centers:
        self.trainable_centers = True
        self.centers = torch.nn.Parameter(
            centers.clone().detach().requires_grad_(True))
      else:
        self.centers = centers
        # centers_init will be saved together w/ the model but centers won't
        # see https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/11
        self.register_buffer('centers_init', centers.clone().detach())
    else:
      del centers

    self.kernel = kernel
    self.phi = Phi(kernel=kernel, in_features=in_features,  
        evaluation=self.evaluation, sigma=sigma)
    self.out_features = out_features
    self.in_features = in_features
    if self.evaluation == 'indirect':
      self.linear = torch.nn.Linear(len(centers), out_features, bias=bias)
    else:
      self.linear = torch.nn.Linear(self.phi.out_features, out_features, bias=bias)


  def forward(self, input):
    if self.evaluation == 'indirect':
      return self.linear(self.phi(input, centers=self.centers))
    else:
      return self.linear(self.phi(input))


class kLinearCommittee(torch.nn.Module):
  """
  A committee of kLinear experts, the responses from whom will be
  summed to create the final model response.

  This is useful for building kLinear models with large centers.
  One can split such a model into several experts and store
  them in a kLinearCommittee container. The resulting
  model is equivalent to the original but does not put
  large tensors (centers) onto GPU memory.

  There is a handy helper function in utils called to_committee
  that, when given a kLinear model, returns an equivalent
  kLinearCommittee model.
  """
  def __init__(self, *args, **kwargs):
    super(kLinearCommittee, self).__init__(*args, **kwargs)
    self.n_experts = 0


  def add_expert(self, expert):
    if not isinstance(expert, kLinear):
      raise TypeError('Expecting the expert to be of ' +
                      'kLinear type, got {} instead.'.format(type(expert)))

    if self.n_experts == 0:
      self.phi = expert.phi
      self.out_features = expert.out_features
      self.evaluation = expert.evaluation
    elif self.out_features != expert.out_features:
      raise ValueError('The expert being added has a different output dimension ' +
                       'than the existing experts, which may cause unexpected ' +
                       'behaviors when evaluating the committee.')

    self.add_module('expert'+str(self.n_experts), expert)
    self.n_experts += 1


  def forward(self, input):
    if self.n_experts == 0:
      raise ValueError('The committee does not have any expert yet.')

    output = self.expert0(input)
    for i in range(1, self.n_experts):
      output = output.add(getattr(self, 'expert'+str(i))(input))

    return output
