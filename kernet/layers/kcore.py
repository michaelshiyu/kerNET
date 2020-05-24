"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch

import kernet_future.utils as utils


class Phi(torch.nn.Module):
  """
  Implements the kernel feature maps for the higher-level kernelized network layers. 

  Perform all sanity checks at this level.
  """
  def __init__(self, kernel='gaussian', in_features=None, evaluation='direct', sigma=1., *args, **kwargs):
    super(Phi, self).__init__(*args, **kwargs)
    self.evaluation = evaluation # sanity check done at network level, now in ['direct', 'indirect']
    k_name = kernel.lower()
    if k_name not in ['gaussian', 'nn_tanh', 'nn_sigmoid', 'nn_relu', 'nn_reapen']:
      raise ValueError('Invalid kernel function: {}'.format(kernel))

    # kernel params
    self.k_params = {
        'sigma': sigma,
        }

    if k_name == 'gaussian': 
      self.k_min, self.k_max = 0., 1.
      if evaluation == 'indirect':
        self.phi_fn = utils.gaussian_phi_fn_indir
      else:
        raise NotImplementedError()
    elif k_name == 'nn_tanh': 
      self.k_min, self.k_max = -1., 1.
      if evaluation == 'indirect':
        raise NotImplementedError()
      else:
        self.phi_fn = utils.nn_tanh_phi_fn_dir
        # in direct mode, out_features will be used by, say, kLinear to
        # determine the shape of the downstream linear layer
        self.out_features = in_features
    elif k_name == 'nn_sigmoid': 
      self.k_min, self.k_max = 0., 1.
      if evaluation == 'indirect':
        raise NotImplementedError()
      else:
        self.phi_fn = utils.nn_sigmoid_phi_fn_dir
        self.out_features = in_features
    elif k_name == 'nn_relu':
      self.k_min, self.k_max = 0., 1.
      if evaluation == 'indirect':
        raise NotImplementedError()
      else:
        self.phi_fn = utils.nn_relu_phi_fn_dir
        self.out_features = in_features
    elif k_name == 'nn_reapen':
      self.k_min, self.k_max = 0., 1.
      if evaluation == 'indirect':
        raise NotImplementedError()
      else:
        self.phi_fn = utils.nn_reapen_phi_fn_dir
        self.out_features = in_features

  def forward(self, input, centers=None):
    """
    If evaluation is 'indirect':
      x -> (k(c_1, x), ..., k(c_m, x)), c_i \in centers, \forall i.

    If evaluation is 'indirect':
      x -> \phi(x).T, where \phi(x) is the feature map of the kernel.

    Shape:
      input: (N, d1, d2, ...)
      centers: (M, d1, d2, ...)
    """
    if self.evaluation == 'indirect':
      if centers.size()[1:] != input.size()[1:]:
        raise ValueError('For input and centers, all dimensions except maybe the first one must match.')

      return self.phi_fn(input, centers, **self.k_params)
    else:
      return self.phi_fn(input, **self.k_params)
      
  def get_k_mtrx(self, input1, input2):
    if self.evaluation == 'indirect':
      return self(input1, centers=input2)
    else:
      return self(input1).mm(self(input2).t())

  def get_ideal_k_mtrx(self, target1, target2, n_classes):
    """
    Returns the "ideal" kernel matrix G* defined as
      (G*)_{ij} = k_min if y_i == y_j;
      (G*)_{ij} = k_max if y_i != y_j.

    Args:
      target1 (tensor): Categorical labels with values in 
        {0, 1, ..., n_classes-1}.
      target2 (tensor): Categorical labels with values in 
        {0, 1, ..., n_classes-1}.
      n_classes (int)

    Shape:
      - Input:
        target1: (n_examples1, 1) or (1,) (singleton set)
        target2: (n_examples2, 1) or (1,) (singleton set)
      - Output: (n_examples1, n_examples2)
    """
    if n_classes < 2:
      raise ValueError('You need at least 2 classes')

    if len(target1.size()) == 1: 
      target1.unsqueeze_(1)
    elif len(target1.size()) > 2:
      raise ValueError('target1 has too many dimensions')
    if len(target2.size()) == 1: 
      target2.unsqueeze_(1)
    elif len(target2.size()) > 2:
      raise ValueError('target2 has too many dimensions')

    if torch.max(target1) + 1 > n_classes:
      raise ValueError('target1 has at least one invalid entry')
    if torch.max(target2) + 1 > n_classes:
      raise ValueError('target2 has at least one invalid entry')

    target1_onehot, target2_onehot = \
      utils.one_hot_encode(target1, n_classes).to(torch.float), \
      utils.one_hot_encode(target2, n_classes).to(torch.float)

    ideal = target1_onehot.mm(target2_onehot.t())

    if self.k_min!=0:
      min_mask = torch.full_like(ideal, self.k_min)
      ideal = torch.where(ideal==0, min_mask, ideal)
    if self.k_max!=1:
      max_mask = torch.full_like(ideal, self.k_max)
      ideal = torch.where(ideal==1, max_mask, ideal)

    return ideal
