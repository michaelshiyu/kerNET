"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn.functional as F

from .misc import *


def gaussian_phi_fn_indir(input, centers, sigma, **kwargs):
  """
  Gaussian kernel (indirect):
    k(x, y) = exp(-||x-y||_2^2 / (2 * sigma^2));
    x -> (k(c_1, x), ..., k(c_m, x)), c_i \in centers, \forall i.
  """
  input = centers.sub(input.unsqueeze(1)).pow(2).view(input.size(0), centers.size(0), -1)
  input = input.sum(dim=-1).mul(-1./(2*sigma**2)).exp()
  return input


def poly2_phi_fn_dir(input, **kwargs):
  """
  Polynomial kernel of order 2 (direct)
  """
  # TODO
  raise NotImplementedError()


def nn_tanh_phi_fn_dir(input, **kwargs):
  """
  Shape:
    input: (n_examples, d)
  """
  output = torch.tanh(input)
  # make sure any nonzero \phi(x) is of unit norm
  return to_unit_vector(output)


def nn_sigmoid_phi_fn_dir(input, **kwargs):
  """
  Shape:
    input: (n_examples, d)
  """
  output = torch.sigmoid(input)
  # make sure any nonzero \phi(x) is of unit norm
  return to_unit_vector(output)


def nn_relu_phi_fn_dir(input, **kwargs):
  """
  Shape:
    input: (n_examples, d)
  """
  output = F.relu(input)
  # make sure any nonzero \phi(x) is of unit norm
  return to_unit_vector(output)


def nn_reapen_phi_fn_dir(input, **kwargs):
  """
  ReLU + average pool 2D + flatten.

  Shape:
    input: (n_examples, d1, d2, d3)
  """
  output = F.relu(input)
  output = F.avg_pool2d(output, 4)
  output = output.view(output.size(0), -1)
  # make sure any nonzero \phi(x) is of unit norm
  return to_unit_vector(output)
