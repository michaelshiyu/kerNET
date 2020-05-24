"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy
import logging

import torch

from .misc import INF
from kernet import datasets
from kernet.layers.klinear import kLinear, kLinearCommittee


logger = logging.getLogger()

def get_centers(opt):
  # TODO for large train sets, do we need to limit the max tensor we load here?
  opt_copy = copy.deepcopy(opt)

  opt_copy.n_workers = 0
  opt_copy.batch_size = 1
  opt_copy.shuffle = True
  opt_copy.is_train = True  # always load the train set as centers
  opt_copy.max_trainset_size = INF

  logger.info('Getting centers: dummy load...')
  loader = datasets.get_dataloaders(opt_copy)

  opt_copy.batch_size = len(loader) # get dataset size
  logger.info('Getting centers...')
  loader = datasets.get_dataloaders(opt_copy)

  centers = next(iter(loader)) # get the entire dataset (input, target) as a tensor

  return centers


def update_centers_eval(model):
  """
  A wrapper around model's update_centers method that sets the model to eval
  and cuts off grad beforehand.
  """
  model.eval()
  with torch.no_grad():
    if hasattr(model, 'update_centers'):
      model.update_centers()


def exclude_during_backward(model):
  logger.debug('Exclude {} during backward...'.format(model.__class__.__name__))
  for p in model.parameters():
    p.requires_grad_(False)


def include_during_backward(model):
  logger.debug('Include {} during backward...'.format(model.__class__.__name__))
  for p in model.parameters():
    p.requires_grad_()


def to_committee(model, expert_size):
  """
  Convert a kLinear model into a committee of experts (a
  kLinearCommittee model) with each expert except possibly the last
  being of size expert_size.

  The new model is numerically equivalent to the original.

  If needed, call this function RIGHT AFTER model initialization.

  This conversion preserves the device allocation of the original model, i.e.,
  if model is on GPU, the returned committee will also be on GPU.
  """
  logger.info('Converting {} to committee w/ expert size {}...'.format(model.__class__.__name__, expert_size))
  if not isinstance(model, kLinear):
    raise TypeError('Expecting the model to be of ' +
                    'kLinear type, got {} instead.'.format(type(model)))

  if not hasattr(model, 'centers'):
    logger.warning('The given model does not have centers, ' +
                  'in which case the conversion to committee ' +
                  'was not performed. The original model ' +
                  'was returned instead.')
    return model

  centers = model.centers
  committee = kLinearCommittee()

  i = 0
  while i * expert_size < len(centers):
    bias = True if model.linear.bias is not None and i == 0 else False
    expert = kLinear(
      out_features=model.out_features,
      in_features=model.in_features,
      kernel=model.kernel,
      bias=bias,
      evaluation=model.evaluation,
      centers=centers[i * expert_size: (i + 1) * expert_size].clone().detach(),
      trainable_centers=getattr(model, 'trainable_centers', False),
      sigma=model.phi.k_params['sigma']
    )

    expert.linear.weight.data = \
      model.linear.weight[:, i * expert_size: (i + 1) * expert_size].clone().detach()
    if bias:
      expert.linear.bias.data = model.linear.bias.clone().detach()

    committee.add_expert(expert)
    i += 1

  return committee


def attach_head(model, opt):
  """
  Attach a trainable, two-layer MLP projection head to model.
  The size of the head is determined dynamically.

  Args:
    model (a torch.nn.Sequential object): The network to be modified. It is important
    that model is a torch.nn.Sequential object since if otherwise model may not be
    subscriptable.

  Returns a new model with a projection head attached to the last module in
  the model.
  """
  if not getattr(opt, 'use_proj_head', None):
    return model

  from kernet.models import Flatten

  device = next(model.parameters()).device
  dummy_input = torch.randn((1,) + eval(opt.data_shape)).to(device)
  dummy_output = model(dummy_input)
  output_size = len(dummy_output.flatten())
  if output_size == opt.head_size:
    return model

  mid = (output_size + opt.head_size) // 2

  proj_head = torch.nn.Sequential(*[
    Flatten(),
    torch.nn.Linear(output_size, mid),
    torch.nn.ReLU(),
    torch.nn.Linear(mid, opt.head_size)
  ]).to(device)

  # only modify the last module of the model
  logger.debug('Before adding projection head:\n')
  logger.debug(str(model))
  logger.debug('Adding projection head...')
  model[-1] = torch.nn.Sequential(*[
    model[-1],
    proj_head
  ])
  logger.debug('After adding projection head:\n')
  logger.debug(str(model))
  return model
