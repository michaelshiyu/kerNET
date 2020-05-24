"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modified from:
  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/__init__.py
"""
import logging
import importlib

import torch

import kernet.utils as utils


logger = logging.getLogger()


class Normalize(torch.nn.Module):
  """
  Make each vector in input have Euclidean norm 1.

  This class is a wrap on utils.to_unit_vector so that it
  can be registered as a module when used in a model.
  """
  def forward(self, input):
    """
    Shape:
      input: (n_examples, d)
    """
    return utils.to_unit_vector(input)


class Flatten(torch.nn.Module):
  """
  A convenient wrap on torch.flatten.
  """
  def forward(self, input):
    """
    Shape:
      input: (n_examples, d1, ...)

    Returns a tensor with shape (n_examples, d1 * ...).
    """
    return torch.flatten(input, start_dim=1)


def find_model_using_name(model_name):
  """
  Import the module "models/[model_name].py".
  In the file, the class called model_name() will
  be instantiated. It has to be a subclass of torch.nn.Module,
  and it is not case-insensitive.
  """
  model_filename = "kernet.models." + model_name
  modellib = importlib.import_module(model_filename)
  model = None
  target_model_name = model_name.replace('_', '')
  for name, cls in modellib.__dict__.items():
    if name.lower() == target_model_name.lower() \
       and issubclass(cls, torch.nn.Module):
      model = cls

  if model is None:
    raise ModuleNotFoundError(
      "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (
        model_filename, target_model_name))

  return model


def get_option_setter(model_name):
  """
  Return the static method <modify_commandline_options> of the model class.
  """
  model_class = find_model_using_name(model_name)
  return getattr(model_class, 'modify_commandline_options', None)


def get_model(opt):
  """
  Create a model given the option.
  This is the main interface between this package and 'train.py'/'test.py'
  Example:
    from models import get_model
    model = get_model(opt)
  """
  model = find_model_using_name(opt.model)

  # check if model contains kernelized components that need centers
  from inspect import signature
  if signature(model).parameters.get('centers'):
    from kernet import utils
    centers = utils.get_centers(opt)
    instance = model(opt, centers)
  else:
    instance = model(opt)

  logger.info("model [%s] was created" % type(instance).__name__)
  return instance
