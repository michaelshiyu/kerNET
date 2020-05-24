"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import torch
import torchvision
import torchvision.transforms as transforms

from kernet import utils

logger = logging.getLogger()

CIFAR10_2 = {
  'cifar10deau': ['deer', 'automobile'],
  'cifar10hotr': ['horse', 'truck'],
  'cifar10detr': ['deer', 'truck'],
  'cifar10hoau': ['horse', 'automobile'],
  'cifar10deca': ['deer', 'cat'],
  'cifar10hodo': ['horse', 'dog'],
  'cifar10dedo': ['deer', 'dog'],
  'cifar10hoca': ['horse', 'cat'],
  'cifar10auca': ['automobile', 'cat'],
  'cifar10trdo': ['truck', 'dog'],
  'cifar10audo': ['automobile', 'dog'],
  'cifar10trca': ['truck', 'cat'],
  'cifar10deho': ['deer', 'horse'],
  'cifar10autr': ['automobile', 'truck'],
  'cifar10cado': ['cat', 'dog']
}
NAMES = locals()


def get_option_setter(dataset_name: str):
  fn_name = dataset_name + '_modify_commandline_options'
  if fn_name in NAMES:
    return NAMES[fn_name]
  else:
    raise NotImplementedError()


def get_dataloaders(opt):
  if opt.is_train:
    l = [
      transforms.ToTensor(),
      transforms.Normalize(opt.normalize_mean, opt.normalize_std)
      ]

    if getattr(opt, 'augment_data', None):
      # opt will have augment_data as an attr during training. During testing,
      # the only case in which training data will be loaded will be for the
      # kernelized modules to get centers. These modules will load saved centers
      # and overwrite the centers loaded here anyway so this expression is safe
      l = [
      transforms.RandomCrop(32 if opt.dataset.startswith('cifar10') else 28, padding=4),
      transforms.RandomHorizontalFlip()
      ] + l

    train_transform = transforms.Compose(l)

    if opt.dataset.startswith('cifar10'):
      trainset = torchvision.datasets.CIFAR10(
          root='./data', train=True,
          download=True, transform=train_transform)
      if opt.dataset in CIFAR10_2:
        trainset = utils.get_cifar10_subset(trainset, CIFAR10_2[opt.dataset])
    elif opt.dataset == 'mnist':
      trainset = torchvision.datasets.MNIST(
          root='./data', train=True,
          download=True, transform=train_transform)
    elif opt.dataset == 'fashionmnist':  
      trainset = torchvision.datasets.FashionMNIST(
          root='./data', train=True,
          download=True, transform=train_transform)
    else:
      raise NotImplementedError()

    logger.info("dataset [train: %s] was created" % (
      type(trainset).__name__,
    ))
    logger.debug("transformations:\n" + str(train_transform))
    trainset = _get_subset(trainset, opt.max_trainset_size, opt.balanced)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size,
        shuffle=opt.shuffle, num_workers=opt.n_workers,
        pin_memory=True)

    return train_loader
  else:
    test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(opt.normalize_mean, opt.normalize_std)
      ])

    if opt.dataset.startswith('cifar10'):
      testset = torchvision.datasets.CIFAR10(
          root='./data', train=False,
          download=True, transform=test_transform)
      if opt.dataset in CIFAR10_2:
        testset = utils.get_cifar10_subset(testset, CIFAR10_2[opt.dataset])
    elif opt.dataset == 'mnist':
      testset = torchvision.datasets.MNIST(
          root='./data', train=False,
          download=True, transform=test_transform)
    elif opt.dataset == 'fashionmnist':  
      testset = torchvision.datasets.FashionMNIST(
          root='./data', train=False,
          download=True, transform=test_transform)
    else:
      raise NotImplementedError()

    logger.info("dataset [test: %s] was created" % (
      type(testset).__name__
    ))
    logger.debug("transformations:\n" + str(test_transform))
    testset = _get_subset(testset, opt.max_testset_size, opt.balanced)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.n_workers,
        pin_memory=True)

    return test_loader


def _get_subset(dataset, size: int, balanced=False):
  """
  Get a random subset from a torch dataset.
  If size is larger than or equal to the actual size of dataset,
  returns dataset directly.

  Args:
    dataset: torch.utils.data.Dataset
      The dataset to sample from.
    size: int
      The size of the subset.
    balanced: (optional) bool
      If True, the sampled subset will have balanced classes.
      Only effective when size is smaller than the actual dataset size.
      Defaults to False.

  Returns:
    A torch.utils.data.Dataset.
  """
  if size < len(dataset):
    if type(size) != int:
      raise TypeError(f'size should be of int type, got {type(size)} instead')
    if not balanced:
      dataset, _ = torch.utils.data.random_split(dataset, [size, len(dataset) - size])
    else:
      logger.warning('Balanced sampling has only been tested on CIFAR-10. The current implementation relies ' +
                     'on how attributes of the given torch Dataset are organized. Tread carefully if you intend to use ' +
                     'this configuration on any dataset other than CIFAR-10.')
      dataset.data, dataset.targets = utils.supervised_sample(
        torch.tensor(dataset.data), torch.tensor(dataset.targets),
        n=size, return_labels=True
      )
      dataset.data, dataset.targets = dataset.data.numpy(), dataset.targets.numpy()
  return dataset


#########
# option modifiers
#########


def cifar10_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=10)
  parser.set_defaults(normalize_mean='0.4914,0.4822,0.4465')
  parser.set_defaults(normalize_std='0.2023,0.1994,0.2010')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10deau_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4714,0.4599,0.4127')
  parser.set_defaults(normalize_std='0.2030,0.1974,0.1928')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10hotr_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.5003,0.4826,0.4475')
  parser.set_defaults(normalize_std='0.2258,0.2288,0.2337')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10detr_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4851,0.4753,0.4281')
  parser.set_defaults(normalize_std='0.2064,0.2044,0.2028')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10hoau_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4866,0.4672,0.4320')
  parser.set_defaults(normalize_std='0.2223,0.2218,0.2237')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10deca_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4835,0.4608,0.3969')
  parser.set_defaults(normalize_std='0.1911,0.1862,0.1803')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10hodo_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.5009,0.4723,0.4167')
  parser.set_defaults(normalize_std='0.2110,0.2108,0.2115')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10dedo_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4858,0.4649,0.3974')
  parser.set_defaults(normalize_std='0.1916,0.1863,0.1805')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10hoca_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4987,0.4681,0.4162')
  parser.set_defaults(normalize_std='0.2104,0.2107,0.2113')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10auca_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4833,0.4555,0.4314')
  parser.set_defaults(normalize_std='0.2223,0.2186,0.2198')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10trdo_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4993,0.4750,0.4473')
  parser.set_defaults(normalize_std='0.2263,0.2256,0.2300')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10audo_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4856,0.4596,0.4319')
  parser.set_defaults(normalize_std='0.2228,0.2187,0.2199')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10trca_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4971,0.4709,0.4468')
  parser.set_defaults(normalize_std='0.2257,0.2255,0.2298')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10deho_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4868,0.4725,0.3975')
  parser.set_defaults(normalize_std='0.1911,0.1895,0.1843')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10autr_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4849,0.4699,0.4626')
  parser.set_defaults(normalize_std='0.2376,0.2367,0.2422')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def cifar10cado_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=2)
  parser.set_defaults(normalize_mean='0.4977,0.4605,0.4160')
  parser.set_defaults(normalize_std='0.2109,0.2075,0.2075')
  parser.set_defaults(data_shape='(3, 32, 32)')
  return parser


def mnist_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=10)
  parser.set_defaults(normalize_mean='0.1307')
  parser.set_defaults(normalize_std='0.3015')
  parser.set_defaults(data_shape='(1, 28, 28)')
  return parser


def fashionmnist_modify_commandline_options(parser, **kwargs):
  parser.set_defaults(n_classes=10)
  parser.set_defaults(normalize_mean='0.2860')
  parser.set_defaults(normalize_std='0.3205')
  parser.set_defaults(data_shape='(1, 28, 28)')
  return parser
