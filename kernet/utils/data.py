"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from typing import List

import numpy as np

import torch
import torchvision


def get_cifar10_subset(cifar10_set, classes: List[str]):
  """
  Provide a list of class names, return the CIFAR-10 subset containing only examples from
  these classes.

  The labels in the returned dataset will be re-encoded to be in the range [0, len(classes) - 1].
  Multiple calls with classes reordered will give the same re-encoded labels in the sense that
  if class c_i is assigned label p when calling with classes being [c_1, ..., c_i, ..., c_k],
  then it will be always assigned label p when calling with classes shuffled in any way.

  This invariance is relevant when, e.g., calling this method twice to first process the
  trainset and then the testset: the user does not have to keep the order of classes unchanged
  between the calls and will still get the correct re-encoded labels.

  Args:
    cifar10_set:
      A torchvision.datasets.CIFAR10 instance. Both train set and test set work.
    classes:
      A list of class names.

  Returns:
    The given cifar10_set with data and targets attributes modified to contain only the
    specified classes.
  """
  str2int = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
  }
  if len(set(classes)) != len(classes):
    raise ValueError('List classes contains duplicate entries.')
  if any(_ not in str2int for _ in classes):
    raise ValueError('Provided class(es) that is not in CIFAR-10.')

  cifar10_set.targets = np.array(cifar10_set.targets)
  idx = [False] * len(cifar10_set.data)
  for c in classes:
    idx += (cifar10_set.targets == str2int[c])
  cifar10_set.data = cifar10_set.data[idx]
  cifar10_set.targets = cifar10_set.targets[idx]

  # reassign the labels. Sort the classes to prevent multiple calls to
  # this method with reordered classes resulting in different label
  # assignments
  for i, c in enumerate(sorted(classes)):
    cifar10_set.targets[cifar10_set.targets == str2int[c]] = i
  cifar10_set.targets = list(cifar10_set.targets)

  return cifar10_set


def get_mean_and_std(dataset):
  """
  Get the sample mean and std of a dataset.
  Adapted from
  https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6?

  Args:
    dataset (torch.utils.data.Dataset): The dataset of interest.
  """
  data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
  )
  mean = 0.
  std = 0.
  n_example = 0

  # assuming this dataset has both input and target
  for x, _ in data_loader:
    x = x.view(x.size(0), x.size(1), -1)
    mean += x.mean(2).sum(0)
    std += x.std(2).sum(0)
    n_example += x.size(0)

  mean /= n_example
  std /= n_example

  return mean, std


if __name__ == '__main__':
  cifar10 = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=torchvision.transforms.ToTensor()
  )
  print(get_mean_and_std(cifar10))
  cifar10deau = get_cifar10_subset(cifar10, ['deer', 'automobile'])
  print(get_mean_and_std(cifar10deau))
