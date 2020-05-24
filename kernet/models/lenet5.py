"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn.functional as F


class LeNet5(torch.nn.Module):
  def __init__(self, opt, *args, **kwargs):
    super(LeNet5, self).__init__(*args, **kwargs)
    if opt.dataset in ['mnist', 'fashionmnist']:
      feat_len = 400
      in_channels = 1
    elif opt.dataset == 'cifar10':
      feat_len = 576
      in_channels = 3
    else:
      raise NotImplementedError()

    if opt.activation == 'tanh':
      self.act = torch.nn.Tanh
    elif opt.activation == 'sigmoid':
      self.act = torch.nn.Sigmoid
    elif opt.activation == 'relu':
      self.act = torch.nn.ReLU
    else:
      raise NotImplementedError()

    self.conv1 = torch.nn.Conv2d(in_channels, 6, 5, padding=2)
    self.conv2 = torch.nn.Conv2d(6, 16, 5)
    self.fc1 = torch.nn.Linear(feat_len, 120)
    self.fc2 = torch.nn.Linear(120, 84)
    self.fc3 = torch.nn.Linear(84, 10)
    

  def forward(self, input, **kwargs):
    output = self.act(self.conv1(input))
    output = F.max_pool2d(output, 2)
    output = self.act(self.conv2(output))
    output = F.max_pool2d(output, 2)
    output = output.view(output.size(0), -1)
    output = self.act(self.fc1(output))
    output = self.act(self.fc2(output))
    output = self.fc3(output)
    return output


  @staticmethod
  def modify_commandline_options(parser, **kwargs):
    return parser

