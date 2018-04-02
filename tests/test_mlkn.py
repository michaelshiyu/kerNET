# -*- coding: utf-8 -*-
# torch 0.3.1

from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../kernet/')
import backend as K
from models.mlkn import MLKNClassifier
from layers.kerlinear import kerLinear

torch.manual_seed(1234)

if __name__=='__main__':
    # toy data
    # X = Variable(torch.FloatTensor([[1, 2], [3, 4]]).type(dtype), requires_grad=False)
    # y = Variable(torch.FloatTensor([[1], [1]]).type(dtype), requires_grad=False)
    pass
