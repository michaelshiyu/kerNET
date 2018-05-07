# -*- coding: utf-8 -*-
# torch 0.3.1

import unittest
import numpy as np
import torch
from torch.autograd import Variable

import sys
sys.path.append('../kernet/')
from models.mlkn import baseMLKN, MLKN, MLKNGreedy, MLKNClassifier
from layers.kerlinear import kerLinear
from layers.ensemble import kerLinearEnsemble


torch.manual_seed(1234)
np.random.seed(1234)

class BaseMLKNTestCase(unittest.TestCase):
    def setUp(self):
        pass



if __name__=='__main__':

    pass
