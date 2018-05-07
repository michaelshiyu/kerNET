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
        # toy data
        dtypeX, dtypeY = torch.FloatTensor, torch.LongTensor
        if torch.cuda.is_available():
            dtypeX, dtypeY = torch.cuda.FloatTensor, torch.cuda.LongTensor
        X = Variable(
            torch.FloatTensor([[1, 2], [3, 4]]).type(dtypeX),
            requires_grad=False
            )
        Y = Variable(
            torch.FloatTensor([[0], [1]]).type(dtypeY),
            requires_grad=False
            )
        mlkn = baseMLKN()
        mlkn.add_layer(kerLinear(
            X=X,
            out_dim=2,
            sigma=3,
            bias=True
        ))
        mlkn.add_layer(kerLinear(
            X=X,
            out_dim=2,
            sigma=2,
            bias=True
        ))
        # manually set some weights
        mlkn.layer0.weight.data = torch.FloatTensor([[.1, .2], [.5, .7]])
        mlkn.layer0.bias.data = torch.FloatTensor([0, 0])
        mlkn.layer1.weight.data = torch.FloatTensor([[1.2, .3], [.2, 1.7]])
        mlkn.layer1.bias.data = torch.FloatTensor([0.1, 0.2])

        

if __name__=='__main__':

    pass
