#ADD REGULARIZATION!!!!!!

import torch.nn as nn
import torch
import math

def my_loss(classifier, regression, points, mode):
    #label is a dictionary with the name, points, mode, and blocked
    #have to adjust alpha
    alpha = 0.5
    MSE = nn.MSELoss()
    MSEl = MSE(regression, points)
    #probably need to divide MSE by the width of the photo
    cross_entropy = nn.CrossEntropyLoss()
    ce = cross_entropy(classifier, mode)
    loss = ce*alpha + MSEl*(1-alpha)
    return loss, MSEl, ce
