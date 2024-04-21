import torch

def L2(model):
    L2_ = 0.
    for p in model.parameters():
        L2_ += torch.sum(p**2)
    return L2_

def rescale(model, alpha):
    for p in model.parameters():
        p.data = alpha * p.data
