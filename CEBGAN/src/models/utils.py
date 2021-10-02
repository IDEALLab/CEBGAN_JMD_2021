import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def strong_convex_func(x, lamb, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / lamb / 2.
    else:
        func = torch.exp(x / lamb) / math.exp(1) * lamb
    return func

def strong_convex_func_normalized(x, lamb, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / 2.
    else:
        func = torch.exp(x / lamb) / math.exp(1)
    return func

def sum_probs_func(x, lamb):
    return torch.mean(torch.maximum(x, 0.0)) / lamb

def first_element(input):
    """Improve compatibility of single and multiple output components.
    """
    if type(input) == tuple or type(input) == list:
        return input[0]
    else:
        return input