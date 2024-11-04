from __future__ import annotations
import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_scores = np.exp(x)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sum_exp
    return probs
