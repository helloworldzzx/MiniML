import numpy as np
from core.Module import Module


class Flatten(Module):

    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        # 记录原始shape，用于backward还原
        self.x_shape = x.shape

        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, grad_output):
        # 还原回 CNN 的 shape
        return grad_output.reshape(self.x_shape)

    def parameters(self):
        return []