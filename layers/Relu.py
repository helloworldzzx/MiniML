import numpy as np
from core.Module import Module

class RelU(Module):
    """
    ReLU :y = max(0,x)
    """
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        dx = grad_output * (self.x > 0)
        return dx

    def parameters(self):
        # ReLU没有参数
        return []