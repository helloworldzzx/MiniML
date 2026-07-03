import numpy as np

from core.Module import Module
from core.Parameter import Parameter


class Linear(Module):
    """
    全连接层

    Y = XW + b

    X : (N, in_features)

    W : (in_features, out_features)

    b : (out_features,)

    Y : (N, out_features)
    """

    def __init__(self, in_features, out_features):

        # Xavier初始化
        weight = np.random.randn(
            in_features,
            out_features
        ) * np.sqrt(2.0 / in_features)

        bias = np.zeros(out_features)

        self.W = Parameter(weight)

        self.b = Parameter(bias)

        # forward时缓存输入
        self.x = None

    def forward(self, x):

        """
        x.shape

        (batch_size, in_features)

        例如：

        (32,10)
        """

        self.x = x

        out = x @ self.W.data + self.b.data

        return out

    def backward(self, grad_output):

        """
        grad_output

        dL/dY

        shape:

        (batch_size, out_features)

        例如：

        (32,5)
        """

        #
        # dW
        #
        # X^T @ grad_output
        #
        # (10,32) @ (32,5)
        #
        # = (10,5)
        #

        self.W.grad = self.x.T @ grad_output

        #
        # db
        #
        # 每个batch加起来
        #
        # (32,5)
        #
        # -> (5,)
        #

        self.b.grad = np.sum(
            grad_output,
            axis=0
        )

        #
        # dx
        #
        # grad_output @ W^T
        #
        # (32,5)
        #
        # @
        #
        # (5,10)
        #
        # =
        #
        # (32,10)
        #

        dx = grad_output @ self.W.data.T

        return dx

    def parameters(self):

        return [
            self.W,
            self.b
        ]