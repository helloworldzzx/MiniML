import numpy as np


class Optimizer:
    """
    所有优化器的基类
    """

    def __init__(self, params):
        # 保存所有需要更新的参数
        self.params = list(params)

    def step(self):
        """
        更新参数
        由子类实现
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        将所有参数梯度清零
        """

        for param in self.params:
            param.grad = np.zeros_like(param.data)