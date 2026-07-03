import numpy as np


class Parameter:
    """
    神经网络中的可训练参数。
    每个参数都包含：
        data : 参数值
        grad : 参数梯度
    """

    def __init__(self, data):
        """
        初始值data : ndarray
        """
        self.data = data

        # 梯度初始化为0
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        """
        梯度清零。
        每次反向传播之前调用。
        """
        self.grad.fill(0)