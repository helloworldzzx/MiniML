from .Optimizer import Optimizer


class SGD(Optimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）
    """

    def __init__(self, params, lr=0.01):
        """
        params : 模型参数
        lr     : 学习率
        """

        # 调用父类构造函数
        super().__init__(params)

        # 保存学习率
        self.lr = lr

    def step(self):
        """
        更新所有参数
        """

        for param in self.params:
            param.data -= self.lr * param.grad