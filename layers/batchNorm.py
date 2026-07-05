import numpy as np
from core.Module import Module
from core.Parameter import Parameter


class BatchNorm(Module):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9):

        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = Parameter(
            np.ones(num_features)
        )

        self.beta = Parameter(
            np.zeros(num_features)
        )

        # 测试使用
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True

    def forward(self, x):

        if self.training:

            N = x.shape[0]

            # Step1
            # mean

            mean = np.mean(
                x,
                axis=0
            )

            # Step2
            # x - mean

            x_mu = x - mean

            # Step3
            # square

            sq = x_mu ** 2

            # Step4
            # variance

            var = np.mean(
                sq,
                axis=0
            )

            # Step5
            # sqrt

            std = np.sqrt(
                var + self.eps
            )

            # Step6
            # inverse

            inv_std = 1.0 / std

            # Step7
            # normalize

            x_hat = x_mu * inv_std

            # Step8
            # gamma * x_hat

            gamma_x = (
                self.gamma.data
                * x_hat
            )

            # Step9
            # + beta

            out = (
                gamma_x
                + self.beta.data
            )

            # 更新running统计量
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * mean
            )

            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * var
            )

            # cache
            self.x = x
            self.mean = mean
            self.x_mu = x_mu
            self.sq = sq
            self.var = var
            self.std = std
            self.inv_std = inv_std
            self.x_hat = x_hat
            self.gamma_x = gamma_x

            return out

        else:

            x_hat = (
                x - self.running_mean
            ) / np.sqrt(
                self.running_var + self.eps
            )

            out = (
                self.gamma.data * x_hat
                + self.beta.data
            )

            return out

    def backward(self, dout):

        N = dout.shape[0]

        # Step1
        # out = gamma_x + beta

        self.beta.grad = np.sum(
            dout,
            axis=0
        )

        dgamma_x = dout

        # Step2
        # gamma_x = gamma * x_hat

        self.gamma.grad = np.sum(
            dgamma_x * self.x_hat,
            axis=0
        )

        dx_hat = (
            dgamma_x
            * self.gamma.data
        )

        # Step3
        # x_hat = x_mu * inv_std

        dx_mu1 = (
            dx_hat
            * self.inv_std
        )

        dinv_std = np.sum(
            dx_hat * self.x_mu,
            axis=0
        )

        # Step4
        # inv = 1 / std

        dstd = (
            dinv_std
            * (-1.0)
            / (self.std ** 2)
        )

        # Step5
        # std = sqrt(var+eps)

        dvar = (
            dstd
            * 0.5
            / self.std
        )
        # Step6
        # var = mean(square)
        # var = (1/N) * sum(square)

        dsq = (
                np.ones_like(self.sq)
                * dvar
                / N
        )

        # Step7
        # square = x_mu^2

        dx_mu2 = (
                2.0
                * self.x_mu
                * dsq
        )

        # Step8
        # 两条路径汇合
        # x_mu 同时来自：
        # x_hat = x_mu * inv
        # square = x_mu^2

        dx_mu = (
                dx_mu1
                + dx_mu2
        )

        # Step9
        # x_mu = x - mean

        dx1 = dx_mu

        dmean = -np.sum(
            dx_mu,
            axis=0
        )

        # Step10
        # mean = (1/N) sum(x)

        dx2 = (
                np.ones_like(self.x)
                * dmean
                / N
        )

        # Step11
        # 两条路径再次汇合

        dx = (
                dx1
                + dx2
        )

        return dx

    def parameters(self):

        return [
            self.gamma,
            self.beta
        ]

    def train(self):

        self.training = True

    def eval(self):

        self.training = False

"""
选择一种语言(推荐Python)，实现一个简单的神经网络框架(可以理解为简化版Tensorflow)，能够实现
1. 全连接层
2. 卷积层
3. BatchNormalization层
4. 你喜欢的任意层 (optional)的**前向传播**和**反向传播**的功能。
 
要求：
1. 仅矩阵运算库，不使用现有的深度学习库。例如在Python中，禁止出现`import tensorflow`或`import torch`等代码。除非你提供了令人信服的理由，否则本task视为未完成。
2. 提高要求：代码运行效率。可以尽量尝试使用更快速的方法来实现这些层，并用适当的方法评估不同方法的运行速度。
3. 请不要用Jupyter Notebook来实现。理想情况下，代码应当被封装为一个库。不要因为cs231n用jupyter就跟着用，锻炼自己的代码能力。
4. 用你实现的框架任意选择一种数据集实现一个神经网络。
5. 撰写任务报告，描述实现过程。
 

"""