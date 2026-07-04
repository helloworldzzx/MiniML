import numpy as np

class CrossEntropyLoss:
    """
    交叉熵损失（内部包含 Softmax）
    输入：
        logits : (batch_size, num_classes)
        target : (batch_size,)
    例如：
    logits =
    [
        [2.3, 1.2, 0.5],
        [0.2, 3.1, 1.8]
    ]
    target =
    [
        0,
        1
    ]
    """

    def __init__(self):

        # Softmax后的概率
        self.prob = None
        # 真实标签
        self.target = None

    def forward(self, logits, target):

        """
        logits : (N, C)
        target : (N,)
        """
        self.target = target
        # 数值稳定
        logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )
        # Softmax
        exp = np.exp(logits)
        self.prob = exp / np.sum(
            exp,
            axis=1,
            keepdims=True
        )
        # 正确类别概率
        batch_size = logits.shape[0]
        correct_prob = self.prob[
            np.arange(batch_size),
            target
        ]

        # Cross Entropy
        loss = -np.mean(
            np.log(correct_prob)
        )
        return loss

    def backward(self):

        """
        返回：
        dL/dLogits
        shape:
        (batch_size, num_classes)
        """

        batch_size = self.prob.shape[0]
        grad = self.prob.copy()

        # 正确类别减1
        grad[
            np.arange(batch_size),
            self.target
        ] -= 1

        grad /= batch_size

        return grad