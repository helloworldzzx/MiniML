import numpy as np

class MSELoss:

    def __init__(self):
        self.pred = None
        self.target = None



    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        loss = np.mean((pred - target) ** 2)
        return loss


    def backward(self):
        grad = 2 *(self.pred - self.target) / self.pred.size
        return grad