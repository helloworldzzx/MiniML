import numpy as np
from core.Module import Module


class MaxPool2D(Module):

    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

        # 用于 backward
        self.x = None
        self.mask = None

    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        H_out = H // S
        W_out = W // S

        out = np.zeros((N, C, H_out, W_out))
        self.mask = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):

                        h_start = i * S
                        h_end = h_start + K

                        w_start = j * S
                        w_end = w_start + K

                        window = x[n, c, h_start:h_end, w_start:w_end]

                        max_val = np.max(window)
                        out[n, c, i, j] = max_val

                        # 记录 max 位置（反向传播用）
                        mask = (window == max_val)
                        self.mask[n, c, h_start:h_end, w_start:w_end] += mask

        return out

    def backward(self, grad_output):

        dx = np.zeros_like(self.x)

        N, C, H_out, W_out = grad_output.shape
        K = self.kernel_size
        S = self.stride

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):

                        h_start = i * S
                        w_start = j * S

                        grad = grad_output[n, c, i, j]

                        dx[n, c,
                           h_start:h_start+K,
                           w_start:w_start+K] += (
                            self.mask[n, c,
                                      h_start:h_start+K,
                                      w_start:w_start+K]
                            * grad
                        )

        return dx

    def parameters(self):
        return []