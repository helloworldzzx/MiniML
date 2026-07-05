import numpy as np
from core.Module import Module
from core.Parameter import Parameter


class Conv2D(Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        K = kernel_size

        weight = np.random.randn(
            out_channels,
            in_channels,
            K,
            K
        ) * np.sqrt(2.0 / (in_channels * K * K))

        bias = np.zeros(out_channels)

        self.W = Parameter(weight)
        self.b = Parameter(bias)

    # =========================
    # im2col
    # =========================
    def im2col(self, x):
        N, C, H, W = x.shape
        K = self.kernel_size

        H_out = H - K + 1
        W_out = W - K + 1

        cols = []

        for n in range(N):
            for i in range(H_out):
                for j in range(W_out):

                    patch = x[
                        n,
                        :,
                        i:i+K,
                        j:j+K
                    ]

                    cols.append(patch.flatten())

        return np.array(cols), H_out, W_out

    # =========================
    # forward
    # =========================
    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        K = self.kernel_size

        self.x_col, H_out, W_out = self.im2col(x)

        W_col = self.W.data.reshape(self.out_channels, -1)

        out = self.x_col @ W_col.T + self.b.data

        out = out.reshape(N, H_out, W_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2)

        return out

    # =========================
    # backward
    # =========================
    def backward(self, grad_output):

        N, C, H, W = self.x.shape
        K = self.kernel_size

        H_out = H - K + 1
        W_out = W - K + 1

        grad_output = grad_output.transpose(0, 2, 3, 1)
        grad_output = grad_output.reshape(-1, self.out_channels)

        W_col = self.W.data.reshape(self.out_channels, -1)

        # =========================
        # 1. bias grad
        # =========================
        self.b.grad = np.sum(grad_output, axis=0)

        # =========================
        # 2. weight grad
        # =========================
        self.W.grad = grad_output.T @ self.x_col
        self.W.grad = self.W.grad.reshape(self.W.data.shape)

        # =========================
        # 3. dx
        # =========================
        dx_col = grad_output @ W_col
        dx = np.zeros_like(self.x)

        idx = 0
        for n in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    dx[n, :, i:i+K, j:j+K] += dx_col[idx].reshape(C, K, K)
                    idx += 1

        return dx

    # =========================
    # parameters
    # =========================
    def parameters(self):
        return [self.W, self.b]