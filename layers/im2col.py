import numpy as np

def im2col(x, kernel_size, stride=1):
    """
    x: (N, C, H, W)
    return:
        col: (N * out_h * out_w, C * K * K)
    """

    N, C, H, W = x.shape
    K = kernel_size

    out_h = (H - K) // stride + 1
    out_w = (W - K) // stride + 1

    cols = []

    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):

                patch = x[
                    n,
                    :,
                    i*stride:i*stride+K,
                    j*stride:j*stride+K
                ]

                cols.append(patch.flatten())

    return np.array(cols)