import numpy as np

from layers.Relu import RelU


x = np.array([[-1, 2, -3], [4, -5, 6], [7, 8, 9]])

relu = RelU()

y = relu.forward(x)
print(y)