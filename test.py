import numpy as np

from layers.Linear import Linear


x = np.random.randn(32,10)

layer = Linear(10,5)

y = layer.forward(x)

print(y.shape)
print(x.shape)
print(layer.W.data.shape)
print(layer.b.data.shape)