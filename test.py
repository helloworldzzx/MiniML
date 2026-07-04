import numpy as np
import matplotlib.pyplot as plt

from layers.Linear import Linear
from losses.MSELoss import MSELoss
from optim.SGD import SGD
from core.Sequential import Sequential
from layers.Relu import RelU

#准备数据
# x
x = np.linspace(
    -2,
    2,
    100
).reshape(-1,1)

y = x ** 2

print(x.shape)
print(y.shape)

#创建模型
model = Sequential(
    Linear(1,16),
    RelU(),
    Linear(16,1)
)


#损失函数
criterion = MSELoss()

#优化器
optimizer = SGD(
    model.parameters(),
    lr=0.001
)

#训练
epochs = 50000
# 记录每一轮的 Loss
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model.forward(x)
    loss = criterion.forward(pred,y)
    loss_history.append(loss)
    grad = criterion.backward()
    model.backward(grad)
    optimizer.step()
    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:4d} "
            f"Loss = {loss:.6f}"
        )


print("\n训练完成！")
for i, p in enumerate(model.parameters()):
    print(f"Parameter {i}:")
    print(p.data)

pred = model.forward(x)
plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))

plt.scatter(
    x,
    y,
    label="Ground Truth",
    s=20
)

plt.plot(
    x,
    pred,
    linewidth=2,
    label="Prediction"
)

plt.title("MLP Function Approximation")

plt.xlabel("x")

plt.ylabel("y")

plt.legend()

plt.grid(True)

plt.show()