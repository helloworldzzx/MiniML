import numpy as np
import matplotlib.pyplot as plt

from core.Sequential import Sequential
from layers.Linear import Linear
from layers.Relu import RelU
from losses.CrossEntropyLoss import CrossEntropyLoss
from optim.SGD import SGD

# 构造数据集
np.random.seed(42)
# 第一类（标签0）
class0 = np.random.randn(100, 2) * 0.5 + np.array([-2, -2])
# 第二类（标签1）
class1 = np.random.randn(100, 2) * 0.5 + np.array([2, 2])
# 合并数据
x = np.vstack((class0, class1))
# 标签
y = np.array([0] * 100 + [1] * 100)

print(x.shape)      # (200,2)
print(y.shape)      # (200,)

# 创建模型
model = Sequential(
    Linear(2, 16),
    RelU(),
    Linear(16, 2)
)

# 损失函数
criterion = CrossEntropyLoss()

# 优化器
optimizer = SGD(
    model.parameters(),
    lr=0.01
)

# 开始训练

epochs = 5000

loss_history = []

for epoch in range(epochs):

    optimizer.zero_grad()

    logits = model.forward(x)

    loss = criterion.forward(logits, y)

    loss_history.append(loss)

    grad = criterion.backward()

    model.backward(grad)

    optimizer.step()

    if epoch % 500 == 0:

        print(f"Epoch {epoch:4d} Loss = {loss:.6f}")

# 预测

logits = model.forward(x)

pred = np.argmax(logits, axis=1)

accuracy = np.mean(pred == y)

print()

print(f"Accuracy: {accuracy * 100:.2f}%")

# Loss 曲线

plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Cross Entropy Loss")

plt.grid(True)

plt.show()

# 分类结果

plt.figure(figsize=(6,6))

plt.scatter(
    x[pred == 0, 0],
    x[pred == 0, 1],
    label="Predict Class 0",
    alpha=0.8
)

plt.scatter(
    x[pred == 1, 0],
    x[pred == 1, 1],
    label="Predict Class 1",
    alpha=0.8
)

plt.legend()

plt.title("Classification Result")

plt.xlabel("x1")

plt.ylabel("x2")

plt.grid(True)

plt.show()

# 绘制决策边界
# 生成二维网格
xx, yy = np.meshgrid(
    np.linspace(-4, 4, 300),
    np.linspace(-4, 4, 300)
)
# 拼成 (90000,2)
grid = np.c_[
    xx.ravel(),
    yy.ravel()
]
# 网络预测
logits = model.forward(grid)
pred = np.argmax(logits, axis=1)
# 恢复成网格
pred = pred.reshape(xx.shape)
# 绘图
plt.figure(figsize=(8,6))
# 背景颜色（决策区域）
plt.contourf(
    xx,
    yy,
    pred,
    alpha=0.3
)
# 第一类
plt.scatter(
    class0[:,0],
    class0[:,1],
    label="Class 0"
)
# 第二类
plt.scatter(
    class1[:,0],
    class1[:,1],
    label="Class 1"
)

plt.title("Decision Boundary")

plt.xlabel("x1")

plt.ylabel("x2")

plt.legend()

plt.grid(True)

plt.show()