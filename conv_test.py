import numpy as np
from core.summary import summary
from layers.Conv2D import Conv2D
from layers.Relu import RelU
from layers.MaxPool2D import MaxPool2D
from layers.Linear import Linear
from core.Sequential import Sequential
from core.Flatten import Flatten
from losses.MSELoss import MSELoss
from optim.SGD import SGD
import time
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# =====================
# 数据
# =====================
mnist = fetch_openml("mnist_784", version=1)

X = mnist.data.values.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

X = X.reshape(-1, 1, 28, 28)

# one-hot !!!
num_classes = 10
Y = np.zeros((y.shape[0], num_classes))
Y[np.arange(y.shape[0]), y] = 1
X = X[:200]
Y = Y[:200]

# =====================
# 模型
# =====================
model = Sequential(
    Conv2D(1, 8, 3),
    RelU(),
    MaxPool2D(2, 2),

    Conv2D(8, 16, 3),
    RelU(),
    MaxPool2D(2, 2),

    Flatten(),
    Linear(16 * 5 * 5, 128),
    RelU(),
    Linear(128, 10)
)
summary(model, (1, 28, 28))

# =====================
# loss + optimizer
# =====================
criterion = MSELoss()

optimizer = SGD(
    model.parameters(),
    lr=0.01
)


# =====================
# accuracy
# =====================
def accuracy(pred, target):
    pred_label = np.argmax(pred, axis=1)
    true_label = np.argmax(target, axis=1)
    return np.mean(pred_label == true_label)


# =====================
# 训练
# =====================
import time

epochs = 50
batch_size = 64

print("开始训练")
loss_history = []
acc_history = []
for epoch in range(epochs):

    start_time = time.time()

    total_loss = 0
    total_acc = 0
    count = 0

    print(f"\n===== Epoch {epoch} 开始 =====", flush=True)

    for i in range(0, len(X), batch_size):

        batch_idx = i // batch_size

        # if batch_idx % 200 == 0:
        print(f'开始第 {batch_idx} 个 batch训练', flush=True)

        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]

        # forward
        pred = model.forward(x_batch)
        loss = criterion.forward(pred, y_batch)

        # backward
        grad = criterion.backward()
        model.backward(grad)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss
        total_acc += accuracy(pred, y_batch)
        count += 1

    end_time = time.time()

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    loss_history.append(avg_loss)
    acc_history.append(avg_acc)
    epoch_time = end_time - start_time

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Acc: {avg_acc:.4f} | "
        f"Time: {epoch_time:.2f}s",
        flush=True
    )

print("训练完成", flush=True)
plt.figure(figsize=(12,5))

# ===== loss =====
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# ===== acc =====
plt.subplot(1,2,2)
plt.plot(acc_history)
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.show()