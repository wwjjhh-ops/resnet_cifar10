import os

import torch

# 服务器上没有图形界面，必须用 Agg 后端把图画成文件
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================
# 1. 加载训练历史
# =====================================

if not os.path.exists("training_history.pth"):
    print("找不到 training_history.pth")
    print("请先运行 train.py 完成训练。")
    exit()


history = torch.load(
    "training_history.pth",
    map_location="cpu"
)


train_losses = history["train_losses"]

train_accuracies = history["train_accuracies"]

val_losses = history["val_losses"]

val_accuracies = history["val_accuracies"]


epochs = range(
    1,
    len(train_losses) + 1
)


# =====================================
# 2. Loss 曲线
# =====================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.title(
    "Training and Validation Loss"
)

plt.savefig(
    "loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =====================================
# 3. Accuracy 曲线
# =====================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.title(
    "Training and Validation Accuracy"
)

plt.savefig(
    "accuracy_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =====================================
# 4. 输出结果
# =====================================

print("曲线已保存：")
print("loss_curve.png")
print("accuracy_curve.png")
