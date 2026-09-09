import torch


# =========================
# 设备
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# 数据集路径
# =========================

# CIFAR-10 数据在服务器上的位置
# torchvision 会在该目录下找 cifar-10-batches-py
root_dir = "/disk/WangJunHao/CIFAR-10_dataset"


# =========================
# 训练参数
# =========================

batch_size = 128

learning_rate = 0.1

epochs = 100

num_workers = 8


# =========================
# 早停
# =========================

patience = 5


# =========================
# 模型保存路径
# =========================

model_path = "resnet18_best.pth"


# =========================
# 随机种子
# =========================

seed = 42
