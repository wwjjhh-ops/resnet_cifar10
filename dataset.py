import torch
from torch.utils.data import (
    DataLoader,
    Subset
)
from torchvision import datasets, transforms

from config import (
    root_dir,
    batch_size,
    num_workers,
    seed
)


# =====================================
# 1. 数据增强与归一化
# =====================================

# 训练集：增强 + 归一化
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# 验证集：不增强，只归一化
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# 测试集：和验证集一样，不增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# =====================================
# 2. 划分训练集与验证集
# =====================================

def split_dataset():

    # 固定随机种子，保证每次划分结果一致
    generator = torch.Generator().manual_seed(seed)

    # 把 50000 个训练样本的索引随机打乱
    indices = torch.randperm(
        50000,
        generator=generator
    )

    # 前 45000 个给训练集，后 5000 个给验证集
    train_indices = indices[:45000].tolist()

    val_indices = indices[45000:].tolist()

    return train_indices, val_indices


# =====================================
# 3. 创建三个 DataLoader
# =====================================

def get_dataloaders():

    # 先算好训练 / 验证的索引
    train_indices, val_indices = split_dataset()

    # 训练集需要增强，验证集不增强，
    # 所以同一份 50000 张图用不同 transform 加载两次
    train_aug_dataset = datasets.CIFAR10(
        root=root_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    train_plain_dataset = datasets.CIFAR10(
        root=root_dir,
        train=True,
        download=False,
        transform=val_transform
    )

    # 用之前算好的索引切片
    train_dataset = Subset(
        train_aug_dataset,
        train_indices
    )

    val_dataset = Subset(
        train_plain_dataset,
        val_indices
    )

    # 官方测试集，共 10000 张
    test_dataset = datasets.CIFAR10(
        root=root_dir,
        train=False,
        download=False,
        transform=test_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("总数据：", len(train_aug_dataset))
    print("训练集：", len(train_dataset))
    print("验证集：", len(val_dataset))
    print("测试集：", len(test_dataset))

    return train_loader, val_loader, test_loader
