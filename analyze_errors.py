import math
from collections import Counter

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

from config import device, model_path, root_dir
from model import build_resnet18


class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# =====================================
# 1. 图片预处理（和验证集一致，不增强）
# =====================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# =====================================
# 2. 加载测试集和模型
# =====================================

print("加载 CIFAR-10 测试集...")

raw_test = datasets.CIFAR10(
    root=root_dir,
    train=False,
    download=False
)

print("加载模型...")

model = build_resnet18()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =====================================
# 3. 遍历测试集，收集错误样本
# =====================================

errors = []

cat_as_dog = []

dog_as_cat = []

confusion_pairs = Counter()


with torch.no_grad():

    for index in range(len(raw_test)):

        image, true_label = raw_test[index]

        image_tensor = transform(image).unsqueeze(0)

        image_tensor = image_tensor.to(device)

        pred_label = model(image_tensor).argmax(1).item()

        if pred_label == true_label:
            continue

        # 记录：索引、图片、真实类别、预测类别
        record = (index, image, true_label, pred_label)

        errors.append(record)

        confusion_pairs[(true_label, pred_label)] += 1

        if true_label == 3 and pred_label == 5:
            cat_as_dog.append(record)

        elif true_label == 5 and pred_label == 3:
            dog_as_cat.append(record)


# =====================================
# 4. 输出统计
# =====================================

print("\n==============================")
print("测试集总错误数：", len(errors))
print()
print("猫(3) -> 狗(5)：", len(cat_as_dog))
print("狗(5) -> 猫(3)：", len(dog_as_cat))
print()
print("最容易混淆的类别对：")

for (true_label, pred_label), count in confusion_pairs.most_common(10):

    print(
        "{:<12s} -> {:<12s} : {}".format(
            class_names[true_label],
            class_names[pred_label],
            count
        )
    )

print("==============================\n")


# =====================================
# 5. 把错误图片拼成网格图
# =====================================

def save_error_grid(records, title, save_path, max_items=15):

    if not records:
        print("没有样本，跳过：", save_path)
        return

    records = records[:max_items]

    cols = 5

    rows = math.ceil(len(records) / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.4, rows * 2.4)
    )

    axes = axes.flatten()

    resample = (
        Image.Resampling.NEAREST
        if hasattr(Image, "Resampling")
        else Image.NEAREST
    )

    for ax in axes:
        ax.axis("off")

    for i, (index, image, true_label, pred_label) in enumerate(records):

        # 32x32 太小，放大一点再展示
        image_large = image.resize(
            (128, 128),
            resample
        )

        axes[i].imshow(image_large)

        axes[i].set_title(
            "idx {}\n{} -> {}".format(
                index,
                class_names[true_label],
                class_names[pred_label]
            ),
            fontsize=9
        )

    fig.suptitle(title, fontsize=13)

    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()

    print("已保存：", save_path)


save_error_grid(
    cat_as_dog,
    "Actually cat, but predicted as dog",
    "error_cat_as_dog.png"
)

save_error_grid(
    dog_as_cat,
    "Actually dog, but predicted as cat",
    "error_dog_as_cat.png"
)
