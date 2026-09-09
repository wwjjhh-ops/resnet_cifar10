import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from config import device, model_path

from model import build_resnet18


# =====================================
# 1. CIFAR-10 类别名
# =====================================

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
# 2. 图片预处理
# =====================================

# 注意：预测时不能使用 train_transform
# 因为里面有随机裁剪和随机翻转。
# 这里使用和验证集 / 测试集相同的预处理方式。

transform = transforms.Compose([
    # 模型要求输入 32x32，任意图片先缩放到这个尺寸
    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# =====================================
# 3. 加载模型
# =====================================

print("当前设备：", device)


model = build_resnet18()

if not os.path.exists(model_path):
    print("找不到模型文件：", model_path)
    print("请先运行 train.py 完成训练。")
    exit()


# 加载训练好的模型参数
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

# 将模型移动到 GPU / CPU
model = model.to(device)

# 开启推理模式
model.eval()

print("模型加载成功！")


# =====================================
# 4. 预测函数
# =====================================

def predict_image(image_path):

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print("找不到图片：", image_path)
        return

    # 读取图片
    image = Image.open(image_path).convert("RGB")

    # 图片预处理
    image_tensor = transform(image)

    # 增加 Batch 维度
    # [3, 32, 32] -> [1, 3, 32, 32]
    image_tensor = image_tensor.unsqueeze(0)

    # 移动到 GPU / CPU
    image_tensor = image_tensor.to(device)

    # 关闭梯度计算
    with torch.no_grad():

        # 模型预测
        output = model(image_tensor)

        # 转换成概率
        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        # 获取概率最大的类别
        predicted_index = torch.argmax(
            probabilities
        ).item()


    # =====================================
    # 5. 输出结果
    # =====================================

    predicted_class = class_names[predicted_index]

    print()
    print("==============================")
    print("图片：", os.path.basename(image_path))
    print()
    print("预测结果：", predicted_class)
    print()

    # 打印每个类别的概率，方便看模型判断依据
    for i, class_name in enumerate(class_names):

        print(
            "{:<12s}: {:.2f}%"
            .format(
                class_name,
                probabilities[i].item() * 100
            )
        )

    print("==============================")


# =====================================
# 6. 输入图片路径
# =====================================

while True:

    print()
    image_path = input(
        "请输入图片路径（输入 q 退出）："
    )

    # 输入 q 退出
    if image_path.lower() == "q":
        print("程序结束。")
        break

    # 去掉可能存在的引号
    image_path = image_path.strip().strip('"')

    # 进行预测
    predict_image(image_path)
