import torch
from torch import nn


# =====================================
# 1. 3x3 卷积封装
# =====================================

def conv3x3(in_channels, out_channels, stride=1):
   return nn.Conv2d(
    in_channels,out_channels,
    3,stride=stride,
    padding=1,bias=False
   )
    


# =====================================
# 2. BasicBlock：一个残差块
# =====================================

class BasicBlock(nn.Module):

    """
    残差块的结构：

        x ── conv1(3x3) ── bn1 ── ReLU ── conv2(3x3) ── bn2 ──┐
          │                                                     │
          └─────────────── 跳跃连接（必要时先 downsample）──────┘
                              -> 相加 -> ReLU -> 输出

    跳跃连接只有在"通道数或特征图尺寸发生变化"时才需要变换，
    因此参数里有一个 downsample。
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        #这是一个残差块
        self.resnet_block = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.resnet_block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        return nn.functional.relu(x + identity)

       


# =====================================
# 3. 把多个 BasicBlock 堆成一个 stage
# =====================================

def make_layer(in_channels, out_channels, num_blocks, stride=1):

    """
    一个 stage 由 num_blocks 个 BasicBlock 组成。

    需要处理的事：
    1. 第一个 block 输入 in_channels、输出 out_channels，
       如果 stride != 1 或通道数变化，需要构造 downsample：
           nn.Sequential(
               nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_channels)
           )
    2. 把第一个 block 放进列表
    3. 后面的 block 输入输出都是 out_channels，不需要 downsample
    4. 返回 nn.Sequential(*layers)
    """
    layers = []
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            #1*1卷积层
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    #第一个block
    layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
    #后续的block
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


# =====================================
# 4. ResNet 主体
# =====================================

class ResNet(nn.Module):

    """
    CIFAR-10 版 ResNet-18 的整体结构：

    stem:
        conv 3x3(3 -> 64) -> BN -> ReLU     图片 32x32 保持不变

    四个 stage（特征图尺寸依次减半）：
        layer1: 64 -> 64，   stride=1，输出 32x32
        layer2: 64 -> 128，  stride=2，输出 16x16
        layer3: 128 -> 256， stride=2，输出 8x8
        layer4: 256 -> 512， stride=2，输出 4x4

    分类头：
        AdaptiveAvgPool2d((1, 1)) -> flatten -> Linear(512, 10)
    """

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            make_layer(64, 64, 2),
            make_layer(64, 128, 2, stride=2),
            make_layer(128, 256, 2, stride=2),
            make_layer(256, 512, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# =====================================
# 5. 创建 ResNet-18 的入口
# =====================================

def build_resnet18(num_classes=10):
    return ResNet(num_classes=num_classes)


# =====================================
# 6. 自测：查看参数量与输出形状
# =====================================

if __name__ == "__main__":
    model = build_resnet18()
    print("参数量：", sum(p.numel() for p in model.parameters()))
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print("输出形状：", out.shape)
    


