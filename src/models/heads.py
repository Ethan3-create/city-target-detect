"""
自研 YOLOv8 检测头（经典架构）
- 训练模式：返回 3 个尺度的原始输出张量 [B, nc+4*reg_max, H, W]
  （与自研损失 MultiModalDetectionLoss 对接，使用 TaskAlignedAssigner）
- 推理模式：返回 (解码预测 [B, 4+nc, N], 特征图列表)
  解码输出 x1y1x2y2 像素坐标 + 类别得分，可直接解析
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import DFL
from ultralytics.utils.tal import make_anchors, dist2bbox


class YOLOv8Head(nn.Module):
    """YOLOv8 检测头（自研，兼容训练/推理双模式）"""

    def __init__(self, nc: int = 12, reg_max: int = 16, ch: tuple = (64, 128, 256)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)                     # 检测尺度数（3）
        self.reg_max = reg_max                # DFL 通道（4*reg_max 回归输出）
        self.no = nc + self.reg_max * 4       # 每个 anchor 的输出通道
        self.stride = torch.zeros(self.nl)    # 各尺度下采样倍数（构建后设置）

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # 回归分支（DFL 分布）
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        # 分类分支
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 推理缓存（make_anchors 结果）
        self.anchors = None
        self.strides = None

    def forward(self, x):
        """x: [P3, P4, P5] 特征图列表"""
        shape = x[0].shape  # B, C, H, W

        # 各尺度拼接 回归+分类 输出 → [B, no, H, W]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x  # 原始输出供损失计算

        # ---- 推理解码 ----
        self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # [B, no, N]
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        # xywh=False → 直接输出 xyxy 像素坐标（与后处理解析一致）
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=False, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)  # [B, 4+nc, N] (xyxy 像素坐标 + 得分)
        return y, x

    def bias_init(self):
        """初始化检测头偏置（可选）"""
        pass


def build_backbone(model_size: str = "n", in_ch: int = 3) -> nn.ModuleList:
    """
    构建 YOLOv8 主干（复用 ultralytics Conv/C2f/SPPF 模块）
    返回 ModuleList（10 层），输出 P3/P4/P5 三个尺度特征

    结构（与 yolov8*.yaml 对齐）:
      P1: Conv(in_ch, c1, 3, 2)                        # 1/2
      P2: Conv(c1, c2, 3, 2), C2f(c2, c2, n1)           # 1/4
      P3: Conv(c2, c3, 3, 2), C2f(c3, c3, n2)           # 1/8
      P4: Conv(c3, c4, 3, 2), C2f(c4, c4, n3)           # 1/16
      P5: Conv(c4, c5, 3, 2), C2f(c5, c5, n4), SPPF(c5) # 1/32
    """
    from ultralytics.nn.modules.block import C2f, SPPF

    cfg = {
        "n": {"width": [16, 32, 64, 128, 256], "depth": [1, 2, 2, 1]},
        "s": {"width": [32, 64, 128, 256, 512], "depth": [1, 2, 2, 1]},
        "m": {"width": [48, 96, 192, 384, 768], "depth": [2, 4, 4, 2]},
    }
    c = cfg.get(model_size, cfg["n"])
    c1, c2, c3, c4, c5 = c["width"]
    n1, n2, n3, n4 = c["depth"]

    layers = nn.ModuleList()
    layers.append(Conv(in_ch, c1, 3, 2))                    # P1
    layers.append(Conv(c1, c2, 3, 2))                       # P2 reduce
    layers.append(C2f(c2, c2, n1, True))                    # P2
    layers.append(Conv(c2, c3, 3, 2))                       # P3 reduce
    layers.append(C2f(c3, c3, n2, True))                    # P3
    layers.append(Conv(c3, c4, 3, 2))                       # P4 reduce
    layers.append(C2f(c4, c4, n3, True))                    # P4
    layers.append(Conv(c4, c5, 3, 2))                       # P5 reduce
    layers.append(C2f(c5, c5, n4, True))                    # P5
    layers.append(SPPF(c5, c5, k=5))                        # P5 sppf
    return layers


def backbone_out_channels(model_size: str = "n") -> list:
    """返回 backbone 输出 P3/P4/P5 通道数"""
    cfg = {
        "n": [64, 128, 256],
        "s": [128, 256, 512],
        "m": [192, 384, 768],
        "c": [256, 512, 768],   # YOLOv9c 等效通道
    }
    return cfg.get(model_size, cfg["n"])
