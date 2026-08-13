"""
YOLOv9 GELAN (Generalized Efficient Layer Aggregation Network) 核心模块
- RepConv: 可重参数化卷积（训练多分支，推理融合为单 3x3）
- RepNCSPELAN4: CSP-ELAN 块（GELAN 核心，含 RepConv 梯度通路）
- SPPELAN: ELAN 风格空间金字塔池化

设计要点：
  - 训练时 RepConv 有 3x3 + 1x1 + identity BN 三分支，梯度流更丰富
  - 推理时三分支可融合为单个 3x3 Conv，零推理开销
  - RepNCSPELAN4 用 CSP 分割 + ELAN 多路聚合，兼顾精度与效率
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class RepConv(nn.Module):
    """
    可重参数化卷积（RepVGG 风格）

    训练时：y = BN(3x3_conv(x)) + BN(1x1_conv(x)) + identity_BN(x)
    推理时：三分支融合为单个 3x3 Conv（通过 fuse_repvgg() 方法）
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, g: int = 1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = s

        # 3x3 分支
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

        # 1x1 分支
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, s, 0, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        # Identity 分支（仅当 in==out 且 stride==1）
        self.identity = nn.BatchNorm2d(in_ch) if in_ch == out_ch and s == 1 else None

        self.act = nn.SiLU()

    def forward(self, x):
        y = self.bn(self.conv(x)) + self.bn1(self.conv1(x))
        if self.identity is not None:
            y = y + self.identity(x)
        return self.act(y)

    def fuse_convs(self):
        """将训练态多分支融合为推理态单个 3x3 Conv（可选优化）"""
        if hasattr(self, 'fused_conv'):
            return
        # 融合 BN 到 Conv
        fused_3x3 = self._fuse_bn(self.conv, self.bn)
        fused_1x1 = self._fuse_bn(self.conv1, self.bn1)
        # 将 1x1 padding 到 3x3
        pad = self.conv.kernel_size[0] // 2
        fused_1x1_padded = torch.nn.functional.pad(fused_1x1.weight, [pad, pad, pad, pad])
        fused_weight = fused_3x3.weight + fused_1x1_padded
        fused_bias = fused_3x3.bias + fused_1x1.bias

        if self.identity is not None:
            id_conv, id_bn = self._fuse_identity(self.identity, self.in_ch)
            fused_weight = fused_weight + id_conv
            fused_bias = fused_bias + id_bn

        self.fused_conv = nn.Conv2d(
            self.in_ch, self.out_ch, 3, self.stride, 1, bias=True)
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias

    @staticmethod
    def _fuse_bn(conv, bn):
        """BN → Conv 融合：w = w * (1/sqrt(var+eps)), b = (b - mean) / sqrt(var+eps)"""
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        fused_weight = conv.weight * scale.reshape(1, -1, 1, 1)
        fused_bias = bn.bias - bn.running_mean * scale
        return nn.Parameter(fused_weight), nn.Parameter(fused_bias)

    @staticmethod
    def _fuse_identity(bn, ch):
        """Identity BN → 3x3 conv 权重"""
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * scale
        # Identity = 中心 1 的 3x3 卷积
        weight = torch.zeros(ch, ch, 3, 3)
        for i in range(ch):
            weight[i, i, 1, 1] = 1.0
        return weight * scale.reshape(1, -1, 1, 1), bias


class RepNCSPELAN4(nn.Module):
    """
    CSP-ELAN 块（GELAN 核心构建块）

    结构：
      input → 1x1 Conv (reduce to c_mid) →
        split:
          path_a: identity (c_mid)
          path_b: RepConv → RepConv (c_mid)
        concat [path_a, path_b] (2*c_mid) →
      1x1 Conv (expand to out_ch) → output

    特点：
      - CSP 分割减少梯度重复路径
      - RepConv 提供更丰富的训练态梯度
      - ELAN 多路聚合增强特征表达
    """

    def __init__(self, in_ch: int, out_ch: int, c_mid: int = None):
        super().__init__()
        if c_mid is None:
            c_mid = out_ch // 2

        # 输入降维
        self.cv_in = Conv(in_ch, c_mid, 1, 1)

        # RepConv 处理路径
        self.rep1 = RepConv(c_mid, c_mid)
        self.rep2 = RepConv(c_mid, c_mid)

        # 输出融合
        self.cv_out = Conv(c_mid * 2, out_ch, 1, 1)

    def forward(self, x):
        y0 = self.cv_in(x)               # 降维
        y1 = self.rep2(self.rep1(y0))    # RepConv 处理
        return self.cv_out(torch.cat([y0, y1], dim=1))  # CSP-ELAN 聚合


class SPPELAN(nn.Module):
    """
    ELAN 风格空间金字塔池化

    多尺度 MaxPool + 1x1 Conv 聚合，增强 P5 全局感受野
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        c_mid = out_ch // 2
        self.cv1 = Conv(in_ch, c_mid, 1, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2),
            nn.MaxPool2d(kernel_size=k * 2 - 1, stride=1, padding=(k * 2 - 1) // 2),  # 9
            nn.MaxPool2d(kernel_size=k * 3 - 2, stride=1, padding=(k * 3 - 2) // 2),  # 13
        ])
        self.cv2 = Conv(c_mid * 4, out_ch, 1, 1)  # 原 1x + 3 pools = 4x

    def forward(self, x):
        y = self.cv1(x)
        return self.cv2(torch.cat([y] + [p(y) for p in self.pools], dim=1))


def build_gelan_backbone(in_ch: int, out_channels: list) -> nn.ModuleList:
    """
    构建 GELAN 主干（替换 CSPDarknet）

    Args:
        in_ch: 输入通道数（来自模态融合后的通道，通常 32）
        out_channels: [P3_ch, P4_ch, P5_ch] 输出通道数

    Returns:
        nn.ModuleList，通过索引访问各层
        结构：
          [0] Conv(in_ch, c3, 3, 2)      # 1/4 → 1/8
          [1] RepNCSPELAN4(c3, c3)        # P3
          [2] Conv(c3, c4, 3, 2)          # 1/8 → 1/16
          [3] RepNCSPELAN4(c4, c4)        # P4
          [4] Conv(c4, c5, 3, 2)          # 1/16 → 1/32
          [5] RepNCSPELAN4(c5, c5)        # P5
          [6] SPPELAN(c5, c5)             # P5 SPP
    """
    c3, c4, c5 = out_channels

    layers = nn.ModuleList()
    # P3 (1/8)
    layers.append(Conv(in_ch, c3, 3, 2))           # 0: downsample
    layers.append(RepNCSPELAN4(c3, c3))             # 1: P3 block
    # P4 (1/16)
    layers.append(Conv(c3, c4, 3, 2))               # 2: downsample
    layers.append(RepNCSPELAN4(c4, c4))             # 3: P4 block
    # P5 (1/32)
    layers.append(Conv(c4, c5, 3, 2))               # 4: downsample
    layers.append(RepNCSPELAN4(c5, c5))             # 5: P5 block
    layers.append(SPPELAN(c5, c5))                   # 6: SPP

    return layers


def gelan_out_channels(model_size: str = "s") -> list:
    """返回 GELAN backbone 输出 P3/P4/P5 通道数（与 CSPDarknet 对齐）"""
    cfg = {
        "n": [64, 128, 256],
        "s": [128, 256, 512],
        "m": [192, 384, 768],
        "c": [256, 512, 768],   # YOLOv9c 等效通道
    }
    return cfg.get(model_size, cfg["s"])
