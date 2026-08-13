"""
分层模态门控融合 (Stage-wise Modality Gating Fusion)

核心思想：在融合点用轻量门控网络动态学习"该信哪个模态"的权重。
- 浅层：几何/边缘信息丰富 → 侧重 Depth + IR
- 深层：语义信息丰富 → 侧重 RGB + 全局 IR 温差

设计原则：
- 单模型、单前向、无集成（满足竞赛约束）
- 门控参数量极少（每级数千参数）
- 可嵌入 YOLOv8 架构任意 stage
"""

import torch
import torch.nn as nn


class ModalityGate(nn.Module):
    """
    单级模态门控单元

    输入: 三模态特征 [B, C, H, W] × 3
    输出: 融合特征 [B, C, H, W]

    原理: 各模态 1×1 压缩 → 拼接 → 1×1 Conv + Sigmoid 生成逐通道权重 → 加权求和 → 投影
    """

    def __init__(self, channels: int, num_modalities: int = 3, reduction: int = 8):
        super().__init__()
        self.num_modalities = num_modalities
        self.channels = channels
        r = max(reduction, 2)
        mid = max(channels // r, 4)

        self.modal_proj = nn.ModuleList([
            nn.Conv2d(channels, mid, 1) for _ in range(num_modalities)
        ])
        self.gate_conv = nn.Sequential(
            nn.Conv2d(mid * num_modalities, channels * num_modalities, 1),
            nn.Sigmoid(),
        )
        self.fuse_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, modal_features):
        """
        Args:
            modal_features: list of [B, C, H, W]，长度 = num_modalities
        Returns:
            fused: [B, C, H, W]
        """
        projected = [self.modal_proj[i](f) for i, f in enumerate(modal_features)]
        concat = torch.cat(projected, dim=1)                 # [B, mid*N, H, W]
        gates = self.gate_conv(concat)                       # [B, C*N, H, W]
        b, _, h, w = gates.shape
        gates = gates.view(b, self.num_modalities, self.channels, h, w)

        fused = sum(gates[:, i] * modal_features[i] for i in range(self.num_modalities))
        return self.fuse_conv(fused)


class StageGateFusion(nn.Module):
    """
    分层门控融合（可覆盖多个 Stage）

    用法:
        fusion = StageGateFusion(stage_channels=[32, 64])
        fused = fusion([[rgb0, ir0, dp0], [rgb1, ir1, dp1]])
    """

    def __init__(self, stage_channels: list, num_modalities: int = 3, reduction: int = 8):
        super().__init__()
        self.num_stages = len(stage_channels)
        self.gates = nn.ModuleList([
            ModalityGate(ch, num_modalities, reduction) for ch in stage_channels
        ])

        # 可学习模态偏好偏置（浅层偏 IR/Depth，深层偏 RGB）
        self.register_parameter(
            "stage_bias", nn.Parameter(torch.zeros(self.num_stages, num_modalities))
        )
        with torch.no_grad():
            for s in range(self.num_stages):
                depth_factor = 1.0 - s / self.num_stages
                self.stage_bias[s, 0] = 0.2 + 0.3 * (s / self.num_stages)  # RGB 浅层低
                self.stage_bias[s, 1] = 0.4 + 0.2 * depth_factor            # IR 始终较高
                self.stage_bias[s, 2] = 0.4 + 0.3 * depth_factor            # Depth 浅层高

    def forward(self, stage_features_list):
        """
        Args:
            stage_features_list: list of (list of 模态特征)
                外层长度 = num_stages；内层 [rgb, ir, depth]
        Returns:
            fused_stages: list of [B, C, H, W]
        """
        fused = []
        for i, (modal_feats, gate) in enumerate(zip(stage_features_list, self.gates)):
            biased = [f * (1.0 + self.stage_bias[i, m]) for m, f in enumerate(modal_feats)]
            fused.append(gate(biased))
        return fused


def make_multimodal_stem(in_channels_list=(3, 1, 1), out_ch: int = 32,
                         down: int = 2) -> nn.ModuleList:
    """
    多模态独立 Stem：各模态独立下采样卷积（保持预训练友好）
    返回 ModuleList（每模态一个 [Conv→BN→SiLU]×down）
    """
    stems = nn.ModuleList()
    for c in in_channels_list:
        layers = []
        cur = c
        for _ in range(down):
            layers.append(nn.Conv2d(cur, out_ch, 3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.SiLU(inplace=True))
            cur = out_ch
        stems.append(nn.Sequential(*layers))
    return stems
