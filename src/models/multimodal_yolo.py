"""
多模态融合目标检测模型
- EarlyFusionYOLO      ：早期融合（RGB+IR+Depth → 5 通道输入，快速基线）
- StageGateFusionYOLO  ：中期融合（三模态独立 Stem + 模态分支 C2f + 分层门控融合，核心方案）
- 统一接口 forward(rgb, ir, depth)：
    训练 → (feats_list, aux_dict)；推理 → (preds, feats)（与经典 YOLO 一致）
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv

from src.models.heads import YOLOv8Head, build_backbone, backbone_out_channels
from src.models.stage_gate_fusion import StageGateFusion


class _BaseMultiModalModel(nn.Module):
    """多模态模型基类：统一接口 + 辅助头 + stride 设置"""

    def __init__(self, num_classes=12, model_size="n", use_aux=True):
        super().__init__()
        self.num_classes = num_classes
        self.model_size = model_size
        self.use_aux = use_aux
        self.head_ch = backbone_out_channels(model_size)  # [P3, P4, P5] 通道
        self.head = YOLOv8Head(nc=num_classes, reg_max=16, ch=tuple(self.head_ch))

        # 辅助预测头（训练时监督伪模态重建）
        if use_aux:
            c = self.head_ch[0]  # P3 通道
            self.aux_ir_head = nn.Conv2d(c, 1, 1)
            self.aux_dm_head = nn.Conv2d(c, 1, 1)
            self.aux_dn_head = nn.Conv2d(c, 2, 1)

    def _init_strides(self, dummy_rgb=None):
        """通过一次前向计算各尺度 stride（标准结构为 8/16/32）"""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            if dummy_rgb is None:
                h, w = 64, 64
                rgb = torch.zeros(1, 3, h, w, device=device)
                ir = torch.zeros(1, 1, h, w, device=device)
                depth = torch.zeros(1, 1, h, w, device=device)
            else:
                rgb, ir, depth = dummy_rgb
                rgb, ir, depth = rgb.to(device), ir.to(device), depth.to(device)
                h, w = rgb.shape[2:]
            feats = self._extract(rgb, ir, depth)
            strides = [round(h / f.shape[2]) for f in feats]
        self.head.stride = torch.tensor(strides, device=device)
        self.train()
        return strides

    def _aux_outputs(self, p3_feat):
        """从 P3 特征预测伪模态（训练用）"""
        return {
            "ir_contrast_pred": self.aux_ir_head(p3_feat),
            "depth_mask_pred": self.aux_dm_head(p3_feat),
            "depth_normal_pred": self.aux_dn_head(p3_feat),
        }

    def forward(self, rgb, ir, depth):
        feats = self._extract(rgb, ir, depth)
        # 注意：YOLOv8Head.forward 会就地修改 feats 列表（替换为拼接后的 [no] 通道张量）
        # 因此辅助头需使用 head 调用前的原始 P3 特征
        p3_raw = feats[0]
        out = self.head(feats)
        if self.training:
            aux = self._aux_outputs(p3_raw) if self.use_aux else None
            return out, aux
        return out


class EarlyFusionYOLO(_BaseMultiModalModel):
    """
    早期融合：三模态拼接为 5 通道输入 → 标准 YOLOv8 backbone → 检测头
    优点：结构最简单、最快；backbone 大部分可复用 COCO 预训练权重
    """

    def __init__(self, num_classes=12, model_size="n", use_aux=True):
        super().__init__(num_classes, model_size, use_aux)
        self.backbone = build_backbone(model_size, in_ch=5)

    def _extract(self, rgb, ir, depth):
        x = torch.cat([rgb, ir, depth], dim=1)  # [B, 5, H, W]
        l = self.backbone
        x = l[0](x)             # P1 1/2
        x = l[1](x)             # P2 1/4
        x = l[2](x)
        x = l[3](x)             # P3 1/8
        p3 = l[4](x)
        x = l[5](p3)            # P4 1/16
        p4 = l[6](x)
        x = l[7](p4)            # P5 1/32
        p5 = l[9](l[8](x))
        return [p3, p4, p5]


class StageGateFusionYOLO(_BaseMultiModalModel):
    """
    中期融合（核心方案）：
      RGB Stem ──→ C2f 分支 ──┐
      IR  Stem ──→ C2f 分支 ──┼→ 门控融合(1/4) → 共享主干 P3/P4/P5 → 检测头
      Dep Stem ──→ C2f 分支 ──┘

    门控融合用轻量权重动态组合三模态特征：
      - 浅层几何信息（Depth/IR）与语义信息（RGB）自适应加权
      - 参数量极小（数千），单模型单前向，满足竞赛约束
    """

    def __init__(self, num_classes=12, model_size="n", use_aux=True,
                 fusion_points: int = 1):
        super().__init__(num_classes, model_size, use_aux)
        self.fusion_points = fusion_points

        # 三模态独立 Stem（各下采样 2 次 → 1/4，通道 32）
        stem_ch = 32
        self.stem_rgb = nn.Sequential(
            Conv(3, stem_ch, 3, 2), Conv(stem_ch, stem_ch, 3, 2))
        self.stem_ir = nn.Sequential(
            Conv(1, stem_ch, 3, 2), Conv(stem_ch, stem_ch, 3, 2))
        self.stem_dp = nn.Sequential(
            Conv(1, stem_ch, 3, 2), Conv(stem_ch, stem_ch, 3, 2))

        # 模态特定特征分支（1/4 分辨率）
        self.branch_rgb = C2f(stem_ch, stem_ch, 1, True)
        self.branch_ir = C2f(stem_ch, stem_ch, 1, True)
        self.branch_dp = C2f(stem_ch, stem_ch, 1, True)

        # 分层门控融合（1/4 分辨率，单融合点）
        self.fusion = StageGateFusion([stem_ch], num_modalities=3)

        # 共享主干（对应标准 YOLOv8 的 P3~P5）
        c3, c4, c5 = self.head_ch
        self.body = nn.ModuleList([
            Conv(stem_ch, c3, 3, 2), C2f(c3, c3, 2, True),       # P3 1/8
            Conv(c3, c4, 3, 2), C2f(c4, c4, 2, True),            # P4 1/16
            Conv(c4, c5, 3, 2), C2f(c5, c5, 1, True), SPPF(c5, c5, 5),  # P5 1/32
        ])

    def _extract(self, rgb, ir, depth):
        xr = self.branch_rgb(self.stem_rgb(rgb))
        xi = self.branch_ir(self.stem_ir(ir))
        xd = self.branch_dp(self.stem_dp(depth))

        # 门控融合（stage_bias 自动施加浅层 Depth/IR 偏好）
        fused = self.fusion([[xr, xi, xd]])[0]

        b = self.body
        x = b[1](b[0](fused))      # P3 1/8
        p3 = x
        x = b[3](b[2](x))          # P4 1/16
        p4 = x
        p5 = b[6](b[5](b[4](x)))   # P5 1/32
        return [p3, p4, p5]


def build_multimodal_model(fusion_mode="stage_gate", num_classes=12,
                           model_size="n", use_aux=True):
    """工厂函数：按融合策略构建模型"""
    if fusion_mode == "early":
        return EarlyFusionYOLO(num_classes, model_size, use_aux)
    elif fusion_mode == "stage_gate":
        return StageGateFusionYOLO(num_classes, model_size, use_aux)
    else:
        raise ValueError(f"未知融合策略: {fusion_mode}（可选 early / stage_gate）")


def load_pretrained_weights(model: nn.Module, weights_path: str,
                            verbose: bool = True) -> int:
    """
    加载 COCO 预训练权重（ultralytics .pt），自动处理通道/结构差异
    匹配规则：
      - key 完全一致且 shape 一致 → 复制
      - key 前缀 'model.' → 按模型类型映射（early: backbone./head./aux*；stage: body.）
      - 第一层卷积 5ch（early 融合）→ 用预训练前 3 通道初始化
    返回加载成功的参数数量
    """
    import numpy as np

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        pt_model = ckpt["model"]
        sd = pt_model.state_dict() if hasattr(pt_model, "state_dict") else pt_model
    elif isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt)
    else:
        sd = ckpt.state_dict()

    own_sd = model.state_dict()
    loaded = 0
    skipped = 0

    # key 映射（ultralytics 'model.N.*' → 本模型命名）
    def map_key(k):
        if isinstance(model, EarlyFusionYOLO):
            return k.replace("model.", "backbone.", 1) if k.startswith("model.") else k
        elif isinstance(model, StageGateFusionYOLO):
            # body 对应预训练 model.3 ~ model.9
            if k.startswith("model."):
                idx = int(k.split(".")[1])
                if 3 <= idx <= 9:
                    return "body." + ".".join([str(idx - 3)] + k.split(".")[2:])
                return None  # model.0~2 为 stem 部分，结构不同
            return k
        return k

    for k, v in sd.items():
        mk = map_key(k)
        if mk is None or mk not in own_sd:
            skipped += 1
            continue
        ov = own_sd[mk]
        if ov.shape == v.shape:
            own_sd[mk].copy_(v)
            loaded += 1
        elif (ov.dim() == 4 and v.dim() == 4
              and ov.shape[1] == 5 and v.shape[1] == 3):
            # 第一层卷积 3→5 通道：前 3 通道用预训练，后 2 通道用小随机
            with torch.no_grad():
                own_sd[mk][:, :3].copy_(v)
                nn.init.kaiming_normal_(own_sd[mk][:, 3:], mode="fan_out", nonlinearity="relu")
            loaded += 1
        else:
            skipped += 1

    if verbose:
        print(f"  预训练加载: 成功 {loaded} 个参数块, 跳过 {skipped} 个不匹配")
    return loaded
