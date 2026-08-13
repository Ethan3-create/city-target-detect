"""
多模态融合目标检测模型
- EarlyFusionYOLO      ：早期融合（RGB+IR+Depth → 5 通道输入，快速基线）
- StageGateFusionYOLO  ：中期融合（三模态独立 Stem + 模态分支 C2f + 分层门控融合，核心方案）
- StageGateFusionYOLOv9：YOLOv9 GELAN backbone + PGI 辅助分支（升级方案）
- 统一接口 forward(rgb, ir, depth)：
    训练 → (feats_list, aux_dict)；推理 → (preds, feats)（与经典 YOLO 一致）
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv

from src.models.heads import YOLOv8Head, build_backbone, backbone_out_channels
from src.models.stage_gate_fusion import StageGateFusion
from src.models.gelan import RepNCSPELAN4, SPPELAN, build_gelan_backbone, gelan_out_channels


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


class PGIBranch(nn.Module):
    """
    PGI (Programmable Gradient Information) 辅助可逆分支

    功能：
      - 轻量级并行路径，从融合特征产生 P3/P4/P5 辅助特征
      - 训练时通过辅助检测头提供额外监督信号
      - 推理时不参与前向，零推理开销

    设计：
      - 使用普通 Conv+C2f（不用 RepConv），保证梯度可逆性
      - 通道数与主分支对齐，便于辅助损失计算
    """

    def __init__(self, in_ch: int, out_channels: list):
        super().__init__()
        c3, c4, c5 = out_channels

        # 轻量级主干（与 GELAN 主干并行，但结构更简单）
        self.body = nn.ModuleList([
            Conv(in_ch, c3, 3, 2), C2f(c3, c3, 1, True),       # P3 1/8
            Conv(c3, c4, 3, 2), C2f(c4, c4, 1, True),           # P4 1/16
            Conv(c4, c5, 3, 2), C2f(c5, c5, 1, True), SPPF(c5, c5, 5),  # P5 1/32
        ])

    def forward(self, x):
        b = self.body
        x = b[1](b[0](x))       # P3
        p3 = x
        x = b[3](b[2](x))       # P4
        p4 = x
        p5 = b[6](b[5](b[4](x)))  # P5
        return [p3, p4, p5]


class StageGateFusionYOLOv9(_BaseMultiModalModel):
    """
    YOLOv9 升级版（GELAN backbone + PGI 辅助分支）：
      RGB Stem ──→ C2f 分支 ──┐
      IR  Stem ──→ C2f 分支 ──┼→ 门控融合(1/4) → GELAN 主干 P3/P4/P5 → 检测头
      Dep Stem ──→ C2f 分支 ──┘                    ↘ PGI 辅助分支 → 辅助检测头（训练）

    升级要点：
      1. backbone: CSPDarknet → GELAN (RepNCSPELAN4 + SPPELAN)
      2. PGI: 训练时启用辅助可逆分支 + 辅助检测头，增强梯度流
      3. Stem + ModalityGate 融合门控：保持不变（已验证有效）
      4. 预训练：支持 yolov9c.pt 权重加载
    """

    def __init__(self, num_classes=12, model_size="c", use_aux=True,
                 fusion_points: int = 1):
        # 使用 GELAN 通道配置
        ch = gelan_out_channels(model_size)
        # 临时设置 head_ch 供 _BaseMultiModalModel.__init__ 使用
        self._gelan_ch = ch
        super().__init__(num_classes, model_size, use_aux)

        # 覆盖 head_ch 和 head（使用 GELAN 通道）
        self.head_ch = ch
        self.head = YOLOv8Head(nc=num_classes, reg_max=16, ch=tuple(ch))

        # 重建辅助头（使用 GELAN 通道）
        if use_aux:
            c = ch[0]
            self.aux_ir_head = nn.Conv2d(c, 1, 1)
            self.aux_dm_head = nn.Conv2d(c, 1, 1)
            self.aux_dn_head = nn.Conv2d(c, 2, 1)

        self.fusion_points = fusion_points

        # 三模态独立 Stem（与 StageGateFusionYOLO 完全相同）
        stem_ch = 32
        self.stem_rgb = nn.Sequential(
            Conv(3, stem_ch, 3, 2), Conv(stem_ch, stem_ch, 3, 2))
        self.stem_ir = nn.Sequential(
            Conv(1, stem_ch, 3, 2), Conv(stem_ch, stem_ch, 3, 2))
        self.stem_dp = nn.Sequential(
            Conv(1, stem_ch, 3, 2), Conv(stem_ch, stem_ch, 3, 2))

        # 模态特定特征分支
        self.branch_rgb = C2f(stem_ch, stem_ch, 1, True)
        self.branch_ir = C2f(stem_ch, stem_ch, 1, True)
        self.branch_dp = C2f(stem_ch, stem_ch, 1, True)

        # 分层门控融合（不变）
        self.fusion = StageGateFusion([stem_ch], num_modalities=3)

        # GELAN 主干（替换 CSPDarknet）
        self.body = build_gelan_backbone(stem_ch, ch)

        # PGI 辅助可逆分支（训练时启用）
        self.pgi_branch = PGIBranch(stem_ch, ch)
        # PGI 辅助检测头
        self.pgi_head = YOLOv8Head(nc=num_classes, reg_max=16, ch=tuple(ch))

    def _init_strides(self, dummy_rgb=None):
        """通过一次前向计算各尺度 stride，同时设置主头和 PGI 辅助头"""
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
        stride_tensor = torch.tensor(strides, device=device)
        self.head.stride = stride_tensor
        self.pgi_head.stride = stride_tensor
        self.train()
        return strides

    def _extract(self, rgb, ir, depth):
        xr = self.branch_rgb(self.stem_rgb(rgb))
        xi = self.branch_ir(self.stem_ir(ir))
        xd = self.branch_dp(self.stem_dp(depth))

        fused = self.fusion([[xr, xi, xd]])[0]

        # GELAN 主干
        b = self.body
        p3 = b[1](b[0](fused))           # RepNCSPELAN4 after Conv downsample
        p4 = b[3](b[2](p3))             # RepNCSPELAN4 after Conv downsample
        p5 = b[6](b[5](b[4](p4)))       # RepNCSPELAN4 + SPPELAN

        # PGI 辅助分支（仅训练时）
        if self.training:
            self._pgi_feats = self.pgi_branch(fused)

        return [p3, p4, p5]

    def forward(self, rgb, ir, depth):
        feats = self._extract(rgb, ir, depth)
        p3_raw = feats[0]

        # 主检测头
        out = self.head(feats)

        if self.training:
            aux = self._aux_outputs(p3_raw) if self.use_aux else None
            # PGI 辅助检测头
            pgi_out = self.pgi_head([f.clone() for f in self._pgi_feats])
            return out, aux, pgi_out
        return out


def build_multimodal_model(fusion_mode="stage_gate", num_classes=12,
                           model_size="n", use_aux=True):
    """工厂函数：按融合策略构建模型"""
    if fusion_mode == "early":
        return EarlyFusionYOLO(num_classes, model_size, use_aux)
    elif fusion_mode == "stage_gate":
        return StageGateFusionYOLO(num_classes, model_size, use_aux)
    elif fusion_mode in ("stage_gate_v9", "stage_gate_gelan"):
        return StageGateFusionYOLOv9(num_classes, model_size, use_aux)
    else:
        raise ValueError(f"未知融合策略: {fusion_mode}（可选 early / stage_gate / stage_gate_v9）")


def load_pretrained_weights(model: nn.Module, weights_path: str,
                            verbose: bool = True) -> int:
    """
    加载 COCO 预训练权重（ultralytics .pt / YOLOv9 .pt），自动处理通道/结构差异
    匹配规则：
      - key 完全一致且 shape 一致 → 复制
      - key 前缀 'model.' → 按模型类型映射（early: backbone.；stage: body.；v9: body.+pgi）
      - 第一层卷积 5ch（early 融合）→ 用预训练前 3 通道初始化
      - GELAN body：先精确 key 映射，再 shape-based 模糊匹配兜底
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
    loaded_keys = set()

    is_v9 = isinstance(model, StageGateFusionYOLOv9)

    # ---- key 映射 ----
    def map_key(k):
        if isinstance(model, EarlyFusionYOLO):
            return k.replace("model.", "backbone.", 1) if k.startswith("model.") else k
        elif isinstance(model, StageGateFusionYOLO):
            if k.startswith("model."):
                idx = int(k.split(".")[1])
                if 3 <= idx <= 9:
                    return "body." + ".".join([str(idx - 3)] + k.split(".")[2:])
                return None
            return k
        elif is_v9:
            # YOLOv9 body mapping: model.N → body.(N-shift)
            # GELAN backbone layers in YOLOv9 are at indices 0~6 in our body
            # YOLOv9 pretrained keys: model.0.conv.*, model.1.cv_in.*, etc.
            if k.startswith("model."):
                idx_str = k.split(".")[1]
                try:
                    idx = int(idx_str)
                except ValueError:
                    return None
                # Map YOLOv9 backbone indices to our body indices
                # YOLOv9c backbone: model.0-2 (stem), model.3-22 (backbone+head)
                # Our body: 0-6 (GELAN backbone)
                # Strategy: try direct index mapping for body layers
                if 0 <= idx <= 6:
                    return "body." + ".".join([str(idx)] + k.split(".")[2:])
                return None
            return k
        return k

    # ---- Pass 1: exact key mapping ----
    for k, v in sd.items():
        mk = map_key(k)
        if mk is None or mk not in own_sd:
            continue
        ov = own_sd[mk]
        if ov.shape == v.shape:
            own_sd[mk].copy_(v)
            loaded += 1
            loaded_keys.add(mk)
        elif (ov.dim() == 4 and v.dim() == 4
              and ov.shape[1] == 5 and v.shape[1] == 3):
            with torch.no_grad():
                own_sd[mk][:, :3].copy_(v)
                nn.init.kaiming_normal_(own_sd[mk][:, 3:], mode="fan_out", nonlinearity="relu")
            loaded += 1
            loaded_keys.add(mk)

    # ---- Pass 2: shape-based fuzzy matching (for GELAN body) ----
    if is_v9:
        # 收集未加载的模型参数（body 部分）
        unloaded = {k: v for k, v in own_sd.items()
                    if k not in loaded_keys and k.startswith("body.")}
        # 收集未使用的预训练参数
        used_pt_keys = {k for k in sd.keys() if map_key(k) in loaded_keys}
        unused_pt = {k: v for k, v in sd.items()
                     if k not in used_pt_keys and v.dim() >= 2}

        # 按 shape 分组匹配
        from collections import defaultdict
        pt_by_shape = defaultdict(list)
        for pk, pv in unused_pt.items():
            pt_by_shape[tuple(pv.shape)].append(pk)

        for mk, mv in unloaded.items():
            shape_key = tuple(mv.shape)
            candidates = pt_by_shape.get(shape_key, [])
            if candidates:
                pk = candidates.pop(0)
                own_sd[mk].copy_(sd[pk])
                loaded += 1
                loaded_keys.add(mk)
                used_pt_keys.add(pk)
            else:
                skipped += 1
    else:
        skipped = len(sd) - loaded

    if verbose:
        total = len(own_sd)
        print(f"  预训练加载: 成功 {loaded}/{total} 个参数块, 跳过 {skipped} 个不匹配")
    return loaded
