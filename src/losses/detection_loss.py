"""
统一检测损失（多模态版，自包含实现）

- 主损失：经典 YOLOv8 三分支
    分类    : BCEWithLogits（正负样本统一，按 TAL 分配的目标得分加权）
    回归    : CIoU（像素空间，fg 锚点）
    DFL     : 分布焦点损失（特征空间，自包含实现，不依赖 ultralytics 内部版本）
  标签分配复用 ultralytics TaskAlignedAssigner（8.2+ 接口，防御式适配 4/5 返回值）

- 辅助损失：伪模态重建监督（IR 温度对比 / Depth 掩码 / Depth 法向）
    迫使浅层特征学习物理语义，提升小目标与远距离目标表现

坐标约定（与数据集/模型统一）：
    - 模型训练输出 feats: 3 尺度 [B, no, H, W] 原始张量
    - GT boxes: letterbox 后 640 像素空间 xyxy（batch 级展平 + batch_idx）
    - 所有匹配与 IoU 计算在像素空间；DFL 在特征空间（除以 stride）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors, dist2bbox, bbox2dist

try:
    from ultralytics.utils.tal import TaskAlignedAssigner
except ImportError:  # 极老版本兜底
    TaskAlignedAssigner = None


class _TALAssigner(nn.Module):
    """TaskAlignedAssigner 防御式封装（适配 ultralytics 8.1~8.4 差异）"""

    def __init__(self, topk=13, num_classes=80, alpha=0.5, beta=6.0, stride=None):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self._assigner = None
        self._has_stride = False

    def _build(self, device):
        if self._assigner is not None:
            return
        kwargs = dict(topk=self.topk, num_classes=self.num_classes,
                      alpha=self.alpha, beta=self.beta)
        # 8.2+ 支持 stride 参数；8.1 不支持 → 降级
        try:
            self._assigner = TaskAlignedAssigner(stride=self.stride, **kwargs)
            self._has_stride = True
        except TypeError:
            self._assigner = TaskAlignedAssigner(**kwargs)
            self._has_stride = False
        self._assigner.to(device)

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points,
                gt_labels, gt_bboxes, batch_size):
        """
        Args:
            pred_scores: [B, N, nc]（已 sigmoid）
            pred_bboxes: [B, N, 4] 像素空间 xyxy
            anchor_points: [N, 2] 像素空间
            gt_labels: [G] 类别（扁平，跨 batch）
            gt_bboxes: [G, 5] (batch_idx, x1, y1, x2, y2) 像素空间
            batch_size: int
        Returns:
            target_bboxes: [B, N, 4] 像素空间
            target_scores: [B, N, nc]
            fg_mask: [B, N] bool
        """
        device = pred_scores.device
        self._build(device)
        self._assigner.num_classes = self.num_classes

        bs = batch_size
        N = pred_scores.shape[1]

        if len(gt_labels) == 0:
            target_bboxes = torch.zeros_like(pred_bboxes)
            target_scores = torch.zeros_like(pred_scores)
            fg_mask = torch.zeros(bs, N, dtype=torch.bool, device=device)
            return target_bboxes, target_scores, fg_mask

        # 扁平 GT → batch 格式 [bs, n_max, ...]（与 ultralytics preprocess 一致）
        batch_idx = gt_bboxes[:, 0].long()      # [G]
        boxes = gt_bboxes[:, 1:5]               # [G, 4] xyxy 像素

        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        n_max = int(counts.max().item())

        gt_labels_batch = torch.zeros(bs, n_max, 1, device=device, dtype=torch.long)
        gt_bboxes_batch = torch.zeros(bs, n_max, 4, device=device, dtype=pred_bboxes.dtype)

        # 用 scatter 索引填充（与 ultralytics preprocess 完全一致）
        offsets = torch.zeros(bs + 1, dtype=torch.long, device=device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(gt_labels), device=device) - offsets[batch_idx]

        gt_labels_batch[batch_idx, within_idx, 0] = gt_labels
        gt_bboxes_batch[batch_idx, within_idx] = boxes

        # 有效 GT 掩码（零框为 padding）
        mask_gt = gt_bboxes_batch.sum(2, keepdim=True).gt_(0.0)  # [bs, n_max, 1]

        out = self._assigner(
            pred_scores, pred_bboxes, anchor_points,
            gt_labels_batch, gt_bboxes_batch, mask_gt)
        target_labels, target_bboxes, target_scores, fg_mask = out[:4]
        return target_bboxes, target_scores, fg_mask


def _df_loss(pred_dist, target):
    """
    分布焦点损失（自包含，与 ultralytics 一致）
    Args:
        pred_dist: [F*4, reg_max]（fg 锚点 DFL 分布，softmax 前）
        target:    [F, 4] 连续 ltrb 距离（特征空间）
    Returns:
        [F, 1] 各锚点 DFL 损失
    """
    tl = target.long()          # [F, 4] 下取整
    tr = tl + 1
    wl = tr - target            # 下取整权重
    wr = 1 - wl
    # cross_entropy: input [F*4, reg_max], target [F*4] → output [F*4]
    ce_lo = F.cross_entropy(pred_dist, tl.reshape(-1), reduction="none")
    ce_hi = F.cross_entropy(pred_dist, tr.reshape(-1), reduction="none")
    ce_lo = ce_lo.view(tl.shape)  # [F, 4]
    ce_hi = ce_hi.view(tr.shape)
    return (ce_lo * wl + ce_hi * wr).mean(-1, keepdim=True)  # [F, 1]


class MultiModalDetectionLoss(nn.Module):
    """
    Args:
        model: 多模态检测模型（需含 .stride / .head(含 .reg_max, .nc)）
        box_w: 回归损失权重
        cls_w: 分类损失权重
        dfl_w: DFL 损失权重
        aux_weights: dict(ir_contrast=, depth_mask=, depth_normal=, total=)
        tal_topk: 标签分配 topk
    """

    def __init__(self, model, box_w=7.5, cls_w=0.5, dfl_w=1.5,
                 aux_weights=None, tal_topk=13, tal_alpha=0.5, tal_beta=6.0,
                 class_weights=None):
        super().__init__()
        device = next(model.parameters()).device
        head = model.head

        self.stride = head.stride.detach().cpu().tolist()
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.no = head.no
        self.device = device

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.box_w = box_w
        self.cls_w = cls_w
        self.dfl_w = dfl_w

        # 类别重加权（缓解类别不平衡）: [nc] 或 None
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float, device=device))
        else:
            self.class_weights = None

        self.assigner = _TALAssigner(
            topk=tal_topk, num_classes=self.nc,
            alpha=tal_alpha, beta=tal_beta, stride=self.stride)

        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

        # 辅助损失权重
        aw = aux_weights or {}
        self.aux_w = {
            "ir": aw.get("ir_contrast", 0.3),
            "dm": aw.get("depth_mask", 0.2),
            "dn": aw.get("depth_normal", 0.15),
            "total": aw.get("total_aux", 0.15),
        }

    def _dfl_decode(self, pred_distri):
        """DFL 解码：分布 → 期望值 [B, N, 4]（特征空间 ltrb）"""
        b, n, c = pred_distri.shape
        pred_distri = pred_distri.view(b, n, 4, self.reg_max).softmax(-1)
        return pred_distri @ self.proj

    def _decode(self, feats):
        """
        feats: 3 尺度 [B, no, H, W] → 像素空间预测
        Returns:
            pred_distri: [B, N, 4*reg_max]（特征空间，DFL 用）
            pred_scores: [B, N, nc]
            pred_bboxes: [B, N, 4] 像素空间 xyxy
            anchor_feat: [N, 2] 特征空间锚点
            anchor_pix:  [N, 2] 像素空间锚点
            stride_tensor: [N]
        """
        pred_distri, pred_scores = torch.cat(
            [xi.view(xi.shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchor_feat, stride_tensor = make_anchors(feats, self.stride, 0.5)
        anchor_pix = anchor_feat * stride_tensor  # [N,2] * [N,1] → [N,2] 像素空间
        # 特征空间解码 → 像素空间（dim=-1：distance [B,N,4] chunk → lt,rb 各 [B,N,2]）
        pred_bboxes = dist2bbox(self._dfl_decode(pred_distri), anchor_feat,
                                xywh=False, dim=-1) * stride_tensor
        return (pred_distri, pred_scores, pred_bboxes,
                anchor_feat, anchor_pix, stride_tensor)

    def forward(self, feats, batch, aux_preds=None, aux_targets=None):
        """
        Args:
            feats: 训练模式模型输出（3 尺度原始张量列表）
            batch: dict(cls=[G], bboxes=[G,4] xyxy 像素, batch_idx=[G])
            aux_preds: 辅助头预测 dict（可选）
            aux_targets: 辅助目标 dict（可选）
        Returns:
            losses: dict(cls, box, dfl, total_aux, total, ...)
        """
        (pred_distri, pred_scores, pred_bboxes,
         anchor_feat, anchor_pix, stride_tensor) = self._decode(feats)

        gt_labels = batch["cls"].to(self.device).long()
        gt_boxes = batch["bboxes"].to(self.device).float()
        batch_idx = batch["batch_idx"].to(self.device).long()
        # [G, 5] (batch_idx, x1, y1, x2, y2) —— TAL 需要 batch 索引
        gt_bboxes = torch.cat([batch_idx.unsqueeze(1), gt_boxes], 1)

        # ---- 标签分配（像素空间匹配）----
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores=pred_scores.detach().sigmoid(),
            pred_bboxes=pred_bboxes.detach(),
            anchor_points=anchor_pix,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            batch_size=pred_scores.shape[0],
        )
        target_scores_sum = max(target_scores.sum(), 1.0)

        # ---- 回归损失：CIoU（像素空间，fg 锚点）----
        pred_boxes_fg = pred_bboxes[fg_mask]
        tgt_boxes_fg = target_bboxes[fg_mask]
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if pred_boxes_fg.numel() > 0:
            iou = bbox_iou(pred_boxes_fg, tgt_boxes_fg, xywh=False, CIoU=True)
            box_loss = ((1.0 - iou) * weight).sum() / target_scores_sum
        else:
            box_loss = pred_bboxes.sum() * 0.0

        # ---- DFL 损失（特征空间）----
        fg_idx = fg_mask.nonzero(as_tuple=False)  # [F, 2] (b, n)
        if len(fg_idx) > 0:
            anc_feat_fg = anchor_feat[fg_idx[:, 1]]                    # [F, 2]
            # stride_tensor 是 [N,1]，索引后 [F,1]，直接除即可广播
            tgt_feat_fg = tgt_boxes_fg / stride_tensor[fg_idx[:, 1]]   # [F, 4]
            target_ltrb = bbox2dist(anc_feat_fg, tgt_feat_fg, self.reg_max - 1)  # [F, 4]
            # ultralytics 约定：pred_distri[fg_mask].view(-1, reg_max) → [F*4, reg_max]
            pred_dfl_fg = pred_distri[fg_mask].view(-1, self.reg_max)
            dfl_loss = (_df_loss(pred_dfl_fg, target_ltrb) * weight).sum() / target_scores_sum
        else:
            dfl_loss = pred_distri.sum() * 0.0

        # ---- 分类损失（BCE，全部锚点）----
        if self.class_weights is not None:
            # 类别重加权：正样本锚点按其目标类别的权重放大梯度
            pos_cls = target_scores.argmax(-1)               # [B, N] 目标类别
            pos_mask = target_scores.sum(-1) > 0             # [B, N] 正锚点
            anchor_w = torch.ones_like(target_scores.sum(-1))
            if pos_mask.any():
                anchor_w[pos_mask] = self.class_weights[pos_cls[pos_mask]]
            bce_all = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
            cls_loss = (bce_all * anchor_w.unsqueeze(-1)).sum() / target_scores_sum
        else:
            cls_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() \
                / target_scores_sum

        # ---- 辅助损失 ----
        aux_loss_total = torch.tensor(0.0, device=self.device)
        aux_items = {}
        if aux_preds is not None and aux_targets is not None:
            aux_loss_total, aux_items = self._aux_loss(aux_preds, aux_targets)

        total = (self.box_w * box_loss + self.cls_w * cls_loss
                 + self.dfl_w * dfl_loss + aux_loss_total)

        losses = {
            "cls": cls_loss,
            "box": box_loss,
            "dfl": dfl_loss,
            "total_aux": aux_loss_total,
            "total": total,
        }
        losses.update(aux_items)
        return losses

    def _aux_loss(self, aux_preds, aux_targets):
        """伪模态重建辅助损失（预测为 P3 1/8 分辨率，目标插值对齐）"""
        ir_pred = aux_preds["ir_contrast_pred"]
        dm_pred = aux_preds["depth_mask_pred"]
        dn_pred = aux_preds["depth_normal_pred"]

        sz = ir_pred.shape[-2:]
        ir_t = F.interpolate(aux_targets["ir_contrast"].to(self.device), size=sz,
                             mode="bilinear", align_corners=False)
        dm_t = F.interpolate(aux_targets["depth_mask"].to(self.device), size=sz,
                             mode="nearest")
        dn_t = F.interpolate(aux_targets["depth_normal"].to(self.device), size=sz,
                             mode="bilinear", align_corners=False)

        l_ir = F.smooth_l1_loss(torch.sigmoid(ir_pred), ir_t)
        l_dm = F.binary_cross_entropy_with_logits(dm_pred, dm_t)
        # 法向余弦相似度（-1~1 → 0~2，取平均余弦相似度损失）
        cos_sim = F.cosine_similarity(dn_pred, dn_t, dim=1)
        l_dn = (1.0 - cos_sim).mean()

        total = (self.aux_w["ir"] * l_ir + self.aux_w["dm"] * l_dm
                 + self.aux_w["dn"] * l_dn) * self.aux_w["total"]
        return total, {"aux_ir": l_ir, "aux_dm": l_dm, "aux_dn": l_dn}
