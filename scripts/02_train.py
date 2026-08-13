#!/usr/bin/env python
"""
02_train.py — 两阶段训练（核心训练脚本）

功能：
  - 两阶段训练：Stage1 冻结主干训练新增模块 → Stage2 全网络微调
  - TaskAlignedAssigner 标签分配 + CIoU + DFL + 辅助损失
  - EMA 权重平滑、AMP 混合精度、梯度裁剪
  - 余弦退火学习率 + 线性 Warmup
  - 验证集 mAP 监控 + 早停 + 断点恢复
  - TensorBoard / 控制台日志

用法:
    python scripts/02_train.py --config configs/train.yaml
    python scripts/02_train.py --config configs/train.yaml --resume weights/checkpoints/last.pt
    python scripts/02_train.py --config configs/train.yaml --fusion_mode early --stage1_only
"""

import argparse
import os
import sys
import time
import math
import random
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.utils.io_utils import get_aligned_stems
from src.models.multimodal_yolo import build_multimodal_model, load_pretrained_weights
from src.losses.detection_loss import MultiModalDetectionLoss
from src.data.multimodal_dataset import MultiModalDataset, collate_fn
from src.utils.metrics import evaluate_coco_map


# ============================================================
# EMA
# ============================================================
class ModelEMA:
    """指数移动平均权重（提升推理稳定性）"""

    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = self.decay * (1 - math.exp(-self.steps / self.warmup_steps))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "steps": self.steps}

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd["ema"])
        self.steps = sd["steps"]


# ============================================================
# 学习率调度
# ============================================================
def build_scheduler(optimizer, cfg, total_steps):
    """余弦退火 + 线性 Warmup"""
    warmup_steps = cfg.get("warmup_epochs", 3) * 0  # 按需设置

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 验证
# ============================================================
@torch.no_grad()
def validate(model, val_loader, device, conf_thresh=0.001, iou_thresh=0.5,
             max_det=100, use_soft_nms=True):
    """验证集 mAP 评估"""
    from src.inference.postprocess import parse_model_outputs

    model.eval()
    preds_all, gts_all = [], []

    for batch in val_loader:
        rgb = batch["rgb"].to(device)
        ir = batch["ir"].to(device)
        depth = batch["depth"].to(device)

        y = model(rgb, ir, depth)
        if isinstance(y, tuple):
            y = y[0]
        dets = parse_model_outputs(y, conf_thresh, iou_thresh, max_det, use_soft_nms)

        for i, det in enumerate(dets):
            preds_all.append(det)
            # GT
            meta = batch["meta"][i]
            orig_h, orig_w = batch["orig_shape"][i].tolist()
            gt_list = []
            mask = batch["batch_idx"] == i
            gt_cls = batch["cls"][mask]
            gt_boxes = batch["bboxes"][mask]
            for c, b in zip(gt_cls, gt_boxes):
                # bboxes 是 letterbox 后 640 像素空间，需转回原图像素
                # 但 mAP 评估在同一空间即可（都是 640 空间）
                gt_list.append({"bbox": b.tolist(), "class_id": int(c)})
            gts_all.append(gt_list)

    result = evaluate_coco_map(
        preds_all, gts_all, class_ids=list(range(12)),
        verbose=True)
    return result


# ============================================================
# 训练循环
# ============================================================
def train_one_stage(model, loss_fn, train_loader, val_loader, optimizer,
                    scheduler, ema, device, cfg, stage_name,
                    epochs, start_epoch=0, best_map=0.0, patience=10):
    """训练单个阶段"""
    scaler = torch.amp.GradScaler("cuda") if cfg.get("amp", True) and device.type == "cuda" else None
    grad_clip = cfg.get("grad_clip", 10.0)
    val_interval = cfg.get("val_interval", 1)
    accum = cfg.get("gradient_accum", 1)
    use_aux = cfg.get("use_aux", True)

    patience_counter = 0
    total_steps = len(train_loader) * epochs

    print(f"\n  [{stage_name}] epochs={epochs}, steps/epoch={len(train_loader)}, "
          f"total_steps={total_steps}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = {"total": 0, "cls": 0, "box": 0, "dfl": 0, "aux": 0}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)

            gt = {
                "cls": batch["cls"],
                "bboxes": batch["bboxes"],
                "batch_idx": batch["batch_idx"],
            }
            aux_targets = None
            if use_aux and "ir_contrast" in batch:
                aux_targets = {
                    "ir_contrast": batch["ir_contrast"].to(device),
                    "depth_mask": batch["depth_mask"].to(device),
                    "depth_normal": batch["depth_normal"].to(device),
                }

            # 前向 + 损失
            if scaler:
                with torch.amp.autocast("cuda"):
                    feats, aux_preds = model(rgb, ir, depth)
                    losses = loss_fn(feats, gt, aux_preds, aux_targets)
                loss = losses["total"] / accum
                scaler.scale(loss).backward()
            else:
                feats, aux_preds = model(rgb, ir, depth)
                losses = loss_fn(feats, gt, aux_preds, aux_targets)
                loss = losses["total"] / accum
                loss.backward()

            # 梯度累积
            if (step + 1) % accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
                if ema:
                    ema.update(model)

            # 日志
            for k in epoch_losses:
                ek = "aux" if k == "aux" else k
                src_key = "total_aux" if k == "aux" else k
                if src_key in losses:
                    epoch_losses[k] += losses[src_key].item()

            if (step + 1) % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                mem = torch.cuda.memory_allocated() / 1024**3 if device.type == "cuda" else 0
                print(f"    E{epoch} S{step+1}/{len(train_loader)} "
                      f"loss={losses['total'].item():.4f} "
                      f"cls={losses['cls'].item():.4f} "
                      f"box={losses['box'].item():.4f} "
                      f"dfl={losses['dfl'].item():.4f} "
                      f"lr={lr:.6f} mem={mem:.1f}G")

        # epoch 统计
        dt = time.time() - t0
        n = len(train_loader)
        print(f"  [{stage_name}] E{epoch} avg: "
              f"total={epoch_losses['total']/n:.4f} "
              f"cls={epoch_losses['cls']/n:.4f} "
              f"box={epoch_losses['box']/n:.4f} "
              f"dfl={epoch_losses['dfl']/n:.4f} "
              f"aux={epoch_losses['aux']/n:.4f} "
              f"({dt:.0f}s)")

        # 验证
        eval_model = ema.ema if ema else model
        if (epoch + 1) % val_interval == 0 and val_loader is not None:
            result = validate(eval_model, val_loader, device,
                              conf_thresh=cfg["inference"]["conf_threshold"],
                              iou_thresh=cfg["inference"]["nms_iou_threshold"],
                              max_det=cfg["inference"]["max_det"],
                              use_soft_nms=cfg["inference"]["use_soft_nms"])
            mAP = result["mAP50_95"]
            print(f"    mAP@50-95={mAP:.4f} mAP@50={result['mAP50']:.4f} mAP@75={result['mAP75']:.4f}")

            if mAP > best_map:
                best_map = mAP
                patience_counter = 0
                save_checkpoint(model, ema, optimizer, scheduler, epoch,
                                cfg, stage_name, is_best=True)
                print(f"    ★ 新最佳 mAP={mAP:.4f}")
            else:
                patience_counter += 1
                if cfg.get("early_stopping", True) and patience_counter >= patience:
                    print(f"    早停: {patience} 轮无提升")
                    break

        # 保存 checkpoint
        save_checkpoint(model, ema, optimizer, scheduler, epoch,
                        cfg, stage_name, is_best=False)

    return best_map


def save_checkpoint(model, ema, optimizer, scheduler, epoch, cfg, stage, is_best):
    """保存 checkpoint"""
    ckpt_dir = Path(cfg.get("checkpoint_dir", "weights/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "config": cfg,
    }
    if ema:
        state["ema"] = ema.state_dict()
    if scheduler:
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, str(ckpt_dir / "last.pt"))
    if is_best:
        best_dir = Path(cfg.get("best_model_dir", "weights/best"))
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(best_dir / "best.pt"))


def load_checkpoint(path, model, ema=None, optimizer=None, scheduler=None, device="cpu"):
    """加载 checkpoint"""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("stage", "")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="多模态检测训练")
    parser.add_argument("--config", default="configs/train.yaml", help="配置文件路径")
    parser.add_argument("--resume", default="", help="从 checkpoint 恢复训练")
    parser.add_argument("--fusion_mode", default="", help="覆盖配置中的融合策略")
    parser.add_argument("--stage1_only", action="store_true", help="仅训练第一阶段")
    parser.add_argument("--data_root", default="", help="覆盖配置中的数据路径")
    parser.add_argument("--subset_ratio", type=float, default=1.0,
                        help="训练数据使用比例 (0,1]，如 0.3 表示只用 30% 数据做快速实验")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.fusion_mode:
        cfg["fusion_mode"] = args.fusion_mode
    if args.data_root:
        cfg["data_root"] = args.data_root

    data_root = cfg["data_root"]
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")

    print("\n" + "=" * 60)
    print("  多模态检测训练")
    print("=" * 60)
    print(f"  融合策略: {cfg['fusion_mode']}")
    print(f"  模型规模: {cfg['model_size']}")
    print(f"  设备: {device}")
    print(f"  数据: {data_root}")
    print(f"  图片尺寸: {cfg['img_size']}")

    # ---- 数据集 ----
    train_ds = MultiModalDataset(
        data_root=data_root, split="train", img_size=cfg["img_size"],
        use_aux=cfg.get("use_aux", True),
        modal_dropout_prob=cfg.get("modal_dropout_prob", 0.3),
        is_training=True, cache_processed=True,
        flip_prob=cfg["augment"]["flip_lr"], scale=cfg["augment"]["scale"],
    )
    val_ds = MultiModalDataset(
        data_root=data_root, split="val", img_size=cfg["img_size"],
        use_aux=False, is_training=False, cache_processed=False,
    ) if get_aligned_stems_safe(data_root, "val") else None

    # ---- 数据子集（快速实验用）----
    if 0 < args.subset_ratio < 1.0:
        n_total = len(train_ds)
        n_use = max(1, int(n_total * args.subset_ratio))
        rng = random.Random(42)  # 固定种子，可复现
        train_ds = Subset(train_ds, rng.sample(range(n_total), n_use))
        print(f"  子集: 使用 {n_use}/{n_total} 样本 ({args.subset_ratio:.0%})")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collate_fn,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_fn,
    ) if val_ds else None

    if val_loader is None:
        print("  ⚠ 无 val/ 数据，请先运行 01_preprocess.py 划分验证集")

    # ---- 模型 ----
    model = build_multimodal_model(
        fusion_mode=cfg["fusion_mode"], num_classes=cfg["num_classes"],
        model_size=cfg["model_size"], use_aux=cfg.get("use_aux", True),
    )
    model.to(device)
    model._init_strides()

    # 预训练权重
    if cfg.get("pretrained", True) and not args.resume:
        try:
            from ultralytics import YOLO
            size_map = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt"}
            pt_name = size_map.get(cfg["model_size"], "yolov8n.pt")
            pt_path = cfg.get("pretrained_path", "") or pt_name
            load_pretrained_weights(model, pt_path)
        except Exception as e:
            print(f"  ⚠ 预训练权重加载失败: {e}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  模型参数量: {params:.2f}M")

    # ---- 损失 ----
    loss_cfg = cfg.get("loss", {})
    loss_fn = MultiModalDetectionLoss(
        model, box_w=loss_cfg.get("box", 7.5), cls_w=loss_cfg.get("cls", 0.5),
        dfl_w=loss_cfg.get("dfl", 1.5),
        aux_weights=loss_cfg.get("aux_weights", {}),
        tal_topk=13,
    )

    # ---- EMA ----
    ema = ModelEMA(model) if cfg.get("ema", True) else None

    # ---- 恢复训练 ----
    start_epoch = 0
    best_map = 0.0
    current_stage = ""
    if args.resume:
        start_epoch, current_stage = load_checkpoint(
            args.resume, model, ema, device=device)
        print(f"  恢复训练: epoch={start_epoch}, stage={current_stage}")

    # ---- Stage 1: 冻结主干 ----
    stage1_epochs = cfg.get("stage1_epochs", 15)
    stage2_epochs = cfg.get("stage2_epochs", 30)

    if not args.stage1_only and stage2_epochs > 0:
        # 冻结共享主干参数（仅训练融合层/检测头/辅助头）
        if cfg["fusion_mode"] == "stage_gate":
            for p in model.body.parameters():
                p.requires_grad = False
            for p in model.head.parameters():
                p.requires_grad = True
            for name, p in model.named_parameters():
                if "stem" in name or "branch" in name or "fusion" in name or "aux" in name:
                    p.requires_grad = True
        elif cfg["fusion_mode"] == "early":
            for p in model.backbone.parameters():
                p.requires_grad = False
            for p in model.head.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"\n  Stage 1: 冻结主干，可训练参数 {trainable:.2f}M")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["stage1_lr"], weight_decay=cfg.get("weight_decay", 5e-4))
        scheduler = build_scheduler(optimizer, cfg, len(train_loader) * stage1_epochs)

        best_map = train_one_stage(
            model, loss_fn, train_loader, val_loader, optimizer,
            scheduler, ema, device, cfg, "Stage1",
            stage1_epochs, start_epoch, best_map,
            cfg.get("patience", 10))

        # 解冻
        for p in model.parameters():
            p.requires_grad = True

    # ---- Stage 2: 全网络微调 ----
    if not args.stage1_only:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["stage2_lr"],
            weight_decay=cfg.get("weight_decay", 5e-4))
        scheduler = build_scheduler(optimizer, cfg, len(train_loader) * stage2_epochs)

        best_map = train_one_stage(
            model, loss_fn, train_loader, val_loader, optimizer,
            scheduler, ema, device, cfg, "Stage2",
            stage2_epochs, 0, best_map,
            cfg.get("patience", 10))
    elif args.stage1_only:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["stage1_lr"], weight_decay=cfg.get("weight_decay", 5e-4))
        scheduler = build_scheduler(optimizer, cfg, len(train_loader) * stage1_epochs)
        best_map = train_one_stage(
            model, loss_fn, train_loader, val_loader, optimizer,
            scheduler, ema, device, cfg, "Stage1-only",
            stage1_epochs, start_epoch, best_map,
            cfg.get("patience", 10))

    print(f"\n  ✓ 训练完成！最佳 mAP@50-95={best_map:.4f}")
    print(f"  最佳模型: {cfg.get('best_model_dir', 'weights/best')}/best.pt")
    print("  下一步: python scripts/03_validate.py --config configs/train.yaml")


def get_aligned_stems_safe(data_root, split):
    """安全版 get_aligned_stems（目录不存在时返回空，不抛异常）"""
    try:
        return get_aligned_stems(data_root, split)
    except Exception as e:
        print(f"  ⚠ 获取 {split} 样本失败: {e}")
        return []


if __name__ == "__main__":
    main()
