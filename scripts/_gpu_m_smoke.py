#!/usr/bin/env python
"""GPU yolov8m + 类别平衡 冒烟测试：模型构建/预训练加载/paste 增强/loss 重加权/前向反向"""
import sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  torch: {torch.__version__}")

    # 1) 构建 yolov8m stage_gate 模型
    from src.models.multimodal_yolo import build_multimodal_model, load_pretrained_weights
    model = build_multimodal_model(fusion_mode="stage_gate", num_classes=12,
                                   model_size="m", use_aux=True)
    model.to(device)
    model._init_strides()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {params:.2f}M  strides: {model.head.stride.tolist()}")

    # 2) 加载预训练 yolov8m
    n_match = load_pretrained_weights(model, "weights/yolov8m.pt")
    print(f"预训练加载: {n_match} 块匹配")

    # 3) 数据集 + paste 增强（只取前 50 个样本验证）
    from src.data.multimodal_dataset import MultiModalDataset, collate_fn
    ds = MultiModalDataset(
        data_root="../dataset", split="train", img_size=640,
        use_aux=True, modal_dropout_prob=0.3, is_training=True,
        cache_processed=True, flip_prob=0.5, scale=0.5,
        paste_aug=True, paste_classes=[1, 7, 9, 10, 11],
        paste_prob=1.0, paste_max_objs=2,
    )
    print(f"训练样本数: {len(ds)}")
    t0 = time.time()
    sample = ds[0]
    dt = time.time() - t0
    print(f"取样本耗时: {dt:.1f}s  (首次含对象库构建)")
    print(f"  cls: {sample['cls'].tolist()}")
    print(f"  bboxes 数: {sample['bboxes'].shape}")
    # 检查是否注入了 paste 对象（类别 1/7/9/10/11）
    injected = [int(c) for c in sample['cls'].tolist() if int(c) in (1, 7, 9, 10, 11)]
    print(f"  paste 注入类别: {injected if injected else '本次未注入(随机)'}")

    # 4) loss 类别重加权
    from src.utils.io_utils import compute_class_weights, count_class_freq
    counts = count_class_freq("../dataset", "train")
    cw = compute_class_weights(counts)
    print(f"类别频次: {counts.tolist()}")
    print(f"重加权: {[round(float(x),2) for x in cw.tolist()]}")

    from src.losses.detection_loss import MultiModalDetectionLoss
    loss_cfg = {"box": 7.5, "cls": 0.5, "dfl": 1.5,
                "aux_weights": {"ir_contrast": 0.3, "depth_mask": 0.2, "depth_normal": 0.15, "total_aux": 0.15}}
    loss_fn = MultiModalDetectionLoss(model, box_w=7.5, cls_w=0.5, dfl_w=1.5,
                                      aux_weights=loss_cfg["aux_weights"],
                                      tal_topk=13, class_weights=cw)
    print(f"loss.class_weights: {loss_fn.class_weights.tolist()}")

    # 5) 前向反向（1 样本重复 4 次模拟 batch 4）
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    items = [ds[i % len(ds)] for i in range(4)]
    batch = collate_fn(items)
    rgb = batch["rgb"].to(device); ir = batch["ir"].to(device); depth = batch["depth"].to(device)
    t0 = time.time()
    feats, aux_preds = model(rgb, ir, depth)
    losses = loss_fn(feats, batch, aux_preds=aux_preds,
                     aux_targets={k: v.to(device) for k, v in batch.items() if k.startswith(("ir_contrast","depth_mask","depth_normal"))})
    opt.zero_grad()
    losses["total"].backward()
    opt.step()
    dt = time.time() - t0
    print(f"前向反向: {dt:.2f}s  total={losses['total'].item():.3f}  cls={losses['cls'].item():.3f}")
    print("SMOKE OK")

if __name__ == "__main__":
    main()
