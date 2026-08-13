#!/usr/bin/env python
"""
03_validate.py — 验证集评估与可视化

功能：
  - 加载训练好的模型，在 val/ 上计算 mAP@50-95
  - 按类别输出 AP，定位薄弱类别
  - 可视化检测结果（三模态 + 检测框叠加）
  - 支持 pycocotools 口径与自研口径对比

用法:
    python scripts/03_validate.py --config configs/train.yaml --weights weights/best/best.pt
    python scripts/03_validate.py --config configs/train.yaml --weights weights/best/best.pt --visualize
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import cv2

from src.models.multimodal_yolo import build_multimodal_model
from src.data.multimodal_dataset import MultiModalDataset, collate_fn
from src.inference.postprocess import parse_model_outputs
from src.utils.metrics import evaluate_coco_map, try_pycocotools
from src.utils.io_utils import get_aligned_stems, load_labels, yolo_to_xyxy


@torch.no_grad()
def validate(model, val_loader, device, cfg, class_names=None):
    """完整验证"""
    model.eval()
    preds_all, gts_all = [], []
    inf_cfg = cfg.get("inference", {})

    for batch in val_loader:
        rgb = batch["rgb"].to(device)
        ir = batch["ir"].to(device)
        depth = batch["depth"].to(device)

        y = model(rgb, ir, depth)
        if isinstance(y, tuple):
            y = y[0]
        dets = parse_model_outputs(
            y,
            conf_thresh=inf_cfg.get("conf_threshold", 0.001),
            iou_thresh=inf_cfg.get("nms_iou_threshold", 0.5),
            max_det=inf_cfg.get("max_det", 100),
            use_soft_nms=inf_cfg.get("use_soft_nms", True),
        )

        for i, det in enumerate(dets):
            preds_all.append(det)
            mask = batch["batch_idx"] == i
            gt_cls = batch["cls"][mask]
            gt_boxes = batch["bboxes"][mask]
            gt_list = []
            for c, b in zip(gt_cls, gt_boxes):
                gt_list.append({"bbox": b.tolist(), "class_id": int(c)})
            gts_all.append(gt_list)

    # 自研 mAP
    print("\n  === 自研 mAP ===")
    result = evaluate_coco_map(
        preds_all, gts_all, class_ids=list(range(cfg["num_classes"])),
        verbose=True)

    # pycocotools mAP（与官方口径一致）
    print("\n  === pycocotools mAP（官方口径）===")
    coco_result = try_pycocotools(preds_all, gts_all, verbose=True)
    if coco_result:
        result.update({"pycoco_" + k: v for k, v in coco_result.items()})

    # 按类别 AP
    if class_names:
        print("\n  === 按类别 AP@50 ===")
        per_class = result.get("per_class_ap", {})
        for cid in sorted(per_class.keys()):
            ap50 = per_class[cid].get(0.5, 0.0)
            name = class_names[cid] if cid < len(class_names) else f"cls_{cid}"
            print(f"    {cid:2d} {name:15s}: {ap50:.4f}")

    return result, preds_all, gts_all


def visualize_results(model, val_ds, device, cfg, output_dir, max_imgs=20):
    """可视化检测结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    inf_cfg = cfg.get("inference", {})
    class_names = cfg.get("class_names", [])

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]

    n = min(len(val_ds), max_imgs)
    for idx in range(n):
        item = val_ds[idx]
        rgb = item["rgb"].unsqueeze(0).to(device)
        ir = item["ir"].unsqueeze(0).to(device)
        depth = item["depth"].unsqueeze(0).to(device)

        y = model(rgb, ir, depth)
        if isinstance(y, tuple):
            y = y[0]
        dets = parse_model_outputs(
            y,
            conf_thresh=inf_cfg.get("conf_threshold", 0.25),
            iou_thresh=inf_cfg.get("nms_iou_threshold", 0.5),
            max_det=inf_cfg.get("max_det", 100),
            use_soft_nms=inf_cfg.get("use_soft_nms", True),
        )[0]

        # 可视化
        img = (item["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cid = d["class_id"]
            score = d["score"]
            color = colors[cid % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            name = class_names[cid] if cid < len(class_names) else str(cid)
            cv2.putText(img, f"{name} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # GT 框（绿色虚线）
        for c, b in zip(item["cls"], item["bboxes"]):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)

        stem = item["meta"]["stem"]
        cv2.imwrite(str(output_dir / f"{stem}_det.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"  ✓ 可视化已保存: {output_dir} ({n} 张)")


def main():
    parser = argparse.ArgumentParser(description="验证集评估")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--visualize", action="store_true", help="保存可视化结果")
    parser.add_argument("--vis_dir", default="outputs/vis_val", help="可视化输出目录")
    parser.add_argument("--data_root", default="", help="覆盖配置中的数据路径")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root

    data_root = cfg["data_root"]
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  验证集评估")
    print("=" * 60)
    print(f"  权重: {args.weights}")
    print(f"  设备: {device}")

    # 加载模型（use_aux 需与训练时一致，否则 state_dict 不匹配）
    model = build_multimodal_model(
        fusion_mode=cfg["fusion_mode"], num_classes=cfg["num_classes"],
        model_size=cfg["model_size"], use_aux=cfg.get("use_aux", True),
    )
    model.to(device)
    model._init_strides()

    ckpt = torch.load(str(args.weights), map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print("  ✓ 模型加载成功")

    # 数据集
    val_ds = MultiModalDataset(
        data_root=data_root, split="val", img_size=cfg["img_size"],
        use_aux=False, is_training=False, cache_processed=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_fn,
    )

    # 评估
    result, preds, gts = validate(
        model, val_loader, device, cfg,
        class_names=cfg.get("class_names"))

    if args.visualize:
        visualize_results(model, val_ds, device, cfg, args.vis_dir)

    print(f"\n  ✓ 评估完成")
    print(f"  mAP@50-95: {result['mAP50_95']:.4f}")
    print(f"  mAP@50:    {result['mAP50']:.4f}")
    print(f"  mAP@75:    {result['mAP75']:.4f}")
    if "pycoco_mAP50_95" in result:
        print(f"  [pycoco] mAP@50-95: {result['pycoco_mAP50_95']:.4f}")

    print("\n  下一步: python scripts/04_infer.py --config configs/train.yaml --weights weights/best/best.pt")


if __name__ == "__main__":
    main()
