#!/usr/bin/env python
"""
04_infer.py — 测试集推理

功能：
  - 在 test/ 上运行推理，生成检测结果
  - 支持多尺度 TTA（缩放 + 水平翻转融合）
  - 模态缺失鲁棒性测试（单模态/双模态推理对比）
  - 结果可视化

用法:
    python scripts/04_infer.py --config configs/train.yaml --weights weights/best/best.pt
    python scripts/04_infer.py --config configs/train.yaml --weights weights/best/best.pt --tta
    python scripts/04_infer.py --config configs/train.yaml --weights weights/best/best.pt --robustness_test
"""

import argparse
import os
import sys
import time
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
from src.inference.postprocess import parse_model_outputs, tta_predict
from src.utils.io_utils import (
    get_aligned_stems, find_file, load_rgb, load_ir, load_depth,
    letterbox_modal,
)
from src.utils.preprocessing import normalize_depth


@torch.no_grad()
def infer_standard(model, test_loader, device, cfg):
    """标准推理（无 TTA）"""
    model.eval()
    inf_cfg = cfg.get("inference", {})
    all_dets = {}
    orig_shapes = {}

    t0 = time.time()
    for batch in test_loader:
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
            stem = batch["meta"][i]["stem"]
            all_dets[stem] = det
            orig_shapes[stem] = batch["orig_shape"][i].tolist()

    dt = time.time() - t0
    n = len(all_dets)
    print(f"  标准推理: {n} 张, {dt:.1f}s ({n/dt:.1f} img/s)")
    return all_dets, orig_shapes


@torch.no_grad()
def infer_tta(model, data_root, split, device, cfg):
    """TTA 推理（多尺度 + 水平翻转）"""
    model.eval()
    inf_cfg = cfg.get("inference", {})
    tta_scales = inf_cfg.get("tta_scales", [640, 800, 960])
    tta_flip = inf_cfg.get("tta_flip", True)

    stems = get_aligned_stems(data_root, split)
    rgb_dir = Path(data_root) / split / "visible"
    ir_dir = Path(data_root) / split / "infrared"
    dp_dir = Path(data_root) / split / "depth"

    all_dets = {}
    orig_shapes = {}
    t0 = time.time()

    for i, stem in enumerate(stems):
        rgb = load_rgb(find_file(rgb_dir, stem))
        ir = load_ir(find_file(ir_dir, stem))
        depth = load_depth(find_file(dp_dir, stem))
        orig_shapes[stem] = rgb.shape[:2]

        dets = tta_predict(
            model, rgb, ir, depth, device,
            scales=tta_scales, flip=tta_flip,
            conf_thresh=inf_cfg.get("conf_threshold", 0.001),
            iou_thresh=inf_cfg.get("nms_iou_threshold", 0.5),
            max_det=inf_cfg.get("max_det", 100),
            use_soft_nms=inf_cfg.get("use_soft_nms", True),
        )
        all_dets[stem] = dets

        if (i + 1) % 100 == 0:
            print(f"    TTA: {i+1}/{len(stems)}")

    dt = time.time() - t0
    n = len(all_dets)
    print(f"  TTA 推理: {n} 张, {dt:.1f}s ({n/dt:.1f} img/s)")
    return all_dets, orig_shapes


@torch.no_grad()
def robustness_test(model, data_root, split, device, cfg, max_samples=50):
    """模态缺失鲁棒性测试"""
    model.eval()
    inf_cfg = cfg.get("inference", {})
    stems = get_aligned_stems(data_root, split)[:max_samples]

    rgb_dir = Path(data_root) / split / "visible"
    ir_dir = Path(data_root) / split / "infrared"
    dp_dir = Path(data_root) / split / "depth"

    configs = {
        "all_modalities": {"rgb": True, "ir": True, "depth": True},
        "no_rgb": {"rgb": False, "ir": True, "depth": True},
        "no_ir": {"rgb": True, "ir": False, "depth": True},
        "no_depth": {"rgb": True, "ir": True, "depth": False},
        "rgb_only": {"rgb": True, "ir": False, "depth": False},
    }

    results = {}
    for config_name, modal_flags in configs.items():
        det_counts = []
        for stem in stems:
            rgb = load_rgb(find_file(rgb_dir, stem))
            ir = load_ir(find_file(ir_dir, stem))
            depth = load_depth(find_file(dp_dir, stem))

            # letterbox
            lb_rgb, r, (dw, dh) = letterbox_modal(rgb, (640, 640), stride=32)
            lb_ir, _, _ = letterbox_modal(ir, (640, 640), color=0, stride=32)
            lb_dp, _, _ = letterbox_modal(depth, (640, 640), color=0, stride=32)

            # 模态替换
            if not modal_flags["rgb"]:
                lb_rgb[:] = 114
            if not modal_flags["ir"]:
                lb_ir[:] = 0
            if not modal_flags["depth"]:
                lb_dp[:] = 0

            rgb_t = torch.from_numpy(lb_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            ir_t = torch.from_numpy(lb_ir.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            dp_t = torch.from_numpy(normalize_depth(lb_dp)).unsqueeze(0).unsqueeze(0).to(device)

            y = model(rgb_t, ir_t, dp_t)
            if isinstance(y, tuple):
                y = y[0]
            dets = parse_model_outputs(
                y,
                conf_thresh=0.01,  # 较高阈值用于统计
                iou_thresh=inf_cfg.get("nms_iou_threshold", 0.5),
                max_det=100,
                use_soft_nms=True,
            )[0]
            det_counts.append(len(dets))

        avg_dets = np.mean(det_counts)
        results[config_name] = avg_dets
        print(f"    {config_name:20s}: avg {avg_dets:.1f} dets/img")

    return results


def main():
    parser = argparse.ArgumentParser(description="测试集推理")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--tta", action="store_true", help="启用 TTA")
    parser.add_argument("--robustness_test", action="store_true", help="模态缺失鲁棒性测试")
    parser.add_argument("--data_root", default="", help="覆盖配置中的数据路径")
    parser.add_argument("--split", default="test", help="推理数据集划分")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root

    data_root = cfg["data_root"]
    if not Path(data_root).is_absolute():
        data_root = str(PROJECT_ROOT / data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  测试集推理")
    print("=" * 60)
    print(f"  权重: {args.weights}")
    print(f"  TTA: {'ON' if args.tta else 'OFF'}")
    print(f"  设备: {device}")

    # 加载模型
    model = build_multimodal_model(
        fusion_mode=cfg["fusion_mode"], num_classes=cfg["num_classes"],
        model_size=cfg["model_size"], use_aux=False,
    )
    model.to(device)
    model._init_strides()

    ckpt = torch.load(str(args.weights), map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print("  ✓ 模型加载成功")

    # 鲁棒性测试
    if args.robustness_test:
        print("\n  === 模态缺失鲁棒性测试 ===")
        robustness_test(model, data_root, args.split, device, cfg)

    # 推理
    if args.tta:
        all_dets, orig_shapes = infer_tta(model, data_root, args.split, device, cfg)
    else:
        test_ds = MultiModalDataset(
            data_root=data_root, split=args.split, img_size=cfg["img_size"],
            use_aux=False, is_training=False, cache_processed=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], collate_fn=collate_fn,
        )
        all_dets, orig_shapes = infer_standard(model, test_loader, device, cfg)

    # 保存中间结果
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"dets": all_dets, "shapes": orig_shapes},
               str(output_dir / "test_predictions.pt"))
    print(f"\n  ✓ 检测结果已保存: {output_dir / 'test_predictions.pt'}")
    print(f"  总检测数: {sum(len(d) for d in all_dets.values())}")
    print("\n  下一步: python scripts/05_export_submission.py --predictions outputs/predictions/test_predictions.pt")


if __name__ == "__main__":
    main()
