#!/usr/bin/env python
"""
01_preprocess.py — 数据预处理与探索性分析

功能：
  1. 从 train/ 按 85/15 比例划分 train/val（硬链接，不复制文件）
  2. 数据探索性分析（EDA）：类别分布、框大小分布、深度格式统计
  3. 伪模态辅助特征预计算与缓存（IR 对比图 / Depth 掩码 / Depth 法向）
  4. 生成 processed/ 缓存目录

用法:
    python scripts/01_preprocess.py --data_root ../dataset --val_split 0.15
    python scripts/01_preprocess.py --data_root ../dataset --eda_only
    python scripts/01_preprocess.py --data_root ../dataset --cache_aux
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    MODALITY_DIRS, get_aligned_stems, find_file,
    load_rgb, load_ir, load_depth, load_labels,
)
from src.utils.preprocessing import (
    compute_ir_contrast_map, compute_depth_mask, compute_depth_normal_map,
    validate_depth_range, depth_is_mm,
)


def create_val_split(data_root, val_split=0.15, seed=42):
    """
    从 train/ 划分 val/（使用硬链接，不占用额外磁盘空间）
    """
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if val_dir.exists():
        val_count = len(get_aligned_stems(data_root, "val"))
        if val_count > 0:
            print(f"  val/ 已存在 ({val_count} 样本)，跳过划分")
            return

    stems = get_aligned_stems(data_root, "train")
    n_val = int(len(stems) * val_split)
    rng = random.Random(seed)
    val_stems = set(rng.sample(stems, n_val))

    print(f"  划分: train={len(stems)-n_val}, val={n_val}")

    for modal in ["visible", "infrared", "depth", "labels"]:
        src_dir = train_dir / modal
        dst_dir = val_dir / modal
        dst_dir.mkdir(parents=True, exist_ok=True)
        for stem in val_stems:
            # 查找源文件（兼容多扩展名）
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG"]:
                src = src_dir / f"{stem}{ext}"
                if src.exists():
                    dst = dst_dir / src.name
                    if not dst.exists():
                        try:
                            os.link(str(src), str(dst))  # 硬链接
                        except OSError:
                            os.symlink(str(src), str(dst))  # 回退到符号链接
                    break

    # 从 train/ 中移除已划分到 val/ 的文件（仅移除硬链接，不删原文件）
    for stem in val_stems:
        for modal in ["visible", "infrared", "depth", "labels"]:
            src_dir = train_dir / modal
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG"]:
                src = src_dir / f"{stem}{ext}"
                if src.exists():
                    os.unlink(str(src))
                    break

    print("  ✓ val/ 划分完成")


def run_eda(data_root):
    """探索性数据分析"""
    data_root = Path(data_root)
    print("\n  === 数据探索性分析 ===\n")

    for split in ["train", "val", "test"]:
        stems = get_aligned_stems(data_root, split)
        if len(stems) == 0:
            continue
        print(f"  [{split}] {len(stems)} 样本")

        # 类别分布
        cls_counter = Counter()
        box_sizes = []
        depth_formats = {"16bit_mm": 0, "8bit_norm": 0}
        ir_formats = {"16bit": 0, "8bit": 0}

        sample_size = min(len(stems), 200)  # 采样以加速
        sampled = np.random.choice(stems, sample_size, replace=False)

        for stem in sampled:
            # 标签
            label_path = data_root / split / "labels" / f"{stem}.txt"
            if label_path.exists():
                labels = load_labels(label_path)
                cls_counter.update(labels[:, 0].astype(int).tolist())
                for _, cx, cy, w, h in labels:
                    box_sizes.append((w, h))

            # 深度格式
            try:
                depth = load_depth(find_file(data_root / split / "depth", stem))
                fmt = "16bit_mm" if depth_is_mm(depth) else "8bit_norm"
                depth_formats[fmt] += 1
            except Exception:
                pass

            # IR 格式
            try:
                ir = load_ir(find_file(data_root / split / "infrared", stem))
                ir_formats["16bit" if ir.max() > 255 else "8bit"] += 1
            except Exception:
                pass

        # 打印统计
        if cls_counter:
            print(f"    类别分布 (采样 {sample_size}):")
            for cid in sorted(cls_counter.keys()):
                print(f"      {cid:2d}: {cls_counter[cid]:4d} ({cls_counter[cid]/sum(cls_counter.values())*100:.1f}%)")

        if box_sizes:
            sizes = np.array(box_sizes)
            print(f"    框大小 (归一化):")
            print(f"      w: mean={sizes[:,0].mean():.4f} std={sizes[:,0].std():.4f} "
                  f"min={sizes[:,0].min():.4f} max={sizes[:,0].max():.4f}")
            print(f"      h: mean={sizes[:,1].mean():.4f} std={sizes[:,1].std():.4f} "
                  f"min={sizes[:,1].min():.4f} max={sizes[:,1].max():.4f}")

        print(f"    深度格式: {depth_formats}")
        print(f"    IR 格式: {ir_formats}")
        print()

    # 保存 EDA 报告
    eda_path = data_root / "eda_report.json"
    report = {
        "splits": {},
        "depth_formats": depth_formats,
        "ir_formats": ir_formats,
    }
    for split in ["train", "val", "test"]:
        stems = get_aligned_stems(data_root, split)
        report["splits"][split] = len(stems)
    with open(eda_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  EDA 报告已保存: {eda_path}")


def cache_aux_features(data_root, split="train"):
    """预计算并缓存伪模态辅助特征"""
    data_root = Path(data_root)
    stems = get_aligned_stems(data_root, split)
    cache_dir = data_root / "processed" / split
    (cache_dir / "ir_contrast").mkdir(parents=True, exist_ok=True)
    (cache_dir / "depth_mask").mkdir(parents=True, exist_ok=True)
    (cache_dir / "depth_normal").mkdir(parents=True, exist_ok=True)

    print(f"\n  缓存辅助特征 [{split}]: {len(stems)} 样本")
    ir_dir = data_root / split / "infrared"
    dp_dir = data_root / split / "depth"

    for i, stem in enumerate(stems):
        ir = load_ir(find_file(ir_dir, stem))
        depth = load_depth(find_file(dp_dir, stem))

        ir_c = compute_ir_contrast_map(ir)
        dm = compute_depth_mask(depth)
        dn = compute_depth_normal_map(depth)

        np.save(str(cache_dir / "ir_contrast" / f"{stem}.npy"), ir_c)
        np.save(str(cache_dir / "depth_mask" / f"{stem}.npy"), dm)
        np.save(str(cache_dir / "depth_normal" / f"{stem}.npy"), dn)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(stems)}")

    print(f"  ✓ 辅助特征缓存完成: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--data_root", default="../dataset",
                        help="数据集根目录")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="验证集划分比例")
    parser.add_argument("--eda_only", action="store_true",
                        help="仅执行 EDA，不划分数据")
    parser.add_argument("--cache_aux", action="store_true",
                        help="预计算并缓存伪模态辅助特征")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / args.data_root).resolve()

    print("\n" + "=" * 60)
    print("  数据预处理 — 多模态目标检测")
    print("=" * 60 + "\n")

    if not args.eda_only:
        create_val_split(data_root, args.val_split, args.seed)

    run_eda(data_root)

    if args.cache_aux:
        for split in ["train", "val"]:
            if get_aligned_stems(data_root, split):
                cache_aux_features(data_root, split)

    print("\n  ✓ 预处理完成")
    print("  下一步: python scripts/02_train.py --config configs/train.yaml")


if __name__ == "__main__":
    main()
