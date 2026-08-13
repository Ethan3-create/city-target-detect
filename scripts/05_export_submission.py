#!/usr/bin/env python
"""
05_export_submission.py — 提交文件导出

功能：
  - 将推理结果导出为竞赛要求的 TXT 格式
  - 每张图一个 TXT 文件，行格式: class_id cx cy w h confidence
  - 坐标按原图宽高归一化（非正方形图正确）
  - 单图最多 100 个检测框
  - 打包为 ZIP 提交文件

用法:
    python scripts/05_export_submission.py --predictions outputs/predictions/test_predictions.pt
    python scripts/05_export_submission.py --predictions outputs/predictions/test_predictions.pt --output submission.zip
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.postprocess import yolo_predictions_to_txt


def export_submission(predictions_path, output_zip="submission.zip", max_det=100):
    """
    导出提交文件

    Args:
        predictions_path: 推理结果文件路径 (.pt)
        output_zip: 输出 ZIP 文件路径
        max_det: 单图最大检测框数
    """
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        print(f"  ✗ 推理结果文件不存在: {predictions_path}")
        return False

    # 加载推理结果
    data = torch.load(str(predictions_path), map_location="cpu", weights_only=False)
    all_dets = data["dets"]
    orig_shapes = data["shapes"]

    print(f"\n  加载推理结果: {len(all_dets)} 张图")

    # 转换为 YOLO TXT 格式
    txt_results = yolo_predictions_to_txt(all_dets, orig_shapes)

    # 创建临时目录
    tmp_dir = Path("outputs/submission_txt")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for stem, lines in txt_results.items():
        # 截断到 max_det
        lines = lines[:max_det]
        total_boxes += len(lines)
        with open(tmp_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))

    print(f"  总检测框: {total_boxes}")
    print(f"  平均每图: {total_boxes / len(txt_results):.1f}")

    # 统计类别分布
    cls_counter = {}
    for lines in txt_results.values():
        for line in lines:
            cid = int(line.split()[0])
            cls_counter[cid] = cls_counter.get(cid, 0) + 1

    print(f"\n  类别分布:")
    for cid in sorted(cls_counter.keys()):
        count = cls_counter[cid]
        print(f"    {cid:2d}: {count:5d} ({count/total_boxes*100:.1f}%)")

    # 打包 ZIP
    output_zip = Path(output_zip)
    if not output_zip.is_absolute():
        output_zip = PROJECT_ROOT / output_zip

    with zipfile.ZipFile(str(output_zip), "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(tmp_dir.glob("*.txt")):
            zf.write(str(txt_file), txt_file.name)

    print(f"\n  ✓ 提交文件已生成: {output_zip}")
    print(f"  文件大小: {output_zip.stat().st_size / 1024:.1f} KB")
    print(f"  包含 {len(txt_results)} 个 TXT 文件")

    # 验证格式
    print("\n  === 格式验证 ===")
    verify_count = min(5, len(txt_results))
    verify_stems = list(txt_results.keys())[:verify_count]
    for stem in verify_stems:
        lines = txt_results[stem]
        print(f"  {stem}.txt: {len(lines)} 行")
        if lines:
            parts = lines[0].split()
            assert len(parts) == 6, f"行格式错误: {lines[0]}"
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5])
            assert 0 <= cx <= 1 and 0 <= cy <= 1, f"坐标越界: {cx}, {cy}"
            assert 0 <= w <= 1 and 0 <= h <= 1, f"尺寸越界: {w}, {h}"
            assert 0 <= conf <= 1, f"置信度越界: {conf}"
            assert 0 <= cls_id <= 11, f"类别越界: {cls_id}"
            print(f"    示例: {lines[0]}")
    print("  ✓ 格式验证通过")

    return True


def main():
    parser = argparse.ArgumentParser(description="提交文件导出")
    parser.add_argument("--predictions", default="outputs/predictions/test_predictions.pt",
                        help="推理结果文件路径")
    parser.add_argument("--output", default="submission.zip",
                        help="输出 ZIP 文件路径")
    parser.add_argument("--max_det", type=int, default=100,
                        help="单图最大检测框数")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = PROJECT_ROOT / args.predictions

    print("\n" + "=" * 60)
    print("  提交文件导出")
    print("=" * 60)

    success = export_submission(str(predictions_path), args.output, args.max_det)

    if success:
        print("\n  ✓ 提交文件导出完成！")
        print(f"  提交文件: {args.output}")
        print("\n  提交说明:")
        print("  1. 将 ZIP 文件上传到竞赛平台")
        print("  2. 确保 ZIP 内每个 TXT 文件对应一张测试图")
        print("  3. 行格式: class_id cx cy w h confidence")
        print("  4. 坐标为归一化值 [0,1]")
        print("  5. 单图最多 100 个检测框")
    else:
        print("\n  ✗ 导出失败，请检查推理结果文件")
        sys.exit(1)


if __name__ == "__main__":
    main()
