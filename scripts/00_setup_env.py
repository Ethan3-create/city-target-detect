#!/usr/bin/env python
"""
00_setup_env.py — 环境检查与依赖验证

检查项：
  - Python 版本 (≥3.9)
  - PyTorch 安装与 CUDA 可用性
  - ultralytics 版本 (≥8.2.0)
  - OpenCV / numpy / PyYAML
  - 数据集目录结构完整性
  - 模型构建与前向冒烟测试

用法:
    python scripts/00_setup_env.py [--data_root ../dataset]
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python():
    """检查 Python 版本"""
    import platform
    ver = sys.version_info
    print(f"[1/6] Python: {platform.python_version()}")
    if ver < (3, 9):
        print(f"  ⚠ 需要 Python ≥3.9，当前 {ver.major}.{ver.minor}")
        return False
    print("  ✓ 版本满足要求")
    return True


def check_torch():
    """检查 PyTorch 与 CUDA"""
    try:
        import torch
        print(f"[2/6] PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  ✓ CUDA 可用: {gpu_name} ({vram:.1f} GB)")
        else:
            print("  ⚠ CUDA 不可用，将使用 CPU（训练速度较慢）")
        return True
    except ImportError:
        print("[2/6] ✗ PyTorch 未安装")
        print("  安装: pip install torch")
        return False


def check_ultralytics():
    """检查 ultralytics 版本"""
    try:
        import ultralytics
        ver = ultralytics.__version__
        print(f"[3/6] ultralytics: {ver}")
        major, minor = int(ver.split(".")[0]), int(ver.split(".")[1])
        if major < 8 or (major == 8 and minor < 2):
            print(f"  ⚠ 需要 ultralytics ≥8.2.0（当前 {ver}）")
            print("  升级: pip install ultralytics>=8.2.0")
            return False
        print("  ✓ 版本满足要求")
        return True
    except ImportError:
        print("[3/6] ✗ ultralytics 未安装")
        print("  安装: pip install ultralytics>=8.2.0")
        return False


def check_other_deps():
    """检查其他依赖"""
    deps = {"cv2": "opencv-python", "numpy": "numpy", "yaml": "PyYAML"}
    all_ok = True
    for mod, pkg in deps.items():
        try:
            __import__(mod)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} 未安装")
            all_ok = False
    print("[4/6] 其他依赖:")
    return all_ok


def check_dataset(data_root):
    """检查数据集目录结构"""
    data_root = Path(data_root)
    print(f"[5/6] 数据集: {data_root}")

    splits = {"train": ["visible", "infrared", "depth", "labels"],
              "test": ["visible", "infrared", "depth"]}
    all_ok = True
    for split, dirs in splits.items():
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  ✗ {split}/ 目录不存在")
            all_ok = False
            continue
        for d in dirs:
            dpath = split_dir / d
            if not dpath.exists():
                print(f"  ✗ {split}/{d}/ 不存在")
                all_ok = False
            else:
                count = len(list(dpath.iterdir()))
                print(f"  ✓ {split}/{d}/ ({count} 文件)")
    return all_ok


def check_model():
    """模型构建与前向冒烟测试"""
    print("[6/6] 模型冒烟测试:")
    try:
        import torch
        from src.models.multimodal_yolo import build_multimodal_model

        for mode in ["early", "stage_gate"]:
            model = build_multimodal_model(mode, num_classes=12, model_size="n", use_aux=True)
            strides = model._init_strides()
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  ✓ {mode}: {params:.2f}M params, strides={strides}")

            # 前向测试
            model.train()
            rgb = torch.randn(1, 3, 640, 640)
            ir = torch.randn(1, 1, 640, 640)
            depth = torch.randn(1, 1, 640, 640)
            feats, aux = model(rgb, ir, depth)
            assert len(feats) == 3, f"expected 3 scales, got {len(feats)}"
            print(f"    train: {len(feats)} scales, aux={'yes' if aux else 'no'}")

            model.eval()
            with torch.no_grad():
                y = model(rgb, ir, depth)
            if isinstance(y, tuple):
                y = y[0]
            assert y.shape[1] == 4 + 12, f"expected 16 ch, got {y.shape[1]}"
            print(f"    eval: output {y.shape}")
        return True
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="环境检查")
    parser.add_argument("--data_root", default="../dataset",
                        help="数据集根目录（默认 ../dataset）")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / args.data_root).resolve()

    print("\n" + "=" * 60)
    print("  环境检查 — 多模态目标检测")
    print("=" * 60 + "\n")

    results = {
        "Python": check_python(),
        "PyTorch": check_torch(),
        "ultralytics": check_ultralytics(),
        "其他依赖": check_other_deps(),
        "数据集": check_dataset(data_root),
        "模型": check_model(),
    }

    print("\n" + "=" * 60)
    print("  检查结果汇总")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    all_ok = all(results.values())
    if all_ok:
        print("\n  ✓ 环境就绪，可以开始训练！")
        print("  下一步: python scripts/01_preprocess.py --data_root <dataset>")
    else:
        print("\n  ✗ 部分检查未通过，请修复后重试")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
