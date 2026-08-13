"""
多模态图像/标签 IO 工具
- 目录常量（与官方数据集命名对齐：visible/infrared/depth）
- 三模态图像加载（RGB 3ch / IR 1ch / Depth 1ch）
- YOLO 标签加载
- 多扩展名文件查找、三模态对齐检查
- letterbox 同步变换（三模态 + 标签）
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# ==================== 目录常量 ====================
# 官方数据集模态目录名（与赛题规则一致，禁止随意改动）
MODALITY_DIRS = {
    "rgb": "visible",      # 可见光
    "ir": "infrared",      # 热红外
    "depth": "depth",      # 深度
    "labels": "labels",    # YOLO 标签
}
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]


def find_file(directory: Union[str, Path], stem: str) -> Path:
    """在目录中按 stem 查找图像文件（兼容多扩展名）"""
    d = Path(directory)
    for ext in IMAGE_EXTS:
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到文件: {d}/{stem}.*")


def list_image_files(directory: Union[str, Path]) -> List[Path]:
    """列出目录下所有图像文件"""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.suffix in IMAGE_EXTS])


def get_aligned_stems(data_root: Union[str, Path], split: str) -> List[str]:
    """
    求三模态文件 stem 交集（保证 RGB/IR/Depth 一一对应）
    兼容 visible/infrared/depth 目录名
    """
    root = Path(data_root)
    split_dir = root / split
    rgb_dir = split_dir / MODALITY_DIRS["rgb"]
    ir_dir = split_dir / MODALITY_DIRS["ir"]
    depth_dir = split_dir / MODALITY_DIRS["depth"]

    rgb_stems = {p.stem for p in list_image_files(rgb_dir)}
    ir_stems = {p.stem for p in list_image_files(ir_dir)}
    dp_stems = {p.stem for p in list_image_files(depth_dir)}

    if not rgb_stems or not ir_stems or not dp_stems:
        return []

    common = rgb_stems & ir_stems & dp_stems
    return sorted(common)


# ==================== 图像加载 ====================

def load_rgb(path: Union[str, Path]) -> np.ndarray:
    """加载可见光图 → RGB uint8 [H, W, 3]"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_ir(path: Union[str, Path]) -> np.ndarray:
    """
    加载热红外图 → 单通道 [H, W]
    支持 8bit / 16bit（官方为 16bit 灰度堆叠）
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]  # 3 通道灰度堆叠 → 取第一通道
    return img


def load_depth(path: Union[str, Path]) -> np.ndarray:
    """加载深度图 → 单通道 [H, W]（16bit，单位毫米）"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def load_labels(path: Union[str, Path]) -> np.ndarray:
    """加载 YOLO 标签 → [N, 5] (class_id, cx, cy, w, h) 归一化坐标"""
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 5), dtype=np.float32)
    boxes = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = float(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append([cls, cx, cy, w, h])
    return np.array(boxes, dtype=np.float32).reshape(-1, 5)


# ==================== letterbox 同步变换 ====================

def letterbox_modal(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
                    color: float = 114, stride: int = 32,
                    scaleup: bool = True) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    单模态 letterbox：等比缩放 + 居中填充
    返回: (处理图, 缩放比例, (pad_w, pad_h))
    """
    if img.ndim == 2:
        h, w = img.shape
        orig_shape = (h, w)
    else:
        h, w = img.shape[:2]
        orig_shape = (h, w)

    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(h * r)), int(round(w * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]
    dw, dh = dw % stride, dh % stride  # 对齐 stride
    dw /= 2
    dh /= 2

    if r != 1 or h != new_unpad[0] or w != new_unpad[1]:
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        if img.ndim == 2:
            img = cv2.resize(img, (new_unpad[1], new_unpad[0]), interpolation=interp)
        else:
            img = cv2.resize(img, (new_unpad[1], new_unpad[0]), interpolation=interp)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if img.ndim == 2:
        canvas = np.full((new_shape[0], new_shape[1]), color, dtype=img.dtype)
    else:
        canvas = np.full((new_shape[0], new_shape[1], img.shape[2]), color, dtype=img.dtype)
    canvas[top:top + new_unpad[0], left:left + new_unpad[1]] = img

    return canvas, r, (dw, dh)


def sync_letterbox(rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
                   new_shape: Tuple[int, int] = (640, 640),
                   stride: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    三模态同步 letterbox（共享同一缩放与填充参数，保证空间对齐）
    返回: (rgb, ir, depth, scale, pad_w, pad_h)
    """
    img, r, (dw, dh) = letterbox_modal(rgb, new_shape, stride=stride)
    ir_s, _, _ = letterbox_modal(ir, new_shape, color=0, stride=stride)
    dp_s, _, _ = letterbox_modal(depth, new_shape, color=0, stride=stride)
    return img, ir_s, dp_s, r, dw, dh


def yolo_to_xyxy(labels: np.ndarray, scale: float, dw: float, dh: float,
                 orig_shape: Tuple[int, int]) -> np.ndarray:
    """
    将归一化 YOLO 标签 (cx, cy, w, h) 转换到 letterbox 后图像的像素坐标 (x1, y1, x2, y2)

    公式（以 x 为例，W = 原图宽）:
        x1 = (cx - w/2) * W * scale + dw

    Args:
        labels:   [N, 5] (class_id, cx, cy, w, h)，归一化坐标（相对原图）
        scale:    letterbox 缩放比例
        dw, dh:   letterbox 左右/上下填充（像素）
        orig_shape: (H, W) 原图尺寸
    Returns:
        out: [N, 5] (class_id, x1, y1, x2, y2) letterbox 后像素坐标
    """
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    H, W = int(orig_shape[0]), int(orig_shape[1])
    cls = labels[:, 0:1]
    cx, cy, w, h = labels[:, 1:2], labels[:, 2:3], labels[:, 3:4], labels[:, 4:5]

    # 原图归一化坐标 → 原图像素 → letterbox 像素
    x1 = (cx - w / 2) * W * scale + dw
    y1 = (cy - h / 2) * H * scale + dh
    x2 = (cx + w / 2) * W * scale + dw
    y2 = (cy + h / 2) * H * scale + dh

    out = np.concatenate([cls, x1, y1, x2, y2], axis=1)
    return out.astype(np.float32)


def xyxy_to_yolo_norm(boxes: np.ndarray, orig_shape: Tuple[int, int]) -> np.ndarray:
    """
    原图像素坐标 [N, 4] (x1,y1,x2,y2) → 归一化 YOLO [N, 4] (cx,cy,w,h)
    按真实原图宽高归一化（宽高可不同）。输入须为「原图像素空间」框。
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    H, W = int(orig_shape[0]), int(orig_shape[1])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def letterbox_xyxy_to_norm(boxes: np.ndarray, scale: float, dw: float, dh: float,
                           orig_shape: Tuple[int, int]) -> np.ndarray:
    """
    letterbox 后像素坐标 [N, 4] (x1,y1,x2,y2) → 原图归一化 YOLO [N, 4]
    （逆 letterbox 变换 + 原图归一化），用于把模型检测框映射回提交坐标系。

    公式: x_orig = (x_lb - dw) / scale ;  归一化 = x_orig / W
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    H, W = int(orig_shape[0]), int(orig_shape[1])
    x1 = (boxes[:, 0] - dw) / scale / W
    y1 = (boxes[:, 1] - dh) / scale / H
    x2 = (boxes[:, 2] - dw) / scale / W
    y2 = (boxes[:, 3] - dh) / scale / H
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    out = np.stack([cx, cy, w, h], axis=1)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def visualize_multimodal(rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
                         save_path: Optional[str] = None) -> np.ndarray:
    """三模态可视化拼接（RGB / IR 热图 / Depth 伪彩）"""
    from src.utils.preprocessing import depth_to_colormap

    ir_vis = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ir_vis = cv2.applyColorMap(ir_vis, cv2.COLORMAP_INFERNO)
    dp_vis = depth_to_colormap(depth)

    h, w = rgb.shape[:2]
    ir_vis = cv2.resize(ir_vis, (w, h))
    dp_vis = cv2.resize(dp_vis, (w, h))

    canvas = np.hstack([rgb, ir_vis, dp_vis])
    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas
