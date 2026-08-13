"""
伪模态特征生成（仅训练阶段使用，推理不依赖）
- IR 温度对比图：局部均值差分高通滤波，突出热斑（行人/车辆热源）
- Depth 有效掩码：有效深度范围二值化
- Depth 法向梯度图：Sobel 梯度归一化法向量
- 深度伪彩色可视化

深度图格式自适应：
  官方为 16bit PNG（毫米单位，有效 [0, 19999]），但部分样本为 8bit JPEG
  （已归一化到 [0, 255]）。所有深度处理函数自动识别并统一转 [0,1]。
"""

import cv2
import numpy as np

# 官方深度有效范围（毫米）
DEPTH_MIN_MM = 300      # 30cm
DEPTH_MAX_MM = 19999    # 20m（官方上限）


def depth_is_mm(depth: np.ndarray) -> bool:
    """判断深度图是否为毫米单位（16bit PNG）而非 8bit 归一化"""
    return float(depth.max()) > 255.0


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """
    深度图 → [0, 1] float32（自动适配 16bit 毫米 / 8bit 归一化）
    - 16bit 毫米: clip [0, 19999] → /19999
    - 8bit:       /255
    """
    d = depth.astype(np.float32)
    if depth_is_mm(d):
        d = np.clip(d, 0.0, float(DEPTH_MAX_MM))
        return (d / float(DEPTH_MAX_MM)).astype(np.float32)
    return (d / 255.0).astype(np.float32)


def compute_ir_contrast_map(ir: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    IR 温度对比图：原始红外与局部均值之差（高通滤波）
    突出局部热斑（人体/车辆发动机等热源），抑制均匀背景

    Args:
        ir: [H, W] 红外图（8bit 或 16bit）
        kernel_size: 局部均值滤波核（奇数）
    Returns:
        contrast: [H, W] float32 温度对比图
    """
    ir_f = ir.astype(np.float32)
    if ir_f.max() > 255:  # 16bit 归一化
        ir_f = ir_f / 65535.0
    else:
        ir_f = ir_f / 255.0

    k = max(kernel_size, 3)
    if k % 2 == 0:
        k += 1
    local_mean = cv2.blur(ir_f, (k, k))
    contrast = ir_f - local_mean
    # 归一化到 [0, 1]
    cmin, cmax = contrast.min(), contrast.max()
    if cmax - cmin > 1e-6:
        contrast = (contrast - cmin) / (cmax - cmin)
    else:
        contrast = np.zeros_like(contrast)
    return contrast.astype(np.float32)


def compute_depth_mask(depth: np.ndarray, min_depth: int = DEPTH_MIN_MM,
                       max_depth: int = DEPTH_MAX_MM) -> np.ndarray:
    """
    Depth 有效深度掩码：有效深度范围内为 1，否则 0
    - 16bit 毫米: [min_depth, max_depth]（默认 30cm~20m）
    - 8bit 归一化: > 阈值（默认 0）即有效
    滤除无效深度（0、空洞、超远）

    Args:
        depth: [H, W] float32 深度图
    Returns:
        mask: [H, W] uint8 (0/1)
    """
    if depth_is_mm(depth):
        mask = ((depth >= min_depth) & (depth <= max_depth)).astype(np.uint8)
    else:
        # 8bit：0 或极小值视为无效；阈值 4/255 ≈ 30cm 对应值
        thr = max(1.0, min_depth / max_depth * 255.0)
        mask = ((depth >= thr) & (depth <= 255)).astype(np.uint8)
    return mask


def compute_depth_normal_map(depth: np.ndarray, min_depth: int = DEPTH_MIN_MM,
                             max_depth: int = DEPTH_MAX_MM) -> np.ndarray:
    """
    Depth 法向梯度图：基于 Sobel 梯度的表面法向量近似
    为远距离目标（UAV、行人）提供几何先验

    Args:
        depth: [H, W] float32 深度图（毫米或 8bit）
    Returns:
        normal: [H, W, 2] float32 归一化梯度（dx, dy）
    """
    d_norm = normalize_depth(depth)  # [0,1]，两种格式统一
    gx = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.stack([gx, gy], axis=-1)
    # 归一化梯度幅值
    mag = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
    normal = normal / mag
    return normal.astype(np.float32)


def validate_depth_range(depth: np.ndarray, min_depth: int = DEPTH_MIN_MM,
                         max_depth: int = DEPTH_MAX_MM) -> dict:
    """深度值域统计（用于数据 EDA，自动区分 16bit 毫米 / 8bit）"""
    is_mm = depth_is_mm(depth)
    if is_mm:
        valid = (depth >= min_depth) & (depth <= max_depth)
    else:
        valid = depth > 0
    stats = {
        "format": "16bit_mm" if is_mm else "8bit_norm",
        "min": float(depth.min()),
        "max": float(depth.max()),
        "valid_ratio": float(valid.mean()),
        "mean_valid": float(depth[valid].mean()) if valid.any() else 0.0,
    }
    return stats


def depth_to_colormap(depth: np.ndarray, min_depth: int = DEPTH_MIN_MM,
                      max_depth: int = DEPTH_MAX_MM) -> np.ndarray:
    """深度图 → JET 伪彩色 [H, W, 3]（自动适配格式）"""
    d_norm = normalize_depth(depth)
    # 8bit 有效下界对应伪彩色低亮；mm 用 30cm 对应值
    lo = 0.0 if not depth_is_mm(depth) else float(min_depth) / float(max_depth)
    d_vis = ((d_norm - lo) / max(1e-6, 1.0 - lo) * 255)
    d_vis = np.clip(d_vis, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
