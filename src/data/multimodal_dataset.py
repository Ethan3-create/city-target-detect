"""
多模态数据集（RGB + IR + Depth + 伪模态）
- 三模态同步加载与 letterbox
- 同步几何增强（翻转 / 缩放）+ RGB 色域增强
- 模态 Dropout（训练时随机失效单模态，提升鲁棒性）
- 伪模态辅助特征（缓存优先，实时计算兜底）
- 标签转换：YOLO 归一化 → letterbox 后像素 xyxy（供 TAL 分配器使用）
"""

import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io_utils import (
    MODALITY_DIRS, find_file, get_aligned_stems, load_rgb, load_ir, load_depth,
    load_labels, sync_letterbox, yolo_to_xyxy,
)
from src.utils.preprocessing import (
    compute_ir_contrast_map, compute_depth_mask, compute_depth_normal_map,
)


class MultiModalDataset(Dataset):
    """
    三模态数据集

    返回 dict:
        rgb:   [3, H, W] float32 [0,1]
        ir:    [1, H, W] float32 [0,1]
        depth: [1, H, W] float32 [0,1] (0.3m~20m 归一化)
        cls:   [N] float32 类别 ID（YOLO 格式 0~11）
        bboxes:[N, 4] float32 (x1,y1,x2,y2) letterbox 后像素坐标
        labels_yolo: [N, 5] 原始归一化标签（用于可视化/提交）
        orig_shape: (H, W) 原图尺寸
        (可选 aux): ir_contrast / depth_mask / depth_normal
        meta: {'stem': ..., 'split': ...}
    """

    def __init__(self, data_root: str, split: str = "train", img_size: int = 640,
                 use_aux: bool = True, modal_dropout_prob: float = 0.3,
                 is_training: bool = True, cache_processed: bool = True,
                 flip_prob: float = 0.5, scale: float = 0.5, seed: int = 0,
                 paste_aug: bool = False, paste_classes=None,
                 paste_prob: float = 0.5, paste_max_objs: int = 2):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.use_aux = use_aux
        self.modal_dropout_prob = modal_dropout_prob
        self.is_training = is_training
        self.cache_processed = cache_processed
        self.flip_prob = flip_prob
        self.scale = scale
        # 复制粘贴增强（弱类别平衡）
        self.paste_aug = paste_aug and is_training
        self.paste_classes = list(paste_classes or [])
        self.paste_prob = paste_prob
        self.paste_max_objs = paste_max_objs
        self._paste_pool = None  # 懒构建

        self.split_dir = self.data_root / split
        self.rgb_dir = self.split_dir / MODALITY_DIRS["rgb"]
        self.ir_dir = self.split_dir / MODALITY_DIRS["ir"]
        self.depth_dir = self.split_dir / MODALITY_DIRS["depth"]
        self.label_dir = self.split_dir / MODALITY_DIRS["labels"]
        self.cache_dir = self.data_root / "processed" / split

        self.stems = get_aligned_stems(data_root, split)
        # 排除已划分到 val/ 的样本（01_preprocess.py 生成 val_stems.txt 清单，
        # 因为 train/ 与 val/ 通过硬链接共享同一 inode，train/ 下仍存在同名文件）
        if split == "train":
            val_list_path = self.data_root / "val_stems.txt"
            if val_list_path.exists():
                val_stems = {line.strip() for line in val_list_path.open()
                             if line.strip()}
                excluded = [s for s in self.stems if s in val_stems]
                if excluded:
                    self.stems = [s for s in self.stems if s not in val_stems]
                    print(f"  [train] 依据 val_stems.txt 排除 {len(excluded)} 个验证样本")
        if len(self.stems) == 0:
            raise RuntimeError(
                f"在 {split} 中未找到三模态对齐数据！\n"
                f"检查目录: {self.rgb_dir} / {self.ir_dir} / {self.depth_dir}\n"
                f"运行: python scripts/01_preprocess.py --data_root {data_root}"
            )
        print(f"  [{split}] 数据集加载完成: {len(self.stems)} 个样本")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # ---- 加载三模态 ----
        rgb = load_rgb(find_file(self.rgb_dir, stem))
        ir = load_ir(find_file(self.ir_dir, stem))
        depth = load_depth(find_file(self.depth_dir, stem))

        # ---- 加载标签（YOLO 归一化，相对原图）----
        labels = np.zeros((0, 5), dtype=np.float32)
        if self.split != "test":
            labels = load_labels(self.label_dir / f"{stem}.txt")
        orig_shape = rgb.shape[:2]

        # ---- 三模态同步 letterbox ----
        rgb_l, ir_l, dp_l, scale, dw, dh = sync_letterbox(
            rgb, ir, depth, (self.img_size, self.img_size))
        rgb, ir, depth = rgb_l, ir_l, dp_l

        # ---- 标签 → letterbox 像素 xyxy（增强与损失统一在 640 像素空间）----
        if len(labels) > 0:
            xyxy = yolo_to_xyxy(labels, scale, dw, dh, orig_shape)  # [N,5] cls+xyxy
        else:
            xyxy = np.zeros((0, 5), dtype=np.float32)

        # ---- 训练时同步增强（像素空间）----
        if self.is_training:
            rgb, ir, depth, xyxy = self._sync_augment(rgb, ir, depth, xyxy)
            # 复制粘贴增强（弱类别对象注入，须在 letterbox 像素空间做）
            if self.paste_aug:
                rgb, ir, depth, xyxy = self._paste_augment(rgb, ir, depth, xyxy)

        # ---- 模态 Dropout ----
        if self.is_training and np.random.rand() < self.modal_dropout_prob:
            rgb, ir, depth = self._apply_modal_dropout(rgb, ir, depth)

        # ---- 伪模态 ----
        aux = None
        if self.use_aux:
            aux = self._load_aux(stem, (self.img_size, self.img_size))

        if len(xyxy) > 0:
            cls = xyxy[:, 0].astype(np.int64)
            boxes = xyxy[:, 1:]
            # 裁剪越界框，避免 TAL 分配异常
            valid = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                     & (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
                     & (boxes[:, 2] <= self.img_size) & (boxes[:, 3] <= self.img_size))
            cls = cls[valid]
            boxes = boxes[valid]
        else:
            cls = np.zeros((0,), dtype=np.int64)
            boxes = np.zeros((0, 4), dtype=np.float32)

        # ---- 归一化 ----
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        ir_t = self._norm_ir(ir)
        dp_t = self._norm_depth(depth)

        item = {
            "rgb": rgb_t,
            "ir": ir_t,
            "depth": dp_t,
            "cls": torch.from_numpy(cls),
            "bboxes": torch.from_numpy(boxes),
            "labels_yolo": torch.from_numpy(labels),
            "orig_shape": torch.tensor(orig_shape, dtype=torch.float32),
            "meta": {"stem": stem, "split": self.split},
        }
        if aux is not None:
            item.update(aux)
        return item

    # ==================== 增强 ====================

    def _sync_augment(self, rgb, ir, depth, xyxy):
        """
        同步几何增强（letterbox 后 640 像素空间）：
        - 三模态共享同一变换，标签（xyxy 像素）同步变换
        - 随机水平翻转 / 缩放+中心裁剪回填 / RGB 色域增强
        """
        h, w = rgb.shape[:2]
        # 随机水平翻转
        if np.random.rand() < self.flip_prob:
            rgb = np.ascontiguousarray(np.flip(rgb, axis=1))
            ir = np.ascontiguousarray(np.flip(ir, axis=1))
            depth = np.ascontiguousarray(np.flip(depth, axis=1))
            if len(xyxy) > 0:
                x1 = xyxy[:, 1].copy()
                xyxy[:, 1] = w - xyxy[:, 3]
                xyxy[:, 3] = w - x1

        # 随机缩放（中心裁剪/填充回原尺寸）
        if np.random.rand() < 0.5 and self.scale > 0:
            s = np.random.uniform(1 - self.scale, 1 + self.scale)
            nw, nh = int(w * s), int(h * s)
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            ir = cv2.resize(ir, (nw, nh), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)

            # 中心裁剪或填充回 (h, w)
            rgb = self._center_crop_or_pad(rgb, (h, w))
            ir = self._center_crop_or_pad(ir, (h, w))
            depth = self._center_crop_or_pad(depth, (h, w))

            # 标签同步：缩放比例 + 中心偏移（像素空间）
            if len(xyxy) > 0:
                sx, sy = nw / w, nh / h
                ox = (w - nw) // 2   # 填充时为正（向右移），裁剪时为负
                oy = (h - nh) // 2
                xyxy[:, 1] = xyxy[:, 1] * sx + ox
                xyxy[:, 2] = xyxy[:, 2] * sy + oy
                xyxy[:, 3] = xyxy[:, 3] * sx + ox
                xyxy[:, 4] = xyxy[:, 4] * sy + oy

        # RGB 色域增强（仅 RGB）
        rgb = self._augment_rgb_color(rgb)
        return rgb, ir, depth, xyxy

    def _center_crop_or_pad(self, img, target_shape):
        """中心对齐：内容小于画布 → 居中填充；大于画布 → 居中裁剪"""
        th, tw = target_shape
        h, w = img.shape[:2]
        y_off = (th - h) // 2   # 可为负（裁剪）或正（填充）
        x_off = (tw - w) // 2
        sy, sx = max(0, -y_off), max(0, -x_off)   # 源裁剪起点
        dy, dx = max(0, y_off), max(0, x_off)     # 目标放置起点
        ch, cw = min(h, th), min(w, tw)

        if img.ndim == 3:
            result = np.full((th, tw, img.shape[2]), 0, dtype=img.dtype)
        else:
            result = np.zeros((th, tw), dtype=img.dtype)
        result[dy:dy + ch, dx:dx + cw] = img[sy:sy + ch, sx:sx + cw]
        return result

    def _augment_rgb_color(self, rgb):
        """RGB 色域增强：亮度/饱和度抖动（不影响 IR/Depth 物理语义）"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] *= np.random.uniform(0.7, 1.3)
        hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _apply_modal_dropout(self, rgb, ir, depth):
        """训练时随机废掉一个模态（模拟模态缺失）"""
        choice = np.random.choice(["rgb", "ir", "depth"])
        if choice == "rgb":
            rgb[:] = 114  # 灰色填充
        elif choice == "ir":
            noise = np.random.normal(0, 0.05 * 255, ir.shape)
            ir[:] = np.clip(ir.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            depth[:] = 0  # 无效深度
        return rgb, ir, depth

    # ==================== 复制粘贴增强（类别平衡） ====================

    def _build_paste_pool(self):
        """
        从训练集构建弱类别对象库（懒加载）：
        遍历 train/ 标签，收集 paste_classes 中各类别的三模态对象 patch。
        patch 从原图（未 letterbox）裁剪，粘贴时再 resize 到目标大小。
        Returns:
            list of dict(cls, rgb, ir, depth, h, w) —— 原始像素 patch
        """
        pool = []
        val_excluded = set()
        val_list_path = self.data_root / "val_stems.txt"
        if val_list_path.exists():
            val_excluded = {l.strip() for l in val_list_path.open() if l.strip()}

        label_dir = self.data_root / "train" / MODALITY_DIRS["labels"]
        rgb_dir = self.data_root / "train" / MODALITY_DIRS["rgb"]
        ir_dir = self.data_root / "train" / MODALITY_DIRS["ir"]
        dp_dir = self.data_root / "train" / MODALITY_DIRS["depth"]

        import glob
        label_files = sorted(glob.glob(str(label_dir / "*.txt")))
        n_sel = 0
        for lp in label_files:
            stem = Path(lp).stem
            if stem in val_excluded:
                continue
            lbl = load_labels(lp)
            if len(lbl) == 0:
                continue
            sel = [int(r[0]) for r in lbl if int(r[0]) in self.paste_classes]
            if not sel:
                continue
            # 每个类别最多取若干实例，控制库大小
            need = any(self._paste_class_count(pool, c) < 40 for c in set(sel))
            if not need:
                continue
            rgb = load_rgb(find_file(rgb_dir, stem))
            ir = load_ir(find_file(ir_dir, stem))
            depth = load_depth(find_file(dp_dir, stem))
            H, W = rgb.shape[:2]
            for row in lbl:
                c = int(row[0])
                if c not in self.paste_classes:
                    continue
                if self._paste_class_count(pool, c) >= 40:
                    continue
                cx, cy, w, h = row[1:5]
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                ir_r = ir if ir.ndim == 2 else ir[:, :, 0]
                pool.append({
                    "cls": c,
                    "rgb": rgb[y1:y2, x1:x2].copy(),
                    "ir": ir_r[y1:y2, x1:x2].copy(),
                    "depth": depth[y1:y2, x1:x2].copy(),
                    "h": y2 - y1, "w": x2 - x1,
                })
                n_sel += 1
        print(f"  [paste] 对象库构建完成: {len(pool)} 个弱类别对象"
              f"（类别 {sorted(set(p['cls'] for p in pool))}）")
        return pool

    @staticmethod
    def _paste_class_count(pool, cls):
        return sum(1 for p in pool if p["cls"] == cls)

    def _paste_augment(self, rgb, ir, depth, xyxy):
        """
        复制粘贴增强：随机从对象库取 1~paste_max_objs 个弱类别对象，
        随机缩放(0.4~1.3)/随机位置粘贴到 640 画布，三模态同步，标签同步追加。
        """
        if self._paste_pool is None:
            self._paste_pool = self._build_paste_pool()
        pool = self._paste_pool
        if len(pool) == 0 or np.random.rand() >= self.paste_prob:
            return rgb, ir, depth, xyxy

        H, W = rgb.shape[:2]
        n_paste = np.random.randint(1, self.paste_max_objs + 1)
        new_rows = []
        for _ in range(n_paste):
            obj = pool[np.random.randint(len(pool))]
            oh, ow = obj["h"], obj["w"]
            # 随机缩放（保持宽高比），限制尺寸避免过大遮挡
            s = np.random.uniform(0.4, 1.3)
            pw = max(12, int(ow * s))
            ph = max(12, int(oh * s))
            if pw >= W or ph >= H:
                continue
            # 随机位置（完整落在画布内）
            x = np.random.randint(0, W - pw)
            y = np.random.randint(0, H - ph)
            # 三模态同步粘贴
            rgb[y:y + ph, x:x + pw] = cv2.resize(obj["rgb"], (pw, ph),
                                                 interpolation=cv2.INTER_LINEAR)
            ir_p = cv2.resize(obj["ir"], (pw, ph), interpolation=cv2.INTER_NEAREST)
            if ir.ndim == 3:
                ir[y:y + ph, x:x + pw, 0] = ir_p
            else:
                ir[y:y + ph, x:x + pw] = ir_p
            depth[y:y + ph, x:x + pw] = cv2.resize(
                obj["depth"], (pw, ph), interpolation=cv2.INTER_NEAREST)
            new_rows.append([float(obj["cls"]), x, y, x + pw, y + ph])

        if new_rows:
            xyxy = np.concatenate([xyxy, np.array(new_rows, dtype=np.float32)], 0)
        return rgb, ir, depth, xyxy

    # ==================== 伪模态 ====================

    def _load_aux(self, stem, size):
        """加载伪模态辅助特征（缓存优先）"""
        cache_ir = self.cache_dir / "ir_contrast" / f"{stem}.npy"
        cache_dm = self.cache_dir / "depth_mask" / f"{stem}.npy"
        cache_dn = self.cache_dir / "depth_normal" / f"{stem}.npy"

        if self.cache_processed and cache_ir.exists():
            ir_c = np.load(str(cache_ir))
            dm = np.load(str(cache_dm))
            dn = np.load(str(cache_dn))
        else:
            ir_raw = load_ir(find_file(self.ir_dir, stem))
            dp_raw = load_depth(find_file(self.depth_dir, stem))
            ir_c = compute_ir_contrast_map(ir_raw)
            dm = compute_depth_mask(dp_raw)
            dn = compute_depth_normal_map(dp_raw)

        if ir_c.shape[:2] != size:
            ir_c = cv2.resize(ir_c, size, interpolation=cv2.INTER_LINEAR)
            dm = cv2.resize(dm.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
            dn = cv2.resize(dn, size, interpolation=cv2.INTER_LINEAR)
            dm = (dm > 0.5).astype(np.float32)

        return {
            "ir_contrast": torch.from_numpy(ir_c.astype(np.float32)).unsqueeze(0),
            "depth_mask": torch.from_numpy(dm.astype(np.float32)).unsqueeze(0),
            "depth_normal": torch.from_numpy(dn.astype(np.float32)).permute(2, 0, 1),
        }

    # ==================== 归一化 ====================

    def _norm_ir(self, ir):
        """IR → [1,H,W] float32 [0,1]（自动适配 8/16bit 灰度堆叠）"""
        if ir.dtype == np.uint16 or ir.max() > 255:
            x = ir.astype(np.float32) / 65535.0
        else:
            x = ir.astype(np.float32) / 255.0
        return torch.from_numpy(x).unsqueeze(0)

    def _norm_depth(self, depth):
        """Depth → [1,H,W] float32 [0,1]（自动适配 16bit 毫米 / 8bit 归一化）"""
        from src.utils.preprocessing import normalize_depth
        x = normalize_depth(depth)
        return torch.from_numpy(x).unsqueeze(0)


def collate_fn(batch):
    """批量拼接（bboxes/cls 变长，展平为 batch 级张量）"""
    rgb = torch.stack([b["rgb"] for b in batch])
    ir = torch.stack([b["ir"] for b in batch])
    depth = torch.stack([b["depth"] for b in batch])

    out = {
        "rgb": rgb,
        "ir": ir,
        "depth": depth,
        "orig_shape": torch.stack([b["orig_shape"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }

    if "ir_contrast" in batch[0]:
        out["ir_contrast"] = torch.stack([b["ir_contrast"] for b in batch])
        out["depth_mask"] = torch.stack([b["depth_mask"] for b in batch])
        out["depth_normal"] = torch.stack([b["depth_normal"] for b in batch])

    # 展平 batch 内所有 GT（附带 batch_idx，供 TAL 分配器定位所属图像）
    cls_list, box_list, yolo_list, idx_list = [], [], [], []
    for i, b in enumerate(batch):
        n = b["cls"].shape[0]
        cls_list.append(b["cls"])
        box_list.append(b["bboxes"])
        yolo_list.append(b["labels_yolo"])
        idx_list.append(torch.full((n,), i, dtype=torch.long))
    out["cls"] = torch.cat(cls_list, dim=0) if len(cls_list) > 0 else torch.zeros(0, dtype=torch.int64)
    out["bboxes"] = torch.cat(box_list, dim=0) if len(box_list) > 0 else torch.zeros(0, 4)
    out["batch_idx"] = torch.cat(idx_list, dim=0) if len(idx_list) > 0 else torch.zeros(0, dtype=torch.long)
    out["labels_yolo"] = yolo_list

    return out
