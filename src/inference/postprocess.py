"""
推理后处理
- 模型输出解析（YOLO 解码结果 → 检测框列表）
- NMS / Soft-NMS（类别内抑制，缓解遮挡漏检）
- 多尺度 TTA（缩放 + 翻转融合，提分手段）
- 检测结果 → COCO JSON
"""

import numpy as np
import torch


def parse_model_outputs(y, conf_thresh: float = 0.001, iou_thresh: float = 0.5,
                        max_det: int = 100, use_soft_nms: bool = True,
                        img_size: int = 640) -> list:
    """
    解析模型推理输出 y: [B, 4+nc, N]（xyxy 像素坐标 + sigmoid 得分）

    Returns:
        list（按 batch）of list[dict] {'bbox':[x1,y1,x2,y2], 'score', 'class_id'}
    """
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = y.detach().cpu().float() if isinstance(y, torch.Tensor) else torch.from_numpy(y).float()
    B, no, N = y.shape
    nc = no - 4

    results = []
    for bi in range(B):
        pred = y[bi]                        # [4+nc, N]
        boxes = pred[:4].t().numpy()        # [N, 4]
        scores = pred[4:].t().numpy()       # [N, nc]

        cls_scores = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        keep = cls_scores > conf_thresh
        boxes, cls_scores, cls_ids = boxes[keep], cls_scores[keep], cls_ids[keep]

        dets = []
        for cid in range(nc):
            m = cls_ids == cid
            if not m.any():
                continue
            b_c = boxes[m]
            s_c = cls_scores[m]
            keep_idx = soft_nms(b_c, s_c, iou_thresh) if use_soft_nms \
                else nms(b_c, s_c, iou_thresh)
            for idx in keep_idx:
                dets.append({
                    "bbox": [float(v) for v in b_c[idx]],
                    "score": float(s_c[idx]),
                    "class_id": int(cid),
                })

        dets.sort(key=lambda d: -d["score"])
        results.append(dets[:max_det])
    return results


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5):
    """标准 NMS（按分数降序）"""
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = _batch_iou(boxes[i][None, :], boxes[order[1:]])
        order = order[1:][ious <= iou_thresh]
    return keep


def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5,
             sigma: float = 0.5, method: str = "gaussian") -> list:
    """
    Soft-NMS：对重叠框分数做衰减而非直接删除，缓解遮挡目标漏检
    method: 'gaussian'（高斯衰减） / 'linear' / 'hard'
    """
    order = np.argsort(-scores).astype(np.int64)
    keep = []
    updated_scores = scores.copy()

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _batch_iou(boxes[i][None, :], boxes[rest])
        if method == "gaussian":
            updated_scores[rest] = updated_scores[rest] * np.exp(-(ious ** 2) / sigma)
        elif method == "linear":
            updated_scores[rest] = updated_scores[rest] * (1 - ious)
        else:
            updated_scores[rest][ious > iou_thresh] = 0.0
        order = rest[updated_scores[rest] > 0.001]
        # 重排序
        order = order[np.argsort(-updated_scores[order])]
    return keep


def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes_a: [1,4], boxes_b: [M,4] → IoU [M]"""
    a1, a2 = boxes_a[:, :2], boxes_a[:, 2:]
    b1, b2 = boxes_b[:, :2], boxes_b[:, 2:]
    inter = np.maximum(0, np.minimum(a2, b2) - np.maximum(a1, b1)).prod(axis=1)
    area_a = (a2 - a1).prod(axis=1)
    area_b = (b2 - b1).prod(axis=1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


# ==================== TTA ====================

@torch.no_grad()
def tta_predict(model, rgb, ir, depth, device, scales=(640, 800, 960),
                flip=True, conf_thresh=0.001, iou_thresh=0.5,
                max_det=100, use_soft_nms=True):
    """
    多尺度 + 水平翻转 TTA 推理

    Args:
        rgb/ir/depth: 原图（未 letterbox），RGB [H,W,3] / IR [H,W] / Depth [H,W]
        scales: 推理尺度列表
    Returns:
        detections: list[dict] {'bbox':[x1,y1,x2,y2], 'score', 'class_id'}
                    坐标已映射回「原图像素空间」
    """
    from src.utils.io_utils import letterbox_modal
    from src.utils.preprocessing import normalize_depth
    from src.models.late_fusion import wbf

    model.eval()
    H, W = rgb.shape[:2]
    all_dets = []

    for scale in scales:
        lb_rgb, r, (dw, dh) = letterbox_modal(rgb, (scale, scale), stride=32)
        lb_ir, _, _ = letterbox_modal(ir, (scale, scale), color=0, stride=32)
        lb_dp, _, _ = letterbox_modal(depth, (scale, scale), color=0, stride=32)

        # 两个视图：原图 / 水平翻转
        views = [(lb_rgb, lb_ir, lb_dp, False)]
        if flip:
            views.append((np.ascontiguousarray(np.flip(lb_rgb, 1)),
                          np.ascontiguousarray(np.flip(lb_ir, 1)),
                          np.ascontiguousarray(np.flip(lb_dp, 1)),
                          True))

        for v_rgb, v_ir, v_dp, is_flip in views:
            ir_f = v_ir.astype(np.float32)
            rgb_t = torch.from_numpy(v_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            ir_t = torch.from_numpy(ir_f / (65535.0 if ir_f.max() > 255 else 255.0)).unsqueeze(0).unsqueeze(0).to(device)
            dp_t = torch.from_numpy(normalize_depth(v_dp)).unsqueeze(0).unsqueeze(0).to(device)

            y = model(rgb_t, ir_t, dp_t)
            dets = parse_model_outputs(y, conf_thresh=0.001, iou_thresh=iou_thresh,
                                       max_det=300, use_soft_nms=False,
                                       img_size=scale)[0]

            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                # letterbox 逆变换 → 原图像素
                ox1 = (x1 - dw) / r
                oy1 = (y1 - dh) / r
                ox2 = (x2 - dw) / r
                oy2 = (y2 - dh) / r
                if is_flip:
                    ox1, ox2 = W - ox2, W - ox1
                d["bbox"] = [max(0.0, float(ox1)), max(0.0, float(oy1)),
                             min(float(W), float(ox2)), min(float(H), float(oy2))]
                all_dets.append(d)

    # 跨视图融合（WBF），输出原图像素空间
    return wbf([all_dets], iou_thr=0.6, max_det=max_det)



# ==================== 格式转换 ====================

def detections_to_coco_json(all_detections: dict) -> list:
    """
    检测结果 → COCO JSON annotations 列表
    all_detections: {image_id: [{'bbox':[x1,y1,x2,y2], 'score', 'class_id'}]}
    """
    results = []
    for img_id, dets in all_detections.items():
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            results.append({
                "image_id": img_id,
                "category_id": int(d["class_id"]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(d["score"]),
            })
    return results


def yolo_predictions_to_txt(all_detections: dict, orig_shapes: dict) -> dict:
    """
    检测结果 → 各图 YOLO 格式文本行列表（提交用，行格式：
    class_id norm_cx norm_cy norm_w norm_h confidence）
    按各图真实原图宽高归一化（非正方形图正确）
    """
    out = {}
    for stem, dets in all_detections.items():
        H, W = orig_shapes.get(stem, (1, 1))
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            lines.append(f"{d['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {d['score']:.4f}")
        out[stem] = lines
    return out
