"""
COCO 风格 mAP 评估（mAP@50-95，101 点插值）
- 优先使用 pycocotools（与官方评测一致）
- 无 pycocotools 时回退到自研 numpy 实现（算法等价：匹配 + 插值平均精度）
"""

import numpy as np


def _compute_iou_matrix(pred_boxes, gt_boxes):
    """预测框与 GT 框 IoU 矩阵 [P, G]"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    p1 = pred_boxes[:, None, :2]
    p2 = pred_boxes[:, None, 2:]
    g1 = gt_boxes[None, :, :2]
    g2 = gt_boxes[None, :, 2:]

    inter = np.maximum(0, np.minimum(p2, g2) - np.maximum(p1, g1)).prod(axis=-1)
    area_p = (p2 - p1).prod(axis=-1)
    area_g = (g2 - g1).prod(axis=-1)
    union = area_p + area_g - inter
    return inter / np.maximum(union, 1e-9)


def _ap_per_class(pred, gt, iou_thr):
    """
    计算单类 AP
    pred: list of dict {'bbox': [x1,y1,x2,y2], 'score': float}（按图像）
    gt:   list of list of [x1,y1,x2,y2]
    """
    # 展平所有预测
    all_pred = []
    for img_id, preds in enumerate(pred):
        for p in preds:
            all_pred.append({
                "img": img_id, "bbox": np.array(p["bbox"], dtype=np.float32),
                "score": float(p["score"]),
            })
    if len(all_pred) == 0:
        return 0.0

    all_pred.sort(key=lambda x: -x["score"])

    n_gt = sum(len(g) for g in gt)
    if n_gt == 0:
        return 0.0

    # 逐预测匹配
    tp = np.zeros(len(all_pred), dtype=bool)
    fp = np.zeros(len(all_pred), dtype=bool)
    gt_matched = [set() for _ in range(len(gt))]

    for i, p in enumerate(all_pred):
        img_id = p["img"]
        gts = gt[img_id]
        if len(gts) == 0:
            fp[i] = True
            continue
        ious = _compute_iou_matrix(p["bbox"][None, :], np.array(gts, dtype=np.float32))[0]
        best_j = int(np.argmax(ious))
        if ious[best_j] >= iou_thr and best_j not in gt_matched[img_id]:
            tp[i] = True
            gt_matched[img_id].add(best_j)
        else:
            fp[i] = True

    # 累计召回率/精确率
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # 101 点插值 AP
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p_at_t = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p_at_t
    return ap / 101.0


def evaluate_coco_map(preds_per_image, gts_per_image, class_ids,
                      iou_thrs=None, verbose=True):
    """
    计算 COCO mAP（所有类别平均）

    Args:
        preds_per_image: list（按图）of list[dict] {'bbox':[x1,y1,x2,y2], 'score', 'class_id'}
        gts_per_image:   list（按图）of list[dict] {'bbox':[x1,y1,x2,y2], 'class_id'}
        class_ids:       类别 ID 列表（如 range(12)）
        iou_thrs:        IoU 阈值列表，默认 [0.5:0.05:0.95]（10 个）
    Returns:
        mAP50_95, mAP50, mAP75, per_class_ap(dict)
    """
    if iou_thrs is None:
        iou_thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)

    per_class_ap = {}
    for cid in class_ids:
        pred_c = []
        gt_c = []
        for img_preds, img_gts in zip(preds_per_image, gts_per_image):
            pred_c.append([p for p in img_preds if p["class_id"] == cid])
            gt_c.append([g["bbox"] for g in img_gts if g["class_id"] == cid])
        per_class_ap[cid] = {
            thr: _ap_per_class(pred_c, gt_c, thr) for thr in iou_thrs
        }

    ap50_95 = float(np.mean([np.mean(list(v.values())) for v in per_class_ap.values()]))
    ap50 = float(np.mean([v[0.5] for v in per_class_ap.values()]))
    ap75 = float(np.mean([v[0.75] for v in per_class_ap.values()])) \
        if 0.75 in iou_thrs else float(ap50)

    if verbose:
        print(f"  mAP@50-95: {ap50_95:.4f} | mAP@50: {ap50:.4f} | mAP@75: {ap75:.4f}")

    return {"mAP50_95": ap50_95, "mAP50": ap50, "mAP75": ap75,
            "per_class_ap": per_class_ap}


def try_pycocotools(preds_per_image, gts_per_image, verbose=True):
    """尝试使用 pycocotools 计算（与官方口径 100% 一致）"""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None

    import json
    import tempfile
    import os

    images, annotations, ann_id = [], [], 0
    for img_id, gts in enumerate(gts_per_image):
        images.append({"id": img_id})
        for g in gts:
            x1, y1, x2, y2 = g["bbox"]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": int(g["class_id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1

    dets = []
    for img_id, preds in enumerate(preds_per_image):
        for p in preds:
            x1, y1, x2, y2 = p["bbox"]
            dets.append({
                "image_id": img_id, "category_id": int(p["class_id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(p["score"]),
            })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": i} for i in range(80)]}, f)
        gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dets, f)
        det_path = f.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(det_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize() if verbose else None
    stats = ev.stats

    os.unlink(gt_path)
    os.unlink(det_path)
    return {"mAP50_95": float(stats[0]), "mAP50": float(stats[1]),
            "mAP75": float(stats[2]), "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]), "mAP_large": float(stats[5])}
