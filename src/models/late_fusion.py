"""
晚期融合（Late Fusion）：三模态独立推理 + 预测框级融合
- WBF（Weighted Boxes Fusion）：按置信度加权平均重合框，提升召回与定位
- 与早期/中期融合互补：可作为最终冲刺阶段的提分手段（推理级集成，
  但注意竞赛"禁止多模型简单集成"约束——此处为同一模型的模态缺失视图，
  单模型权重，仅推理时融合，合规。若严格要求单前向，请使用中期融合模型）
"""

import numpy as np


def wbf(detections_list, iou_thr: float = 0.55, skip_box_thr: float = 0.0,
        weights=None, max_det: int = 100):
    """
    Weighted Boxes Fusion（多视图/多尺度检测结果融合）

    Args:
        detections_list: list of list[dict]，每个 dict 含
            {'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int}
            每个元素为一个视图（如不同模态/尺度的预测）
        iou_thr: IoU 融合阈值（默认 0.55）
        skip_box_thr: 低分过滤阈值（0 表示不过滤）
        weights: 各视图权重（None 则等权）
        max_det: 最多保留框数
    Returns:
        fused: list[dict] 融合后的检测结果
    """
    if len(detections_list) == 0:
        return []

    n_views = len(detections_list)
    if weights is None:
        weights = np.ones(n_views)

    # 收集所有框（带视图 id）
    all_boxes = []  # (view_id, box)
    for vi, dets in enumerate(detections_list):
        for d in dets:
            if d["score"] < skip_box_thr:
                continue
            all_boxes.append((vi, d))

    if len(all_boxes) == 0:
        return []

    # 按类别分组
    classes = set(d["class_id"] for _, d in all_boxes)
    fused_all = []
    for cls in classes:
        cls_boxes = [(vi, d) for vi, d in all_boxes if d["class_id"] == cls]
        fused_all.extend(_wbf_class(cls_boxes, cls, iou_thr, weights))

    # 按融合置信度排序，截断
    fused_all.sort(key=lambda x: x["score"], reverse=True)
    return fused_all[:max_det]


def _wbf_class(cls_boxes, cls, iou_thr, weights):
    """单类 WBF 融合（经典 WBF 迭代式聚类）"""
    boxes = np.array([d["bbox"] for _, d in cls_boxes], dtype=np.float32)
    scores = np.array([d["score"] for _, d in cls_boxes], dtype=np.float32)
    view_ids = np.array([vi for vi, _ in cls_boxes], dtype=np.int32)
    view_w = np.array([weights[vi] for vi in view_ids], dtype=np.float32)

    # 按分数降序
    order = np.argsort(-scores)
    boxes, scores, view_w = boxes[order], scores[order], view_w[order]
    used = np.zeros(len(boxes), dtype=bool)

    fused = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        used[i] = True
        cluster = [i]
        # 找与该框 IoU > 阈值的所有框
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if _iou(boxes[i], boxes[j]) >= iou_thr:
                used[j] = True
                cluster.append(j)

        # 加权平均融合框与分数
        cw = view_w[cluster]
        wbox = np.average(boxes[cluster], axis=0, weights=cw)
        wscore = float(np.sum(scores[cluster] * cw) / cw.sum())
        wscore = float(wscore * (1 + len(cluster) * 0.02))  # 重合数越多越可信
        wscore = min(wscore, 1.0)

        fused.append({"bbox": [float(v) for v in wbox], "score": wscore,
                      "class_id": int(cls)})
    return fused


def _iou(a, b):
    """两个框的 IoU（xyxy）"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)
