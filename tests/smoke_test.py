#!/usr/bin/env python
"""
冒烟测试：验证损失前向 + 真实数据加载 + 端到端训练步

测试项：
  1. 合成 batch 损失前向（TAL 分配 + CIoU + DFL + 辅助损失）
  2. 真实数据集加载（MultiModalDataset + collate_fn）
  3. 端到端单步训练（前向 → 损失 → 反向 → 更新）
  4. 推理前向 + 后处理（parse_model_outputs → NMS）
"""

import sys
import os
import traceback

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

PYTHON = sys.executable
print(f"Python: {PYTHON}")
print(f"PyTorch: {torch.__version__}")

try:
    import ultralytics
    print(f"ultralytics: {ultralytics.__version__}")
except:
    print("ultralytics: NOT FOUND")

from src.models.multimodal_yolo import build_multimodal_model
from src.losses.detection_loss import MultiModalDetectionLoss
from src.data.multimodal_dataset import MultiModalDataset, collate_fn
from src.inference.postprocess import parse_model_outputs
from src.utils.io_utils import get_aligned_stems

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")


# ============================================================
# Test 1: 合成 batch 损失前向
# ============================================================
def test_loss_forward():
    print("=" * 60)
    print("[Test 1] 损失前向（合成 batch）")
    print("=" * 60)

    model = build_multimodal_model("stage_gate", num_classes=12, model_size="n", use_aux=True)
    model.to(DEVICE)
    model.train()
    strides = model._init_strides()
    print(f"  strides: {strides}")

    loss_fn = MultiModalDetectionLoss(
        model, box_w=7.5, cls_w=0.5, dfl_w=1.5,
        aux_weights={"ir_contrast": 0.3, "depth_mask": 0.2, "depth_normal": 0.15, "total_aux": 0.15},
        tal_topk=13,
    )

    B = 2
    H, W = 640, 640
    rgb = torch.randn(B, 3, H, W, device=DEVICE)
    ir = torch.randn(B, 1, H, W, device=DEVICE)
    depth = torch.randn(B, 1, H, W, device=DEVICE)

    # 合成 GT：每张图 3 个框，letterbox 后 640 像素空间 xyxy
    cls_list = []
    box_list = []
    idx_list = []
    for i in range(B):
        for j in range(3):
            cls_list.append(torch.tensor(j % 12, dtype=torch.long))
            x1 = torch.tensor(50.0 + j * 100 + i * 30)
            y1 = torch.tensor(80.0 + j * 80)
            x2 = x1 + torch.tensor(60.0)
            y2 = y1 + torch.tensor(80.0)
            box_list.append(torch.stack([x1, y1, x2, y2]))
            idx_list.append(torch.tensor(i, dtype=torch.long))

    batch = {
        "cls": torch.stack(cls_list),
        "bboxes": torch.stack(box_list).float(),
        "batch_idx": torch.stack(idx_list),
    }

    # 辅助目标
    aux_targets = {
        "ir_contrast": torch.rand(B, 1, H // 8, W // 8, device=DEVICE),
        "depth_mask": torch.rand(B, 1, H // 8, W // 8, device=DEVICE),
        "depth_normal": torch.randn(B, 2, H // 8, W // 8, device=DEVICE),
    }

    feats, aux_preds = model(rgb, ir, depth)
    print(f"  feats shapes: {[(f.shape[0], f.shape[1], f.shape[2], f.shape[3]) for f in feats]}")
    print(f"  aux keys: {list(aux_preds.keys()) if aux_preds else None}")

    losses = loss_fn(feats, batch, aux_preds, aux_targets)
    for k, v in losses.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k}: {val:.6f}")

    assert torch.isfinite(losses["total"]), "total loss is not finite!"
    assert losses["total"].requires_grad, "total loss has no grad!"

    # 反向传播测试
    losses["total"].backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"  backward grad check: {'OK' if grad_ok else 'FAIL'}")

    print("  [PASS] 损失前向 + 反向传播\n")
    return model, loss_fn


# ============================================================
# Test 2: 真实数据集加载
# ============================================================
def test_dataset_loading():
    print("=" * 60)
    print("[Test 2] 真实数据集加载")
    print("=" * 60)

    data_root = os.path.join(PROJECT_ROOT, "..", "dataset")
    data_root = os.path.abspath(data_root)
    print(f"  data_root: {data_root}")

    # 检查路径存在
    assert os.path.isdir(data_root), f"data_root not found: {data_root}"

    stems = get_aligned_stems(data_root, "train")
    print(f"  aligned stems: {len(stems)}")
    assert len(stems) > 0, "No aligned stems found!"

    dataset = MultiModalDataset(
        data_root=data_root,
        split="train",
        img_size=640,
        use_aux=True,
        modal_dropout_prob=0.0,  # 关闭 dropout 以验证干净管线
        is_training=True,
        cache_processed=False,
    )
    print(f"  dataset size: {len(dataset)}")

    # 加载第一个样本
    item = dataset[0]
    print(f"\n  Sample 0 keys: {list(item.keys())}")
    print(f"    rgb:        {item['rgb'].shape} dtype={item['rgb'].dtype} "
          f"min={item['rgb'].min():.4f} max={item['rgb'].max():.4f}")
    print(f"    ir:         {item['ir'].shape} dtype={item['ir'].dtype} "
          f"min={item['ir'].min():.4f} max={item['ir'].max():.4f}")
    print(f"    depth:      {item['depth'].shape} dtype={item['depth'].dtype} "
          f"min={item['depth'].min():.4f} max={item['depth'].max():.4f}")
    print(f"    cls:        {item['cls'].shape} values={item['cls'].tolist()}")
    print(f"    bboxes:     {item['bboxes'].shape}")
    if len(item['bboxes']) > 0:
        print(f"    bbox[0]:    {item['bboxes'][0].tolist()}")
        # 验证坐标在 [0, 640] 范围内
        b = item['bboxes']
        assert b[:, 0].min() >= 0 and b[:, 2].max() <= 640, "bbox x out of range!"
        assert b[:, 1].min() >= 0 and b[:, 3].max() <= 640, "bbox y out of range!"
        assert (b[:, 2] > b[:, 0]).all(), "x2 <= x1!"
        assert (b[:, 3] > b[:, 1]).all(), "y2 <= y1!"
        print(f"    bbox range: x=[{b[:,0].min():.1f}, {b[:,2].max():.1f}] "
              f"y=[{b[:,1].min():.1f}, {b[:,3].max():.1f}]")
    print(f"    orig_shape: {item['orig_shape'].tolist()}")
    print(f"    meta:       {item['meta']}")

    if "ir_contrast" in item:
        print(f"    ir_contrast:  {item['ir_contrast'].shape}")
        print(f"    depth_mask:   {item['depth_mask'].shape}")
        print(f"    depth_normal: {item['depth_normal'].shape}")

    # collate_fn 测试
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\n  Batch:")
    print(f"    rgb: {batch['rgb'].shape}")
    print(f"    ir:  {batch['ir'].shape}")
    print(f"    depth: {batch['depth'].shape}")
    print(f"    cls: {batch['cls'].shape} values={batch['cls'].unique().tolist()}")
    print(f"    bboxes: {batch['bboxes'].shape}")
    print(f"    batch_idx: {batch['batch_idx'].shape} unique={batch['batch_idx'].unique().tolist()}")
    print(f"    ir_contrast: {batch.get('ir_contrast', 'N/A')}")

    assert batch['rgb'].shape[0] == 4
    assert batch['batch_idx'].max() == 3

    print("  [PASS] 数据集加载 + collate\n")
    return data_root


# ============================================================
# Test 3: 端到端单步训练
# ============================================================
def test_end2end_step(data_root):
    print("=" * 60)
    print("[Test 3] 端到端单步训练（真实数据）")
    print("=" * 60)

    model = build_multimodal_model("stage_gate", num_classes=12, model_size="n", use_aux=True)
    model.to(DEVICE)
    model.train()
    model._init_strides()

    loss_fn = MultiModalDetectionLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    dataset = MultiModalDataset(
        data_root=data_root, split="train", img_size=640,
        use_aux=True, modal_dropout_prob=0.0, is_training=True, cache_processed=False,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    batch = next(iter(loader))
    rgb = batch["rgb"].to(DEVICE)
    ir = batch["ir"].to(DEVICE)
    depth = batch["depth"].to(DEVICE)

    gt = {
        "cls": batch["cls"],
        "bboxes": batch["bboxes"],
        "batch_idx": batch["batch_idx"],
    }
    aux_targets = {
        "ir_contrast": batch["ir_contrast"].to(DEVICE),
        "depth_mask": batch["depth_mask"].to(DEVICE),
        "depth_normal": batch["depth_normal"].to(DEVICE),
    }

    optimizer.zero_grad()
    feats, aux_preds = model(rgb, ir, depth)
    losses = loss_fn(feats, gt, aux_preds, aux_targets)
    print(f"  losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    print(f"  optimizer step: OK")

    # 验证梯度统计
    grad_norms = [p.grad.norm().item() for p in model.parameters()
                  if p.grad is not None and p.requires_grad]
    print(f"  grad norms: min={min(grad_norms):.6f} max={max(grad_norms):.4f} "
          f"mean={np.mean(grad_norms):.4f}")

    print("  [PASS] 端到端训练步\n")


# ============================================================
# Test 4: 推理前向 + 后处理
# ============================================================
def test_inference(model):
    print("=" * 60)
    print("[Test 4] 推理前向 + 后处理")
    print("=" * 60)

    model.eval()
    B = 2
    rgb = torch.randn(B, 3, 640, 640, device=DEVICE)
    ir = torch.randn(B, 1, 640, 640, device=DEVICE)
    depth = torch.randn(B, 1, 640, 640, device=DEVICE)

    with torch.no_grad():
        y = model(rgb, ir, depth)

    if isinstance(y, tuple):
        y = y[0]
    print(f"  output shape: {y.shape}")
    assert y.shape[0] == B
    assert y.shape[1] == 4 + 12  # xyxy + 12 classes

    dets = parse_model_outputs(y, conf_thresh=0.001, iou_thresh=0.5,
                               max_det=100, use_soft_nms=True)
    print(f"  detections per image: {[len(d) for d in dets]}")
    if len(dets[0]) > 0:
        d0 = dets[0][0]
        print(f"  det[0][0]: bbox={d0['bbox']} score={d0['score']:.4f} cls={d0['class_id']}")

    print("  [PASS] 推理 + 后处理\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  多模态检测系统 — 综合冒烟测试")
    print("=" * 60 + "\n")

    results = {}

    # Test 1: 损失前向
    try:
        model, loss_fn = test_loss_forward()
        results["loss_forward"] = True
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["loss_forward"] = False
        model = None

    # Test 2: 数据集加载
    try:
        data_root = test_dataset_loading()
        results["dataset"] = True
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["dataset"] = False
        data_root = None

    # Test 3: 端到端训练步
    if results.get("loss_forward") and data_root:
        try:
            test_end2end_step(data_root)
            results["e2e_train"] = True
        except Exception as e:
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            results["e2e_train"] = False

    # Test 4: 推理
    if model is not None:
        try:
            test_inference(model)
            results["inference"] = True
        except Exception as e:
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            results["inference"] = False

    # Summary
    print("\n" + "=" * 60)
    print("  SMOKE TEST SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"    {k}: {status}")
    all_pass = all(results.values())
    print(f"\n  {'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
