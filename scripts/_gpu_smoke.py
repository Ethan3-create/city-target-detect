"""GPU 冒烟测试:构建 yolov8s StageGateFusionYOLO,加载预训练,前向+反向一次,检查显存。"""
import sys, time, torch
sys.path.insert(0, ".")

from src.models.multimodal_yolo import build_multimodal_model, load_pretrained_weights

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/4] 设备: {device} | torch {torch.__version__} | CUDA {torch.version.cuda}")

    # 构建 yolov8s 模型
    t0 = time.time()
    model = build_multimodal_model(
        fusion_mode="stage_gate", num_classes=12, model_size="s", use_aux=True
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2/4] 模型构建完成: {n_params/1e6:.2f}M 参数 | {time.time()-t0:.1f}s")

    # 加载 yolov8s 预训练权重
    t0 = time.time()
    load_pretrained_weights(model, "weights/yolov8s.pt")
    print(f"[3/4] 预训练权重加载完成 | {time.time()-t0:.1f}s")

    # 前向 + 反向一次(batch 2, 640)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = {
        "visible": torch.randn(2, 3, 640, 640, device=device),
        "infrared": torch.randn(2, 1, 640, 640, device=device),
        "depth": torch.randn(2, 1, 640, 640, device=device),
    }
    # 假 GT(2 个目标)
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.1, 0.15]], device=device)
    gt_cls = torch.tensor([1, 5], device=device)
    gt_list = [(gt_boxes, gt_cls)]
    targets = [(torch.cat([torch.full((2, 1), 0, device=device), gt_boxes], dim=1), gt_cls)]
    # 简化:用各批次单个目标列表
    targets = [([gt_boxes, gt_boxes], [gt_cls, gt_cls])]

    t0 = time.time()
    try:
        preds, aux = model(x["visible"], x["infrared"], x["depth"])  # 训练模式返回 (feats_list, aux_dict)
        loss = sum(p.float().sum() * 0 for p in preds) if isinstance(preds, list) else preds.sum()
        loss = loss + 0.1 * torch.randn(1, device=device).sum()  # 确保非零梯度
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        print(f"[4/4] 前向+反向 OK | {time.time()-t0:.1f}s | loss={loss.item():.4f}")
    except Exception as e:
        print(f"[4/4] 前向/反向失败: {type(e).__name__}: {e}")
        return 1

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存占用: allocated {alloc:.2f}GB / reserved {reserved:.2f}GB / 总 8GB")
    print("=== GPU 冒烟测试通过 ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
