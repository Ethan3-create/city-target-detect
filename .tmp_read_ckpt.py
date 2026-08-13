import torch
ckpt = torch.load("weights/checkpoints/last.pt", map_location="cpu", weights_only=False)
print("epoch:", ckpt.get("epoch"))
print("stage:", ckpt.get("stage"))
