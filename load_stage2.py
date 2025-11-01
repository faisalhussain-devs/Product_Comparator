import torch
ckpt_small = torch.load(r"D:\stage2_compact.pt", map_location="cpu", weights_only=False)
print(ckpt_small.keys())
print(ckpt_small["sub_thresholds"])
