import torch

# Paths
ckpt_path = r"D:\latest_checkpoint (1).pt"
save_path = r"D:\stage2_compact.pt"

# 🔧 Allow full loading (your own file, so it's safe)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Extract sub-thresholds (dict)
sub_thresholds = ckpt.get("sub_thresholds", None)

if sub_thresholds is not None:
    # Load best weights
    model_state = torch.load(r"D:\best_stage2 (1).pt", map_location="cpu")

    # Create compact checkpoint (weights + thresholds only)
    compact_ckpt = {
        "model_state_dict": model_state,
        "sub_thresholds": sub_thresholds
    }

    torch.save(compact_ckpt, save_path)
    print(f"💾 Saved compact checkpoint (weights + thresholds) to: {save_path}")
else:
    print("⚠️ 'sub_thresholds' not found in checkpoint keys.")
    print("Available keys:", ckpt.keys())
