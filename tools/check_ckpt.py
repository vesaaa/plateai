"""Inspect a plate_rec_color .pth file to verify cfg detection works."""
import sys
import torch

sys.path.insert(0, "/opt/plateai-build")
from plateai.model import detect_cfg_from_checkpoint  # noqa: E402

PATH = "/opt/plateai-build/weights/plate_rec_color.pth"
ck = torch.load(PATH, map_location="cpu", weights_only=False)
print("top-level keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck))
print("has 'cfg' key:", "cfg" in ck if isinstance(ck, dict) else False)
sd = ck.get("state_dict", ck) if isinstance(ck, dict) else None
if isinstance(sd, dict):
    fw = sd.get("feature.0.weight")
    if fw is not None:
        print("feature.0.weight shape:", tuple(fw.shape))
print("detected cfg:", detect_cfg_from_checkpoint(PATH))
