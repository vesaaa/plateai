"""myNet_ocr_color: dual-head CRNN that emits CTC plate logits + 5-class color.

Architecture taken from we0091234/crnn_plate_recognition (plate_color branch);
kept byte-compatible so that pretrained weights load cleanly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNetOcrColor(nn.Module):
    """Plate recognizer with optional color head.

    Args:
        num_classes: number of CTC classes (default 78 for the platex charset).
        export: if True, returns raw conv features ready for ONNX export.
        cfg: VGG-like channel layout. Default is the medium model used by
             plate_rec_color.pth.
        color_num: if not None, also emit a color head with this many classes.
    """

    def __init__(self, cfg=None, num_classes: int = 78, export: bool = False, color_num: int | None = None):
        super().__init__()
        if cfg is None:
            cfg = [32, 32, 64, 64, "M", 128, 128, "M", 196, 196, "M", 256, 256]
        self.cfg = cfg
        self.feature = self._make_layers(cfg, batch_norm=True)
        self.export = export
        self.color_num = color_num
        self.conv_out_num = 12

        if self.color_num:
            self.conv1 = nn.Conv2d(cfg[-1], self.conv_out_num, kernel_size=3, stride=2)
            self.bn1 = nn.BatchNorm2d(self.conv_out_num)
            self.relu1 = nn.ReLU(inplace=True)
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.color_classifier = nn.Conv2d(self.conv_out_num, self.color_num, kernel_size=1, stride=1)
            self.color_bn = nn.BatchNorm2d(self.color_num)
            self.flatten = nn.Flatten()

        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)

    @staticmethod
    def _make_layers(cfg, batch_norm: bool = False):
        layers = []
        in_channels = 3
        for i, v in enumerate(cfg):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                continue
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=(1, 1), stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.feature(x)

        x_color = None
        if self.color_num:
            c = self.conv1(feat)
            c = self.bn1(c)
            c = self.relu1(c)
            c = self.color_classifier(c)
            c = self.color_bn(c)
            c = self.gap(c)
            x_color = self.flatten(c)

        loc = self.loc(feat)
        out = self.newCnn(loc)

        if self.export:
            # ONNX-friendly path used by platex:
            # squeeze to (B, num_classes, W) and transpose to (B, W, num_classes).
            conv = out.squeeze(2)
            conv = conv.transpose(2, 1)
            if self.color_num:
                return conv, x_color
            return conv

        # Training path: return log-softmax over the time axis for CTC loss.
        b, c, h, w = out.size()
        assert h == 1, f"the height of conv must be 1, got {h}"
        conv = out.squeeze(2)
        conv = conv.permute(2, 0, 1)  # (T, B, C) for CTC
        log_probs = F.log_softmax(conv, dim=2)
        if self.color_num:
            return log_probs, x_color
        return log_probs


def load_pretrained(model: MyNetOcrColor, ckpt_path: str, strict: bool = False) -> dict:
    """Load weights from a we0091234-style .pth file."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def detect_cfg_from_checkpoint(ckpt_path: str) -> list | None:
    """Inspect a .pth file and return the model.cfg used to train it.

    The bundled plate_rec_color checkpoints store ``cfg`` directly in the
    pickled dict; if that's missing we fall back to inferring the channel
    counts from the first conv layer (``feature.0.weight``) since each cfg
    preset has a unique fingerprint at that layer.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "cfg" in state and isinstance(state["cfg"], list):
        return list(state["cfg"])
    if isinstance(state, dict) and "state_dict" in state and isinstance(state.get("cfg"), list):
        return list(state["cfg"])

    # Fallback: peek at the first conv weight shape to identify the preset.
    sd = state.get("state_dict", state) if isinstance(state, dict) else None
    if not isinstance(sd, dict):
        return None
    w = sd.get("feature.0.weight")
    if w is None:
        return None
    out_channels = w.shape[0]
    presets = {
        8: [8, 8, 16, 16, "M", 32, 32, "M", 48, 48, "M", 64, 128],
        16: [16, 16, 32, 32, "M", 64, 64, "M", 96, 96, "M", 128, 256],
        32: [32, 32, 64, 64, "M", 128, 128, "M", 196, 196, "M", 256, 256],
    }
    return presets.get(int(out_channels))
