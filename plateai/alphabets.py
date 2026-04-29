"""Character set used by the plate_rec_color recognizer.

This must stay byte-for-byte identical to the upstream we0091234 charset so
that ONNX outputs from this trainer are drop-in compatible with platex.
"""

# Index 0 is the CTC blank token; the remaining 77 indices map to the
# province characters, special suffixes, digits and letters used on
# Chinese license plates.
PLATE_CHR = (
    "#"
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危"
    "0123456789"
    "ABCDEFGHJKLMNPQRSTUVWXYZ"
    "险品"
)

# 5 colors emitted by the dual-head model.
PLATE_COLORS = ["黑色", "蓝色", "绿色", "白色", "黄色"]

NUM_CLASSES = len(PLATE_CHR)  # 78 (1 blank + 77 real chars)
NUM_COLORS = len(PLATE_COLORS)  # 5

assert NUM_CLASSES == 78, f"PLATE_CHR length must be 78, got {NUM_CLASSES}"


def char_to_index(ch: str) -> int:
    """Map a single character to its CTC index. Returns 0 (blank) for unknown."""
    return PLATE_CHR.index(ch) if ch in PLATE_CHR else 0


def label_to_indices(label: str) -> list[int]:
    """Encode a plate label string as a list of CTC indices, skipping unknowns."""
    return [PLATE_CHR.index(ch) for ch in label if ch in PLATE_CHR]


def indices_to_label(idxs: list[int]) -> str:
    """Decode a list of CTC indices back into a plate string (greedy, no merge)."""
    return "".join(PLATE_CHR[i] for i in idxs if 0 < i < NUM_CLASSES)


def color_name(idx: int) -> str:
    if 0 <= idx < NUM_COLORS:
        return PLATE_COLORS[idx]
    return "未知"
