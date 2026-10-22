"""PlateCRNN: small CNN + BiLSTM CTC recognizer for Chinese plates.

Input is a 32x160 BGR crop normalized to [-1, 1]; output is 40 CTC steps over
``len(CHARS) + 1`` classes (index 0 = blank). 40 steps leave room for 8-char
NEV plates with repeated characters, which the 20/21-step HyperLPR3 and
plate_rec_color models cannot represent reliably.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CHARS = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危"
    "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品应急"
)
NUM_CLASSES = len(CHARS) + 1  # 0 = CTC blank
INPUT_H, INPUT_W = 32, 160
MEAN, STD = 0.5, 0.5  # applied to BGR/255

DEFAULT_CHANNELS = (32, 64, 96, 128)


def _cbr(i, o, k=3, s=1, p=1):
    return [nn.Conv2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class PlateCRNN(nn.Module):
    def __init__(self, c=DEFAULT_CHANNELS, hidden: int = 128):
        super().__init__()
        c1, c2, c3, c4 = c
        self.cnn = nn.Sequential(
            *_cbr(3, c1), nn.MaxPool2d(2, 2),                              # 16x80
            *_cbr(c1, c2), nn.MaxPool2d(2, 2),                             # 8x40
            *_cbr(c2, c3), *_cbr(c3, c3), nn.MaxPool2d((2, 1), (2, 1)),    # 4x40
            *_cbr(c3, c4), *_cbr(c4, c4), nn.MaxPool2d((2, 1), (2, 1)),    # 2x40
            *_cbr(c4, c4, k=(2, 1), p=0),                                  # 1x40
        )
        self.rnn = nn.LSTM(c4, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, NUM_CLASSES)

    def forward(self, x):
        """x: (B,3,32,160) normalized -> logits (B,T,C)."""
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        f, _ = self.rnn(f)
        return self.fc(f)


def decode(logits):
    """Greedy CTC decode; returns [(text, min_char_prob)] per batch item."""
    prob = logits.softmax(-1)
    p, idx = prob.max(-1)
    out = []
    for b in range(idx.size(0)):
        s, ps, prev = [], [], 0
        for t, i in enumerate(idx[b].tolist()):
            if i != 0 and i != prev:
                s.append(CHARS[i - 1])
                ps.append(p[b, t].item())
            prev = i
        out.append(("".join(s), min(ps) if ps else 0.0))
    return out


def parse_channels(s: str) -> tuple:
    return tuple(int(v) for v in s.split(","))
