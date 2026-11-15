"""不合规样本的三种构造方式（方案第五节）：交换、插入、整段提前。

所有位置参数从 1 开始计数，均相对于扰动前的序列。
每个扰动返回 (新序列, 参数字典)；参数中 involved_positions 记录涉及的位置，用于判定边界难例。
"""

from __future__ import annotations

import random
from typing import Sequence

from boltcheck.grid import Coord, layer_of

Range = tuple[int, int]  # 闭区间，从 1 开始

METHODS = ("swap", "insert", "block")


def swap(seq: Sequence[Coord], i: int, j: int) -> list[Coord]:
    out = list(seq)
    out[i - 1], out[j - 1] = out[j - 1], out[i - 1]
    return out


def insert(seq: Sequence[Coord], src: int, dst: int) -> list[Coord]:
    """把第 src 步取出，插到第 dst 步的位置（dst < src），中间的步骤顺延。"""
    if not dst < src:
        raise ValueError("插入位置必须早于原位置")
    out = list(seq)
    item = out.pop(src - 1)
    out.insert(dst - 1, item)
    return out


def block(seq: Sequence[Coord], src: int, k: int, dst: int) -> list[Coord]:
    """把第 src..src+k-1 步整段取出，插到第 dst 步的位置（dst < src）。"""
    if not dst < src:
        raise ValueError("插入位置必须早于原位置")
    out = list(seq)
    seg = out[src - 1 : src - 1 + k]
    del out[src - 1 : src - 1 + k]
    out[dst - 1 : dst - 1] = seg
    return out


def _clip(r: Range, n: int) -> Range:
    return max(1, r[0]), min(n, r[1])


def random_perturbation(
    seq: Sequence[Coord],
    method: str,
    rng: random.Random,
    src_range: Range = (1, 64),
    dst_range: Range = (1, 64),
    k_range: Range = (2, 6),
    attempts: int = 200,
) -> tuple[list[Coord], dict] | None:
    """在给定范围内随机做一次能制造越序的扰动；找不到时返回 None。

    交换：dst 位置与 src 位置的层级不同，且外层螺栓被换到前面。
    插入：被移动的螺栓比 [dst, src) 区间内至少一颗螺栓更靠外。
    整段提前：被移动的 k 步同属一层，且比 [dst, src) 区间内至少一颗螺栓更靠外。
    """
    n = len(seq)
    layers = [layer_of(p) for p in seq]
    s_lo, s_hi = _clip(src_range, n)
    d_lo, d_hi = _clip(dst_range, n)
    for _ in range(attempts):
        src = rng.randint(s_lo, s_hi)
        dst = rng.randint(d_lo, d_hi)
        if dst >= src:
            continue
        if method == "swap":
            if layers[src - 1] <= layers[dst - 1]:
                continue
            return swap(seq, dst, src), {
                "method": "swap",
                "positions": [dst, src],
                "involved_positions": [dst, src],
            }
        if method == "insert":
            if layers[src - 1] <= min(layers[dst - 1 : src - 1]):
                continue
            return insert(seq, src, dst), {
                "method": "insert",
                "src": src,
                "dst": dst,
                "involved_positions": [dst, src],
            }
        if method == "block":
            k = rng.randint(k_range[0], k_range[1])
            if src + k - 1 > s_hi:  # 整段都要落在 src_range 内
                continue
            seg_layers = set(layers[src - 1 : src - 1 + k])
            if len(seg_layers) != 1:
                continue
            if seg_layers.pop() <= min(layers[dst - 1 : src - 1]):
                continue
            return block(seq, src, k, dst), {
                "method": "block",
                "src": src,
                "k": k,
                "dst": dst,
                "involved_positions": [dst, *range(src, src + k)],
            }
        raise ValueError(f"未知的构造方式: {method}")
    return None
