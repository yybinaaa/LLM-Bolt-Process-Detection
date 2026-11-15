"""程序规则基线：违规步的形式化定义（方案第四节）与分组标签（方案第五节）。

记第 t 步所操作螺栓的层级为 L(t)：
- 第 t 步违规 ⇔ 存在 s > t 使 L(s) < L(t)；
- 首次违规步为最小的违规 t；不存在则合规；
- 跨层距离 = L(t) - min_{s>t} L(s)，t 为首次违规步。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from boltcheck.grid import LAYER_BOUNDARIES, N_BOLTS, layer_of, sequence_from_matrix

FRONT_LAST_STEP = 16  # 前段：第 1~16 步（第 1、2 层施工期间）
LATEST_FIRST_VIOLATION = 36  # 首次违规步的理论上限


@dataclass
class Judgement:
    compliant: bool
    first_violation_step: int | None
    first_violation_coordinate: list[int] | None
    n_violation_steps: int
    layer_gap: int | None
    layers: list[int] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "合规" if self.compliant else "不合规"

    @property
    def position(self) -> str | None:
        if self.first_violation_step is None:
            return None
        return "front" if self.first_violation_step <= FRONT_LAST_STEP else "back"


def layers_of(seq: Sequence[Sequence[int]]) -> list[int]:
    return [layer_of(p) for p in seq]


def violation_steps(seq: Sequence[Sequence[int]]) -> list[int]:
    """返回所有违规步（从 1 开始）。"""
    layers = layers_of(seq)
    suffix_min = [10] * (len(layers) + 1)
    for i in range(len(layers) - 1, -1, -1):
        suffix_min[i] = min(suffix_min[i + 1], layers[i])
    return [t + 1 for t in range(len(layers)) if suffix_min[t + 1] < layers[t]]


def judge(seq: Sequence[Sequence[int]]) -> Judgement:
    if len(seq) != N_BOLTS or len({tuple(p) for p in seq}) != N_BOLTS:
        raise ValueError("序列必须恰好包含 64 个不同坐标")
    layers = layers_of(seq)
    steps = violation_steps(seq)
    if not steps:
        return Judgement(True, None, None, 0, None, layers)
    t = steps[0]
    gap = layers[t - 1] - min(layers[t:])
    x, y = seq[t - 1]
    return Judgement(False, t, [int(x), int(y)], len(steps), gap, layers)


def judge_matrix(matrix: Sequence[Sequence[int]]) -> Judgement:
    return judge(sequence_from_matrix(matrix))


def near_same_boundary(positions: Sequence[int], window: int = 2) -> bool:
    """所有位置都在同一层交界附近：交界 b|b+1 两侧各 window 步以内，即 [b-window, b+1+window]。"""
    return any(all(b - window <= p <= b + 1 + window for p in positions) for b in LAYER_BOUNDARIES)


def group_labels(j: Judgement, perturbations: Sequence[dict]) -> dict:
    """计算方案第五节的分组标签。合规样本各维度为 None。"""
    if j.compliant:
        return {"layer_gap": None, "position": None, "n_perturb": None, "boundary_hard": False}
    n = len(perturbations)
    boundary = (
        n == 1
        and j.layer_gap == 1
        and near_same_boundary(perturbations[0]["involved_positions"])
    )
    return {
        "layer_gap": "1" if j.layer_gap == 1 else "2-3",
        "position": j.position,
        "n_perturb": "single" if n == 1 else "multi",
        "boundary_hard": boundary,
    }
