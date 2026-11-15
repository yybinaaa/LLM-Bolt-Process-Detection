"""8×8 螺栓阵列的坐标、层级与步骤矩阵约定（方案第三、四节）。

约定：
- 坐标 (x, y)，x 向右增大，y 向下增大，取值 10..80，间距 10，中心 (45, 45)。
- 行号 r（自上而下）对应 y = 10r，列号 c（自左向右）对应 x = 10c，r、c 取 1..8。
- 步骤矩阵 order_matrix[r-1][c-1] 为该位置的操作步数（1..64）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

N = 8
SPACING = 10
N_BOLTS = N * N
LAYER_SIZES = {1: 4, 2: 12, 3: 20, 4: 28}
# 合规序列中各层最后一步的位置：层交界位于第 4/5、16/17、36/37 步之间
LAYER_BOUNDARIES = (4, 16, 36)

Coord = tuple[int, int]

ALL_COORDS: list[Coord] = [(SPACING * c, SPACING * r) for r in range(1, N + 1) for c in range(1, N + 1)]


def rc_of(coord: Sequence[int]) -> tuple[int, int]:
    """坐标 -> (行号, 列号)，均从 1 开始。"""
    x, y = int(coord[0]), int(coord[1])
    if not is_valid_coord((x, y)):
        raise ValueError(f"不是阵列上的坐标: {coord!r}")
    return y // SPACING, x // SPACING


def coord_of(r: int, c: int) -> Coord:
    """(行号, 列号) -> 坐标。"""
    return (SPACING * c, SPACING * r)


def is_valid_coord(coord: Sequence[int]) -> bool:
    try:
        x, y = int(coord[0]), int(coord[1])
    except (TypeError, ValueError, IndexError):
        return False
    return (
        len(coord) == 2
        and x % SPACING == 0
        and y % SPACING == 0
        and 1 <= x // SPACING <= N
        and 1 <= y // SPACING <= N
    )


def layer_rc(r: int, c: int) -> int:
    """layer(r, c) = max(|r - 4.5|, |c - 4.5|) + 0.5，用整数运算避免浮点误差。"""
    return (max(abs(2 * r - (N + 1)), abs(2 * c - (N + 1))) + 1) // 2


def layer_of(coord: Sequence[int]) -> int:
    return layer_rc(*rc_of(coord))


def layer_grid() -> list[list[int]]:
    return [[layer_rc(r, c) for c in range(1, N + 1)] for r in range(1, N + 1)]


def order_matrix(seq: Sequence[Sequence[int]]) -> list[list[int]]:
    """坐标序列（第 i 个为第 i 步）-> 8×8 步骤矩阵。"""
    if len(seq) != N_BOLTS:
        raise ValueError(f"序列长度应为 {N_BOLTS}，实际 {len(seq)}")
    m = [[0] * N for _ in range(N)]
    for step, coord in enumerate(seq, start=1):
        r, c = rc_of(coord)
        if m[r - 1][c - 1]:
            raise ValueError(f"坐标重复: {tuple(coord)}")
        m[r - 1][c - 1] = step
    return m


def is_valid_matrix(m: object) -> bool:
    """8×8 整数矩阵，且恰好包含 1..64 各一次。"""
    if not is_matrix_shape(m):
        return False
    values = sorted(v for row in m for v in row)  # type: ignore[union-attr]
    return values == list(range(1, N_BOLTS + 1))


def is_matrix_shape(m: object) -> bool:
    """8×8 且元素都是整数（不检查是否为 1..64 的排列）。"""
    return (
        isinstance(m, list)
        and len(m) == N
        and all(isinstance(row, list) and len(row) == N for row in m)
        and all(isinstance(v, int) and not isinstance(v, bool) for row in m for v in row)
    )


def sequence_from_matrix(m: Sequence[Sequence[int]]) -> list[Coord]:
    """合法步骤矩阵 -> 坐标序列。"""
    if not is_valid_matrix(m):
        raise ValueError("步骤矩阵必须是 8×8 且恰好包含 1..64")
    seq: list[Coord | None] = [None] * N_BOLTS
    for r in range(1, N + 1):
        for c in range(1, N + 1):
            seq[m[r - 1][c - 1] - 1] = coord_of(r, c)
    return [p for p in seq if p is not None]


def transpose(m: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*m)]


def as_coords(seq: Iterable[Sequence[int]]) -> list[Coord]:
    return [(int(p[0]), int(p[1])) for p in seq]
