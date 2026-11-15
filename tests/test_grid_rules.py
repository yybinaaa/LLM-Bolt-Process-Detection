import random

from boltcheck.dataset import compliant_sequence
from boltcheck.grid import (
    ALL_COORDS, LAYER_SIZES, coord_of, layer_grid, layer_of, order_matrix, rc_of,
    sequence_from_matrix, transpose,
)
from boltcheck.perturb import METHODS, random_perturbation
from boltcheck.rules import LATEST_FIRST_VIOLATION, judge, near_same_boundary, violation_steps

PLAN_GRID = [
    [4, 4, 4, 4, 4, 4, 4, 4],
    [4, 3, 3, 3, 3, 3, 3, 4],
    [4, 3, 2, 2, 2, 2, 3, 4],
    [4, 3, 2, 1, 1, 2, 3, 4],
    [4, 3, 2, 1, 1, 2, 3, 4],
    [4, 3, 2, 2, 2, 2, 3, 4],
    [4, 3, 3, 3, 3, 3, 3, 4],
    [4, 4, 4, 4, 4, 4, 4, 4],
]


def test_layer_grid_matches_plan():
    assert layer_grid() == PLAN_GRID
    counts = {k: sum(row.count(k) for row in PLAN_GRID) for k in (1, 2, 3, 4)}
    assert counts == LAYER_SIZES


def test_matrix_convention():
    # 第 r 行对应 y = 10r，第 c 列对应 x = 10c
    assert rc_of((10, 80)) == (8, 1)
    assert coord_of(8, 1) == (10, 80)
    seq = list(ALL_COORDS)
    m = order_matrix(seq)
    assert m[0][:3] == [1, 2, 3]  # 第一行是 y=10 的一排
    assert m[7][0] == 57  # 左下角 (10, 80)
    assert sequence_from_matrix(m) == seq
    assert transpose(m) != m


def naive_violations(seq):
    """按方案文字定义：操作某颗螺栓时，若还有更内层的螺栓未操作，则该步违规。"""
    out = []
    for t, p in enumerate(seq):
        remaining = seq[t + 1:]
        if any(layer_of(q) < layer_of(p) for q in remaining):
            out.append(t + 1)
    return out


def test_formal_definition_equals_text_definition():
    rng = random.Random(0)
    for _ in range(300):
        seq = list(ALL_COORDS)
        rng.shuffle(seq)
        assert violation_steps(seq) == naive_violations(seq)


def test_compliant_sequences():
    rng = random.Random(1)
    for _ in range(50):
        j = judge(compliant_sequence(rng))
        assert j.compliant and j.first_violation_step is None and j.n_violation_steps == 0


def test_boundary_swap_first_violation():
    seq = compliant_sequence(random.Random(2))
    seq[15], seq[16] = seq[16], seq[15]  # 第 2 层最后一颗与第 3 层第一颗互换
    j = judge(seq)
    assert not j.compliant
    assert j.first_violation_step == 16
    assert j.layer_gap == 1
    assert j.first_violation_coordinate == list(seq[15])


def test_first_violation_bounds():
    """首次违规不晚于第 36 步；跨层距离 ≥ 2 时不晚于第 16 步。"""
    rng = random.Random(3)
    for _ in range(3000):
        seq = compliant_sequence(rng)
        for _ in range(rng.randint(1, 3)):
            res = random_perturbation(seq, rng.choice(METHODS), rng)
            if res:
                seq = res[0]
        j = judge(seq)
        if not j.compliant:
            assert j.first_violation_step <= LATEST_FIRST_VIOLATION
            if j.layer_gap >= 2:
                assert j.first_violation_step <= 16


def test_perturbations_keep_permutation_and_violate():
    rng = random.Random(4)
    for method in METHODS:
        for _ in range(200):
            base = compliant_sequence(rng)
            res = random_perturbation(base, method, rng)
            assert res is not None
            seq, params = res
            assert sorted(seq) == sorted(base)
            assert not judge(seq).compliant, (method, params)


def test_near_same_boundary():
    assert near_same_boundary([16, 17])
    assert near_same_boundary([14, 19])
    assert not near_same_boundary([13, 17])
    assert not near_same_boundary([4, 17])  # 不同交界


def forward_check(seq):
    """提示词 v1.2 写给模型的检查方法：维护各层剩余数，从前往后扫。"""
    remaining = {1: 4, 2: 12, 3: 20, 4: 28}
    out = []
    for t, p in enumerate(seq, start=1):
        m = min(k for k, v in remaining.items() if v > 0)
        if layer_of(p) > m:
            out.append(t)
        remaining[layer_of(p)] -= 1
    return out


def test_prompt_procedure_equals_formal_definition():
    rng = random.Random(5)
    for _ in range(500):
        seq = list(ALL_COORDS)
        rng.shuffle(seq)
        assert forward_check(seq) == violation_steps(seq)
    for _ in range(300):
        seq = compliant_sequence(rng)
        res = random_perturbation(seq, rng.choice(METHODS), rng)
        seq = res[0] if res else seq
        assert forward_check(seq) == violation_steps(seq)


def test_prompt_layer_formula():
    from boltcheck.grid import layer_rc
    assert all(layer_rc(r, c) == 5 - min(r, c, 9 - r, 9 - c) for r in range(1, 9) for c in range(1, 9))


def test_block_check_equals_formal_definition():
    """提示词 v1.3 的区块检查（中心 2×2 / 4×4 / 6×6 最大数）与违规定义等价。"""
    from boltcheck.rules import block_check, block_maxima

    rng = random.Random(9)
    for _ in range(1500):
        seq = list(ALL_COORDS)
        rng.shuffle(seq)
        assert block_check(order_matrix(seq)) == violation_steps(seq)
    for _ in range(1500):
        seq = compliant_sequence(rng)
        for _ in range(rng.randint(0, 3)):
            res = random_perturbation(seq, rng.choice(METHODS), rng)
            if res:
                seq = res[0]
        m = order_matrix(seq)
        assert block_check(m) == violation_steps(seq)
        assert (block_maxima(m) == [4, 16, 36]) == judge(seq).compliant
