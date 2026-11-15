"""各组对照的提示词（方案第六、八节）。修改模板时必须同步提高 PROMPT_VERSION。

v1.2 相对 v1.1（依据提示词 v1.1 的 Qwen3-4B 两阶段 5 条试跑）：
- 坐标逐行标出步数；写明中心不是螺栓位置；层数改用公式；先写每步层级、后下结论。

v1.3 相对 v1.2（依据提示词 v1.2 的同一设置 5 条试跑：复原仍输出数字套路，step_layers 退化成一长串 4）：
- 复原：输入每行改成坐标在前"(50,50) → 第1步"；输出先按格子顺序逐个写 cells（"(10,10)": 步数），
  再排成 order_matrix。每写一个坐标就能在输入里找到同一个坐标、抄下后面的步数，不需要数数或心算查找。
- 判断：不再逐步标层，改为在步骤矩阵上按区块检查：中心 2×2、中间 4×4、中间 6×6 的最大数是否为 4、16、36；
  区块外比区块最大数小的数都是违规步，最小的就是首次违规步。只需输出 3 个数（block_max），与违规定义等价（测试验证）。
"""

from __future__ import annotations

from typing import Sequence

PROMPT_VERSION = "v1.3"

GROUPS = {
    "joint": "第 2 组：联合任务（还原矩阵 + 合规判断）",
    "restore": "第 3 组：仅做还原",
    "judge": "第 4 组：仅做判断（给出标准步骤矩阵）",
    "pipeline": "第 5 组：两阶段（先复原，再按自己复原的矩阵判断）",
}

# 第 5 组两次调用分别使用的提示词组：第一阶段同第 3 组，第二阶段同第 4 组（矩阵换成模型自己复原的）
PIPELINE_STAGES = ("restore", "judge")

COORD_DESC = (
    "节点是一个 8×8 的规则螺栓阵列，共 64 颗螺栓。\n"
    "坐标系中 x 向右增大，y 向下增大。x 和 y 的取值都是 10, 20, 30, 40, 50, 60, 70, 80，相邻螺栓间距为 10。\n"
    "阵列中心为 (45,45)，位于中间四颗螺栓之间，本身不是螺栓位置。"
)

MATRIX_CONVENTION = "第 r 行对应 y = 10r（第 1 行是最上面一排），第 c 列对应 x = 10c（第 1 列是最左边一列）"

RULES = (
    "初拧判断规则：\n"
    "1. 把螺栓由内向外划为四层：中心 2×2（第 4～5 行、第 4～5 列）为第 1 层；"
    "中间 4×4（第 3～6 行、第 3～6 列）中除第 1 层外为第 2 层；\n"
    "   中间 6×6（第 2～7 行、第 2～7 列）中除前两层外为第 3 层；最外一圈为第 4 层。\n"
    "2. 必须完成当前层的所有螺栓，才能操作下一层。\n"
    "3. 同一层内的操作顺序不限。\n"
    "4. 如果某一步操作时，还有比它更内层的螺栓没有操作，这一步就是违规步。\n"
    "检查方法（在步骤矩阵上按区块检查）：\n"
    "（1）分别找出中心 2×2、中间 4×4、中间 6×6 三个区块里最大的数，记为 M1、M2、M3。\n"
    "（2）若 M1 = 4、M2 = 16、M3 = 36，说明每一层都在外层之前完成，合规。\n"
    "（3）否则，对每个最大数超出的区块（M1 > 4、M2 > 16 或 M3 > 36），区块外面所有比该区块最大数小的数都是违规步。\n"
    "    所有违规步中最小的那个数就是首次违规步。"
)

VERDICT_FIELDS = (
    "block_max：[M1, M2, M3]，三个整数；\n"
    "verdict：合规、不合规或无法判断，三选一；\n"
    "first_violation_step：首次违规的步骤（整数），合规时为 null；\n"
    "first_violation_coordinate：首次违规步所在格子的坐标，第 r 行第 c 列的坐标是 (10c, 10r)，用 [x, y] 数组表示，合规时为 null；\n"
    "reason：一句话说明判断依据，不超过 80 字。\n"
    "字段请按以上顺序输出：先写 block_max，再写结论。"
)

RESTORE_FIELDS = (
    "cells：按格子顺序列出每个坐标对应的步数，共 64 项。从第 1 行（y = 10）开始，每行从左到右（x = 10, 20, …, 80），"
    "写法为 \"(x,y)\": 步数。每写一个坐标，就到操作序列里找到同一个坐标，抄下它右边的步数；\n"
    f"order_matrix：8×8 步骤矩阵，把 cells 里的步数按顺序每 8 个排成一行，共 8 行。{MATRIX_CONVENTION}。"
    "矩阵里 1～64 每个数恰好出现一次"
)


def format_coords(coords: Sequence[Sequence[int]]) -> str:
    """每行一步，坐标在前：(50,50) → 第1步。"""
    return "\n".join(f"({x},{y}) → 第{k}步" for k, (x, y) in enumerate(coords, start=1))


def format_matrix(matrix: Sequence[Sequence[int]]) -> str:
    return "\n".join("[" + ", ".join(str(v) for v in row) + "]" for row in matrix)


def _sequence_block(coords: Sequence[Sequence[int]]) -> str:
    return (
        "操作序列（按初拧先后排列，共 64 步；每行左边是螺栓坐标，右边是它在第几步拧）：\n"
        f"{format_coords(coords)}"
    )


def build_judge_prompt(matrix: Sequence[Sequence[int]]) -> str:
    """第 4 组，以及第 5 组第二阶段（矩阵为模型自己复原的）共用的判断提示词。"""
    return (
        f"{COORD_DESC}\n"
        f"下面是 8×8 步骤矩阵，每个格子是该位置的初拧步数。{MATRIX_CONVENTION}。\n"
        f"{format_matrix(matrix)}\n"
        f"{RULES}\n"
        "请只输出一个 JSON 对象，包含以下字段：\n"
        f"{VERDICT_FIELDS}"
    )


def build_prompt(group: str, sample: dict) -> str:
    if group == "joint":
        return (
            f"{COORD_DESC}\n"
            f"{_sequence_block(sample['coords'])}\n"
            f"{RULES}\n"
            "请只输出一个 JSON 对象，依次包含以下字段：\n"
            f"{RESTORE_FIELDS}；\n"
            f"{VERDICT_FIELDS}"
        )
    if group == "restore":
        return (
            f"{COORD_DESC}\n"
            f"{_sequence_block(sample['coords'])}\n"
            "请只输出一个 JSON 对象，依次包含两个字段：\n"
            f"{RESTORE_FIELDS}。"
        )
    if group == "judge":
        return build_judge_prompt(sample["order_matrix"])
    if group == "pipeline":
        return build_prompt("restore", sample)  # 第一阶段；第二阶段用 build_judge_prompt(模型复原的矩阵)
    raise ValueError(f"未知的对照组: {group}（可选 {', '.join(GROUPS)}）")
