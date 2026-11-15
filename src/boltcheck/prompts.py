"""各组对照的提示词（方案第六、八节）。修改模板时必须同步提高 PROMPT_VERSION。

v1.2 相对 v1.1 的改动（依据 2026-11-15 Qwen3-4B 两阶段 5 条试跑）：
- 坐标序列逐行标出步数（"第1步 (40,40)"），复原时只需查表抄数，不用数坐标排在第几个；
- 写明阵列中心位于中间四颗螺栓之间，本身不是螺栓位置；
- 分层规则写成可直接套用的公式：层数 = 5 − min(r, c, 9−r, 9−c)；
- 规定检查步骤：先写出每一步所在的层（step_layers），再按"各层剩余数"往后扫；
- 字段顺序改为先 step_layers、后结论，reason 限制在 80 字内，避免先下结论再长篇推理。
"""

from __future__ import annotations

from typing import Sequence

PROMPT_VERSION = "v1.2"

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
    "1. 把螺栓由内向外划为四层。第 r 行第 c 列（r、c 取 1～8）的层数 = 5 − min(r, c, 9−r, 9−c)，\n"
    "   即离最近一条边越远，层数越小：中央 2×2 为第 1 层，最外一圈为第 4 层。\n"
    "2. 必须完成当前层的所有螺栓，才能操作下一层。\n"
    "3. 同一层内的操作顺序不限。\n"
    "4. 如果某一步操作时，还有比它更内层的螺栓没有操作，这一步就是违规步。\n"
    "检查方法：\n"
    "（1）按第 1～64 步依次写出每一步螺栓所在的层，得到 step_layers。\n"
    "（2）从第 1 步往后检查。开始时各层剩余：第 1 层 4 颗、第 2 层 12 颗、第 3 层 20 颗、第 4 层 28 颗。\n"
    "    每一步先找出还有剩余的最内层 m；如果这一步的层数大于 m，这一步就是违规步。然后把这一步所在层的剩余数减 1。\n"
    "（3）第一个违规步就是首次违规步；没有违规步则合规。"
)

VERDICT_FIELDS = (
    "step_layers：长度为 64 的整数数组，第 k 个数是第 k 步螺栓所在的层（1～4）；\n"
    "verdict：合规、不合规或无法判断，三选一；\n"
    "first_violation_step：首次违规的步骤（整数），合规时为 null；\n"
    "first_violation_coordinate：首次违规那一步的坐标，用 [x, y] 数组表示，合规时为 null；\n"
    "reason：一句话说明判断依据，不超过 80 字。\n"
    "字段请按以上顺序输出：先写 step_layers，再写结论。"
)

RESTORE_FIELD = (
    f"order_matrix：8×8 步骤矩阵。{MATRIX_CONVENTION}。\n"
    "坐标 (x, y) 对应第 y/10 行、第 x/10 列的格子，格子里填这个坐标在操作序列中标注的步数，"
    "矩阵里 1～64 每个数恰好出现一次"
)


def format_coords(coords: Sequence[Sequence[int]]) -> str:
    """逐行标出步数：第1步 (40,40)。"""
    return "\n".join(f"第{k}步 ({x},{y})" for k, (x, y) in enumerate(coords, start=1))


def format_matrix(matrix: Sequence[Sequence[int]]) -> str:
    return "\n".join("[" + ", ".join(str(v) for v in row) + "]" for row in matrix)


def _sequence_block(coords: Sequence[Sequence[int]]) -> str:
    return f"操作序列（按初拧先后排列，共 64 步，每行是一步）：\n{format_coords(coords)}"


def build_judge_prompt(matrix: Sequence[Sequence[int]]) -> str:
    """第 4 组，以及第 5 组第二阶段（矩阵为模型自己复原的）共用的判断提示词。"""
    return (
        f"{COORD_DESC}\n"
        f"下面是 8×8 步骤矩阵，每个格子是该位置的初拧步数。{MATRIX_CONVENTION}。\n"
        "第 k 步螺栓的位置，就是矩阵里数字 k 所在的格子。\n"
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
            "请只输出一个 JSON 对象，包含以下字段：\n"
            f"{RESTORE_FIELD}；\n"
            f"{VERDICT_FIELDS}"
        )
    if group == "restore":
        return (
            f"{COORD_DESC}\n"
            f"{_sequence_block(sample['coords'])}\n"
            "请只输出一个 JSON 对象，只包含一个字段：\n"
            f"{RESTORE_FIELD}。"
        )
    if group == "judge":
        return build_judge_prompt(sample["order_matrix"])
    if group == "pipeline":
        return build_prompt("restore", sample)  # 第一阶段；第二阶段用 build_judge_prompt(模型复原的矩阵)
    raise ValueError(f"未知的对照组: {group}（可选 {', '.join(GROUPS)}）")
