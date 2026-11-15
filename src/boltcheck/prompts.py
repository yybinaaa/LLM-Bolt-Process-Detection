"""四组对照的提示词（方案第六、八节）。修改模板时必须同步提高 PROMPT_VERSION。"""

from __future__ import annotations

from typing import Sequence

PROMPT_VERSION = "v1.1"

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
    "坐标系中 x 向右增大，y 向下增大。x 和 y 的取值都是 10, 20, 30, 40, 50, 60, 70, 80，\n"
    "相邻螺栓间距为 10，阵列中心为 (45,45)。"
)

RULES = (
    "初拧判断规则：\n"
    "1. 将螺栓按距离阵列中心的位置划为四个方形层：\n"
    "   中央 2×2 为第 1 层，向外依次为第 2、3、4 层。\n"
    "2. 必须完成当前层的所有螺栓，才能操作下一层。\n"
    "3. 同一层内的操作顺序不限。\n"
    "4. 如果某一步操作时，还有比它更内层的螺栓没有操作，这一步就是违规步。"
)

MATRIX_CONVENTION = "第 r 行对应 y = 10r（第 1 行是最上面一排），第 c 列对应 x = 10c（第 1 列是最左边一列）"

VERDICT_FIELDS = (
    "verdict：合规、不合规或无法判断，三选一；\n"
    "first_violation_step：首次违规的步骤（整数），合规时为 null；\n"
    "first_violation_coordinate：首次违规那一步的坐标，用 [x, y] 数组表示，合规时为 null；\n"
    "reason：判断依据。"
)


def format_coords(coords: Sequence[Sequence[int]]) -> str:
    return "[" + ", ".join(f"({x},{y})" for x, y in coords) + "]"


def format_matrix(matrix: Sequence[Sequence[int]]) -> str:
    return "\n".join("[" + ", ".join(str(v) for v in row) + "]" for row in matrix)


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
            "下面的坐标按初拧先后排列，第 1 个坐标代表第 1 步。\n"
            f"操作序列：{format_coords(sample['coords'])}\n"
            f"{RULES}\n"
            "请只输出一个 JSON 对象，包含以下字段：\n"
            f"order_matrix：8×8 步骤矩阵，每个格子填该位置的操作步数。{MATRIX_CONVENTION}；\n"
            f"{VERDICT_FIELDS}"
        )
    if group == "restore":
        return (
            f"{COORD_DESC}\n"
            "下面的坐标按初拧先后排列，第 1 个坐标代表第 1 步。\n"
            f"操作序列：{format_coords(sample['coords'])}\n"
            "请只输出一个 JSON 对象，只包含一个字段：\n"
            f"order_matrix：8×8 步骤矩阵，每个格子填该位置的操作步数。{MATRIX_CONVENTION}。"
        )
    if group == "judge":
        return build_judge_prompt(sample["order_matrix"])
    if group == "pipeline":
        return build_prompt("restore", sample)  # 第一阶段；第二阶段用 build_judge_prompt(模型复原的矩阵)
    raise ValueError(f"未知的对照组: {group}（可选 {', '.join(GROUPS)}）")
