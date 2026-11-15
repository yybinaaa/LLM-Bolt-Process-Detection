"""解析模型输出并做格式校验（方案第七、八节"格式有效率"）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from boltcheck.grid import N_BOLTS, coord_of, is_matrix_shape, is_valid_coord, is_valid_matrix

VERDICTS = ("合规", "不合规", "无法判断")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


@dataclass
class Parsed:
    obj: dict | None = None
    matrix: list | None = None
    matrix_shape_ok: bool = False  # 8×8 整数
    matrix_valid: bool = False  # 8×8 且恰好 1..64
    verdict: str | None = None
    verdict_valid: bool = False
    first_step: int | None = None
    first_coord: list[int] | None = None
    step_layers: list[int] | None = None  # 提示词 v1.2 要求输出；不影响格式有效性，单独统计
    cells: dict | None = None  # 逐格清单 {(x, y): 步数}；v1.4 起由它排出 matrix
    cells_lenient: dict | None = None  # JSON 写坏时用宽松提取得到的逐格清单，只用于辅助指标
    block_max: list[int] | None = None  # 提示词 v1.3 判断时写的三个区块最大数；不影响格式有效性
    format_valid: bool = False
    errors: list[str] = field(default_factory=list)


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # 只有结束标记（模板把 <think> 放在了提示里）
        text = text.split("</think>")[-1]
    return text.strip()


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """从回复中取出 JSON 对象：优先代码块，其次最后一个能解析的 {...}。"""
    body = strip_thinking(text or "")
    for m in reversed(_FENCE_RE.findall(body)):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    found = None
    i = 0
    while i < len(body):
        if body[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(body[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            found = obj  # 取最后一个完整的顶层对象
        i += end  # 跳过整个对象，避免把内部嵌套的对象（如 cells）当成答案
    if found is None:
        return None, "未找到可解析的 JSON 对象"
    return found, None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v)
    return None


def _as_layers(v: Any) -> list[int] | None:
    """长度 64、元素为 1～4 的整数数组；否则视为没给。"""
    if not isinstance(v, list) or len(v) != N_BOLTS:
        return None
    out = [_as_int(t) for t in v]
    if any(t is None or not 1 <= t <= 4 for t in out):
        return None
    return out  # type: ignore[return-value]


_PAIR_RE = re.compile(r'"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?"\s*:\s*(\d+)')


def lenient_cells(text: str) -> dict | None:
    """不管 JSON 是否合法，抠出所有 "(x,y)": 步数 形式的项（同一坐标取第一次出现），只用于辅助指标。"""
    body = strip_thinking(text)
    out: dict = {}
    for x, y, k in _PAIR_RE.findall(body):
        xy = (int(x), int(y))
        if is_valid_coord(xy):
            out.setdefault(xy, int(k))
    return out or None


def cells_to_matrix(cells: dict) -> list[list[int]]:
    return [[cells[coord_of(r, c)] for c in range(1, 9)] for r in range(1, 9)]


def _as_cells(v: Any) -> dict | None:
    """逐格清单：键为坐标字符串（如 "(10,10)"），值为步数；只保留能解析的项，一项都没有则视为没给。"""
    if not isinstance(v, dict):
        return None
    out = {}
    for key, val in v.items():
        nums = re.findall(r"\d+", str(key))
        step = _as_int(val)
        if len(nums) == 2 and step is not None and is_valid_coord([int(nums[0]), int(nums[1])]):
            out[(int(nums[0]), int(nums[1]))] = step
    return out or None


def _as_coord(v: Any) -> list[int] | None:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        xy = [_as_int(t) for t in v]
    elif isinstance(v, str):
        nums = re.findall(r"\d+", v)
        xy = [int(t) for t in nums] if len(nums) == 2 else [None]
    else:
        return None
    if None in xy or not is_valid_coord(xy):
        return None
    return xy  # type: ignore[return-value]


def parse_output(text: str, group: str) -> Parsed:
    p = Parsed()
    if group in ("joint", "restore"):
        p.cells_lenient = lenient_cells(text or "")
    obj, err = extract_json(text)
    if obj is None:
        p.errors.append(err or "解析失败")
        return p
    p.obj = obj

    if group in ("joint", "restore"):
        p.cells = _as_cells(obj.get("cells"))
        if p.cells and all(coord_of(r, c) in p.cells for r in range(1, 9) for c in range(1, 9)):
            # v1.4 起：由程序按坐标把逐格清单排成 8×8 矩阵（只排版，不做判断）
            m = cells_to_matrix(p.cells)
            source = "cells"
        else:
            m = obj.get("order_matrix")  # 兼容 v1.1～v1.3 的回复
            source = "order_matrix"
        p.matrix = m if isinstance(m, list) else None
        p.matrix_shape_ok = is_matrix_shape(m)
        p.matrix_valid = is_valid_matrix(m)
        if not p.matrix_shape_ok:
            if p.cells:
                missing = sum(coord_of(r, c) not in p.cells for r in range(1, 9) for c in range(1, 9))
                p.errors.append(f"cells 缺少 {missing} 个格子")
            else:
                p.errors.append("没有可用的 cells 或 order_matrix")
        elif not p.matrix_valid:
            p.errors.append(f"{source} 没有恰好包含 1～{N_BOLTS}")

    if group in ("joint", "judge"):
        v = obj.get("verdict")
        v = v.strip() if isinstance(v, str) else v
        p.verdict = v if isinstance(v, str) else None
        p.verdict_valid = v in VERDICTS
        if not p.verdict_valid:
            p.errors.append(f"verdict 不在 {VERDICTS} 中: {v!r}")
        p.first_step = _as_int(obj.get("first_violation_step"))
        p.first_coord = _as_coord(obj.get("first_violation_coordinate"))
        p.step_layers = _as_layers(obj.get("step_layers"))
        bm = obj.get("block_max")
        if isinstance(bm, list) and len(bm) == 3 and all(_as_int(v) is not None for v in bm):
            p.block_max = [_as_int(v) for v in bm]  # type: ignore[misc]
        if p.verdict == "不合规" and not (p.first_step and 1 <= p.first_step <= N_BOLTS):
            p.errors.append("判为不合规但 first_violation_step 不是 1～64 的整数")

    if group == "restore":
        p.format_valid = p.matrix_valid
    elif group == "judge":
        p.format_valid = not p.errors
    else:
        p.format_valid = p.matrix_valid and not p.errors
    return p
