"""解析模型输出并做格式校验（方案第七、八节"格式有效率"）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from boltcheck.grid import N_BOLTS, is_matrix_shape, is_valid_coord, is_valid_matrix

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
    for i, ch in enumerate(body):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(body[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found = obj  # 取最后一个完整对象
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
    obj, err = extract_json(text)
    if obj is None:
        p.errors.append(err or "解析失败")
        return p
    p.obj = obj

    if group in ("joint", "restore"):
        m = obj.get("order_matrix")
        p.matrix = m if isinstance(m, list) else None
        p.matrix_shape_ok = is_matrix_shape(m)
        p.matrix_valid = is_valid_matrix(m)
        if not p.matrix_shape_ok:
            p.errors.append("order_matrix 不是 8×8 整数矩阵")
        elif not p.matrix_valid:
            p.errors.append(f"order_matrix 没有恰好包含 1～{N_BOLTS}")

    if group in ("joint", "judge"):
        v = obj.get("verdict")
        v = v.strip() if isinstance(v, str) else v
        p.verdict = v if isinstance(v, str) else None
        p.verdict_valid = v in VERDICTS
        if not p.verdict_valid:
            p.errors.append(f"verdict 不在 {VERDICTS} 中: {v!r}")
        p.first_step = _as_int(obj.get("first_violation_step"))
        p.first_coord = _as_coord(obj.get("first_violation_coordinate"))
        if p.verdict == "不合规" and not (p.first_step and 1 <= p.first_step <= N_BOLTS):
            p.errors.append("判为不合规但 first_violation_step 不是 1～64 的整数")

    if group == "restore":
        p.format_valid = p.matrix_valid
    elif group == "judge":
        p.format_valid = not p.errors
    else:
        p.format_valid = p.matrix_valid and not p.errors
    return p
