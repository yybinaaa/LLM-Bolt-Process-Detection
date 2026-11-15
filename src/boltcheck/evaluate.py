"""评测：方案第八节的全部指标、自洽率、错误归因和分组报告。"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from boltcheck.dataset import load_dataset
from boltcheck.grid import transpose
from boltcheck.parse import Parsed, parse_output
from boltcheck.rules import judge_matrix

MATRIX_GROUPS = ("joint", "restore")
VERDICT_GROUPS = ("joint", "judge")

METRIC_NAMES = {
    "format_valid_rate": "格式有效率",
    "api_error_rate": "调用失败率",
    "truncated_rate": "长度截断率",
    "matrix_cell_acc": "矩阵位置准确率",
    "matrix_whole_acc": "整矩阵正确率",
    "matrix_transposed_rate": "转置矩阵比例",
    "verdict_acc": "合规判断准确率",
    "detection_rate": "违规检出率",
    "false_alarm_rate": "合规误报率",
    "unable_rate": "\"无法判断\"比例",
    "first_step_acc": "首次违规步定位准确率",
    "first_coord_acc": "首次违规坐标准确率",
    "self_consistency_rate": "自洽率",
}

ATTRIBUTION_NAMES = {
    "full_correct": "完全正确",
    "reasoning_error": "规则推理错误",
    "restore_error": "还原错误，推理本身成立",
    "both_error": "还原和推理都有问题",
}

GROUP_DIM_NAMES = {
    "construction": "构造方式",
    "layer_gap": "跨层距离",
    "position": "首次违规位置",
    "n_perturb": "扰动处数",
    "boundary_hard": "边界难例",
}


def _mean(xs: list) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _conclusion_ok(verdict: str | None, step: int | None, truth_verdict: str, truth_step: int | None) -> bool:
    if verdict != truth_verdict:
        return False
    return truth_verdict != "不合规" or step == truth_step


def score_row(sample: dict, out: dict | None, group: str) -> dict:
    """单条样本、单次重复的评分。缺失或出错的输出视为格式无效、全部答错。"""
    text = (out or {}).get("text") or ""
    p: Parsed = parse_output(text, group) if out and not out.get("error") else Parsed(errors=["无输出或调用失败"])
    row: dict = {
        "id": sample["id"],
        "compliant": sample["compliant"],
        "format_valid": p.format_valid,
        "api_error": out is None or out.get("error") is not None,
        "truncated": (out or {}).get("finish_reason") == "length",
        "errors": p.errors,
        "latency_s": (out or {}).get("latency_s"),
        "prompt_tokens": (out or {}).get("prompt_tokens"),
        "completion_tokens": (out or {}).get("completion_tokens"),
        "gpu_peak_mib": (out or {}).get("gpu_peak_mib"),
        "groups": {
            "construction": "合规" if sample["compliant"] else sample["construction"],
            **{k: v for k, v in sample["groups"].items() if k != "boundary_hard"},
            "boundary_hard": "是" if sample["groups"]["boundary_hard"] else ("否" if not sample["compliant"] else None),
        },
    }
    truth_m = sample["order_matrix"]
    if group in MATRIX_GROUPS:
        if p.matrix_shape_ok:
            row["cell_acc"] = sum(a == b for ra, rb in zip(p.matrix, truth_m) for a, b in zip(ra, rb)) / 64
        else:
            row["cell_acc"] = 0.0
        row["whole_ok"] = p.matrix_shape_ok and p.matrix == truth_m
        row["transposed"] = p.matrix_shape_ok and p.matrix == transpose(truth_m)
    if group in VERDICT_GROUPS:
        row["verdict"] = p.verdict if p.verdict_valid else None
        row["verdict_ok"] = row["verdict"] == sample["verdict"]
        row["conclusion_ok"] = _conclusion_ok(row["verdict"], p.first_step, sample["verdict"], sample["first_violation_step"])
        row["coord_ok"] = row["verdict"] == "不合规" and p.first_coord == sample["first_violation_coordinate"]
    if group == "joint" and p.format_valid:
        own = judge_matrix(p.matrix)
        row["consistent"] = _conclusion_ok(p.verdict, p.first_step, own.verdict, own.first_violation_step)
        m_ok, c_ok = row["whole_ok"], row["conclusion_ok"]
        if m_ok and c_ok:
            row["attribution"] = "full_correct"
        elif m_ok:
            row["attribution"] = "reasoning_error"
        elif row["consistent"]:
            row["attribution"] = "restore_error"
        else:
            row["attribution"] = "both_error"
    return row


def compute_metrics(rows: list[dict], group: str) -> dict:
    m: dict = {
        "n": len(rows),
        "format_valid_rate": _mean([r["format_valid"] for r in rows]),
        "api_error_rate": _mean([r["api_error"] for r in rows]),
        "truncated_rate": _mean([r["truncated"] for r in rows]),
    }
    if group in MATRIX_GROUPS:
        m["matrix_cell_acc"] = _mean([r["cell_acc"] for r in rows])
        m["matrix_whole_acc"] = _mean([r["whole_ok"] for r in rows])
        m["matrix_transposed_rate"] = _mean([r["transposed"] for r in rows])
    if group in VERDICT_GROUPS:
        nc = [r for r in rows if not r["compliant"]]
        cp = [r for r in rows if r["compliant"]]
        m["verdict_acc"] = _mean([r["verdict_ok"] for r in rows])
        m["detection_rate"] = _mean([r["verdict"] == "不合规" for r in nc])
        m["false_alarm_rate"] = _mean([r["verdict"] == "不合规" for r in cp])
        m["unable_rate"] = _mean([r["verdict"] == "无法判断" for r in rows])
        m["first_step_acc"] = _mean([r["conclusion_ok"] for r in nc])
        m["first_coord_acc"] = _mean([r["coord_ok"] for r in nc])
    if group == "joint":
        m["self_consistency_rate"] = _mean([r["consistent"] for r in rows if "consistent" in r])
    return m


def grouped_metrics(rows: list[dict]) -> dict:
    out: dict = {}
    for dim in GROUP_DIM_NAMES:
        buckets: dict = defaultdict(list)
        for r in rows:
            v = r["groups"].get(dim)
            if v is not None:
                buckets[v].append(r)
        out[dim] = {
            str(v): {
                "n": len(rs),
                "verdict_acc": _mean([r["verdict_ok"] for r in rs]),
                "first_step_acc": _mean([r["conclusion_ok"] for r in rs if not r["compliant"]]),
            }
            for v, rs in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        }
    return out


def _perf(rows: list[dict]) -> dict:
    lat = sorted(r["latency_s"] for r in rows if r["latency_s"] is not None and not r["api_error"])
    pt = [r["prompt_tokens"] for r in rows if r["prompt_tokens"] is not None]
    ct = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
    gpu = [r["gpu_peak_mib"] for r in rows if r["gpu_peak_mib"] is not None]

    def pct(q: float):
        return lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None

    return {
        "latency_mean_s": _mean(lat),
        "latency_p50_s": pct(0.5),
        "latency_p95_s": pct(0.95),
        "prompt_tokens_mean": _mean(pt),
        "completion_tokens_mean": _mean(ct),
        "completion_tokens_max": max(ct) if ct else None,
        "gpu_peak_mib": max(gpu) if gpu else None,
    }


def evaluate_run(run_dir: str | Path, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    group = meta["group"]
    samples = load_dataset(meta["dataset"]["path"])
    if meta.get("limit"):
        samples = samples[: meta["limit"]]
    outputs: dict = {}
    out_path = run_dir / "outputs.jsonl"
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                outputs[(rec["sample_id"], rec["repeat"])] = rec

    repeats = meta.get("repeats", 1)
    all_rows: list[dict] = []
    per_repeat = []
    for rep in range(repeats):
        rows = [score_row(s, outputs.get((s["id"], rep)), group) for s in samples]
        for r in rows:
            r["repeat"] = rep
        all_rows.extend(rows)
        per_repeat.append(compute_metrics(rows, group))

    summary: dict = {}
    for key in per_repeat[0]:
        vals = [m[key] for m in per_repeat]
        if key == "n" or any(v is None for v in vals):
            summary[key] = vals[0] if key == "n" else None
            continue
        summary[key] = {
            "mean": _mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "per_repeat": vals,
        }

    metrics: dict = {
        "run_id": meta["run_id"],
        "group": group,
        "n_samples": len(samples),
        "repeats": repeats,
        "n_outputs_found": sum(1 for s in samples for rep in range(repeats) if (s["id"], rep) in outputs),
        "summary": summary,
        "perf": _perf(all_rows),
    }
    if group == "joint":
        counts = {k: 0 for k in ATTRIBUTION_NAMES}
        for r in all_rows:
            if "attribution" in r:
                counts[r["attribution"]] += 1
        metrics["attribution"] = counts
    if group in VERDICT_GROUPS:
        metrics["grouped"] = grouped_metrics(all_rows)
    metrics["invalid_examples"] = [
        {"id": r["id"], "repeat": r["repeat"], "errors": r["errors"]} for r in all_rows if not r["format_valid"]
    ][:10]

    if write:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (run_dir / "report.md").write_text(render_report(meta, metrics), encoding="utf-8")
        (run_dir / "log_row.md").write_text(render_log_row(meta, metrics) + "\n", encoding="utf-8")
        from boltcheck.viewer import run_view_data, write_view

        write_view(run_view_data(run_dir), run_dir / "view.html")
    return metrics


def _pct(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        s = f"{v['mean'] * 100:.1f}%"
        return s + (f" ± {v['std'] * 100:.1f}" if v.get("std") else "")
    return f"{v * 100:.1f}%"


def _num(v, fmt="{:.1f}") -> str:
    return "—" if v is None else fmt.format(v)


def render_report(meta: dict, metrics: dict) -> str:
    git = meta.get("git") or {}
    commit = (git.get("commit") or "")[:7] + ("（有未提交改动）" if git.get("dirty") else "")
    lines = [
        f"# 运行报告：{meta['run_id']}",
        "",
        f"- 对照组：{meta['group_desc']}",
        f"- 模型：{meta['model']}（后端 {meta['backend']}，配置 {meta['profile']}）"
        + (f"；权重 {meta['hf_repo']}" if meta.get("hf_repo") else "")
        + (f"；{meta['quantization']}" if meta.get("quantization") else ""),
        f"- 重复次数：{metrics['repeats']}；样本数：{metrics['n_samples']}；找到输出：{metrics['n_outputs_found']}",
        f"- 数据集：{meta['dataset'].get('version')}（sha256 {meta['dataset']['sha256'][:12]}）",
        f"- 提示词 {meta['prompt_version']} · 代码 {meta['code_version']} · 提交 {commit or '—'}",
        f"- 运行时间：{meta['created_at']}" + (f"；备注：{meta['note']}" if meta.get("note") else ""),
        "",
        "## 主要指标",
        "",
        "| 指标 | 结果 |",
        "|---|---|",
    ]
    for key, name in METRIC_NAMES.items():
        if key in metrics["summary"]:
            lines.append(f"| {name} | {_pct(metrics['summary'][key])} |")
    lines.append("")
    lines.append("格式无效和调用失败的样本都计入分母、按答错处理；多次重复时显示均值 ± 标准差（百分点）。")

    if "attribution" in metrics:
        total = sum(metrics["attribution"].values())
        lines += ["", "## 错误归因（格式有效的样本）", "", "| 归因 | 数量 | 占比 |", "|---|---|---|"]
        for key, name in ATTRIBUTION_NAMES.items():
            c = metrics["attribution"][key]
            lines.append(f"| {name} | {c} | {_pct(c / total) if total else '—'} |")

    if "grouped" in metrics:
        lines += ["", "## 分组结果", "", "| 维度 | 取值 | 样本数 | 判断准确率 | 首次违规步准确率 |", "|---|---|---|---|---|"]
        for dim, name in GROUP_DIM_NAMES.items():
            for v, g in metrics["grouped"].get(dim, {}).items():
                lines.append(f"| {name} | {v} | {g['n']} | {_pct(g['verdict_acc'])} | {_pct(g['first_step_acc'])} |")
        lines.append("")
        lines.append("多次重复时样本数为各次合计。")

    perf = metrics["perf"]
    lines += [
        "",
        "## 性能",
        "",
        f"- 延迟：均值 {_num(perf['latency_mean_s'], '{:.2f}')} s，P50 {_num(perf['latency_p50_s'], '{:.2f}')} s，P95 {_num(perf['latency_p95_s'], '{:.2f}')} s",
        f"- token：输入均值 {_num(perf['prompt_tokens_mean'], '{:.0f}')}，输出均值 {_num(perf['completion_tokens_mean'], '{:.0f}')}，输出最大 {_num(perf['completion_tokens_max'], '{:.0f}')}",
        f"- 峰值显存：{_num(perf['gpu_peak_mib'], '{:.0f}')} MiB",
    ]
    if metrics["invalid_examples"]:
        lines += ["", "## 格式无效示例（最多 10 条）", ""]
        for ex in metrics["invalid_examples"]:
            lines.append(f"- {ex['id']} 第{ex['repeat'] + 1}次：{'；'.join(ex['errors'])}")
    return "\n".join(lines) + "\n"


LOG_HEADER = (
    "| 日期 | 运行 ID | 提交 | 数据 | 组 | 模型 / 配置 | 重复 | 格式有效 | 判断准确 | 检出率 | 误报率 | 首违步 | 整矩阵 | 自洽 | 备注 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def render_log_row(meta: dict, metrics: dict) -> str:
    s = metrics["summary"]
    git = meta.get("git") or {}
    commit = (git.get("commit") or "")[:7] + ("*" if git.get("dirty") else "")

    def g(key):
        return _pct(s.get(key)) if key in s else "—"

    return (
        f"| {meta['created_at'][:10]} | {meta['run_id']} | {commit or '—'} | {meta['dataset'].get('version')} "
        f"| {meta['group']} | {meta['model']} / {meta['profile']} | {metrics['repeats']} | {g('format_valid_rate')} "
        f"| {g('verdict_acc')} | {g('detection_rate')} | {g('false_alarm_rate')} | {g('first_step_acc')} "
        f"| {g('matrix_whole_acc')} | {g('self_consistency_rate')} | {meta.get('note') or ''} |"
    )
