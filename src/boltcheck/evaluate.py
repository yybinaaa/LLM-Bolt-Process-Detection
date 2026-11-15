"""评测：方案第八节的全部指标、自洽率、错误归因和分组报告。"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from boltcheck.dataset import load_dataset, select_samples
from boltcheck.grid import transpose
from boltcheck.parse import Parsed, parse_output
from boltcheck.grid import coord_of
from boltcheck.rules import BLOCK_KEYS, block_maxima, block_slices, judge_matrix

MATRIX_GROUPS = ("joint", "restore", "pipeline")
VERDICT_GROUPS = ("joint", "judge", "pipeline")
SELF_CHECK_GROUPS = ("joint", "pipeline")  # 模型自己输出矩阵和结论的组：可算自洽率与错误归因

METRIC_NAMES = {
    "format_valid_rate": "格式有效率",
    "api_error_rate": "调用失败率",
    "truncated_rate": "长度截断率",
    "matrix_cell_acc": "矩阵位置准确率",
    "matrix_whole_acc": "整矩阵正确率",
    "matrix_transposed_rate": "转置矩阵比例",
    "cells_acc": "逐格清单准确率",
    "cells_acc_lenient": "逐格清单准确率（宽松提取，辅助）",
    "cells_order_violation_rate": "cells 未按模板顺序的比例",
    "verdict_acc": "合规判断准确率",
    "detection_rate": "违规检出率",
    "false_alarm_rate": "合规误报率",
    "unable_rate": "\"无法判断\"比例",
    "first_step_acc": "首次违规步定位准确率",
    "first_coord_acc": "首次违规坐标准确率",
    "self_consistency_rate": "自洽率",
    "block_max_acc": "区块最大数准确率",
    "block_max_whole_acc": "区块最大数全对比例",
    "block_copy_acc": "区块抄写准确率",
    "block_max_copy_consistent_rate": "区块最大数与所抄区块一致的比例",
    "layer_label_acc": "层级标注准确率",
    "layer_label_whole_acc": "层级标注全对比例",
    "acc_when_matrix_ok": "复原正确时的判断准确率",
    "acc_when_matrix_wrong": "复原错误时的判断准确率",
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


def parse_record(out: dict | None, group: str) -> Parsed:
    """解析一条输出记录。第 5 组把第一阶段的矩阵和第二阶段的结论合并成一个结果。"""
    if not out or out.get("error"):
        return Parsed(errors=["无输出或调用失败"])
    if group != "pipeline":
        return parse_output(out.get("text") or "", group)
    p1 = parse_output((out.get("stage1") or {}).get("text") or "", "restore")
    stage2 = out.get("stage2")
    if stage2 is None:
        return Parsed(
            matrix=p1.matrix, matrix_shape_ok=p1.matrix_shape_ok, matrix_valid=p1.matrix_valid, cells=p1.cells,
            cells_lenient=p1.cells_lenient, cells_order_ok=p1.cells_order_ok, errors=[f"复原：{e}" for e in p1.errors] + ["第一阶段没有得到 8×8 矩阵，未进行判断"],
        )
    p2 = parse_output(stage2.get("text") or "", "judge")
    return Parsed(
        obj=p2.obj, matrix=p1.matrix, matrix_shape_ok=p1.matrix_shape_ok, matrix_valid=p1.matrix_valid, cells=p1.cells,
        cells_lenient=p1.cells_lenient, cells_order_ok=p1.cells_order_ok,
        verdict=p2.verdict, verdict_valid=p2.verdict_valid, first_step=p2.first_step, first_coord=p2.first_coord,
        step_layers=p2.step_layers, block_max=p2.block_max, blocks=p2.blocks,
        format_valid=p1.format_valid and p2.format_valid,
        errors=[f"复原：{e}" for e in p1.errors] + [f"判断：{e}" for e in p2.errors],
    )


def score_row(sample: dict, out: dict | None, group: str) -> dict:
    """单条样本、单次重复的评分。缺失或出错的输出视为格式无效、全部答错。"""
    p: Parsed = parse_record(out, group)
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
        "gpu_apps_peak_mib": (out or {}).get("gpu_apps_peak_mib"),
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
        if p.cells is not None and len(p.cells) == 64:
            row["cells_order_ok"] = p.cells_order_ok
        if p.cells is not None:
            row["cells_acc"] = sum(p.cells.get(coord_of(r + 1, c + 1)) == truth_m[r][c] for r in range(8) for c in range(8)) / 64
        if p.cells_lenient is not None:  # JSON 写坏时也统计内容，只作辅助参考，不影响格式有效性
            row["cells_acc_lenient"] = sum(
                p.cells_lenient.get(coord_of(r + 1, c + 1)) == truth_m[r][c] for r in range(8) for c in range(8)) / 64
    if group in VERDICT_GROUPS:
        # 格式无效（含第 5 组第一阶段矩阵无效）时，判断类指标一律按答错计
        row["verdict"] = p.verdict if (p.verdict_valid and p.format_valid) else None
        row["verdict_ok"] = row["verdict"] == sample["verdict"]
        row["conclusion_ok"] = _conclusion_ok(row["verdict"], p.first_step, sample["verdict"], sample["first_violation_step"])
        row["coord_ok"] = row["verdict"] == "不合规" and p.first_coord == sample["first_violation_coordinate"]
        # 区块类指标与"模型所判断的那张矩阵"比较——第 4 组是标准矩阵，第 2、5 组是模型自己复原的矩阵。
        # 代码 0.8.0 起只要求该矩阵是 8×8 整数（区块最大数对任意整数矩阵都有定义），不再要求恰好包含 1～64，
        # 以便在复原不完全正确时也能看出判断阶段是否真的读了矩阵
        ref_shape = truth_m if group == "judge" else (p.matrix if p.matrix_shape_ok else None)
        ref = truth_m if group == "judge" else (p.matrix if p.matrix_valid else None)
        if ref_shape is not None and p.block_max is not None:
            ref_max = block_maxima(ref_shape)
            row["block_acc"] = sum(a == b for a, b in zip(p.block_max, ref_max)) / 3
            row["block_whole_ok"] = p.block_max == ref_max
        if ref_shape is not None and p.blocks is not None:  # 提示词 v1.5：抄写的 56 个数是否与矩阵一致
            want = block_slices(ref_shape)
            pairs = [(a, b) for k in BLOCK_KEYS for ra, rb in zip(p.blocks[k], want[k]) for a, b in zip(ra, rb)]
            row["block_copy_acc"] = sum(a == b for a, b in pairs) / len(pairs)
        if p.blocks is not None and p.block_max is not None:  # 写出的最大数是否来自自己抄的区块（区分抄错与照搬"应有值"）
            row["block_max_copy_consistent"] = p.block_max == [max(max(r) for r in p.blocks[k]) for k in BLOCK_KEYS]
        if ref is not None and p.step_layers is not None:
            ref_layers = judge_matrix(ref).layers
            got = p.step_layers or []
            row["layer_acc"] = sum(a == b for a, b in zip(got, ref_layers)) / 64
            row["layer_whole_ok"] = got == ref_layers
    if group in SELF_CHECK_GROUPS and p.format_valid:
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
        if any("cells_acc" in r or "cells_acc_lenient" in r for r in rows):  # 只有要求输出 cells 的提示词（v1.3 起）才统计；没给按 0 计
            m["cells_acc"] = _mean([r.get("cells_acc", 0.0) for r in rows])
            m["cells_acc_lenient"] = _mean([r.get("cells_acc_lenient", 0.0) for r in rows])
        if any("cells_order_ok" in r for r in rows):  # 分母只算给齐 64 格的样本
            m["cells_order_violation_rate"] = _mean([not r["cells_order_ok"] for r in rows if "cells_order_ok" in r])
    if group in VERDICT_GROUPS:
        nc = [r for r in rows if not r["compliant"]]
        cp = [r for r in rows if r["compliant"]]
        m["verdict_acc"] = _mean([r["verdict_ok"] for r in rows])
        m["detection_rate"] = _mean([r["verdict"] == "不合规" for r in nc])
        m["false_alarm_rate"] = _mean([r["verdict"] == "不合规" for r in cp])
        m["unable_rate"] = _mean([r["verdict"] == "无法判断" for r in rows])
        m["first_step_acc"] = _mean([r["conclusion_ok"] for r in nc])
        m["first_coord_acc"] = _mean([r["coord_ok"] for r in nc])
        if any("block_acc" in r for r in rows):
            m["block_max_acc"] = _mean([r["block_acc"] for r in rows if "block_acc" in r])
            m["block_max_whole_acc"] = _mean([r["block_whole_ok"] for r in rows if "block_whole_ok" in r])
        if any("block_copy_acc" in r for r in rows):
            m["block_copy_acc"] = _mean([r["block_copy_acc"] for r in rows if "block_copy_acc" in r])
        if any("block_max_copy_consistent" in r for r in rows):
            m["block_max_copy_consistent_rate"] = _mean(
                [r["block_max_copy_consistent"] for r in rows if "block_max_copy_consistent" in r])
        if any("layer_acc" in r for r in rows):
            m["layer_label_acc"] = _mean([r["layer_acc"] for r in rows if "layer_acc" in r])
            m["layer_label_whole_acc"] = _mean([r["layer_whole_ok"] for r in rows if "layer_whole_ok" in r])
    if group in SELF_CHECK_GROUPS:
        m["self_consistency_rate"] = _mean([r["consistent"] for r in rows if "consistent" in r])
        m["acc_when_matrix_ok"] = _mean([r["conclusion_ok"] for r in rows if r["whole_ok"]])
        m["acc_when_matrix_wrong"] = _mean([r["conclusion_ok"] for r in rows if not r["whole_ok"]])
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
    gpu_apps = [r["gpu_apps_peak_mib"] for r in rows if r.get("gpu_apps_peak_mib") is not None]

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
        "gpu_apps_peak_mib": max(gpu_apps) if gpu_apps else None,
    }


def evaluate_run(run_dir: str | Path, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    group = meta["group"]
    samples = select_samples(load_dataset(meta["dataset"]["path"]), meta.get("limit"), meta.get("ids"))
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
    if group in SELF_CHECK_GROUPS:
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
    lines.append("格式无效和调用失败的样本都计入分母，判断类指标按答错处理（第 5 组第一阶段矩阵无效时，即使第二阶段给出了结论也按答错计）；"
                 "多次重复时显示均值 ± 标准差（百分点）。区块最大数、层级标注与模型所判断的那张矩阵比较"
                 "（第 4 组为标准矩阵，第 2、5 组为模型自己复原的矩阵）。")

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
        f"- 峰值显存：整卡 {_num(perf['gpu_peak_mib'], '{:.0f}')} MiB（含其他程序）；"
        f"计算进程 {_num(perf.get('gpu_apps_peak_mib'), '{:.0f}')} MiB（通常就是 vLLM，Windows 上可能读不到）",
    ]
    if metrics["invalid_examples"]:
        lines += ["", "## 格式无效示例（最多 10 条）", ""]
        for ex in metrics["invalid_examples"]:
            lines.append(f"- {ex['id']} 第{ex['repeat'] + 1}次：{'；'.join(ex['errors'])}")
    return "\n".join(lines) + "\n"


LOG_HEADER = (
    "| 日期 | 运行 ID | 提交 | 数据 | 提示词 | 组 | 模型 / 配置 | 重复 | 格式有效 | 判断准确 | 检出率 | 误报率 | 首违步 | 整矩阵 | 自洽 | 备注 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def render_log_row(meta: dict, metrics: dict) -> str:
    s = metrics["summary"]
    git = meta.get("git") or {}
    commit = (git.get("commit") or "")[:7] + ("*" if git.get("dirty") else "")

    def g(key):
        return _pct(s.get(key)) if key in s else "—"

    return (
        f"| {meta['created_at'][:10]} | {meta['run_id']} | {commit or '—'} | {meta['dataset'].get('version')} "
        f"| {meta.get('prompt_version', '—')} | {meta['group']} | {meta['model']} / {meta['profile']} | {metrics['repeats']} | {g('format_valid_rate')} "
        f"| {g('verdict_acc')} | {g('detection_rate')} | {g('false_alarm_rate')} | {g('first_step_acc')} "
        f"| {g('matrix_whole_acc')} | {g('self_consistency_rate')} | {meta.get('note') or ''} |"
    )
