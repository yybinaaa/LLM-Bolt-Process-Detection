import json
from pathlib import Path

import pytest

from boltcheck.backends import BaselineBackend, NoisyBackend
from boltcheck.dataset import load_dataset
from boltcheck.evaluate import evaluate_run, score_row
from boltcheck.grid import transpose
from boltcheck.parse import extract_json, parse_output
from boltcheck.prompts import GROUPS, build_prompt
from boltcheck.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v1.1"
SAMPLES = load_dataset(DATA)
COMPLIANT = next(s for s in SAMPLES if s["compliant"])
VIOLATING = next(s for s in SAMPLES if not s["compliant"])


def answer(matrix=None, verdict=None, step=None, coord=None):
    return json.dumps(
        {"order_matrix": matrix, "verdict": verdict, "first_violation_step": step,
         "first_violation_coordinate": coord, "reason": ""},
        ensure_ascii=False,
    )


def test_extract_json_variants():
    assert extract_json('<think>先想想 {"a": 1}</think>\n{"b": 2}')[0] == {"b": 2}
    assert extract_json('推理过程……</think>```json\n{"c": 3}\n```')[0] == {"c": 3}
    assert extract_json('结果如下：{"d": 4} 以上')[0] == {"d": 4}
    assert extract_json('{"order_matrix": [[1, 2')[0] is None


def test_parse_format_rules():
    m = VIOLATING["order_matrix"]
    ok = parse_output(answer(m, "不合规", 5, [40, 40]), "joint")
    assert ok.format_valid
    bad_perm = [row[:] for row in m]
    bad_perm[0][0] = bad_perm[0][1]
    assert not parse_output(answer(bad_perm, "合规"), "joint").format_valid
    assert not parse_output(answer(m, "基本合规"), "joint").format_valid
    assert not parse_output(answer(m, "不合规", None), "joint").format_valid  # 不合规必须给步骤
    assert parse_output(answer(None, "合规"), "judge").format_valid
    assert parse_output(json.dumps({"order_matrix": m}), "restore").format_valid
    assert parse_output(answer(m, "不合规", "12", "(40,40)"), "joint").first_coord == [40, 40]


def test_attribution_categories():
    s = VIOLATING
    truth = s["order_matrix"]
    good = score_row(s, {"text": answer(truth, s["verdict"], s["first_violation_step"])}, "joint")
    assert good["attribution"] == "full_correct"
    reasoning = score_row(s, {"text": answer(truth, "合规")}, "joint")
    assert reasoning["attribution"] == "reasoning_error"
    # 转置矩阵 + 按转置矩阵自己的结论回答 -> 还原错误、推理成立
    from boltcheck.rules import judge_matrix
    tm = transpose(truth)
    own = judge_matrix(tm)
    restore = score_row(s, {"text": answer(tm, own.verdict, own.first_violation_step)}, "joint")
    assert restore["transposed"] and restore["attribution"] == "restore_error"
    wrong_both = score_row(s, {"text": answer(tm, "合规" if own.verdict == "不合规" else "不合规", 7)}, "joint")
    assert wrong_both["attribution"] == "both_error"
    missing = score_row(s, None, "joint")
    assert not missing["format_valid"] and missing["api_error"]


@pytest.mark.parametrize("group", list(GROUPS))
def test_prompts_contain_no_answers(group):
    p = build_prompt(group, VIOLATING)
    assert "不合规\"" not in p and str(VIOLATING["first_violation_coordinate"]) not in p
    if group != "judge":
        assert "(45,45)" in p and '"(10,10)": ?' in p and '"(80,80)": ?' in p


@pytest.mark.parametrize("group", list(GROUPS))
def test_baseline_run_scores_perfectly(tmp_path, group):
    info = {"backend": "baseline", "model": "rule-baseline", "profile": "-", "repeats": 1, "params": {}}
    run_dir = run_experiment(str(DATA), group, BaselineBackend(), info, out_root=str(tmp_path), log=lambda *_: None)
    m = evaluate_run(run_dir)
    for key, v in m["summary"].items():
        if key in ("n", "api_error_rate", "truncated_rate", "matrix_transposed_rate", "false_alarm_rate", "unable_rate"):
            continue
        if v is None:  # 例如"复原错误时的判断准确率"：基线没有复原错误的样本
            continue
        assert v["mean"] == 1.0, key
    assert (run_dir / "report.md").exists() and (run_dir / "log_row.md").exists()


def test_resume_only_reruns_missing(tmp_path):
    info = {"backend": "noisy", "model": "noisy", "profile": "-", "repeats": 1, "params": {}}
    run_dir = run_experiment(str(DATA), "joint", NoisyBackend(), info, out_root=str(tmp_path), limit=20, log=lambda *_: None)
    out = run_dir / "outputs.jsonl"
    lines = out.read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[3])
    broken["error"] = "模拟超时"
    lines[3] = json.dumps(broken, ensure_ascii=False)
    out.write_text("\n".join(lines[:10]) + "\n", encoding="utf-8")  # 只保留 10 条，其中 1 条出错
    calls = []

    class Counting(NoisyBackend):
        def generate(self, prompt, sample, group, repeat):
            calls.append(sample["id"])
            return super().generate(prompt, sample, group, repeat)

    run_experiment(str(DATA), "joint", Counting(), info, limit=20, resume_dir=str(run_dir), log=lambda *_: None)
    assert len(calls) == 11
    assert evaluate_run(run_dir)["n_outputs_found"] == 20


def test_pipeline_two_stage_calls_and_metrics(tmp_path):
    """第 5 组：每条样本两次调用；第二阶段拿到的是第一阶段复原的矩阵；第一阶段失败时不调用第二阶段。"""
    calls = []

    class Recording(NoisyBackend):
        def generate(self, prompt, sample, group, repeat):
            calls.append((sample["id"], group, prompt))
            return super().generate(prompt, sample, group, repeat)

    info = {"backend": "noisy", "model": "noisy", "profile": "-", "repeats": 1, "params": {}}
    run_dir = run_experiment(str(DATA), "pipeline", Recording(), info, out_root=str(tmp_path), limit=40, log=lambda *_: None)
    recs = [json.loads(l) for l in (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 40
    for rec in recs:
        stages = [c for c in calls if c[0] == rec["sample_id"]]
        assert stages[0][1] == "restore" and "操作序列" in stages[0][2]
        if rec["stage2"] is None:
            assert len(stages) == 1  # 第一阶段没得到矩阵，不做判断
        else:
            assert len(stages) == 2 and stages[1][1] == "judge"
            m1 = parse_output(rec["stage1"]["text"], "restore").matrix
            assert all(str(row)[1:-1] in stages[1][2] for row in m1)  # 判断提示词里是模型自己复原的矩阵
            assert "操作序列" not in stages[1][2]
    assert any(r["stage2"] is None for r in recs)
    m = evaluate_run(run_dir)
    s = m["summary"]
    for key in ("matrix_whole_acc", "verdict_acc", "self_consistency_rate", "acc_when_matrix_ok", "acc_when_matrix_wrong"):
        assert s[key] is not None, key
    assert sum(m["attribution"].values()) > 0
    data_view = (run_dir / "view.html").read_text(encoding="utf-8")
    assert "第 5 组" in data_view


def test_prompt_v14_content():
    from boltcheck.prompts import PROMPT_VERSION, build_judge_prompt, cells_template

    assert PROMPT_VERSION == "v1.4"
    s = VIOLATING
    p = build_prompt("restore", s)
    x, y = s["coords"][0]
    assert f"({x},{y}) → 第1步" in p and "抄下它右边的步数" in p
    assert cells_template() in p and "order_matrix" not in p  # 只填 64 格，矩阵由程序排
    lines = cells_template().splitlines()
    assert lines[1].strip().startswith('"(10,10)": ?') and lines[8].strip().startswith('"(10,80)": ?')  # 每行一排
    assert cells_template().count(": ?") == 64
    j = build_judge_prompt(s["order_matrix"])
    assert "中心 2×2" in j and j.index("block_max：") < j.index("verdict：")
    jt = build_prompt("joint", s)
    assert jt.index('"(80,80)": ?') < jt.index("block_max：") < jt.index("verdict：")


def test_cells_build_matrix_and_lenient_extraction():
    s = VIOLATING
    m = s["order_matrix"]
    cells = {f"({10 * (c + 1)},{10 * (r + 1)})": m[r][c] for r in range(8) for c in range(8)}
    p = parse_output(json.dumps({"cells": cells}), "restore")
    assert p.format_valid and p.matrix == m  # 程序按坐标排成矩阵
    # v1.3 试跑里的写法：cells 用了方括号，JSON 无效；宽松提取仍能统计内容
    broken = '{"cells": [\n' + ",\n".join(f'  "{k[1:-1]}": {v}' for k, v in cells.items()) + "\n]}"
    p = parse_output(broken, "restore")
    assert not p.format_valid and p.cells_lenient and len(p.cells_lenient) == 64
    row = score_row(s, {"text": broken}, "restore")
    assert row["cells_acc_lenient"] == 1.0 and row.get("cells_acc", 0.0) == 0.0  # 严格口径按 0 计
    # 缺格子时给出明确错误
    del cells["(80,80)"]
    p = parse_output(json.dumps({"cells": cells}), "restore")
    assert not p.format_valid and "缺少 1 个格子" in "".join(p.errors)


def test_extract_json_keeps_outer_object():
    obj, _ = extract_json('{"cells": {"(10,10)": 1}, "order_matrix": [[1]]}')
    assert set(obj) == {"cells", "order_matrix"}


def test_cells_parse_and_metric():
    s = VIOLATING
    m = s["order_matrix"]
    cells = {f"({10 * (c + 1)},{10 * (r + 1)})": m[r][c] for r in range(8) for c in range(8)}
    text = json.dumps({"cells": cells, "order_matrix": m})
    row = score_row(s, {"text": text}, "restore")
    assert row["cells_acc"] == 1.0 and row["whole_ok"]
    cells["(10,10)"] = 0
    row = score_row(s, {"text": json.dumps({"cells": cells, "order_matrix": m})}, "restore")
    assert row["cells_acc"] == 63 / 64


def test_step_layers_still_scored_for_old_runs():
    """v1.2 运行重新评测时仍能统计层级标注；长度不对的 step_layers 视为没给，不统计。"""
    s = VIOLATING
    obj = json.loads(answer(None, s["verdict"], s["first_violation_step"]))
    obj["step_layers"] = s["layers"]
    row = score_row(s, {"text": json.dumps(obj, ensure_ascii=False)}, "judge")
    assert row["layer_acc"] == 1.0 and row["layer_whole_ok"]
    obj["step_layers"] = s["layers"] + [4] * 40  # 试跑中出现的"一长串 4"
    assert parse_output(json.dumps(obj), "judge").step_layers is None
    row = score_row(s, {"text": json.dumps(obj, ensure_ascii=False)}, "judge")
    assert "layer_acc" not in row and row["format_valid"]


def test_block_max_metric(tmp_path):
    info = {"backend": "noisy", "model": "noisy", "profile": "-", "repeats": 1, "params": {}}
    run_dir = run_experiment(str(DATA), "judge", NoisyBackend(), info, out_root=str(tmp_path), limit=60, log=lambda *_: None)
    s = evaluate_run(run_dir)["summary"]
    assert 0.8 < s["block_max_acc"]["mean"] < 1.0
    assert s["block_max_whole_acc"]["mean"] < 1.0
    assert "layer_label_acc" not in s


def test_invalid_pipeline_output_counts_as_wrong():
    """第一阶段矩阵无效时，即使第二阶段的结论碰巧与标准答案一致，也按答错计（2026-11-15 v1.2 试跑发现的口径问题）。"""
    s = VIOLATING
    bad = [row[:] for row in s["order_matrix"]]
    bad[0][0] = bad[0][1]  # 有重复，不是 1～64 的排列
    out = {
        "stage1": {"text": json.dumps({"order_matrix": bad})},
        "stage2": {"text": answer(None, s["verdict"], s["first_violation_step"], s["first_violation_coordinate"])},
        "error": None,
    }
    row = score_row(s, out, "pipeline")
    assert not row["format_valid"]
    assert row["verdict"] is None and not row["verdict_ok"] and not row["conclusion_ok"] and not row["coord_ok"]


def test_gpu_monitor_peaks():
    from boltcheck.runner import GpuMonitor

    m = GpuMonitor()
    m.samples = [(1.0, 6000), (2.0, 6600), (3.0, 6100)]
    m.app_samples = [(1.0, 4300), (2.0, 4700)]
    assert m.peak_between(0.5, 2.5) == 6600
    assert m.peak_between(0.5, 2.5, apps=True) == 4700
