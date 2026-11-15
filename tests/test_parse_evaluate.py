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
        assert "(45,45)" in p and "第 r 行对应 y = 10r" in p


@pytest.mark.parametrize("group", list(GROUPS))
def test_baseline_run_scores_perfectly(tmp_path, group):
    info = {"backend": "baseline", "model": "rule-baseline", "profile": "-", "repeats": 1, "params": {}}
    run_dir = run_experiment(str(DATA), group, BaselineBackend(), info, out_root=str(tmp_path), log=lambda *_: None)
    m = evaluate_run(run_dir)
    for key, v in m["summary"].items():
        if key in ("n", "api_error_rate", "truncated_rate", "matrix_transposed_rate", "false_alarm_rate", "unable_rate"):
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
