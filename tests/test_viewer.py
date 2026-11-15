import json
import re
from pathlib import Path

from boltcheck.backends import NoisyBackend
from boltcheck.dataset import load_dataset
from boltcheck.evaluate import evaluate_run
from boltcheck.runner import run_experiment
from boltcheck.viewer import dataset_view_data, describe_sample, render_html, run_view_data, text_grid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v1.1"


def embedded(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_dataset_view_contains_all_samples():
    html = render_html(dataset_view_data(DATA))
    data = embedded(html)
    assert data["mode"] == "dataset"
    assert len(data["items"]) == 200
    item = next(i for i in data["items"] if not i["compliant"])
    assert item["first"] == min(item["viol"])
    assert "</script>" not in html.split("const DATA = ")[1].split(";\n")[0]


def test_dataset_view_filter_by_ids():
    data = dataset_view_data(DATA, ["v1.1-0005", "v1.1-0008"])
    assert [i["id"] for i in data["items"]] == ["v1.1-0005", "v1.1-0008"]


def test_run_view_marks_model_output(tmp_path):
    info = {"backend": "noisy", "model": "noisy", "profile": "-", "repeats": 1, "params": {}}
    run_dir = run_experiment(str(DATA), "joint", NoisyBackend(), info, out_root=str(tmp_path), limit=30, log=lambda *_: None)
    evaluate_run(run_dir)
    assert (run_dir / "view.html").exists()
    data = run_view_data(run_dir)
    assert data["mode"] == "run" and len(data["items"]) == 30
    reps = [i["model"][0] for i in data["items"]]
    assert any(r["m"] is None for r in reps)  # 模拟的格式错误
    assert any(r["wholeOk"] for r in reps)


def test_text_grid_layout_and_marks():
    s = next(x for x in load_dataset(DATA) if not x["compliant"])
    lines = text_grid(s["order_matrix"], [s["first_violation_step"]], s["first_violation_step"]).splitlines()
    assert len(lines) == 9
    assert lines[0].split()[-8:] == [str(10 * c) for c in range(1, 9)]
    assert f"{s['first_violation_step']}!" in "".join(lines)
    truth = s["order_matrix"]
    wrong = [row[:] for row in truth]
    wrong[0][0], wrong[0][1] = wrong[0][1], wrong[0][0]
    assert text_grid(wrong, truth=truth).count("×") == 2
    assert "首次违规" in describe_sample(s)
