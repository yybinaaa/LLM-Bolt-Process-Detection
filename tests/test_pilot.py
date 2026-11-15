"""自动试跑清单（configs/pilot.toml）与按样本 ID 选样。"""
from pathlib import Path

import pytest

from boltcheck.dataset import load_dataset, select_samples
from boltcheck.runner import load_pilot, run_pilot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v1.1"
SAMPLES = load_dataset(DATA)


def test_repo_pilot_config_is_balanced():
    cfg = load_pilot(ROOT / "configs" / "pilot.toml")
    chosen = select_samples(SAMPLES, ids=cfg["ids"])
    assert [s["id"] for s in chosen] == cfg["ids"]
    assert [s["id"] for s in chosen[:5]] == [s["id"] for s in SAMPLES[:5]]  # 与此前 --limit 5 的试跑可比
    assert sum(s["compliant"] for s in chosen) == sum(not s["compliant"] for s in chosen) == 4
    assert {s["construction"] for s in chosen if not s["compliant"]} == {"swap", "insert", "block"}
    assert {r["group"] for r in cfg["runs"]} >= {"judge", "pipeline"}


def test_select_samples_by_ids():
    ids = ["v1.1-0019", "v1.1-0002"]
    assert [s["id"] for s in select_samples(SAMPLES, ids=ids)] == ["v1.1-0002", "v1.1-0019"]  # 保持数据集顺序
    with pytest.raises(ValueError):
        select_samples(SAMPLES, ids=["v1.1-9999"])


def test_run_pilot_runs_each_entry_and_survives_failures(tmp_path):
    cfg = tmp_path / "pilot.toml"
    cfg.write_text(
        'note = "测试"\nids = ["v1.1-0001", "v1.1-0005", "v1.1-0008"]\n'
        '[[runs]]\ngroup = "judge"\nbackend = "noisy"\npurpose = "判断"\n'
        '[[runs]]\ngroup = "pipeline"\nbackend = "openai"\nmodel = "qwen3-4b-instruct-2507-int4"\nprofile = "no-such-profile"\n'
        '[[runs]]\ngroup = "pipeline"\nbackend = "baseline"\n',
        encoding="utf-8",
    )
    results = run_pilot(cfg, str(ROOT / "configs" / "models.toml"), str(DATA), str(tmp_path / "runs"),
                        skip_check=True, log=lambda *_: None)
    assert [("error" in r) for r in results] == [False, True, False]
    import json
    for r in (results[0], results[2]):
        meta = json.loads(Path(r["run_dir"], "meta.json").read_text(encoding="utf-8"))
        metrics = json.loads(Path(r["run_dir"], "metrics.json").read_text(encoding="utf-8"))
        assert meta["ids"] == ["v1.1-0001", "v1.1-0005", "v1.1-0008"] and metrics["summary"]["n"] == 3
        assert "测试" in meta["note"] and "|" in r["log_row"]
    assert "no-such-profile" in results[1]["error"]
