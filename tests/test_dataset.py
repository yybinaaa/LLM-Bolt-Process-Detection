import json
from pathlib import Path

from boltcheck.dataset import (
    check_quotas, dumps_jsonl, generate, load_config, sha256_text, summarize, verify,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "dataset_v1.1.toml"
DATA = ROOT / "data" / "v1.1"


def test_generation_meets_quotas_and_is_deterministic():
    cfg = load_config(CFG)
    a = generate(cfg)
    b = generate(cfg)
    assert dumps_jsonl(a) == dumps_jsonl(b)
    assert check_quotas(a, cfg) == []
    s = summarize(a)
    assert s["compliant"] == 100 and s["noncompliant"] == 100
    assert s["construction"] == {"block": 33, "insert": 33, "swap": 34}


def test_committed_dataset_is_reproducible():
    """仓库中的 data/v1.1 必须能由配置原样复现（不同 Python 版本也一样）。"""
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert sha256_text(dumps_jsonl(generate(load_config(CFG)))) == manifest["sha256"]
    assert verify(DATA) == []


def test_cross_layer_samples_are_front():
    for r in generate(load_config(CFG)):
        if r["groups"]["layer_gap"] == "2-3":
            assert r["groups"]["position"] == "front"
