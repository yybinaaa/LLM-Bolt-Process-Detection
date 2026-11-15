"""自动试跑清单（configs/pilot.toml）与按样本 ID 选样。"""
from pathlib import Path

import pytest

from boltcheck.dataset import load_dataset, select_samples
from boltcheck.runner import load_pilot, run_pilot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v1.1"
SAMPLES = load_dataset(DATA)


def test_ten_sample_lists_share_ids():
    """10 条清单（本地、大模型、自动试跑）用同一组样本，便于逐条对比。"""
    ids = {tuple(load_pilot(ROOT / "configs" / p)["ids"]) for p in ("pilot.toml", "runs/qwen3-4b-10.toml", "runs/api-10.toml")}
    assert len(ids) == 1 and len(next(iter(ids))) == 10


def test_repo_pilot_config_is_balanced():
    cfg = load_pilot(ROOT / "configs" / "pilot.toml")
    chosen = select_samples(SAMPLES, ids=cfg["ids"])
    assert [s["id"] for s in chosen] == cfg["ids"]
    assert [s["id"] for s in chosen[:8]] == [s["id"] for s in SAMPLES[:8]]  # 数据集前 8 条，前 5 条与此前 --limit 5 的试跑相同
    assert sum(s["compliant"] for s in chosen) == sum(not s["compliant"] for s in chosen) == 5
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


@pytest.mark.parametrize("cfg_path", sorted((ROOT / "configs" / "runs").glob("*.toml")) + [ROOT / "configs" / "pilot.toml"])
def test_run_lists_reference_valid_models(cfg_path, monkeypatch):
    """各运行清单里的模型、配置都能在 configs/models.toml 中找到并组合使用（密钥用占位值）。"""
    from boltcheck.runner import resolve_backend

    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "ZAI_API_KEY"):
        monkeypatch.setenv(env, "test-key")
    cfg = load_pilot(cfg_path)
    if cfg.get("ids"):
        select_samples(SAMPLES, ids=cfg["ids"])
    for r in cfg["runs"]:
        resolve_backend(r.get("backend", "openai"), str(ROOT / "configs" / "models.toml"), r.get("model", cfg.get("model")), r["profile"])


def test_api_models_send_clean_params(monkeypatch):
    """远程 API：去掉 vLLM 扩展字段和 seed、不设温度；OpenAI 把 max_tokens 改名为 max_completion_tokens。"""
    from boltcheck.runner import resolve_backend

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = str(ROOT / "configs" / "models.toml")
    gpt, _ = resolve_backend("openai", cfg, "gpt", "api")
    assert gpt.params == {"max_completion_tokens": 32768} and gpt.seed is None and not gpt.local
    claude, info = resolve_backend("openai", cfg, "claude-opus", "api")
    assert claude.params == {"max_tokens": 32768} and claude.seed is None and info["served_model"] == "claude-opus-5-5"
    monkeypatch.setenv("ZAI_API_KEY", "k")
    glm, _ = resolve_backend("openai", cfg, "glm", "api-nothink")
    assert glm.params == {"max_tokens": 8192, "thinking": {"type": "disabled"}} and glm.seed is None
    local, _ = resolve_backend("openai", cfg, "qwen3-4b-instruct-2507-int4", "greedy")
    assert local.local and local.seed == 1234 and "chat_template_kwargs" in local.params
