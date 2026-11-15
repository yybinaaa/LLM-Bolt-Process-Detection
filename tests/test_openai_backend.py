"""用本地假服务器检查 OpenAI 兼容接口的请求体和响应解析，不需要 GPU 或网络。"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from boltcheck.backends import build_openai_backend
from boltcheck.runner import resolve_backend


class FakeVLLM(BaseHTTPRequestHandler):
    requests: list = []

    def do_GET(self):
        data = json.dumps({"data": [{"id": "qwen3-vl-2b-instruct"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        resp = {
            "choices": [{
                "message": {"content": '{"verdict": "合规"}', "reasoning_content": "想了一下"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 900, "completion_tokens": 30},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server(monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    srv = HTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeVLLM.requests = []
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_request_body_and_response(server, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    model_cfg = {"base_url": server, "model": "qwen3-vl-2b-instruct", "api_key_env": "TEST_KEY", "drop_params": ["top_k"]}
    profile = {"seed": 7, "params": {"temperature": 0.6, "top_k": 20, "max_tokens": 100}}
    be = build_openai_backend(model_cfg, profile)
    gen = be.generate("提示词", {"id": "x"}, "judge", repeat=2)
    req = FakeVLLM.requests[-1]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == "qwen3-vl-2b-instruct"
    assert req["body"]["messages"] == [{"role": "user", "content": "提示词"}]
    assert req["body"]["temperature"] == 0.6 and "top_k" not in req["body"]
    assert req["body"]["seed"] == 9
    assert gen.text == '{"verdict": "合规"}' and gen.reasoning == "想了一下"
    assert gen.prompt_tokens == 900 and gen.completion_tokens == 30 and gen.error is None


def test_connection_error_is_recorded(monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(k, raising=False)
    be = build_openai_backend({"base_url": "http://127.0.0.1:9/v1", "model": "m", "retries": 0, "timeout_s": 2}, {"params": {}})
    gen = be.generate("p", {"id": "x"}, "joint", 0)
    assert gen.error and gen.text == ""


def test_models_config_profiles():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
    _, info = resolve_backend("openai", cfg_path, "qwen3-4b-instruct-2507-int4", "greedy")
    assert info["repeats"] == 1 and info["params"]["temperature"] == 0.0
    assert info["params"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert info["hf_repo"] and info["quantization"]
    _, info = resolve_backend("openai", cfg_path, "qwen3-1.7b-int8", "thinking")
    assert info["repeats"] == 3 and info["params"]["max_tokens"] >= 16384
    assert info["params"]["chat_template_kwargs"] == {"enable_thinking": True}
    with pytest.raises(ValueError):
        resolve_backend("openai", cfg_path, "qwen3-4b-instruct-2507-int4", "thinking")


def test_models_config_is_consistent():
    """每个模型声明的配置都存在；本地模型都写明了权重来源和量化方式。"""
    from boltcheck.dataset import load_config

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml"))
    for name, m in cfg["models"].items():
        for prof in m.get("profiles", []):
            assert prof in cfg["profiles"], (name, prof)
        if "localhost" in m["base_url"]:
            assert m.get("hf_repo") and m.get("quantization"), name
    for name, prof in cfg["profiles"].items():
        if prof["params"].get("chat_template_kwargs", {}).get("enable_thinking"):
            assert prof["params"]["max_tokens"] >= 16384, name


def test_preflight_check(server):
    ok = build_openai_backend({"base_url": server, "model": "qwen3-vl-2b-instruct"}, {"params": {}})
    ok.check()
    wrong = build_openai_backend({"base_url": server, "model": "qwen3-vl-2b-thinking"}, {"params": {}})
    with pytest.raises(RuntimeError, match="served-model-name"):
        wrong.check()
    down = build_openai_backend({"base_url": "http://127.0.0.1:9/v1", "model": "m"}, {"params": {}})
    with pytest.raises(RuntimeError, match="无法连接"):
        down.check()


def test_running_guide_matches_models_config():
    """运行指南里每个 vllm serve 命令的 --served-model-name 都要能在 models.toml 里找到，反之亦然。"""
    import re

    from boltcheck.dataset import load_config

    root = os.path.join(os.path.dirname(__file__), "..")
    guide = open(os.path.join(root, "docs", "running.md"), encoding="utf-8").read()
    served = set(re.findall(r"--served-model-name\s+(\S+)", guide))
    cfg = load_config(os.path.join(root, "configs", "models.toml"))
    local = {m["model"] for m in cfg["models"].values() if "localhost" in m["base_url"]}
    assert served == local


def test_structured_output_sends_schema(server):
    """greedy-json：按调用的组附上 JSON Schema；restore 的 cells 要求正好 64 个格子、按格子顺序。"""
    from boltcheck.prompts import response_schema

    be = build_openai_backend({"base_url": server, "model": "m"}, {"structured_output": True, "params": {"temperature": 0.0}})
    be.generate("p", {"id": "x"}, "restore", 0)
    rf = FakeVLLM.requests[-1]["body"]["response_format"]
    assert rf["type"] == "json_schema"
    cells = rf["json_schema"]["schema"]["properties"]["cells"]
    assert len(cells["required"]) == 64 and cells["required"][:2] == ["(10,10)", "(20,10)"]
    be.generate("p", {"id": "x"}, "judge", 0)
    judge = FakeVLLM.requests[-1]["body"]["response_format"]["json_schema"]["schema"]
    assert list(judge["properties"])[:3] == ["blocks", "block_max", "verdict"]  # 提示词 v1.5：先抄区块
    center6 = judge["properties"]["blocks"]["properties"]["center6"]
    assert center6["minItems"] == center6["maxItems"] == 6 and center6["items"]["maxItems"] == 6
    assert "structured_output" not in FakeVLLM.requests[-1]["body"]
    assert set(response_schema("joint")["properties"]) >= {"cells", "block_max", "verdict"}
    plain = build_openai_backend({"base_url": server, "model": "m"}, {"params": {}})
    plain.generate("p", {"id": "x"}, "restore", 0)
    assert "response_format" not in FakeVLLM.requests[-1]["body"]
