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
    cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
    _, info = resolve_backend("openai", cfg, "qwen3-vl-2b-instruct", "greedy")
    assert info["repeats"] == 1 and info["params"]["temperature"] == 0.0
    _, info = resolve_backend("openai", cfg, "qwen3-vl-2b-thinking", "thinking")
    assert info["repeats"] == 3 and info["params"]["max_tokens"] == 16384
    with pytest.raises(ValueError):
        resolve_backend("openai", cfg, "qwen3-vl-2b-instruct", "thinking")


def test_preflight_check(server):
    ok = build_openai_backend({"base_url": server, "model": "qwen3-vl-2b-instruct"}, {"params": {}})
    ok.check()
    wrong = build_openai_backend({"base_url": server, "model": "qwen3-vl-2b-thinking"}, {"params": {}})
    with pytest.raises(RuntimeError, match="served-model-name"):
        wrong.check()
    down = build_openai_backend({"base_url": "http://127.0.0.1:9/v1", "model": "m"}, {"params": {}})
    with pytest.raises(RuntimeError, match="无法连接"):
        down.check()
