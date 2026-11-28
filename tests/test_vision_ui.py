"""视觉判断台（tools/vision_ui/server.py）的单元测试：不启动服务、不发网络请求。"""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "tools" / "vision_ui" / "server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("vision_ui_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses 需要能按 __module__ 找到它
    spec.loader.exec_module(module)
    return module


server = _load_server()

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def make_model(**kwargs) -> "server.Model":
    fields = {
        "id": "api:test",
        "name": "test-model",
        "source": "api",
        "base_url": "https://example.com/v1",
    }
    fields.update(kwargs)
    return server.Model(**fields)


@pytest.mark.parametrize(("data", "expected"), [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")])
def test_image_mime_accepts_supported_formats(data: bytes, expected: str) -> None:
    assert server.image_mime(b64(data)) == expected


@pytest.mark.parametrize("image", [None, "", "not-base64!!", b64(b"GIF89a" + b"\x00" * 8)])
def test_image_mime_rejects_bad_input(image) -> None:
    with pytest.raises(ValueError):
        server.image_mime(image)


def test_image_mime_rejects_oversize(monkeypatch) -> None:
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", len(PNG) - 1)
    with pytest.raises(ValueError, match="20 MB"):
        server.image_mime(b64(PNG))


def test_validate_prompt_trims_and_rejects_empty() -> None:
    assert server.validate_prompt({"prompt": "  判断顺序  "}) == "判断顺序"
    for body in ({}, {"prompt": ""}, {"prompt": "   "}, {"prompt": 123}):
        with pytest.raises(ValueError):
            server.validate_prompt(body)


def test_validate_prompt_rejects_too_long(monkeypatch) -> None:
    monkeypatch.setattr(server, "MAX_PROMPT_CHARS", 3)
    with pytest.raises(ValueError):
        server.validate_prompt({"prompt": "abcd"})


def test_prepare_chat_requires_image_for_vision_model() -> None:
    with pytest.raises(ValueError, match="请先选择一张图片"):
        server.prepare_chat(make_model(vision=True), {"prompt": "看图"})


def test_prepare_chat_rejects_image_for_text_model() -> None:
    with pytest.raises(ValueError, match="只接收文字"):
        server.prepare_chat(make_model(vision=False), {"prompt": "你好", "image": b64(PNG)})


def test_prepare_chat_text_only_model() -> None:
    payload = server.prepare_chat(make_model(), {"prompt": "你好"})
    assert payload["messages"] == [{"role": "user", "content": "你好"}]
    assert payload["max_tokens"] == 8192


def test_prepare_chat_api_vision_embeds_data_uri() -> None:
    payload = server.prepare_chat(make_model(vision=True), {"prompt": "看图", "image": b64(PNG)})
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0] == {"type": "text", "text": "看图"}
    assert message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["max_tokens"] == 8192


def test_prepare_chat_honours_token_field() -> None:
    payload = server.prepare_chat(
        make_model(vision=True, token_field="max_completion_tokens"),
        {"prompt": "看图", "image": b64(JPEG)},
    )
    assert payload["max_completion_tokens"] == 8192
    assert "max_tokens" not in payload


def test_prepare_chat_ollama_uses_images_field() -> None:
    model = make_model(source="ollama", vision=True, base_url="http://127.0.0.1:11434")
    payload = server.prepare_chat(model, {"prompt": "看图", "image": b64(WEBP)})
    message = payload["messages"][0]
    assert message["content"] == "看图"
    assert message["images"] == [b64(WEBP)]
    assert "max_tokens" not in payload


def test_api_key_prefers_environment(monkeypatch) -> None:
    monkeypatch.setenv("VISION_UI_TEST_KEY", "env-key")
    model = make_model(key_env="VISION_UI_TEST_KEY")
    assert server.api_key_for(model, {"api_key": "page-key"}) == "env-key"


def test_api_key_accepts_page_key(monkeypatch) -> None:
    monkeypatch.delenv("VISION_UI_TEST_KEY", raising=False)
    model = make_model(key_env="VISION_UI_TEST_KEY")
    assert server.api_key_for(model, {"api_key": "  page-key  "}) == "page-key"


def test_api_key_rejects_missing_or_invalid(monkeypatch) -> None:
    monkeypatch.delenv("VISION_UI_TEST_KEY", raising=False)
    model = make_model(key_env="VISION_UI_TEST_KEY")
    for body in ({}, {"api_key": ""}, {"api_key": "中文"}, {"api_key": "a" * 5000}):
        with pytest.raises(ValueError):
            server.api_key_for(model, body)


def test_api_key_local_model_ignores_page_key() -> None:
    assert server.api_key_for(make_model(source="vllm", key_env=""), {"api_key": "x"}) == ""


def test_configured_models_reads_aliases_and_vision() -> None:
    local, remote = server.configured_models()
    local_ids = {model.id for models in local.values() for model in models}
    remote_by_id = {model.id: model for model in remote}
    assert "vllm:qwen3-4b-instruct-2507-int4" in local_ids
    assert remote_by_id["api:deepseek-flash"].vision is True
    assert remote_by_id["api:qwen3-vl-plus"].vision is True
    assert remote_by_id["api:qwen3-vl-flash"].vision is True
    assert remote_by_id["api:deepseek"].vision is False
    assert remote_by_id["api:gpt"].token_field == "max_completion_tokens"


def test_model_public_reports_host() -> None:
    model = make_model(base_url="https://api.example.com/v1")
    public = model.public()
    assert public["id"] == "api:test"
    assert public["host"] == "api.example.com"
    assert public["vision"] is False


def test_resolve_model_returns_none_for_unknown_api_model(monkeypatch) -> None:
    monkeypatch.setattr(server, "configured_models", lambda: ({}, []))
    assert server.resolve_model("api:missing") is None


class HandlerStub:
    def __init__(self, host: str, origin: str | None = None):
        self.server = SimpleNamespace(server_port=8765)
        self.headers = {"Host": host}
        if origin is not None:
            self.headers["Origin"] = origin
        self.responses: list[tuple[int, dict]] = []

    def send_json(self, status: int, value: dict) -> None:
        self.responses.append((status, value))


def test_check_local_request_accepts_local_page() -> None:
    handler = HandlerStub("127.0.0.1:8765", "http://127.0.0.1:8765")
    assert server.VisionHandler.check_local_request(handler) is True
    assert handler.responses == []


def test_check_local_request_rejects_other_host() -> None:
    handler = HandlerStub("evil.example:8765")
    assert server.VisionHandler.check_local_request(handler) is False
    assert handler.responses[0][0] == 403


def test_check_local_request_rejects_foreign_origin() -> None:
    handler = HandlerStub("localhost:8765", "https://evil.example")
    assert server.VisionHandler.check_local_request(handler) is False
    assert handler.responses[0][0] == 403


def test_static_assets_exist() -> None:
    for name in ("index.html", "styles.css", "app.js"):
        assert (server.ASSET_DIR / name).is_file()
