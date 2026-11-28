"""Serve the workbench and route submissions to the selected model."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the project's tomli dependency.
    import tomli as tomllib


DEFAULT_MODEL_NAME = "qwen2.5vl:3b"
DEFAULT_MODEL_ID = f"ollama:{DEFAULT_MODEL_NAME}"
OLLAMA_URL = "http://127.0.0.1:11434"
VLLM_URL = "http://localhost:8000/v1"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
MODEL_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.toml"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PROMPT_CHARS = 16_000
MAX_KEY_CHARS = 4096
MAX_BODY_BYTES = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + MAX_PROMPT_CHARS * 12 + MAX_KEY_CHARS + 1024
ASSET_DIR = Path(__file__).resolve().parent / "public"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    source: str
    base_url: str
    vision: bool = False
    status: str = "ready"
    key_env: str = ""
    token_field: str = "max_tokens"

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "vision": self.vision,
            "status": self.status,
            "host": urlsplit(self.base_url).hostname,
            "key_env": self.key_env,
        }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def model_request(url: str, payload: dict | None = None, timeout: int = 3, key: str = "") -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(url, data=data, headers=headers)
    proxy = ProxyHandler({}) if urlsplit(url).hostname in LOCAL_HOSTS else ProxyHandler()
    with build_opener(proxy, NoRedirect()).open(request, timeout=timeout) as response:
        return json.load(response)


def ollama_models(selected_id: str | None = None) -> list[Model]:
    try:
        tags = model_request(OLLAMA_URL + "/api/tags").get("models", [])
    except (OSError, ValueError):
        return []
    models = []
    for tag in tags:
        name = tag.get("name")
        if not name or tag.get("remote_host") or tag.get("remote_model") or name.endswith(("-cloud", ":cloud")):
            continue
        model_id = f"ollama:{name}"
        if selected_id is not None and model_id != selected_id:
            continue
        try:
            details = model_request(OLLAMA_URL + "/api/show", {"model": name})
        except (OSError, ValueError):
            models.append(Model(model_id, name, "ollama", OLLAMA_URL, status="offline"))
            continue
        if details.get("remote_host") or details.get("remote_model"):
            continue
        models.append(Model(model_id, name, "ollama", OLLAMA_URL, vision="vision" in details.get("capabilities", [])))
    return models


def configured_models() -> tuple[dict[str, list[Model]], list[Model]]:
    with MODEL_CONFIG.open("rb") as source:
        configs = tomllib.load(source).get("models", {})
    local: dict[str, list[Model]] = {VLLM_URL: []}
    remote = []
    for alias, config in configs.items():
        base_url = config["base_url"].rstrip("/")
        url = urlsplit(base_url)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            continue
        name = config["model"]
        vision = config.get("vision") is True
        key_env = config.get("api_key_env", "")
        if url.hostname in LOCAL_HOSTS and url.scheme in ("http", "https"):
            local.setdefault(base_url, []).append(Model(f"vllm:{alias}", name, "vllm", base_url, vision, key_env=key_env))
        elif url.scheme == "https":
            status = "key" if key_env and not os.environ.get(key_env) else "ready"
            token_field = config.get("rename_params", {}).get("max_tokens", "max_tokens")
            remote.append(Model(f"api:{alias}", name, "api", base_url, vision, status, key_env, token_field))
    return local, remote


def vllm_models(configured: dict[str, list[Model]], selected_id: str | None = None) -> list[Model]:
    models = []
    for url, entries in configured.items():
        key = next((os.environ[model.key_env] for model in entries if os.environ.get(model.key_env)), "")
        try:
            served = {item["id"] for item in model_request(url + "/models", key=key).get("data", [])}
        except (OSError, ValueError):
            served = None
        for entry in entries:
            if selected_id is not None and entry.id != selected_id:
                continue
            status = "offline" if served is None else "ready" if entry.name in served else "inactive"
            models.append(Model(entry.id, entry.name, entry.source, url, entry.vision, status, entry.key_env))
        if url != VLLM_URL or served is None:
            continue
        for name in sorted(served - {entry.name for entry in entries}):
            model_id = f"vllm:served:{name}"
            if selected_id is None or selected_id == model_id:
                models.append(Model(model_id, name, "vllm", url))
    return models


def resolve_model(model_id: str) -> Model | None:
    if model_id.startswith("ollama:"):
        return next(iter(ollama_models(model_id)), None)
    local, remote = configured_models()
    models = vllm_models(local, model_id) if model_id.startswith("vllm:") else remote
    return next((model for model in models if model.id == model_id), None)


def validate_prompt(body: dict) -> str:
    if not isinstance(body, dict):
        raise ValueError("请求格式无效。")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("请输入提示词。")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"提示词最多 {MAX_PROMPT_CHARS:,} 字符。")
    return prompt.strip()


def image_mime(image: str) -> str:
    if not isinstance(image, str) or not image:
        raise ValueError("请先选择一张图片。")
    try:
        image_bytes = base64.b64decode(image, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("图片数据无效，请重新选择。") from error
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("图片大小请控制在 20 MB 以内。")
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("请选择 JPG、PNG 或 WebP 图片。")


def prepare_chat(model: Model, body: dict) -> dict:
    prompt = validate_prompt(body)
    image = body.get("image")
    if image is not None and not model.vision:
        raise ValueError("所选模型只接收文字。")
    message = {"role": "user", "content": prompt}
    payload = {"model": model.name, "messages": [message], "stream": False}
    if image is not None or model.vision:
        mime = image_mime(image)
        if model.source == "ollama":
            message["images"] = [image]
        else:
            message["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
            ]
    if model.source == "api":
        payload[model.token_field] = 8192
    return payload


def api_key_for(model: Model, body: dict) -> str:
    saved_key = os.environ.get(model.key_env, "")
    if model.source != "api" or not model.key_env or saved_key:
        return saved_key
    key = body.get("api_key", "")
    if not isinstance(key, str):
        raise ValueError("请输入 API 密钥。")
    key = key.strip()
    if not key or len(key) > MAX_KEY_CHARS or any(not 33 <= ord(char) <= 126 for char in key):
        raise ValueError("请输入有效的 API 密钥。")
    return key


def generate(model: Model, payload: dict, key: str) -> tuple[str, str]:
    path = "/api/chat" if model.source == "ollama" else "/chat/completions"
    data = model_request(model.base_url + path, payload, timeout=300, key=key)
    if model.source == "ollama":
        message = data.get("message", {})
    else:
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
    answer = message.get("content") or ""
    return answer.strip(), data.get("model") or model.name


class VisionHandler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        self.request.settimeout(20)
        super().setup()

    def send_body(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status: int, value: dict) -> None:
        self.send_body(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def check_local_request(self) -> bool:
        port = self.server.server_port
        host = self.headers.get("Host")
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self.send_json(403, {"error": "请从本机地址打开页面。"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            self.send_json(403, {"error": "请在视觉判断台页面内提交。"})
            return False
        return True

    def do_GET(self) -> None:
        if not self.check_local_request():
            return
        if self.path == "/api/models":
            try:
                local, remote = configured_models()
                models = ollama_models() + vllm_models(local) + remote
            except (OSError, ValueError, KeyError):
                self.send_json(500, {"error": "无法读取 configs/models.toml，请检查模型配置。"})
            else:
                self.send_json(200, {"default": DEFAULT_MODEL_ID, "models": [model.public() for model in models]})
            return
        if self.path == "/api/status":
            # Pages already open during an upgrade can still use the original default.
            model = resolve_model(DEFAULT_MODEL_ID)
            self.send_json(200, {"model": DEFAULT_MODEL_NAME, "status": model.status if model else "missing"})
            return
        asset = ASSETS.get(self.path)
        if asset is None:
            self.send_json(404, {"error": "页面不存在。"})
            return
        filename, content_type = asset
        self.send_body(200, (ASSET_DIR / filename).read_bytes(), content_type)

    def do_POST(self) -> None:
        if not self.check_local_request():
            return
        if self.path != "/api/judge":
            self.send_json(404, {"error": "接口不存在。"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "请选择图片和提示词后重试。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("请求太大或为空，请检查图片和提示词。")
            body = json.loads(self.rfile.read(length))
            validate_prompt(body)
            model_id = body.get("model", DEFAULT_MODEL_ID)
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("请选择模型。")
        except (UnicodeError, json.JSONDecodeError, TimeoutError):
            self.send_json(400, {"error": "请求读取失败，请重新提交。"})
            return
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        try:
            model = resolve_model(model_id)
        except (OSError, ValueError, KeyError):
            self.send_json(500, {"error": "无法读取模型配置，请刷新模型列表。"})
            return
        if model is None or model.status in {"offline", "inactive"}:
            self.send_json(503, {"error": "所选模型暂不可用。启动对应的服务后，点击「刷新模型」。"})
            return
        try:
            payload = prepare_chat(model, body)
            key = api_key_for(model, body)
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        try:
            answer, served_model = generate(model, payload, key)
        except HTTPError as error:
            with error:
                message = f"模型服务返回 HTTP {error.code}。"
                if error.code in (401, 403):
                    message += "请检查密钥与模型访问权限。"
                elif error.code == 429:
                    message += "请检查调用频率或账户额度。"
                elif model.source != "api":
                    try:
                        detail = json.load(error).get("error", "")
                    except (OSError, ValueError):
                        detail = ""
                    if detail:
                        message += f"\n{str(detail)[:500]}"
            self.send_json(502, {"error": message})
            return
        except TimeoutError:
            self.send_json(504, {"error": "模型响应超时，请稍后再试。"})
            return
        except (URLError, OSError):
            self.send_json(503, {"error": "无法连接所选模型的服务，请检查连接后重试。"})
            return
        except (ValueError, AttributeError, TypeError):
            self.send_json(502, {"error": "模型服务返回了无法识别的结果。"})
            return
        if not answer:
            self.send_json(502, {"error": "模型没有返回文字，请调整提示词后再试。"})
            return
        self.send_json(200, {"answer": answer, "model": served_model, "source": model.source})


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地螺栓视觉判断台")
    parser.add_argument("--port", type=int, default=8765, help="网页端口，默认 8765")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), VisionHandler) as server:
        print(f"本地页面 http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
