"""模型调用后端。

- openai：OpenAI 兼容接口（本地 vLLM 或远程 API），只用标准库 urllib，无额外依赖。
- baseline：程序规则基线（第 1 组），直接给出标准答案，用于检验整条流水线。
- noisy：按固定种子模拟各类错误，用于在没有 GPU 时检查评测与报告。
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

from boltcheck.grid import transpose
from boltcheck.rules import block_maxima, judge_matrix


@dataclass
class Generation:
    text: str
    reasoning: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    finish_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _cells(matrix: list) -> dict:
    return {f"({10 * (c + 1)},{10 * (r + 1)})": matrix[r][c] for r in range(8) for c in range(8)}


def _answer_json(
    group: str, matrix: list | None, verdict: str | None, step: int | None, coord: list | None,
    block_max: list | None = None,
) -> str:
    """按提示词 v1.3 的字段顺序生成模拟回复。"""
    obj: dict = {}
    if group in ("joint", "restore"):
        obj["cells"] = _cells(matrix) if matrix else None
        obj["order_matrix"] = matrix
    if group in ("joint", "judge"):
        obj.update(
            {
                "block_max": block_max,
                "verdict": verdict,
                "first_violation_step": step,
                "first_violation_coordinate": coord,
                "reason": "程序规则基线" if verdict else "",
            }
        )
    return json.dumps(obj, ensure_ascii=False)


class BaselineBackend:
    name = "baseline"

    def generate(self, prompt: str, sample: dict, group: str, repeat: int) -> Generation:
        t0 = time.perf_counter()
        text = _answer_json(
            group,
            sample["order_matrix"],
            sample["verdict"],
            sample["first_violation_step"],
            sample["first_violation_coordinate"],
            block_maxima(sample["order_matrix"]),
        )
        return Generation(text=text, latency_s=time.perf_counter() - t0, finish_reason="stop")


class NoisyBackend:
    """模拟输出：格式错误、转置矩阵、换错两个格子、结论翻转、回答无法判断等。"""

    name = "noisy"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def generate(self, prompt: str, sample: dict, group: str, repeat: int) -> Generation:
        rng = random.Random(f"{self.seed}:{sample['id']}:{repeat}")
        if rng.random() < 0.08:
            return Generation(text='{"order_matrix": [[1, 2, 3', finish_reason="length", latency_s=0.01)
        matrix = [row[:] for row in sample["order_matrix"]]
        u = rng.random()
        if u < 0.10:
            matrix = transpose(matrix)
        elif u < 0.30:
            (r1, c1), (r2, c2) = rng.sample([(r, c) for r in range(8) for c in range(8)], 2)
            matrix[r1][c1], matrix[r2][c2] = matrix[r2][c2], matrix[r1][c1]
        if group == "judge":
            matrix = sample["order_matrix"]
        j = judge_matrix(matrix)
        verdict, step, coord = j.verdict, j.first_violation_step, j.first_violation_coordinate
        bmax = block_maxima(matrix)
        if rng.random() < 0.15:  # 模拟区块最大数写错一个
            k = rng.randrange(3)
            bmax[k] += rng.choice([-1, 1])
        v = rng.random()
        if v < 0.05:
            verdict, step, coord = "无法判断", None, None
        elif v < 0.17:
            if verdict == "合规":
                verdict, step, coord = "不合规", rng.randint(5, 36), None
            else:
                verdict, step, coord = "合规", None, None
        return Generation(text=_answer_json(group, matrix, verdict, step, coord, bmax), finish_reason="stop", latency_s=0.01)


class OpenAICompatBackend:
    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        params: dict,
        api_key: str | None = None,
        timeout_s: float = 600,
        retries: int = 3,
        seed: int | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        host = urllib.parse.urlparse(self.url).hostname or ""
        # 本地 vLLM 不走系统代理，避免设置了 HTTP_PROXY 时请求被转发出去
        local = host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if local else urllib.request.build_opener()
        self.model = model
        self.params = params
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.retries = retries
        self.seed = seed

    def _request(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers)
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def check(self) -> None:
        """运行前检查服务是否可达、模型名是否存在，避免整批调用都超时重试。"""
        url = self.url.rsplit("/chat/completions", 1)[0] + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        try:
            with self._opener.open(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise RuntimeError(f"无法连接 {url}（{e}）。请先启动模型服务，或检查 configs/models.toml 中的 base_url。") from e
        names = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        if names and self.model not in names:
            raise RuntimeError(f"服务上没有模型 {self.model!r}，可用的是 {names}。请让 --served-model-name 与配置一致。")

    def generate(self, prompt: str, sample: dict, group: str, repeat: int) -> Generation:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], **self.params}
        if self.seed is not None:
            body["seed"] = self.seed + repeat
        last_err = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                data = self._request(body)
                latency = time.perf_counter() - t0
                choice = data["choices"][0]
                msg = choice.get("message") or {}
                usage = data.get("usage") or {}
                return Generation(
                    text=msg.get("content") or "",
                    reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    latency_s=latency,
                    finish_reason=choice.get("finish_reason"),
                )
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code not in (408, 429, 500, 502, 503, 504):
                    break  # 参数错误等不重试
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, KeyError, ValueError) as e:
                last_err = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                time.sleep(min(30, 2**attempt))
        return Generation(text="", error=last_err)


def build_openai_backend(model_cfg: dict, profile: dict) -> OpenAICompatBackend:
    params = {**profile.get("params", {}), **model_cfg.get("params_override", {})}
    for key in model_cfg.get("drop_params", []):
        params.pop(key, None)
    key_env = model_cfg.get("api_key_env") or ""
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(f"需要设置环境变量 {key_env}（API 密钥）")
    return OpenAICompatBackend(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        params=params,
        api_key=api_key,
        timeout_s=model_cfg.get("timeout_s", 600),
        retries=model_cfg.get("retries", 3),
        seed=profile.get("seed"),
    )
