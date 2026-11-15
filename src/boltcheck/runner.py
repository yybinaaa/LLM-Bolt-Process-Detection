"""执行一次实验运行，把所有原始输出和运行信息写入 runs/<run_id>/。

目录内容：
- meta.json      运行配置快照：Git 提交、数据集哈希、模型与采样参数、提示词版本等
- outputs.jsonl  每条样本每次重复一行，保存原始回复、token 数、延迟、峰值显存
- metrics.json / report.md / log_row.md  评测结果（运行结束后自动生成，也可用 evaluate 重算）
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from boltcheck import PLAN_VERSION, __version__
from boltcheck.backends import BaselineBackend, NoisyBackend, build_openai_backend
from boltcheck.dataset import dataset_sha256, load_config, load_dataset
from boltcheck.parse import parse_output
from boltcheck.prompts import GROUPS, PROMPT_VERSION, build_judge_prompt, build_prompt


class GpuMonitor:
    """定时调用 nvidia-smi 记录显存占用（MiB，所有 GPU 求和），同时记录两个读数：
    - 整卡占用：包含桌面等其他程序；
    - 计算进程占用：nvidia-smi --query-compute-apps 列出的进程之和，通常就是 vLLM（Windows 上可能读不到）。

    注意：vLLM 启动时按 --gpu-memory-utilization 预分配显存，这里记录的是进程实际占用的总量，
    主要用于确认服务没有超出显存预算（3～5GB），不等于模型推理的最小需求。
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[tuple[float, int]] = []
        self.app_samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    def _read(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return sum(int(x) for x in out.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _read_apps(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            vals = [int(x) for x in out.split() if x.strip().isdigit()]
            return sum(vals) if vals else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            v = self._read()
            if v is not None:
                self.samples.append((now, v))
            a = self._read_apps()
            if a is not None:
                self.app_samples.append((now, a))
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuMonitor":
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def peak_between(self, t0: float, t1: float, apps: bool = False) -> int | None:
        samples = self.app_samples if apps else self.samples
        vals = [v for t, v in samples if t0 <= t <= t1]
        if not vals and samples:
            vals = [min(samples, key=lambda s: abs(s[0] - t1))[1]]
        return max(vals) if vals else None


def git_info(cwd: Path | None = None) -> dict:
    def g(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except OSError:
            return None

    status = g("status", "--porcelain")
    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def resolve_backend(backend: str, models_config: str | None, model: str | None, profile: str | None, noisy_seed: int = 0):
    """返回 (后端对象, 运行信息)。"""
    if backend == "baseline":
        return BaselineBackend(), {"backend": "baseline", "model": "rule-baseline", "profile": "-", "repeats": 1, "params": {}}
    if backend == "noisy":
        return NoisyBackend(noisy_seed), {"backend": "noisy", "model": f"noisy-seed{noisy_seed}", "profile": "-", "repeats": 1, "params": {}}
    if backend != "openai":
        raise ValueError(f"未知后端: {backend}")
    if not (models_config and model and profile):
        raise ValueError("openai 后端需要 --models-config、--model 和 --profile")
    cfg = load_config(models_config)
    if model not in cfg.get("models", {}):
        raise ValueError(f"{models_config} 中没有模型 {model}（可选：{', '.join(cfg.get('models', {}))}）")
    if profile not in cfg.get("profiles", {}):
        raise ValueError(f"{models_config} 中没有配置 {profile}（可选：{', '.join(cfg.get('profiles', {}))}）")
    model_cfg, prof = cfg["models"][model], cfg["profiles"][profile]
    allowed = model_cfg.get("profiles")
    if allowed and profile not in allowed:
        raise ValueError(f"模型 {model} 只能使用这些配置：{', '.join(allowed)}")
    be = build_openai_backend(model_cfg, prof)
    info = {
        "backend": "openai",
        "models_config": str(models_config),
        "model": model,
        "served_model": model_cfg["model"],
        "hf_repo": model_cfg.get("hf_repo"),
        "quantization": model_cfg.get("quantization"),
        "base_url": model_cfg["base_url"],
        "profile": profile,
        "repeats": prof.get("repeats", 1),
        "seed": prof.get("seed"),
        "params": be.params,
    }
    return be, info


def run_pipeline(backend, sample: dict, rep: int, prompt1: str, sha) -> dict:
    """第 5 组：第一阶段复原矩阵；矩阵是 8×8 整数时，把它交回给模型做第二阶段判断。"""
    g1 = backend.generate(prompt1, sample, "restore", rep)
    stage1 = {"prompt_sha256": sha(prompt1), **g1.to_dict()}
    stage2 = None
    if g1.error is None:
        p1 = parse_output(g1.text, "restore")
        if p1.matrix_shape_ok:
            prompt2 = build_judge_prompt(p1.matrix)
            # 后端拿到的样本里，矩阵换成模型自己复原的（只影响 baseline / noisy 模拟后端）
            g2 = backend.generate(prompt2, {**sample, "order_matrix": p1.matrix}, "judge", rep)
            stage2 = {"prompt_sha256": sha(prompt2), **g2.to_dict()}
    stages = [st for st in (stage1, stage2) if st]

    def total(key):
        vals = [st.get(key) for st in stages if st.get(key) is not None]
        return sum(vals) if vals else None

    return {
        "sample_id": sample["id"],
        "repeat": rep,
        "prompt_sha256": stage1["prompt_sha256"],
        "text": stage2["text"] if stage2 else "",
        "reasoning": None,
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "latency_s": total("latency_s") or 0.0,
        "finish_reason": "length" if any(st.get("finish_reason") == "length" for st in stages) else (stages[-1].get("finish_reason")),
        "error": stage1["error"] or (stage2 or {}).get("error"),
        "stage1": stage1,
        "stage2": stage2,
    }


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in s)


def run_experiment(
    dataset_dir: str,
    group: str,
    backend,
    info: dict,
    out_root: str = "runs",
    limit: int | None = None,
    repeats: int | None = None,
    resume_dir: str | None = None,
    gpu_monitor: bool = False,
    workers: int = 1,
    note: str = "",
    log=print,
) -> Path:
    if group not in GROUPS:
        raise ValueError(f"未知的对照组: {group}（可选 {', '.join(GROUPS)}）")
    samples = load_dataset(dataset_dir)
    if limit:
        samples = samples[:limit]
    n_repeats = repeats or info.get("repeats", 1)

    if resume_dir:
        run_dir = Path(resume_dir)
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        if meta["dataset"]["sha256"] != dataset_sha256(dataset_dir):
            raise RuntimeError("数据集与原运行不一致，不能续跑")
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"_{_safe(info['profile'])}" if info.get("profile") not in (None, "-") else ""
        run_dir = Path(out_root) / f"{ts}_{group}_{_safe(info['model'])}{suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = Path(dataset_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        meta = {
            "run_id": run_dir.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "group": group,
            "group_desc": GROUPS[group],
            **info,
            "repeats": n_repeats,
            "limit": limit,
            "dataset": {
                "path": str(dataset_dir),
                "version": manifest.get("dataset_version"),
                "sha256": dataset_sha256(dataset_dir),
                "n_samples": len(samples),
            },
            "prompt_version": PROMPT_VERSION,
            "plan_version": PLAN_VERSION,
            "code_version": __version__,
            "git": git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "note": note,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if meta["git"].get("dirty"):
            log("提示：工作区有未提交的改动，本次运行会在 meta.json 中标记 dirty=true，正式实验请先提交代码。")

    out_path = run_dir / "outputs.jsonl"
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        kept = []
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error") is None:
                done.add((rec["sample_id"], rec["repeat"]))
                kept.append(line)
        out_path.write_text("".join(k + "\n" for k in kept), encoding="utf-8")  # 去掉出错的记录，重新调用

    jobs = [(s, r) for r in range(n_repeats) for s in samples if (s["id"], r) not in done]
    total = len(samples) * n_repeats
    log(f"运行目录：{run_dir}")
    log(f"共 {total} 次调用，已完成 {total - len(jobs)}，本次需要 {len(jobs)}")

    monitor = GpuMonitor().start() if gpu_monitor else None
    if gpu_monitor and monitor and not monitor.available:
        log("提示：未找到 nvidia-smi，不记录显存")
    lock = threading.Lock()
    counter = {"n": total - len(jobs)}

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def work(job):
        sample, rep = job
        prompt = build_prompt(group, sample)
        t0 = time.perf_counter()
        if group == "pipeline":
            rec = run_pipeline(backend, sample, rep, prompt, sha)
            gen_error = rec["error"]
            latency = rec["latency_s"]
        else:
            gen = backend.generate(prompt, sample, group, rep)
            rec = {"sample_id": sample["id"], "repeat": rep, "prompt_sha256": sha(prompt), **gen.to_dict()}
            gen_error, latency = gen.error, gen.latency_s
        t1 = time.perf_counter()
        rec["gpu_peak_mib"] = monitor.peak_between(t0, t1) if monitor else None
        rec["gpu_apps_peak_mib"] = monitor.peak_between(t0, t1, apps=True) if monitor else None
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter["n"] += 1
            status = f"错误：{gen_error[:80]}" if gen_error else f"{latency:.1f}s"
            log(f"[{counter['n']}/{total}] {sample['id']} 第{rep + 1}次 {status}")

    try:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(work, jobs))
        else:
            for job in jobs:
                work(job)
    finally:
        if monitor:
            monitor.stop()
    return run_dir
