"""命令行入口：boltcheck <命令>（或 python -m boltcheck <命令>）。

  generate     按配置生成数据集
  verify-data  校验数据集可复现、标签正确、配额满足
  show-prompt  打印某条样本在某组下的提示词
  show-sample  在终端把某条样本展开成 8×8 表格
  view         生成网页查看器（数据集，或某次运行的标准答案与模型复原对比）
  run          调用模型跑一组对照，结束后自动评测
  evaluate     重新评测一个运行目录
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from boltcheck import __version__


def _cmd_generate(a) -> int:
    from boltcheck.dataset import generate, load_config, summarize, write_dataset

    cfg = load_config(a.config)
    out = Path(a.out or f"data/{cfg['version']}")
    if (out / "dataset.jsonl").exists() and not a.force:
        print(f"{out}/dataset.jsonl 已存在。数据集一旦用于正式实验就不应覆盖；确需重建请加 --force。")
        return 1
    records = generate(cfg)
    manifest = write_dataset(records, cfg, out)
    print(f"已生成 {len(records)} 条样本 -> {out}")
    print(f"sha256: {manifest['sha256']}")
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    return 0


def _cmd_verify(a) -> int:
    from boltcheck.dataset import verify

    problems = verify(a.data)
    if problems:
        print("校验未通过：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{a.data} 校验通过：哈希一致、可按配置复现、标签与规则程序一致、配额满足。")
    return 0


def _cmd_show_prompt(a) -> int:
    from boltcheck.dataset import load_dataset
    from boltcheck.prompts import build_prompt

    samples = {s["id"]: s for s in load_dataset(a.data)}
    sample = samples.get(a.id) if a.id else next(iter(samples.values()))
    if sample is None:
        print(f"找不到样本 {a.id}")
        return 1
    print(build_prompt(a.group, sample))
    return 0


def _cmd_show_sample(a) -> int:
    from boltcheck.dataset import load_dataset
    from boltcheck.viewer import describe_sample

    samples = load_dataset(a.data)
    picked = [s for s in samples if s["id"] in a.ids] if a.ids else samples[:1]
    if not picked:
        print(f"找不到样本 {' '.join(a.ids)}")
        return 1
    print("\n\n".join(describe_sample(s) for s in picked))
    return 0


def _cmd_view(a) -> int:
    from boltcheck.viewer import dataset_view_data, run_view_data, write_view

    if a.run:
        out = write_view(run_view_data(a.run), a.out or Path(a.run) / "view.html")
    else:
        out = write_view(dataset_view_data(a.data, a.ids), a.out or Path(a.data) / "view.html")
    print(f"已生成 {out}，用浏览器打开即可（不需要联网）。")
    return 0


def _cmd_run(a) -> int:
    from boltcheck.evaluate import evaluate_run
    from boltcheck.runner import resolve_backend, run_experiment

    if a.resume:
        meta = json.loads((Path(a.resume) / "meta.json").read_text(encoding="utf-8"))
        backend, info = resolve_backend(
            meta["backend"], meta.get("models_config"), meta.get("model"), meta.get("profile"), a.noisy_seed
        )
        group, data, limit, repeats = meta["group"], meta["dataset"]["path"], meta.get("limit"), meta["repeats"]
    else:
        if not a.group:
            print("需要 --group（joint / restore / judge）")
            return 2
        backend, info = resolve_backend(a.backend, a.models_config, a.model, a.profile, a.noisy_seed)
        group, data, limit, repeats = a.group, a.data, a.limit, a.repeats
    if hasattr(backend, "check") and not a.skip_check:
        backend.check()
    run_dir = run_experiment(
        data, group, backend, info,
        out_root=a.out, limit=limit, repeats=repeats, resume_dir=a.resume,
        gpu_monitor=a.gpu_monitor, workers=a.workers, note=a.note,
    )
    metrics = evaluate_run(run_dir)
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    print("实验日志行（正式实验请粘贴到 docs/experiment-log.md）：")
    print((run_dir / "log_row.md").read_text(encoding="utf-8"))
    print(f"逐条查看标准答案与模型复原：用浏览器打开 {run_dir / 'view.html'}")
    return 0 if metrics["n_outputs_found"] else 1


def _cmd_evaluate(a) -> int:
    from boltcheck.evaluate import evaluate_run

    for d in a.run_dirs:
        evaluate_run(d)
        print(Path(d, "report.md").read_text(encoding="utf-8"))
        print(Path(d, "log_row.md").read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="boltcheck", description="高强螺栓初拧顺序判断实验")
    p.add_argument("--version", action="version", version=f"boltcheck {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="按配置生成数据集")
    g.add_argument("--config", default="configs/dataset_v1.1.toml")
    g.add_argument("--out", help="输出目录，默认 data/<版本>")
    g.add_argument("--force", action="store_true", help="覆盖已有数据")
    g.set_defaults(func=_cmd_generate)

    v = sub.add_parser("verify-data", help="校验数据集")
    v.add_argument("data", nargs="?", default="data/v1.1")
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("show-prompt", help="打印提示词")
    s.add_argument("--data", default="data/v1.1")
    s.add_argument("--group", default="joint", choices=["joint", "restore", "judge"])
    s.add_argument("--id", help="样本 ID，默认第一条")
    s.set_defaults(func=_cmd_show_prompt)

    ss = sub.add_parser("show-sample", help="在终端把样本展开成 8×8 表格")
    ss.add_argument("ids", nargs="*", help="样本 ID，可给多个；默认第一条")
    ss.add_argument("--data", default="data/v1.1")
    ss.set_defaults(func=_cmd_show_sample)

    vw = sub.add_parser("view", help="生成网页查看器")
    vw.add_argument("--data", default="data/v1.1", help="数据集目录")
    vw.add_argument("--run", help="运行目录；给出时生成标准答案与模型复原的对比视图")
    vw.add_argument("--ids", nargs="*", help="只包含这些样本")
    vw.add_argument("--out", help="输出文件，默认 <数据集或运行目录>/view.html")
    vw.set_defaults(func=_cmd_view)

    r = sub.add_parser("run", help="跑一组对照并评测")
    r.add_argument("--group", choices=["joint", "restore", "judge"], help="joint=第2组 restore=第3组 judge=第4组")
    r.add_argument("--backend", default="openai", choices=["openai", "baseline", "noisy"],
                   help="baseline=第1组程序规则基线；noisy=模拟错误输出，用于检查评测")
    r.add_argument("--models-config", default="configs/models.toml")
    r.add_argument("--model", help="configs/models.toml 中的模型名")
    r.add_argument("--profile", help="configs/models.toml 中的采样配置名")
    r.add_argument("--data", default="data/v1.1")
    r.add_argument("--limit", type=int, help="只跑前 N 条（试跑用）")
    r.add_argument("--repeats", type=int, help="覆盖配置中的重复次数")
    r.add_argument("--workers", type=int, default=1, help="并发请求数，默认 1（便于逐条记录显存）")
    r.add_argument("--gpu-monitor", action="store_true", help="用 nvidia-smi 记录峰值显存")
    r.add_argument("--resume", help="续跑一个中断的运行目录（只补跑缺失和出错的调用）")
    r.add_argument("--out", default="runs")
    r.add_argument("--note", default="", help="备注，写入 meta.json 和实验日志行")
    r.add_argument("--noisy-seed", type=int, default=0)
    r.add_argument("--skip-check", action="store_true", help="跳过运行前的服务连通性检查")
    r.set_defaults(func=_cmd_run)

    e = sub.add_parser("evaluate", help="重新评测运行目录")
    e.add_argument("run_dirs", nargs="+")
    e.set_defaults(func=_cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
