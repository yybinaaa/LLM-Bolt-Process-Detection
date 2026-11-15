"""按配额生成数据集（方案第五节），并支持复现校验。"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from boltcheck import __version__
from boltcheck.grid import ALL_COORDS, LAYER_BOUNDARIES, Coord, as_coords, layer_of, order_matrix
from boltcheck.perturb import random_perturbation
from boltcheck.rules import LATEST_FIRST_VIOLATION, group_labels, judge

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

GROUP_DIMENSIONS = {
    "layer_gap": ("1", "2-3"),
    "position": ("front", "back"),
    "n_perturb": ("single", "multi"),
}


class QuotaError(RuntimeError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def compliant_sequence(rng: random.Random) -> list[Coord]:
    seq: list[Coord] = []
    for layer in (1, 2, 3, 4):
        group = [p for p in ALL_COORDS if layer_of(p) == layer]
        rng.shuffle(group)
        seq.extend(group)
    return seq


def _first_ranges(kind: str, rng: random.Random, cfg: dict) -> dict:
    k_full = (cfg["perturb"]["block_k_min"], cfg["perturb"]["block_k_max"])
    if kind == "boundary":
        b = rng.choice(LAYER_BOUNDARIES)
        return {"src_range": (b + 1, b + 3), "dst_range": (b - 2, b), "k_range": (2, 3)}
    if kind == "cross":
        return {"src_range": (17, 64), "dst_range": (1, 16), "k_range": k_full}
    if kind == "back":
        return {"src_range": (37, 64), "dst_range": (17, 36), "k_range": k_full}
    return {"src_range": (2, 64), "dst_range": (1, 63), "k_range": k_full}


def _extra_ranges(kind: str, cfg: dict) -> dict:
    k_full = (cfg["perturb"]["block_k_min"], cfg["perturb"]["block_k_max"])
    if kind == "back":
        return {"src_range": (18, 64), "dst_range": (17, 63), "k_range": k_full}
    return {"src_range": (2, 64), "dst_range": (1, 63), "k_range": k_full}


def _matches(kind: str, rec: dict) -> bool:
    g = rec["groups"]
    if rec["compliant"]:
        return False
    if kind == "boundary":
        return g["boundary_hard"]
    if kind == "cross":
        return g["layer_gap"] == "2-3"
    if kind == "back":
        return g["position"] == "back"
    return True


def make_record(seq: list[Coord], perturbations: list[dict], construction: str) -> dict:
    j = judge(seq)
    return {
        "id": None,
        "coords": [[x, y] for x, y in seq],
        "order_matrix": order_matrix(seq),
        "layers": j.layers,
        "compliant": j.compliant,
        "verdict": j.verdict,
        "first_violation_step": j.first_violation_step,
        "first_violation_coordinate": j.first_violation_coordinate,
        "n_violation_steps": j.n_violation_steps,
        "construction": construction,
        "perturbations": perturbations,
        "groups": group_labels(j, perturbations),
    }


def _make_noncompliant(rng: random.Random, method: str, kind: str, multi: bool, cfg: dict, seen: set) -> dict:
    p = cfg["perturb"]
    for _ in range(cfg["max_attempts"]):
        seq = compliant_sequence(rng)
        first = random_perturbation(seq, method, rng, **_first_ranges(kind, rng, cfg))
        if first is None:
            continue
        seq, params = first
        perts = [params]
        if multi:
            ok = True
            for _ in range(rng.randint(p["extra_min"], p["extra_max"])):
                nxt = random_perturbation(seq, method, rng, **_extra_ranges(kind, cfg))
                if nxt is None:
                    ok = False
                    break
                seq, params = nxt
                perts.append(params)
            if not ok:
                continue
        key = tuple(seq)
        if key in seen:
            continue
        rec = make_record(seq, perts, method)
        if _matches(kind, rec):
            seen.add(key)
            return rec
    raise QuotaError(f"{method}/{kind}/{'多处' if multi else '单处'}：{cfg['max_attempts']} 次尝试内未生成满足条件的样本")


def generate(cfg: dict) -> list[dict]:
    rng = random.Random(cfg["seed"])
    seen: set = set()
    records: list[dict] = []

    while len(records) < cfg["n_compliant"]:
        seq = compliant_sequence(rng)
        if tuple(seq) in seen:
            continue
        seen.add(tuple(seq))
        records.append(make_record(seq, [], "none"))

    targets = cfg["targets"]
    for method, count in cfg["methods"].items():
        n_targeted = targets["boundary"] + targets["cross"] + targets["back"]
        if n_targeted > count:
            raise QuotaError(f"{method} 的定向样本数 {n_targeted} 超过分配数 {count}")
        kinds = (
            ["boundary"] * targets["boundary"]
            + ["cross"] * targets["cross"]
            + ["back"] * targets["back"]
            + ["free"] * (count - n_targeted)
        )
        non_boundary_idx = 0
        for kind in kinds:
            if kind == "boundary":
                multi = False
            else:
                multi = non_boundary_idx % 2 == 1  # 非边界样本单处、多处交替
                non_boundary_idx += 1
            records.append(_make_noncompliant(rng, method, kind, multi, cfg, seen))

    rng.shuffle(records)
    for i, rec in enumerate(records, start=1):
        rec["id"] = f"{cfg['version']}-{i:04d}"

    problems = check_quotas(records, cfg)
    if problems:
        raise QuotaError("配额未满足：" + "；".join(problems))
    return records


def summarize(records: list[dict]) -> dict:
    nc = [r for r in records if not r["compliant"]]
    out: dict[str, Any] = {
        "total": len(records),
        "compliant": len(records) - len(nc),
        "noncompliant": len(nc),
        "construction": dict(sorted(Counter(r["construction"] for r in nc).items())),
        "boundary_hard": sum(r["groups"]["boundary_hard"] for r in nc),
        "max_first_violation_step": max((r["first_violation_step"] for r in nc), default=None),
    }
    for dim in GROUP_DIMENSIONS:
        out[dim] = dict(sorted(Counter(r["groups"][dim] for r in nc).items()))
    return out


def check_quotas(records: list[dict], cfg: dict) -> list[str]:
    s = summarize(records)
    problems = []
    if s["compliant"] != cfg["n_compliant"]:
        problems.append(f"合规样本 {s['compliant']} 条，应为 {cfg['n_compliant']}")
    for method, count in cfg["methods"].items():
        if s["construction"].get(method, 0) != count:
            problems.append(f"{method} 构造 {s['construction'].get(method, 0)} 条，应为 {count}")
    for dim, values in GROUP_DIMENSIONS.items():
        for v in values:
            if s[dim].get(v, 0) < cfg["min_per_group"]:
                problems.append(f"{dim}={v} 只有 {s[dim].get(v, 0)} 条，少于 {cfg['min_per_group']}")
    if s["boundary_hard"] < cfg["min_boundary_hard"]:
        problems.append(f"边界难例只有 {s['boundary_hard']} 条，少于 {cfg['min_boundary_hard']}")
    if s["max_first_violation_step"] and s["max_first_violation_step"] > LATEST_FIRST_VIOLATION:
        problems.append("出现晚于第 36 步的首次违规，规则实现有误")
    if len({tuple(map(tuple, r["coords"])) for r in records}) != len(records):
        problems.append("存在重复样本")
    return problems


def dumps_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_dataset(records: list[dict], cfg: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = dumps_jsonl(records)
    (out / "dataset.jsonl").write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "dataset_version": cfg["version"],
        "generator_version": __version__,
        "sha256": sha256_text(text),
        "summary": summarize(records),
        "config": cfg,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_dataset(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "dataset.jsonl"
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_sha256(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        p = p / "dataset.jsonl"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(data_dir: str | Path) -> list[str]:
    """校验：文件哈希、按 manifest 配置重新生成是否一致、标签是否与规则程序一致、配额。"""
    d = Path(data_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    actual_sha = dataset_sha256(d)
    if actual_sha != manifest["sha256"]:
        problems.append(f"dataset.jsonl 哈希与 manifest 不一致（{actual_sha[:12]} ≠ {manifest['sha256'][:12]}），文件可能被改动")
    regenerated = sha256_text(dumps_jsonl(generate(manifest["config"])))
    if regenerated != manifest["sha256"]:
        problems.append("按 manifest 中的配置重新生成的数据与登记的不一致（检查 Python 版本或生成代码是否变化）")
    records = load_dataset(d)
    for r in records:
        j = judge(as_coords(r["coords"]))
        if (
            j.compliant != r["compliant"]
            or j.first_violation_step != r["first_violation_step"]
            or r["order_matrix"] != order_matrix(as_coords(r["coords"]))
        ):
            problems.append(f"{r['id']} 的标签与规则程序不一致")
    problems += check_quotas(records, manifest["config"])
    return problems


def select_samples(samples: list[dict], limit: int | None = None, ids: list[str] | None = None) -> list[dict]:
    """按样本 ID（保持数据集中的顺序）和/或前 N 条挑选样本；运行、评测、查看器共用，保证三者一致。"""
    if ids:
        known = {s["id"] for s in samples}
        missing = [i for i in ids if i not in known]
        if missing:
            raise ValueError(f"数据集中没有这些样本：{', '.join(missing)}")
        wanted = set(ids)
        samples = [s for s in samples if s["id"] in wanted]
    if limit:
        samples = samples[:limit]
    return samples
