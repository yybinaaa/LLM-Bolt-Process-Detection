"""把坐标序列展开成 8×8 表格，每格标上安装顺序，便于直观查看全局分布。

- 数据集视图：标准步骤矩阵，底色表示层级，标出违规步和首次违规步，可以逐步播放。
- 运行视图：标准答案与模型复原的矩阵并排，标出复原错的格子和模型的结论。
- 终端表格：show-sample 命令用的纯文本版本。
"""

from __future__ import annotations

import json
from pathlib import Path

from boltcheck.dataset import load_dataset
from boltcheck.grid import as_coords, layer_grid
from boltcheck.rules import violation_steps

LAYERS = layer_grid()


def _sample_view(s: dict) -> dict:
    return {
        "id": s["id"],
        "m": s["order_matrix"],
        "compliant": s["compliant"],
        "first": s["first_violation_step"],
        "firstCoord": s["first_violation_coordinate"],
        "viol": violation_steps(as_coords(s["coords"])),
        "construction": s["construction"],
        "perturbations": [{k: v for k, v in p.items() if k != "involved_positions"} for p in s["perturbations"]],
        "groups": s["groups"],
    }


def dataset_view_data(data_dir: str | Path, ids: list[str] | None = None) -> dict:
    samples = load_dataset(data_dir)
    if ids:
        wanted = set(ids)
        samples = [s for s in samples if s["id"] in wanted]
    manifest_path = Path(data_dir) / "manifest.json"
    version = json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_version") if manifest_path.exists() else None
    return {
        "mode": "dataset",
        "title": f"数据集 {version or Path(data_dir).name}",
        "subtitle": f"{len(samples)} 条样本",
        "layers": LAYERS,
        "items": [_sample_view(s) for s in samples],
    }


def run_view_data(run_dir: str | Path) -> dict:
    from boltcheck.evaluate import parse_record, score_row

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    group = meta["group"]
    samples = load_dataset(meta["dataset"]["path"])
    if meta.get("limit"):
        samples = samples[: meta["limit"]]
    outputs: dict = {}
    out_path = run_dir / "outputs.jsonl"
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                outputs[(rec["sample_id"], rec["repeat"])] = rec

    items = []
    for s in samples:
        item = _sample_view(s)
        reps = []
        for rep in range(meta.get("repeats", 1)):
            out = outputs.get((s["id"], rep))
            row = score_row(s, out, group)
            p = parse_record(out, group) if out and not out.get("error") else None
            reps.append({
                "repeat": rep,
                "m": p.matrix if p and p.matrix_shape_ok else None,
                "verdict": row.get("verdict"),
                "step": p.first_step if p else None,
                "formatValid": row["format_valid"],
                "errors": row["errors"],
                "verdictOk": row.get("verdict_ok"),
                "conclusionOk": row.get("conclusion_ok"),
                "wholeOk": row.get("whole_ok"),
                "transposed": row.get("transposed"),
                "attribution": row.get("attribution"),
            })
        item["model"] = reps
        items.append(item)
    return {
        "mode": "run",
        "group": group,
        "title": f"运行 {meta['run_id']}",
        "subtitle": f"{meta['group_desc']} · {meta['model']} / {meta['profile']} · {len(samples)} 条样本 × {meta.get('repeats', 1)} 次",
        "layers": LAYERS,
        "items": items,
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", data["title"]).replace("__DATA__", payload)


def write_view(data: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    return out


def text_grid(matrix: list, violations: list[int] | None = None, first: int | None = None, truth: list | None = None) -> str:
    """终端用的 8×8 表格。* 违规步，! 首次违规步；给出 truth 时，× 表示与标准答案不同。"""
    violations = set(violations or [])
    lines = [_HEADER]
    for r in range(8):
        cells = []
        for c in range(8):
            v = matrix[r][c]
            mark = " "
            if truth is not None and v != truth[r][c]:
                mark = "×"
            elif v == first:
                mark = "!"
            elif v in violations:
                mark = "*"
            cells.append(f"{v:>4}{mark} ")
        lines.append(f"{10 * (r + 1):>5}  " + "".join(cells).rstrip())
    return "\n".join(lines)


_HEADER = "y↓ x→  " + "".join(f"{10 * c:>4}  " for c in range(1, 9)).rstrip()


def describe_sample(s: dict) -> str:
    head = f"{s['id']}  {s['verdict']}"
    if not s["compliant"]:
        x, y = s["first_violation_coordinate"]
        head += f"  首次违规：第 {s['first_violation_step']} 步 ({x},{y})  违规步共 {s['n_violation_steps']} 个"
        head += f"  构造：{s['construction']} " + "；".join(
            json.dumps({k: v for k, v in p.items() if k not in ("method", "involved_positions")}, ensure_ascii=False)
            for p in s["perturbations"]
        )
    grid = text_grid(s["order_matrix"], violation_steps(as_coords(s["coords"])), s["first_violation_step"])
    layer_rows = "\n".join([_HEADER] + [f"{10 * (r + 1):>5}  " + "".join(f"{v:>4}  " for v in row).rstrip() for r, row in enumerate(LAYERS)])
    return f"{head}\n\n步骤矩阵（* 违规步，! 首次违规步）：\n{grid}\n\n层级：\n{layer_rows}"


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#1d2433;--muted:#667085;--line:#d9dee7;--bg:#f6f7f9;--card:#fff;
--l1:#9fc3e6;--l2:#c4daf0;--l3:#e3edf8;--l4:#ffffff;--bad:#c62828;--badbg:#fde4e2;--ok:#1b7f4b;--okbg:#e3f4ea;--warn:#a15c00;--warnbg:#fdf0d9}
*{box-sizing:border-box}
body{margin:0;font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:14px 24px 10px}
h1{font-size:19px;font-weight:600;margin:0}
.sub{color:var(--muted);font-size:13px;margin-top:3px}
.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-top:10px;font-size:13px}
.bar select,.bar input{font:inherit;font-size:13px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:#fff}
.legend{display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:12px;color:var(--muted);margin-top:8px}
.sw{display:inline-block;width:14px;height:14px;border:1px solid var(--line);border-radius:3px;vertical-align:-3px;margin-right:4px}
main{padding:18px 24px;display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
main.run{grid-template-columns:repeat(auto-fill,minmax(720px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.id{font-weight:600;font-size:14px;font-variant-numeric:tabular-nums}
.tag{font-size:12px;padding:1px 8px;border-radius:10px;background:#eef1f5;color:var(--muted)}
.tag.bad{background:var(--badbg);color:var(--bad)}.tag.ok{background:var(--okbg);color:var(--ok)}.tag.warn{background:var(--warnbg);color:var(--warn)}
.info{font-size:12px;color:var(--muted);margin:6px 0 8px;line-height:1.6}
.grids{display:flex;gap:14px;flex-wrap:wrap}
.gwrap{max-width:350px}
.gwrap h3{font-size:12px;font-weight:600;margin:0 0 4px;color:var(--muted);line-height:1.9}
table.g{border-collapse:separate;border-spacing:2px;font-variant-numeric:tabular-nums}
table.g th{font-size:10px;font-weight:400;color:var(--muted);padding:0 2px;text-align:center}
table.g td{width:34px;height:30px;text-align:center;font-size:13px;border:1px solid var(--line);border-radius:4px;position:relative;transition:opacity .12s}
table.g td.v{color:var(--bad);font-weight:700}
table.g td.f{outline:2.5px solid var(--bad);outline-offset:-2px}
table.g td.wrong{background:var(--badbg)!important;color:var(--bad);font-weight:700}
table.g td.wrong small{position:absolute;right:2px;bottom:0;font-size:9px;color:var(--muted);font-weight:400}
table.g td.fut{opacity:.18}
table.g td.cur{box-shadow:0 0 0 2px var(--ink) inset}
.play{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px;color:var(--muted)}
.play input{flex:1}
.play button{font:inherit;font-size:12px;border:1px solid var(--line);background:#fff;border-radius:6px;padding:2px 10px;cursor:pointer}
.rep{font-size:12px;margin-left:auto}
.empty{color:var(--muted);padding:40px;text-align:center;grid-column:1/-1}
.more{grid-column:1/-1;text-align:center}
.more button{font:inherit;padding:6px 18px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="sub" id="subtitle"></div>
  <div class="bar" id="filters"></div>
  <div class="legend" id="legend"></div>
</header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const L = DATA.layers;
const isRun = DATA.mode === "run";
const NAMES = {swap:"交换", insert:"插入", block:"整段提前", none:"—", front:"前段", back:"后段", single:"单处", multi:"多处",
  full_correct:"完全正确", reasoning_error:"规则推理错误", restore_error:"还原错误", both_error:"还原和推理都错"};
const $ = (s, el=document) => el.querySelector(s);
document.title = DATA.title;
$("#title").textContent = "螺栓初拧顺序查看器 · " + DATA.title;
$("#subtitle").textContent = DATA.subtitle + "。每格数字是该位置的安装顺序（第几步），坐标 x 向右、y 向下。";
if (isRun) $("#list").classList.add("run");

const legend = [
  ...[1,2,3,4].map(k => `<span><i class="sw" style="background:var(--l${k})"></i>第 ${k} 层</span>`),
  `<span><b style="color:var(--bad)">红字</b> 违规步</span>`,
  `<span><i class="sw" style="outline:2.5px solid var(--bad);outline-offset:-2px"></i>首次违规步</span>`,
];
if (isRun) legend.push(`<span><i class="sw" style="background:var(--badbg)"></i>模型复原错误（右下角小字为正确值）</span>`);
$("#legend").innerHTML = legend.join("");

const filters = [
  ["verdict", "结论", [["", "全部"], ["ok", "合规"], ["bad", "不合规"]]],
  ["construction", "构造", [["", "全部"], ["swap", "交换"], ["insert", "插入"], ["block", "整段提前"]]],
  ["gap", "跨层距离", [["", "全部"], ["1", "1"], ["2-3", "2～3"]]],
  ["pos", "首次违规位置", [["", "全部"], ["front", "前段"], ["back", "后段"]]],
  ["hard", "边界难例", [["", "全部"], ["1", "是"]]],
];
if (isRun) filters.push(["result", "模型结果", [["", "全部"], ["wrong", "判断或复原有错"], ["verdict", "判断错误"], ["matrix", "复原错误"], ["invalid", "格式无效"], ["transposed", "转置"]]]);
$("#filters").innerHTML = filters.map(([k, label, opts]) =>
  `<label>${label} <select data-k="${k}">${opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select></label>`
).join("") + `<label>编号 <input id="q" placeholder="如 v1.1-0007" size="12"></label><span id="count" style="color:var(--muted)"></span>`;
const state = {};
document.querySelectorAll("#filters select").forEach(s => s.onchange = () => { state[s.dataset.k] = s.value; render(); });
$("#q").oninput = e => { state.q = e.target.value.trim(); render(); };

function repBad(r, kind) {
  if (kind === "invalid") return !r.formatValid;
  if (kind === "transposed") return !!r.transposed;
  if (kind === "verdict") return r.verdictOk === false || r.conclusionOk === false;
  if (kind === "matrix") return r.wholeOk === false;
  if (kind === "wrong") return !r.formatValid || r.verdictOk === false || r.conclusionOk === false || r.wholeOk === false;
  return true;
}

function matches(it) {
  const g = it.groups;
  if (state.verdict === "ok" && !it.compliant) return false;
  if (state.verdict === "bad" && it.compliant) return false;
  if (state.construction && it.construction !== state.construction) return false;
  if (state.gap && g.layer_gap !== state.gap) return false;
  if (state.pos && g.position !== state.pos) return false;
  if (state.hard && !g.boundary_hard) return false;
  if (state.q && !it.id.includes(state.q)) return false;
  if (isRun && state.result && !it.model.some(r => repBad(r, state.result))) return false;
  return true;
}

function gridHTML(m, opts = {}) {
  let h = `<table class="g"><tr><th></th>${[1,2,3,4,5,6,7,8].map(c => `<th>${c*10}</th>`).join("")}</tr>`;
  for (let r = 0; r < 8; r++) {
    h += `<tr><th>${(r+1)*10}</th>`;
    for (let c = 0; c < 8; c++) {
      const v = m ? m[r][c] : "";
      const cls = [];
      if (opts.viol && opts.viol.has(v)) cls.push("v");
      if (opts.first && v === opts.first) cls.push("f");
      let inner = v;
      if (opts.truth && m && v !== opts.truth[r][c]) { cls.push("wrong"); inner = `${v}<small>${opts.truth[r][c]}</small>`; }
      h += `<td data-s="${v}" class="${cls.join(" ")}" style="background:var(--l${L[r][c]})" title="(${(c+1)*10},${(r+1)*10}) 第 ${L[r][c]} 层">${inner}</td>`;
    }
    h += "</tr>";
  }
  return h + "</table>";
}

function pertText(p) {
  if (p.method === "swap") return `交换第 ${p.positions[0]}、${p.positions[1]} 步`;
  if (p.method === "insert") return `第 ${p.src} 步插到第 ${p.dst} 步`;
  if (p.method === "block") return `第 ${p.src}～${p.src + p.k - 1} 步整段插到第 ${p.dst} 步`;
  return "";
}

function card(it) {
  const tags = [it.compliant ? `<span class="tag ok">合规</span>` : `<span class="tag bad">不合规</span>`];
  if (!it.compliant) {
    tags.push(`<span class="tag">${NAMES[it.construction]}</span>`);
    tags.push(`<span class="tag">跨层 ${it.groups.layer_gap}</span>`);
    tags.push(`<span class="tag">${NAMES[it.groups.position]}</span>`);
    if (it.groups.boundary_hard) tags.push(`<span class="tag warn">边界难例</span>`);
  }
  let info = it.compliant ? "各层依次完成，无违规步。" :
    `首次违规：第 ${it.first} 步 (${it.firstCoord.join(",")})；违规步共 ${it.viol.length} 个；扰动：${it.perturbations.map(pertText).join("；")}`;
  const viol = new Set(it.viol);
  let grids = `<div class="gwrap"><h3>${isRun ? "标准答案" : "步骤矩阵"}</h3>${gridHTML(it.m, {viol, first: it.first})}</div>`;
  let repSel = "";
  if (isRun) {
    if (it.model.length > 1) repSel = `<select class="rep">${it.model.map(r => `<option value="${r.repeat}">第 ${r.repeat+1} 次</option>`).join("")}</select>`;
    grids += `<div class="gwrap model"></div>`;
  }
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<div class="top"><span class="id">${it.id}</span>${tags.join("")}${repSel}</div>
    <div class="info">${info}</div><div class="grids">${grids}</div>
    <div class="play"><button>播放</button><input type="range" min="0" max="64" value="64"><span class="stepv">全部 64 步</span></div>`;
  if (isRun) {
    const draw = rep => {
      const r = it.model[rep];
      const isPipe = DATA.group === "pipeline";
      const vLabel = isPipe ? "第二步，模型按自己复原的表格判断：" : "模型结论：";
      const verdict = r.verdict ? `${vLabel}${r.verdict}${r.verdict === "不合规" && r.step ? `，首次违规第 ${r.step} 步（红框）` : ""}` : `${vLabel}—`;
      const badges = [];
      if (!r.formatValid) badges.push(`<span class="tag bad">格式无效</span>`);
      if (r.verdictOk === false || r.conclusionOk === false) badges.push(`<span class="tag bad">判断错误</span>`);
      else if (r.verdictOk) badges.push(`<span class="tag ok">判断正确</span>`);
      if (r.wholeOk === false && r.m) badges.push(`<span class="tag bad">复原错误</span>`);
      if (r.transposed) badges.push(`<span class="tag warn">行列写反</span>`);
      if (r.attribution) badges.push(`<span class="tag">${NAMES[r.attribution]}</span>`);
      const showGrid = DATA.group !== "judge";
      const mLabel = isPipe ? "第一步，模型复原" : (DATA.group === "judge" ? "模型判断" : "模型复原");
      const claimed = r.verdict === "不合规" ? r.step : null;
      $(".model", el).innerHTML = `<h3>${mLabel} ${badges.join(" ")}</h3>` +
        (showGrid ? (r.m ? gridHTML(r.m, {truth: it.m, first: claimed}) : `<div class="info">没有可解析的 8×8 矩阵：${(r.errors || []).join("；")}</div>`) : "") +
        `<div class="info">${DATA.group !== "restore" ? verdict : ""}</div>`;
      applyStep(el, +$("input[type=range]", el).value);
    };
    // 按"模型结果"筛选时，默认显示第一次出问题的那一次重复
    const first = state.result ? Math.max(0, it.model.findIndex(r => repBad(r, state.result))) : 0;
    draw(first);
    const sel = $(".rep", el);
    if (sel) { sel.value = String(first); sel.onchange = () => draw(+sel.value); }
  }
  const range = $("input[type=range]", el), btn = $("button", el);
  range.oninput = () => applyStep(el, +range.value);
  let timer = null;
  btn.onclick = () => {
    if (timer) { clearInterval(timer); timer = null; btn.textContent = "播放"; return; }
    if (+range.value >= 64) range.value = 0;
    btn.textContent = "暂停";
    timer = setInterval(() => {
      range.value = +range.value + 1; applyStep(el, +range.value);
      if (+range.value >= 64) { clearInterval(timer); timer = null; btn.textContent = "播放"; }
    }, 250);
  };
  return el;
}

function applyStep(el, t) {
  el.querySelectorAll("td[data-s]").forEach(td => {
    const s = +td.dataset.s;
    td.classList.toggle("fut", t < 64 && s > t);
    td.classList.toggle("cur", t < 64 && s === t);
  });
  $(".stepv", el).textContent = t >= 64 ? "全部 64 步" : `前 ${t} 步`;
}

const PAGE = 60;
let shown = PAGE;
function render(reset = true) {
  if (reset) shown = PAGE;
  const list = $("#list");
  list.innerHTML = "";
  const items = DATA.items.filter(matches);
  $("#count").textContent = `共 ${items.length} 条`;
  if (!items.length) { list.innerHTML = `<div class="empty">没有符合条件的样本</div>`; return; }
  items.slice(0, shown).forEach(it => list.appendChild(card(it)));
  if (items.length > shown) {
    const more = document.createElement("div");
    more.className = "more";
    more.innerHTML = `<button>再显示 ${Math.min(PAGE, items.length - shown)} 条（还剩 ${items.length - shown} 条）</button>`;
    more.querySelector("button").onclick = () => { shown += PAGE; render(false); };
    list.appendChild(more);
  }
}
render();
</script>
</body>
</html>
"""
