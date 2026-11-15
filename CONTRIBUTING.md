# 迭代规范

本仓库同时管理三样东西：实验方案、数据集、代码。三者各有版本号，改动都要能追溯。

## 版本号

| 对象 | 版本形式 | 位置 | 何时升级 |
|---|---|---|---|
| 实验方案 | v1.1、v1.2… | `docs/experiment-plan.md` 修订记录表；`boltcheck.PLAN_VERSION` | 实验设计、判定口径、模型选择变化 |
| 数据集 | 目录名 `data/v1.1/` | `data/<版本>/manifest.json` | 数据规格或生成逻辑变化 |
| 提示词 | `PROMPT_VERSION` | `src/boltcheck/prompts.py` | 任何模板文字变化 |
| 代码 | 语义化版本 0.1.0… | `boltcheck.__version__` | 每次合并到 main 并打标签 |

规则：

- 数据集一旦登记过正式实验，就不再覆盖。生成逻辑或配置变了，就新建一个数据版本目录（如 `data/v1.2/`）和对应配置文件。CI 会检查已提交的数据能被原样复现。
- 改提示词必须同时提高 `PROMPT_VERSION`，否则不同运行之间无法比较。
- 方案改动先改 `docs/experiment-plan.md`（含修订记录表），再改代码。

## 分支与合并

- `main` 始终可运行、测试通过，不直接推送。
- 新工作从 `main` 拉分支，命名为 `类型/简述`，例如 `feat/v1.1-experiment-pipeline`、`fix/parse-coordinate`、`docs/plan-v1.3`、`exp/2b-greedy-pilot`。
- 通过 Pull Request 合并。PR 描述写清楚改了什么、为什么、怎么验证；CI 通过后再合并。
- 合并后在 main 上打代码版本标签：`git tag v0.1.0 && git push origin v0.1.0`。

## 提交信息

格式：`类型: 简述`，简述用中文，正文按需列要点。

| 类型 | 用途 |
|---|---|
| feat | 新功能 |
| fix | 修复问题 |
| docs | 方案、说明文档 |
| test | 测试 |
| refactor | 不改行为的重构 |
| data | 新增数据集版本 |
| exp | 登记实验结果（修改 `docs/experiment-log.md`） |
| chore | 配置、依赖、CI 等杂项 |

## 每个 PR 的检查清单

- [ ] `pytest -q` 通过
- [ ] 改了数据生成逻辑：新建了数据版本目录，`boltcheck verify-data` 通过
- [ ] 改了提示词：提高了 `PROMPT_VERSION`
- [ ] 改了方案：更新了 `docs/experiment-plan.md` 的修订记录表和 `PLAN_VERSION`
- [ ] 在 `CHANGELOG.md` 记录了改动

## 实验运行与登记

- 每次运行的完整记录在 `runs/<运行 ID>/`（不进仓库）：`meta.json` 保存 Git 提交、数据集哈希、模型与采样参数、提示词版本；`outputs.jsonl` 保存每条原始回复。
- 正式实验前先提交代码。工作区有未提交改动时，运行会在 `meta.json` 中记 `dirty: true`，日志行的提交号后面会带 `*`，这样的结果不作为正式结论。
- 正式实验跑完后，把命令行最后打印的"实验日志行"粘贴到 `docs/experiment-log.md`，用 `exp:` 类型提交。需要长期保存的运行目录，压缩后另行存档（网盘或 Release 附件），在日志备注里写明位置。
- 试跑（`--limit`）不登记，或在备注中注明"试跑"。
