# LLM-Bolt-Process-Detection

用大模型判断工人安装高强螺栓的过程是否符合作业要求。输入为时序图片或视频；推理服务显存控制在 3～5GB（开发机为 8GB 显卡）。

## 当前阶段

第一版实验：以模拟坐标代替目标检测结果，验证文字大模型能否还原 8×8 螺栓节点的初拧顺序，并按"由中间向四周"的分层规则判断是否合规。本地主测模型为 Qwen3-4B-Instruct-2507（INT4），对比 Phi-4-mini-instruct（INT4）和 Qwen3-1.7B（INT8，含思考模式）。

## 快速开始

```bash
pip install -e ".[dev]"
pytest -q
boltcheck verify-data                               # 校验数据集
boltcheck show-sample v1.1-0005                     # 在终端把样本展开成 8×8 表格，标出安装顺序
boltcheck view                                      # 生成网页查看器 data/v1.1/view.html
boltcheck run --backend baseline --group joint      # 不用 GPU，跑通第 1 组程序规则基线
```

接上本地 vLLM 后运行正式实验：

```bash
boltcheck run --model qwen3-4b-instruct-2507-int4 --profile greedy --group joint --gpu-monitor
```

完整步骤见 [本地运行指南](docs/running.md)。

## 文档

- [实验方案](docs/experiment-plan.md)：当前版本 v1.3，修改历史见 Git 提交记录
- Word 版本：[v1.3](docs/experiment-plan-v1.3.docx)（当前）、[v1.2](docs/experiment-plan-v1.2.docx)、[v1.1](docs/experiment-plan-v1.1.docx)、[v1.0](docs/experiment-plan-v1.0.docx)（原稿）
- [本地运行指南](docs/running.md)
- [实验日志](docs/experiment-log.md)：正式实验结果登记
- [迭代规范](CONTRIBUTING.md)：版本号、分支、提交信息、实验登记
- [变更日志](CHANGELOG.md)

## 仓库结构

```
configs/
  dataset_v1.1.toml           数据集生成配置
  models.toml                 模型地址与采样配置
data/v1.1/                    数据集（dataset.jsonl + manifest.json），已提交、可复现
src/boltcheck/
  grid.py                     坐标、层级、步骤矩阵约定
  rules.py                    程序规则基线：违规定义与分组标签
  perturb.py                  交换 / 插入 / 整段提前三种违规构造
  dataset.py                  按配额生成与校验数据集
  prompts.py                  四组对照的提示词
  backends.py                 模型调用（OpenAI 兼容接口 / 规则基线 / 模拟错误）
  runner.py                   执行运行、记录原始输出与运行信息
  parse.py                    解析模型输出、校验格式
  evaluate.py                 指标、自洽率、错误归因、分组报告
  viewer.py                   8×8 表格可视化（网页查看器、终端表格）
  cli.py                      命令行入口
tests/                        单元测试与端到端测试
docs/                         方案、运行指南、实验日志
runs/                         运行输出（不进仓库）
```
