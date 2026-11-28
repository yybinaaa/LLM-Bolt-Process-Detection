# LLM-Bolt-Process-Detection

用大模型判断工人安装高强螺栓的过程是否符合作业要求。输入为时序图片或视频；推理服务显存控制在 3～5GB（开发机为 8GB 显卡）。

## 项目概述

我们计划先针对普通对接钢板上的高强螺栓初拧顺序开展实验。假设钢板上有 8×8、共 64 颗螺栓。第一阶段暂不接入目标检测模型，而是模拟生成每颗螺栓的二维坐标，并按照初拧先后顺序排列，形成一条坐标序列。坐标只表示位置，不预先标注每颗螺栓属于哪一层，也不需要在实际螺栓上标注编号。

随后由文字大模型分两步完成判断。第一步，将完整的坐标序列输入大模型，由它根据全部坐标复原螺栓的 8×8 空间布局，并结合坐标在序列中出现的先后，在每个位置标出对应的操作步骤，形成一张步骤表。第二步，将这张步骤表和初拧判断规则一起输入大模型，由它依据提示词中给出的划分方法判断各位置所在的层，再结合操作步骤检查是否满足由中心向四周施拧的要求。

为了让第一版实验有明确的标准答案，我们暂时将阵列由内向外划分为四层，规定同一层内顺序不限，但须完成内层后才能进入外层。大模型据此输出顺序是否合规；如果不合规，需要指出首次违规的步骤、位置和判断依据。

实验中将生成多种合规和不合规的坐标序列，并通过独立的规则程序生成标准答案，分别与大模型的布局还原结果和规则判断结果进行比较；同时设置只做还原、只做判断等对照实验，以区分错误出在还原环节还是判断环节。这里的严格四层划分只是便于实验的数据与判定假设，不能直接等同于实际工程规范或验收规则。等第一阶段验证完成后，再考虑接入目标检测模型获取真实坐标，并逐步研究终拧和视觉大模型方案。

## 当前进度

第一版实验：以模拟坐标代替目标检测结果，验证文字大模型能否还原 8×8 螺栓节点的初拧顺序，并按"由中间向四周"的分层规则判断是否合规。本地主测模型为 Qwen3-4B-Instruct-2507（INT4），对比 Phi-4-mini-instruct（INT4）和 Qwen3-1.7B（INT8，含思考模式）。判断流程的主线是第 5 组：模型先把坐标复原成标好安装顺序的 8×8 表格，再按这张表对照规则判断。

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

## 本地视觉判断台

判断台使用浏览器界面选择图片、输入提示词和切换模型。要用本机 Ollama 看图，先启动 Ollama。首次使用时可通过 `ollama pull qwen2.5vl:3b` 下载图文模型。

在仓库根目录使用已配置的 Python 环境运行以下命令，然后打开 [本地视觉判断台](http://127.0.0.1:8765/)。

```powershell
.\.venv\Scripts\python.exe .\tools\vision_ui\server.py
```

在 Linux 或 macOS 上，激活项目的 Python 环境后运行 `python tools/vision_ui/server.py`。可用 `--port` 指定网页端口。

在「模型」下拉菜单选择本机 Ollama、vLLM 或远程 API。Ollama 已下载的本地模型会自动列出，启动 vLLM 或新增模型后可点「刷新模型」。图文模型支持拖入、点选或粘贴图片，文字模型只提交提示词。换模型时，图片和提示词留在页面。

远程模型可在页面临时填入密钥，也可按 [本地运行指南](docs/running.md) 配置对应环境变量。提交前页面会显示接收数据的服务商域名。添加其他 OpenAI 兼容模型时，修改 [模型配置](configs/models.toml) 并在页面刷新模型列表。图文模型须在对应条目中设置 `vision = true`。

前端由原生 HTML、CSS 和 JavaScript 实现。Python 服务监听 `127.0.0.1`，提供模型列表和判断接口，并将请求转发到选中的 Ollama、vLLM 或 OpenAI 兼容 API。模型的实际名称显示在回答上方。 [判断台源码](tools/vision_ui/)

## 文档

- [实验方案](docs/experiment-plan.md)：当前版本 v1.15（提示词 v1.6），修改历史见 Git 提交记录
- Word 版本：[v1.15](docs/experiment-plan-v1.15.docx)（当前）、[v1.14](docs/experiment-plan-v1.14.docx)、[v1.13](docs/experiment-plan-v1.13.docx)、[v1.12](docs/experiment-plan-v1.12.docx)、[v1.11](docs/experiment-plan-v1.11.docx)、[v1.10](docs/experiment-plan-v1.10.docx)、[v1.9](docs/experiment-plan-v1.9.docx)、[v1.8](docs/experiment-plan-v1.8.docx)、[v1.7](docs/experiment-plan-v1.7.docx)、[v1.6](docs/experiment-plan-v1.6.docx)、[v1.5](docs/experiment-plan-v1.5.docx)、[v1.4](docs/experiment-plan-v1.4.docx)、[v1.3](docs/experiment-plan-v1.3.docx)、[v1.2](docs/experiment-plan-v1.2.docx)、[v1.1](docs/experiment-plan-v1.1.docx)、[v1.0](docs/experiment-plan-v1.0.docx)（原稿）
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
