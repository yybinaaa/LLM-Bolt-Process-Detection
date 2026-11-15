# 本地运行指南

按顺序做下来，大约十分钟能跑通第一组正式实验。

## 1. 安装

本工具只依赖 Python 标准库（3.10 及以上），测试需要 pytest。

```bash
git clone https://github.com/yybinaaa/LLM-Bolt-Process-Detection.git
cd LLM-Bolt-Process-Detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                          # 应全部通过
```

vLLM 依赖较重，建议装在单独的环境里，只负责起模型服务。vLLM 需要 Linux，Windows 上请用 WSL2。

## 2. 检查数据

```bash
boltcheck verify-data                              # 校验 data/v1.1 可复现、标签正确、配额满足
boltcheck show-prompt --group joint --id v1.1-0001 # 看看发给模型的提示词
```

`data/v1.1/dataset.jsonl` 每行一条样本，`manifest.json` 记录生成配置、哈希和各分组的样本数。

## 3. 直观查看样本

把坐标序列展开成 8×8 表格，每格标上安装顺序（第几步）。

```bash
boltcheck show-sample v1.1-0005 v1.1-0008   # 在终端打印表格：* 违规步，! 首次违规步，下方附层级表
boltcheck view                              # 生成 data/v1.1/view.html，用浏览器打开
```

网页查看器里每条样本是一张 8×8 表格：底色深浅表示层级（中心深、外圈浅），红字是违规步，红框是首次违规步；拖动下方滑块或点"播放"，可以看到螺栓按顺序一颗颗拧上去。可以按结论、构造方式、跨层距离、首次违规位置、边界难例筛选。

## 4. 不用 GPU 先跑通流水线

```bash
boltcheck run --backend baseline --group joint     # 第 1 组：程序规则基线，所有准确率应为 100%
boltcheck run --backend noisy --group joint        # 模拟各种错误输出，看看报告长什么样
```

## 5. 启动模型服务

部署约束是推理服务显存 3～5GB。开发机是 8GB 显卡，下面的命令都用 `--gpu-memory-utilization 0.6`（8GB × 0.6 ≈ 4.8GB）把 vLLM 限制在 5GB 以内；放不下指定的上下文长度时，vLLM 启动就会报错，相当于一次硬性检验。`--enforce-eager` 关闭 CUDA Graph，能省几百 MB 显存，代价是稍慢。

一次只起一个模型，都用 8000 端口，换模型时先停掉上一个。`--served-model-name` 必须和 `configs/models.toml` 里的 `model` 一致。

**主测：Qwen3-4B-Instruct-2507，INT4**（预估约 4～4.5GB）

```bash
vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit \
  --served-model-name qwen3-4b-instruct-2507-int4 \
  --port 8000 --max-model-len 4096 \
  --gpu-memory-utilization 0.6 --enforce-eager
```

**跨家族对比：Phi-4-mini-instruct，INT4**（预估约 4～4.5GB，加载时量化，启动会慢一些）

```bash
vllm serve microsoft/Phi-4-mini-instruct \
  --quantization bitsandbytes \
  --served-model-name phi-4-mini-instruct-int4 \
  --port 8000 --max-model-len 4096 \
  --gpu-memory-utilization 0.6 --enforce-eager
```

**低配档与思考模式对比：Qwen3-1.7B，INT8**（非思考约 3GB，思考约 4.5～5GB）

思考模式输出很长，上下文要开到 18432；同一个服务既能跑非思考也能跑思考，由请求里的 `enable_thinking` 决定。

```bash
vllm serve Qwen/Qwen3-1.7B-GPTQ-Int8 \
  --served-model-name qwen3-1.7b-int8 \
  --port 8000 --max-model-len 18432 \
  --gpu-memory-utilization 0.6 --enforce-eager \
  --reasoning-parser qwen3
```

如果启动时报 KV cache 放不下，可以加 `--kv-cache-dtype fp8` 把推理缓存减半（需要显卡和 vLLM 版本支持），或者只跑非思考模式时把 `--max-model-len` 改成 4096。旧版 vLLM 的推理解析器写法是 `--enable-reasoning --reasoning-parser deepseek_r1`。

**可选，量化对照：Qwen3-1.7B，原精度**（只跑非思考模式，预估约 4.5GB）

```bash
vllm serve Qwen/Qwen3-1.7B \
  --served-model-name qwen3-1.7b-bf16 \
  --port 8000 --max-model-len 4096 \
  --gpu-memory-utilization 0.6 --enforce-eager
```

确认服务正常：

```bash
curl http://localhost:8000/v1/models
```

## 6. 正式运行

建议顺序：

1. 先试跑 5 条，看原始回复是否正常：

   ```bash
   boltcheck run --model qwen3-4b-instruct-2507-int4 --profile greedy --group joint --limit 5 --note 试跑
   ```

   打开 `runs/<运行 ID>/outputs.jsonl` 看 `text` 字段。如果大量出现重复内容或长度截断，换用 `--profile sampled`。

2. 主测模型跑四组（第 2、3、4、5 组）：

   ```bash
   boltcheck run --model qwen3-4b-instruct-2507-int4 --profile greedy --group joint   --gpu-monitor
   boltcheck run --model qwen3-4b-instruct-2507-int4 --profile greedy --group restore --gpu-monitor
   boltcheck run --model qwen3-4b-instruct-2507-int4 --profile greedy --group judge   --gpu-monitor
   ```

   第 5 组两阶段：模型先把坐标复原成标好安装顺序的表格，再拿这张表按规则判断。每条样本调用两次，耗时约为其他组的两倍：

   ```bash
   boltcheck run --model qwen3-4b-instruct-2507-int4 --profile greedy --group pipeline --gpu-monitor
   ```

   第一阶段没得到 8×8 矩阵时不做第二阶段，记为格式无效。报告里的"复原正确 / 错误时的判断准确率"可以看出复原出错对判断的影响；和第 4 组（给标准矩阵判断）比较，差距就是复原误差传递造成的损失。`boltcheck show-prompt --group pipeline` 可以看两个阶段的提示词。

3. 换成 Phi-4-mini，同样四组，`--model phi-4-mini-instruct-int4`。

4. 换成 Qwen3-1.7B，先跑非思考（`--profile greedy`），再跑思考（`--profile thinking`，每条样本 3 次，耗时明显更长）：

   ```bash
   boltcheck run --model qwen3-1.7b-int8 --profile greedy   --group joint --gpu-monitor
   boltcheck run --model qwen3-1.7b-int8 --profile thinking --group joint --gpu-monitor
   ```

5. 每次运行结束，命令行最后会打印一行实验日志。正式结果把它粘贴到 `docs/experiment-log.md`，用 `exp:` 类型提交。报告和 `meta.json` 会记下权重来源和量化方式。

| 方案中的组 | 命令参数 |
|---|---|
| 第 1 组 程序规则基线 | `--backend baseline --group joint` |
| 第 2 组 联合任务 | `--group joint` |
| 第 3 组 仅做还原 | `--group restore` |
| 第 4 组 仅做判断 | `--group judge` |
| 第 5 组 两阶段（先复原，再按自己复原的表格判断） | `--group pipeline` |

## 7. 中断与续跑

运行中断（断电、服务崩溃、Ctrl+C）后，用原来的运行目录续跑，只会补跑缺失和出错的调用：

```bash
boltcheck run --resume runs/<运行 ID>
```

改了评测代码后，可以对已有运行重新评测，不用重新调用模型：

```bash
boltcheck evaluate runs/<运行 ID>
```

## 8. 运行目录里有什么

| 文件 | 内容 |
|---|---|
| `meta.json` | Git 提交号、是否有未提交改动、数据集哈希、模型与采样参数、提示词版本 |
| `outputs.jsonl` | 每条样本每次重复一行：原始回复、推理内容、token 数、延迟、峰值显存、错误信息；第 5 组另有 `stage1`、`stage2` 两个字段，分别保存两次调用的原始回复 |
| `metrics.json` | 全部指标、错误归因、分组结果 |
| `report.md` | 便于阅读的报告 |
| `log_row.md` | 粘贴到实验日志的一行 |
| `view.html` | 网页查看器：标准答案与模型复原的矩阵并排，复原错的格子标红并给出正确值，可筛选判断错误、复原错误、格式无效、行列写反的样本 |

## 9. 常见问题

- **连接失败**：先用 `curl` 确认服务已启动、端口正确。本地地址（localhost、127.0.0.1）会自动绕过系统代理。
- **显存不够**：调小 `--max-model-len`，或加 `--kv-cache-dtype fp8`。注意 `--gpu-memory-utilization` 是占整张卡的比例，桌面程序占用的显存也要算进去。
- **显存读数偏高**：vLLM 启动时会按 `--gpu-memory-utilization` 预先占满给它的显存，`--gpu-monitor` 记录的是整张卡的实际占用（含其他程序），主要用于确认服务没有超过 5GB，不等于模型的最小需求。
- **用 Ollama、llama.cpp 或 LM Studio 代替 vLLM**：如果最终部署在 3～5GB 的小卡上，这类运行方式通常比 vLLM 更省显存。它们都提供 OpenAI 兼容接口，在 `configs/models.toml` 里新增一个模型条目，把 `base_url` 和 `model` 改成对应的地址和名称即可。部分采样参数（如 `top_k`、`seed`）可能不生效，结果需在备注中注明。
- **用远程 API 做上限参考**：在 `configs/models.toml` 的 `api-reference` 里填好地址和模型名，设置环境变量 `LLM_API_KEY` 后运行 `--model api-reference --profile greedy`。
