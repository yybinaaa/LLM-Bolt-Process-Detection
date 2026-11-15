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

## 3. 不用 GPU 先跑通流水线

```bash
boltcheck run --backend baseline --group joint     # 第 1 组：程序规则基线，所有准确率应为 100%
boltcheck run --backend noisy --group joint        # 模拟各种错误输出，看看报告长什么样
```

## 4. 启动模型服务

8GB 显存一次只起一个模型，两个模型都用 8000 端口，换模型时先停掉上一个。

Instruct 模型（主测）：

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --served-model-name qwen3-vl-2b-instruct \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Thinking 模型（思考模式对比），要加推理解析器，并放大上下文长度：

```bash
vllm serve Qwen/Qwen3-VL-2B-Thinking \
  --served-model-name qwen3-vl-2b-thinking \
  --port 8000 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3
```

确认服务正常：

```bash
curl http://localhost:8000/v1/models
```

`--served-model-name` 必须和 `configs/models.toml` 里的 `model` 一致。第一版只输入文字，显存紧张时可以关掉图像和视频输入来节省显存，例如加 `--limit-mm-per-prompt '{"image": 0, "video": 0}'`；这个参数的写法随 vLLM 版本不同，以 `vllm serve --help` 为准。

## 5. 正式运行

建议顺序：

1. 先试跑 5 条，看原始回复是否正常：

   ```bash
   boltcheck run --model qwen3-vl-2b-instruct --profile greedy --group joint --limit 5 --note 试跑
   ```

   打开 `runs/<运行 ID>/outputs.jsonl` 看 `text` 字段。如果大量出现重复内容或长度截断，换用 `--profile instruct-sampled`。

2. Instruct 模型跑三组：

   ```bash
   boltcheck run --model qwen3-vl-2b-instruct --profile greedy --group joint   --gpu-monitor
   boltcheck run --model qwen3-vl-2b-instruct --profile greedy --group restore --gpu-monitor
   boltcheck run --model qwen3-vl-2b-instruct --profile greedy --group judge   --gpu-monitor
   ```

3. 换成 Thinking 模型，同样跑三组（每条样本 3 次，耗时明显更长）：

   ```bash
   boltcheck run --model qwen3-vl-2b-thinking --profile thinking --group joint --gpu-monitor
   ```

4. 每次运行结束，命令行最后会打印一行实验日志。正式结果把它粘贴到 `docs/experiment-log.md`，用 `exp:` 类型提交。

| 方案中的组 | 命令参数 |
|---|---|
| 第 1 组 程序规则基线 | `--backend baseline --group joint` |
| 第 2 组 联合任务 | `--group joint` |
| 第 3 组 仅做还原 | `--group restore` |
| 第 4 组 仅做判断 | `--group judge` |

## 6. 中断与续跑

运行中断（断电、服务崩溃、Ctrl+C）后，用原来的运行目录续跑，只会补跑缺失和出错的调用：

```bash
boltcheck run --resume runs/<运行 ID>
```

改了评测代码后，可以对已有运行重新评测，不用重新调用模型：

```bash
boltcheck evaluate runs/<运行 ID>
```

## 7. 运行目录里有什么

| 文件 | 内容 |
|---|---|
| `meta.json` | Git 提交号、是否有未提交改动、数据集哈希、模型与采样参数、提示词版本 |
| `outputs.jsonl` | 每条样本每次重复一行：原始回复、推理内容、token 数、延迟、峰值显存、错误信息 |
| `metrics.json` | 全部指标、错误归因、分组结果 |
| `report.md` | 便于阅读的报告 |
| `log_row.md` | 粘贴到实验日志的一行 |

## 8. 常见问题

- **连接失败**：先用 `curl` 确认服务已启动、端口正确。本地地址（localhost、127.0.0.1）会自动绕过系统代理。
- **显存不够**：调小 `--max-model-len` 或 `--gpu-memory-utilization`，或关掉多模态输入。
- **显存读数偏高**：vLLM 启动时会按 `--gpu-memory-utilization` 预先占满显存，`--gpu-monitor` 记录的是实际占用总量，主要用于确认没有超过 8GB，不等于模型的最小需求。
- **用 Ollama 或 LM Studio 代替 vLLM**：两者都提供 OpenAI 兼容接口，在 `configs/models.toml` 里新增一个模型条目，把 `base_url` 和 `model` 改成对应的地址和名称即可。部分采样参数（如 `top_k`、`seed`）可能不生效，结果需在备注中注明。
- **用远程 API 做上限参考**：在 `configs/models.toml` 的 `api-reference` 里填好地址和模型名，设置环境变量 `LLM_API_KEY` 后运行 `--model api-reference --profile greedy`。
