# LLM-Bolt-Process-Detection

用大模型判断工人安装高强螺栓的过程是否符合作业要求。输入为时序图片或视频，部署约束为单卡 8GB 显存。

## 当前阶段

第一版实验：以模拟坐标代替目标检测结果，验证文字大模型能否还原 8×8 螺栓节点的初拧顺序，并按"由中间向四周"的分层规则判断是否合规。

## 文档

- [实验方案](docs/experiment-plan.md)：当前版本 v1.0，修改历史见 Git 提交记录
- Word 版本：[v1.0](docs/experiment-plan-v1.0.docx)
- [变更日志](CHANGELOG.md)

## 仓库结构

```
docs/
  experiment-plan.md          实验方案（持续更新，以此为准）
  experiment-plan-vX.Y.docx   各版本的 Word 文档
CHANGELOG.md                  变更日志
```
