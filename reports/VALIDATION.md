# 验收记录

日期：2026-10-23（会话日期）。

## 已实际运行

- Python 3.12 / PyTorch 2.14.1+cpu / NumPy 2.3.5 / FAISS 1.15.1；无 CUDA。
- `make demo`：10,000 条合成曝光，200 用户、200 物品，训练/验证/测试 7,000 / 1,500 / 1,500。
- 召回与精排各训练 3 epoch，保存 last 和 validation-best checkpoint，再进行 test 评测。
- 原始指标见 `synthetic-metrics.json`，训练日志见两个 JSONL 文件。
- `pytest`：11 passed、1 skipped。跳过的是多 worker DataLoader：当前 host 禁止 tensor IPC 需要的 Unix socket。未将失败掩盖为已通过。
- 因果 mask、严格历史时间、训练词表隔离、缺失标签、极端联合转化 loss、小目录 HNSW、全量精确 top-k、模型加载、API 输入验证均通过。
- `ruff check` 通过。
- 默认索引构建完成；带/不带 ANN 的服务入口都有测试/运行验证。
- FastAPI TestClient 100 次请求，10 次预热；单线程 CPU、候选 50、返回 10，p50≈7.48ms、p95≈11.65ms、p99≈12.92ms。仅进程内接口耗时，非网络 QPS，见 `latency-synthetic.json`。

## 结果解读

| 指标 | 本次值 |
|---|---:|
| CTR AUC（4096 桶近似） | 0.4982 |
| CTR LogLoss | 0.4851 |
| 召回 Hit@10 | 0.0789 |
| 召回 NDCG@10 | 0.0407 |
| 候选 Recall@50 | 0.2975 |
| 精排后 Hit@10 | 0.0681 |
| 精排后 NDCG@10 | 0.0351 |
| Popularity Hit@10 | 0.0466 |

短训练合成实验中精排未超过召回排序，CTR AUC 接近随机。**本记录用于工程链路验收，不支持“效果提升”结论。**

## GitHub CI 补充验收

[Actions run 39933028074](https://github.com/WRF2004/seqrank/actions/runs/39933028074) 全部通过，验证源码提交 `787e293bc768177b4460bbab3b410ee7228da622`：

- `ruff` 通过。
- **12 tests passed**，包括本地受限而跳过的多 worker DataLoader。
- 完整 `make demo` 通过。
- 召回与精排各完成 **2 进程 CPU DDP / 3 epoch**，world_size=2；这不是多 GPU 验证。
- 合成数据指标已作为 Actions artifact 上传。

## 未完成的验证

- 本地双进程 torchrun 因 host 网络隔离无法建立 rendezvous；后续已在 GitHub runner 完成 CPU DDP 验证。
- 没有真实数据训练，没有 CUDA AMP / 多卡 / 大目录 ANN / 工业压测 / Docker runtime 验证。
- 没有训练集之外的新物品冷启动能力，没有生产高可用部署。
- 上述代码提交的 CI 已成功；后续版本请查看对应 Actions 状态。

复现依赖版本见 `environment.txt`。随机种子固定，但跨平台浮点与计时会变化。
