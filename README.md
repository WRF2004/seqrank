# SeqRank：序列召回、多任务精排与观看时长研究系统

面向公开日志、可复现实验和推荐算法学习的 Python / PyTorch 工程。v2 在原有 Transformer、DIN、MMoE、ESMM、FAISS、DDP、mmap 基础上增加真实数据协议与时长专项。**本次升级只编写代码与测试，未运行测试、训练或真实性能实验；不能视作已验证的生产系统。** 历史报告保留在 `reports/`，不代表 v2 的性能。

核心能力：

- KuaiRand / CSV / Parquet 转换、严格全局时间划分、训练词表、延迟反馈历史、OOV 和数据质量报告。
- Popularity、ItemKNN、GRU4Rec-inspired、SASRec-inspired；精确全目录 Top-K、可选 FAISS HNSW。
- DNN、DIN-style、Shared-Bottom、MMoE；真实 CTR、可选 ESMM、六类观看时长目标。
- CREAD-inspired 自适应离散化、阈值分类、秒数恢复、序约束；训练集拟合边界与视频时长分组分析。
- 磁盘外排序精确 AUC / GAUC / AP、ECE、秒数误差、分组校准、固定候选集代理评估。
- CPU / 单 GPU / DDP、BF16 / FP16、梯度累积、epoch 边界恢复、模型/数据/索引指纹。
- 配置驱动实验矩阵、多种子汇总、梯度冲突诊断、FastAPI 与批量推理。

**标签语义**：KuaiRand 的 `is_click` 在不同展示界面可能表示点击或有效播放。本项目默认实验为 **Valid-Play + Watch Time**，报告不会将其命名为 CTR。真正的 CTR + Watch Time 需要核实并筛选 click 语义流量，或使用有真实点击标签的通用数据。KuaiRand 不启用购买转化头。

## 安装

Python 3.10+（建议 3.11/3.12），在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU 环境；GPU 环境用与你的驱动兼容的官方 PyTorch CUDA wheel 替换此步
pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,data,ann]'
```

GPU 安装方式见 [REPRODUCE](docs/REPRODUCE.md)。FAISS 为 CPU HNSW，训练仍可使用 GPU。没有下载模型或隐式合成数据回退。依赖使用兼容范围而非假称经过 H200 验证的锁文件；实验时保存 `pip freeze`。

## 从已下载的真实数据开始

推荐首先用 KuaiRand-Pure 或 1K 的少量用户，跑一轮 CPU/单 GPU 闭环。Pure 的序列本身不完整，结果不能冒充完整日志实验。先确认你解压后的文件名、覆盖日期和字段，按需修改 `adapter.sources` 与时间分界。

```bash
export RAW_ROOT=/path/to/KuaiRand-Pure
export DATA_ROOT=/path/to/seqrank-data/debug
export RUN_ROOT=/path/to/seqrank-runs/debug
export GPU_IDS=0                      # 只使用你获分配的卡；默认配置仍为 CPU
export CONFIG=configs/datasets/kuairand_debug.yaml
bash scripts/research_pipeline.sh
```

该脚本依次检查、转换、预处理、校验哈希、报告质量、训练两阶段模型、验证/测试、候选重排评估。输出目录必须是新的；重复执行时明确选取后续命令或更换输出目录，避免覆盖旧实验。

每一步也可独立执行：

```bash
python -m seqrank.cli schema --config "$CONFIG"
python -m seqrank.cli convert --config "$CONFIG"
python -m seqrank.cli prepare-data --config "$CONFIG"
python -m seqrank.cli verify-data --config "$CONFIG"
python -m seqrank.cli quality --config "$CONFIG" --set quality.plots=true
python -m seqrank.cli train-retrieval --config "$CONFIG"
python -m seqrank.cli evaluate-retrieval --config "$CONFIG" --split valid
python -m seqrank.cli evaluate-retrieval --config "$CONFIG" --split test
python -m seqrank.cli train-ranking --config "$CONFIG"
python -m seqrank.cli evaluate-ranking --config "$CONFIG" --split valid
python -m seqrank.cli evaluate-ranking --config "$CONFIG" --split test
python -m seqrank.cli evaluate-e2e --config "$CONFIG" --split test
```

默认使用训练日志流量 `log_standard*.csv`；随机曝光数据必须单独配置、单独预处理和报告，不自动生成 IPS。

## 单卡、可选 DDP 和时长专项

```bash
export CUDA_VISIBLE_DEVICES=2   # 示例：调度器分配给你的物理卡，进程内对应 cuda:0
# 已经准备好的数据可复用；以下配置不重新预处理
python -m seqrank.cli train-retrieval --config configs/h200.yaml
python -m seqrank.cli train-ranking --config configs/h200.yaml --set watchtime.mode=cread_inspired
# 恢复时，保持原训练配置、world size 与总 epochs 不变
python -m seqrank.cli train-ranking --config configs/h200.yaml --set watchtime.mode=cread_inspired \
  --resume "$RUN_ROOT/ranking/last.pt"
# 可选两卡；不会自动占满 8 卡
GPU_IDS=2,3 NPROC=2 STAGE=ranking bash scripts/research_ddp.sh
```

不同方法使用不同 `RUN_ROOT`。单卡训练默认 BF16、batch 256，仍应根据你的物品规模和共享资源调整；不承诺特定显存或吞吐。断点恢复仅支持 **epoch 边界**。

## 模型对比、消融与报告

```bash
# 不加 --execute 只写可审阅的逐种子 resolved.json
python -m seqrank.cli experiments --config configs/experiments/retrieval.yaml
python -m seqrank.cli experiments --config configs/experiments/retrieval.yaml --execute
python -m seqrank.cli report --config configs/experiments/retrieval.yaml
python -m seqrank.cli experiments --config configs/experiments/ablation.yaml --execute
python -m seqrank.cli report --config configs/experiments/ablation.yaml
python -m seqrank.cli diagnose-gradients --config "$CONFIG"
```

矩阵包含四类召回、均匀/流行度负采样，DNN、DIN、Dwell-only、Shared-Bottom、MMoE、raw/log/normalized/bucket/CREAD-inspired/duration-conditioned 时长方法，以及注意力/序约束/恢复项消融。详情与 CTR 版本配置方式见 [EXPERIMENTS](docs/EXPERIMENTS.md)。矩阵按顺序运行，不并发占用所有 GPU。

## 索引、服务和性能测量

```bash
python -m seqrank.cli build-ann --config "$CONFIG"
python -m seqrank.cli benchmark-ann --config "$CONFIG"
python -m seqrank.cli benchmark --config "$CONFIG"
python -m seqrank.cli serve-research --config "$CONFIG"
```

API 文档：`http://127.0.0.1:8000/docs`。`/health` 返回 dataset/model 指纹；请求必须携带 `dataset_version`。`/predict` 处理已知曝光特征，`/recommend` 和 `/recommend-batch` 从训练物品目录召回再精排。默认本机监听；不提供身份认证或公网部署。

## 你在服务器上应运行的验证

```bash
pytest -q tests/research
pytest -q                    # 含旧兼容测试；本次均未运行
```

`tests/research` 的 CPU 集成测试自行在临时目录生成确定性合成样本，贯穿预处理、训练、独立评估、检查点、API；不要求某个业务提升。真实实验命令不引用此测试数据。

## 阅读与迁移

- [审查清单](docs/UPGRADE_AUDIT.md)、[架构](docs/ARCHITECTURE.md)、[数据契约](docs/DATASET.md)
- [模型与论文差异](docs/MODELS.md)、[损失公式](docs/LOSSES.md)、[实验协议](docs/EXPERIMENTS.md)
- [服务器复现](docs/REPRODUCE.md)、[问题排查](docs/TROUBLESHOOTING.md)、[源码导读](docs/CODE_WALKTHROUGH.md)

旧 `demo-data / prepare / train / evaluate / build-index / serve` 保留，见 [v1 README](docs/LEGACY.md)。v2 使用独立 schema/checkpoint，不能混装。旧 KuaiRand 转换脚本现在要求 `--verified-click-semantics`；建议迁移到显式 tab 语义的 v2 adapter。v1 的常驻 CVR 头与融合逻辑仅保留兼容，不用于无转化标签的真实实验。

限制：本地 SQLite 预处理与精确评估更重正确性而非集群吞吐；ItemKNN、全 embedding AdamW、全目录精确评估仍受单机内存限制。没有稀疏分布式 embedding、在线实验平台或严格 CWM 复现。不能由本项目离线指标推断在线 CTR、观看时长或因果收益。
