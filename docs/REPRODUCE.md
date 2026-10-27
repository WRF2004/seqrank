# Linux 服务器从零复现

## 1. 安装与资源

```bash
git clone https://github.com/WRF2004/seqrank.git
cd seqrank
git checkout upgrade/research-v2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CPU：`pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu`。

GPU：先用服务器驱动/调度环境确定兼容的官方 wheel 地址，再安装，避免在共享机更换系统 CUDA/驱动：

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124  # 示例，须与你选定 PyTorch/驱动兼容
pip install 'torch>=2.4,<3' --index-url "$TORCH_INDEX_URL"
pip install -e '.[dev,data,ann]'
export CUDA_VISIBLE_DEVICES=2   # 仅分配给你的卡；进程内 cuda:0
```

该 URL 仅是安装路径示例，不声称其是最新 wheel 或适配你的驱动。官方安装选择器：https://pytorch.org/get-started/locally/ 。不要安装会改变其他用户运行环境的系统包。依赖范围在 pyproject.toml；本次未在 H200 上验证具体版本组合。FAISS 用 CPU，无需 GPU FAISS。

## 2. 数据下载完成后

```bash
export RAW_ROOT=/datasets/KuaiRand-Pure
export DATA_ROOT=/work/$USER/seqrank/debug-data
export RUN_ROOT=/work/$USER/seqrank/debug-run
export CONFIG=configs/datasets/kuairand_debug.yaml
python -m seqrank.cli schema --config "$CONFIG"
```

核对字段/日期。默认 sources 匹配 log_standard*.csv，不混随机日志。默认 train_end=2022-04-22 00:00+08、valid_end=2022-05-01 00:00+08，确保三个 split 都覆盖你的 release。需要内容 tag 时按 DATASET.md 开启 immutable join；默认仅使用日志可用内容，不以缺标签填 0。

```bash
python -m seqrank.cli convert --config "$CONFIG"
python -m seqrank.cli prepare-data --config "$CONFIG"
python -m seqrank.cli verify-data --config "$CONFIG"
python -m seqrank.cli quality --config "$CONFIG" --set quality.plots=true
```

查看 quality.json：实际用户/物品/曝光、primary 正负比例、missing、history、时长和 OOV；不正常就先修数据语义/切分。转换输出与 prepared 非空目录拒绝覆盖；修改适配/特征/历史协议后换目录，不改已有 manifest。

## 3. 最小真实闭环

debug 默认 2% 用户（保留其全部输入事件）、max_len20、dim32、1 epoch、CPU。Pure/1K 若抽样太小导致空 split，用 `--set adapter.user_sample_fraction=1` 转换到新目录。首次可先跑 CPU；如目录太大转单卡，并降低 batch/catalog_chunk。

```bash
python -m seqrank.cli train-retrieval --config "$CONFIG"
python -m seqrank.cli train-ranking --config "$CONFIG" --set watchtime.mode=cread_inspired
python -m seqrank.cli evaluate-retrieval --config "$CONFIG" --split valid
python -m seqrank.cli evaluate-ranking --config "$CONFIG" --split valid
python -m seqrank.cli evaluate-retrieval --config "$CONFIG" --split test
python -m seqrank.cli evaluate-ranking --config "$CONFIG" --split test
python -m seqrank.cli evaluate-e2e --config "$CONFIG" --split test
pip freeze > "$RUN_ROOT/environment.txt"
```

评价阶段通过 checkpoint 恢复 CREAD-inspired 配置；无需重写评估 YAML。运行目录有输出则拒绝覆盖，重复评估选择新 evaluation.output。

## 4. 扩展到单 H200

复用同一 prepared 数据比较硬件/模型；若扩大训练用户则另建 DATA_ROOT 重新转换。单卡配置：

```bash
export RUN_ROOT=/work/$USER/seqrank/h200-run
python -m seqrank.cli train-retrieval --config configs/h200.yaml
python -m seqrank.cli train-ranking --config configs/h200.yaml --set watchtime.mode=cread_inspired
python -m seqrank.cli evaluate-ranking --config configs/h200.yaml --split test
python -m seqrank.cli diagnose-gradients --config configs/h200.yaml
```

BF16 无 GradScaler；FP16 使用 scaler；FP32 支持 CPU。若要限制卡，设置 CUDA_VISIBLE_DEVICES 而不是让程序自行发现全部 GPU。模型/训练配置在 resolved checkpoint 中保存；日志含 epoch 秒数、样本吞吐（含 sampler 补齐）和 rank0 峰值显存。计时含验证与 epoch 管理，不是纯 kernel benchmark。

## 5. DDP 与恢复

```bash
GPU_IDS=2,3 NPROC=2 CONFIG=configs/ddp.yaml STAGE=ranking bash scripts/research_ddp.sh
GPU_IDS=2,3 NPROC=2 CONFIG=configs/ddp.yaml STAGE=ranking bash scripts/research_ddp.sh \
  --resume "$RUN_ROOT/ranking/last.pt"
```

保持同样 world size、seed、batch、总 epochs、loss/模型配置与 dataset fingerprint。最后完成的 epoch 才是恢复点；中途停止丢弃该 epoch，重跑。不能修改 epochs 以接着训练而宣称精确恢复（scheduler 总步数已变）；新训练计划应开新 run。CPU DDP 可 `torchrun --standalone --nproc_per_node=2 -m seqrank.cli train-ranking --config "$CONFIG"`，仍由用户自行验证。

## 6. 评估、部署与报告

```bash
python -m seqrank.cli build-ann --config configs/h200.yaml
python -m seqrank.cli benchmark-ann --config configs/h200.yaml
python -m seqrank.cli benchmark --config configs/h200.yaml
python -m seqrank.cli serve-research --config configs/h200.yaml
```

ANN 默认不是 serving 开关；启用时另设 `--set serving.ann=/absolute/path/hnsw.faiss`。索引绑定模型 bytes；更换权重必须重建。性能评估至少需 2 个 validation batch，首批预热不计。

API 示例：先 GET /health 取 dataset_version，然后 POST /recommend：

```json
{"user_id":"123","timestamp":1651900000000,"dataset_version":"替换为实际hash","history":[],"k":10,"candidates":100}
```

历史每项含 raw item_id、timestamp、available_at、已完成 watch/duration、positive=true；调用者负责它确为配置 primary-positive。建议使用 /docs。`/predict` 另需 item_id、预测时已知 duration/category，用于和单曝光离线预测对齐。服务无在线行为存储，不以请求数据更新模型。

模型对比/消融完整命令见 EXPERIMENTS.md。产出 comparison.md/.csv/.json 由实际 metrics 聚合。最后记录环境、Git SHA、数据来源与配置，以及未完成/失败实验，不补造结论。

## 7. 验证入口（本次没有执行）

`pytest -q tests/research`：CPU fixture 含真实接口全链路；`pytest -q`：包含旧行为回归；FAISS 用 importorskip。多 GPU、真实 schema 变体、BF16/FP16、数据规模和服务负载需在你的环境验证。上传提交标记 [skip ci]，避免替你启动测试/训练。

可选双进程 CPU DDP 集成测试：`SEQRANK_DDP_TESTS=1 pytest -q tests/research/test_ddp.py`。普通 pytest 默认跳过这两个启动多进程的测试；测试代码仍是实际完整流程。

默认 training.deterministic=true，启用 PyTorch deterministic algorithms、关闭 cuDNN benchmark 并在 CUDA 初始化前设置 CUBLAS_WORKSPACE_CONFIG（未被外部指定时）。遇不支持的确定性算子会报错；显式改为 false 后不应再宣称数值精确复现。跨硬件/驱动仍不保证逐 bit 一致。
