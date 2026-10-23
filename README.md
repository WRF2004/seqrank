# SeqRank — 大规模序列推荐与多任务精排系统

一个可复现的两阶段推荐工程：**因果序列召回 → 候选集 → 目标感知多任务精排 → 推荐 API**。
包含训练、验证、完整候选库评测、ANN 索引、断点续训、服务和自动化测试。

**项目状态：研究与工程验证版本。** 已在 CPU 上跑通 10,000 条合成曝光的完整链路；不代表工业规模验证、线上收益或竞赛成绩。DDP/AMP 入口已实现，但当前环境限制跨进程通信且无 GPU，尚未验证多卡训练。真实业务效果需要在真实曝光数据上测量。

## 实现范围

| 层级 | 实现 | 关键约束 |
|---|---|---|
| 数据 | CSV → SQLite 外部排序 → NPY 分片 / mmap | 训练期词表、全局时间切分、严格过去点击历史 |
| 序列召回 | SASRec-inspired 因果 Transformer + sampled binary loss | BOS 处理空历史；不声称原论文逐项复现 |
| 检索 | 分块精确 top-k；可选 FAISS HNSW 内积索引 | 默认允许重复消费；索引绑定模型 SHA256 |
| 精排 | DIN-style 目标注意力 + MMoE 三个任务门控 | CTR、条件 CVR、点击后 log-dwell |
| 转化建模 | ESMM-style `pCTCVR = pCTR × pCVR` | CTR 与曝光到转化标签监督；CVR 无直接标签监督 |
| 训练 | AdamW、梯度裁剪、CUDA FP16 AMP、DDP | 最优验证损失选模；epoch 边界续训 |
| 评估 | Hit/NDCG、候选召回率、AUC/GAUC、LogLoss、log-dwell MAE | 全训练目录评测；未知目标记为 miss；AUC 为分桶近似 |
| 服务 | FastAPI `/recommend`、`/health`、Docker | raw ID 映射、seen 过滤、模型版本校验 |

## 一次跑通

Python 3.10+，推荐 3.12。CPU 环境：

```bash
git clone https://github.com/WRF2004/seqrank.git
cd seqrank
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,ann]'
make test
make demo
make serve
```

Windows 可用 `.venv\Scripts\activate`，没有 make 时依次执行 Makefile 中各条 `python -m seqrank.cli ...` 命令。
`make demo` 不覆盖已有数据；再次运行请通过 `make demo DATA=data/prepared2 RUN=runs/demo2` 指定新目录。
服务默认仅监听 `127.0.0.1:8000`；Swagger 在 `/docs`。

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"0","history":["0","10","20"],"k":10,"candidates":50,"exclude_seen":true}'
```

请求里的 ID 均为原始字符串，历史按从旧到新排列，由上游提供，仅含请求之前的点击。
返回 item_id、融合分数、p_click、p_conversion、expected_log_dwell、模型版本与本次耗时。
空历史使用 BOS；未知用户映射 OOV。冷启动目前只有默认表示，没有内容特征冷启动模型，不保证个性化质量。

## 数据契约

```csv
user_id,item_id,timestamp,click,conversion,dwell
u1,i10,100,1,0,12.5
u1,i20,110,0,0,0
u2,i10,120,1,1,35.0
```

- 每行是一条**曝光**，不能只用正反馈日志训练 CTR。
- `timestamp` 是统一单位的整数事件时间；切分边界使用相同单位。
- `click ∈ {0,1}`；`conversion ∈ {0,1,-1}` 且转化必须发生在点击后。`-1` / 空列表示未知。
- `dwell` 为秒，非负，未知为 `-1`。停留损失只在点击且标签有效的曝光上计算。
- 必须等待转化归因窗口成熟再导出标签。当前实现不做延迟反馈校正，也不会把未成熟标签自动当作负样本。
- 历史仅使用严格早于当前时间的点击。同一用户同一时间的事件互不可见。
- train: `t < train_end`；valid: `train_end <= t < valid_end`；test: `t >= valid_end`。
- 采用 rolling-history 协议：验证/测试期间已发生的点击可以进入后续请求历史；模型不更新。非固定训练期历史协议。
- 词表仅用训练期曝光拟合，0=padding，1=OOV。验证/测试新物品无法召回，报告其覆盖率。

```bash
python -m seqrank.cli prepare --source data/events.csv --out data/production \
  --train-end 1700000000 --valid-end 1700600000 --max-len 200 --shard-size 100000
```

边界仅为格式示例，需按实际数据时间范围设置。原始数据、模型权重和预处理文件均不进入 Git。
数据目录内 `events.db` 在预处理结束后只保留 ID 映射，原始曝光表会被删除。

### 真实数据接入：KuaiRand

提供流式转换器，数据从 [官方项目](https://github.com/chongminggao/KuaiRand) 自行下载并遵守数据许可：

```bash
python scripts/convert_kuairand.py --input /path/to/log_standard.csv --output data/kuairand.csv
```

映射 `time_ms → timestamp`、`video_id → item_id`、`is_click → click`、`play_time_ms / 1000 → dwell`。
官方 `is_click` 在不同 tab 下分别有点击/有效播放含义，因此需按场景过滤后解释指标。
**KuaiRand 没有购买转化标签，本项目不伪造 CVR 标签。** 用该数据时复制 scale.yaml 并将 `conversion_weight: 0`；转化头不参与融合、API 转化概率返回 null。此时是点击与观看时长双任务，不能声称做过真实转化预测。
标准流量和随机曝光流量请分开评估；该转换器不会自动执行 IPS 或无偏评估。

## 训练与多卡入口

```bash
python -m seqrank.cli train --stage retrieval --data data/production --out runs/exp1 --config configs/scale.yaml
python -m seqrank.cli train --stage ranker --data data/production --out runs/exp1 --config configs/scale.yaml

# 示例：8 卡（请先在自己的 GPU 环境完成两卡冒烟测试）
python -m torch.distributed.run --standalone --nproc-per-node=8 -m seqrank.cli train \
  --stage retrieval --data data/production --out runs/ddp --config configs/scale.yaml
python -m torch.distributed.run --standalone --nproc-per-node=8 -m seqrank.cli train \
  --stage ranker --data data/production --out runs/ddp --config configs/scale.yaml

# 中断后从完整 epoch 的 last checkpoint 恢复；配置/数据/进程数保持相同
python -m seqrank.cli train --stage ranker --data data/production --out runs/exp1 \
  --config configs/scale.yaml --resume runs/exp1/ranker-last.pt
```

GPU PyTorch 请使用与服务器驱动相容的官方构建；上方 CPU 安装命令不适用于 GPU 训练。
每个 stage 输出 `*-last.pt`、`*-best.pt` 和 JSONL 日志。checkpoint 含模型、优化器、scaler、配置和数据 manifest；原子替换避免半写文件。
严格续训仅支持 epoch 边界，不恢复半个 epoch；跨硬件/版本不承诺 bitwise 一致。

## 评估与服务

```bash
python -m seqrank.cli evaluate --data data/production \
  --retrieval runs/exp1/retrieval-best.pt --ranker runs/exp1/ranker-best.pt \
  --k 10 --candidates 200 --out runs/exp1/test.json

python -m seqrank.cli build-index --retrieval runs/exp1/retrieval-best.pt --out runs/exp1/catalog.index
python -m seqrank.cli serve --data data/production \
  --retrieval runs/exp1/retrieval-best.pt --ranker runs/exp1/ranker-best.pt --ann runs/exp1/catalog.index
```

评测默认使用精确检索，确保模型质量不与 ANN 近似误差混淆。召回每个点击曝光的目标物品，单正例场景中 Hit@K 等于 Recall@K；不使用随机负样本评测。历史已见物品仍为合法候选。
精排评测还包含曝光分布上的 CTR/CTCVR 和召回候选上的最终 Hit/NDCG；没有强行把目标塞回候选集。
AUC 和 GAUC 使用 4096 桶近似，单类别群体不计入 GAUC；未知用户不进入 GAUC。停留指标在 log1p 秒空间计算，不是原始秒 MAE。
默认融合为 `pCTR + pCTCVR + 0.01 × E[log(1+dwell)|click]`，关闭的辅助任务不参加融合。这是可审查的初始规则，不是学得的最优业务权重。

```bash
python scripts/benchmark.py --data data/prepared --run runs/demo --requests 100

docker build -t seqrank .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data:ro" -v "$PWD/runs:/app/runs:ro" seqrank \
  --data data/prepared --retrieval runs/demo/retrieval-best.pt --ranker runs/demo/ranker-best.pt
```

Docker 构建文件已提供，当前环境未运行 Docker。服务需在部署时接入网关认证、限流和外部历史特征存储；不要将裸服务直接用于生产。

## 真实运行记录与边界

查看 [验收记录](reports/VALIDATION.md)、[合成数据指标](reports/synthetic-metrics.json) 和 [架构说明](docs/ARCHITECTURE.md)。
当前 3 epoch 合成数据实验中精排 CTR AUC 约 0.498，没有验证精排优于基线；这些数值只证明管线执行，不证明推荐质量。
大规模机制与完成的大规模实验是两件事：本项目没有宣称亿级样本、百万 QPS、线上 CTR 提升或比赛排名。

推荐后续实验：真实曝光数据 + 3 个随机种子；Popularity vs 序列召回；MMoE experts=1 vs 4；关闭辅助目标；序列长度 20/50/200；ANN top-k 与精确 top-k 的重合率、延迟、内存。测试集只在确定方案后使用。

## 目录

```text
src/seqrank/       数据、模型、训练、检索、评估、API、CLI
configs/          demo / scale 配置
scripts/          KuaiRand 转换、延迟测量
tests/           泄漏、因果 mask、损失、索引、训练与服务回归测试
reports/          本次实测与限制
docs/             架构、扩展约束与面试说明
.github/workflows CPU CI
```

## 参考与许可

以下方法作为架构依据，本仓库自行实现代码，没有复制这些项目源码：

- [SASRec 官方实现](https://github.com/kang205/SASRec)：因果序列表示。
- [MMoE, Google Research](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)：共享专家与任务门控。
- [DIN](https://arxiv.org/abs/1706.06978)：目标相关兴趣聚合。
- [ESMM](https://arxiv.org/abs/1804.07931)：CTR × 条件 CVR 的联合转化建模。
- [KuaiRand](https://kuairand.com/)：真实序列曝光数据接入格式。

代码 MIT；外部数据与模型遵循各自许可。
