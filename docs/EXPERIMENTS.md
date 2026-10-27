# 实验协议、比较与消融

本文件是执行协议，不包含预填的真实结果。`reports/` 的旧结果不可用作 v2 结果。

## 三种离线问题

1. `evaluate-retrieval`：对真实 primary-positive 曝光，以此前合法历史召回当前目标；完整训练目录准确 Top-K。Hit=Recall（每请求一个目标），NDCG=1/log2(rank+1)，MRR=1/rank。OOV 目标计 miss。Coverage 是被推荐训练物品数/训练目录数。冷用户=不在 train 词表；长尾阈值仅来自训练正例计数分位点。请求按曝光加权，非用户均权。
2. `evaluate-ranking`：只对真实曝光预测；click/valid_play 的 AUC、GAUC、AP、LogLoss、ECE；有合法标签时才评估 conditional CVR/CTCVR；时长在所配 condition 下评估 MAE/RMSE/log-MAE。
3. `evaluate-e2e`：固定召回候选上重排，不强行插入目标；当前曝光正例为唯一已知相关目标，其余候选未知。指标是目标排名代理，**没有把其他物品标为已知负反馈训练 CTR**。不能解释成在线 CTR/时长提升。

random policy 不等于拥有可用 propensity。当前系统不估算 IPS/SNIPS，亦不提供假装满足支持集条件的反事实结果。真实业务收益须在线随机实验。

## 指标细节

精确 AUC 通过 SQLite GROUP BY score 排序，ties 给予 0.5 信用；单类别全集返回 null。PR-AUC 输出名 `average_precision`，采用 step-integral AP，不是梯形 PR 面积。GAUC 按 raw user 的独立 group ID 汇总，过滤单类别用户，以其有效曝光数加权；输出 eligible user/row 数。不会把所有 OOV 用户合为同一人。

ECE 用 20 个等宽概率桶，Σ n_b/N·|mean(p)_b−mean(y)_b|。`BinaryAccumulator` 单独使用时是 4096 桶近似 AUC/AP；正式 `MetricStore` 输出被精确 SQL 指标替换并标识 exact 方法。没有把 histogram 近似伪装成精确。

观看秒数指标不裁剪 test 标签；单独计数超过训练 cap 的观测。watch_groups 包含 train duration quartile（缺时长组 -1）、过去序列长度、历史 log-watch 偏好、训练目标桶。watch_calibration 使用训练边界按预测值分桶，输出均值预测、均值标签和残差；duration_rank_bias_proxy 比较每用户预测与观测 watch 的平均 tie percentile，按 duration 分组。这是曝光分布的描述性排序偏差，不是因果识别。

## 选模与版本

每 epoch 只在 valid 上计算与训练一致的加权 loss，最低者写 best.pt。test 由独立命令在训练后读取 best.pt，从不反传或参与选 checkpoint。召回采样 loss 的验证负样本用固定 seed，每 epoch 一致；完整目录排名指标由独立评估产生。当前默认不是以 Recall@K 早停；需要按验证排名选择超参数时应显式记录该过程，test 仅最终评估。

检查点记录 resolved config、seed、数据/代码版本、optimizer、scheduler、scaler、每 rank RNG、epoch。配置矩阵另外记录 variant 与 override。不同数据指纹/历史协议不能混合汇总；报告分组保留这些字段。样本标准差 ddof=1，单 seed 为 null，不伪造置信区间或显著性。

## 命令矩阵

在仓库根目录，准备数据并设置 DATA_ROOT 后：

```bash
export CUDA_VISIBLE_DEVICES=0
# 默认 h200 基础配置；CPU 调试可覆盖 experiments.base
python -m seqrank.cli experiments --config configs/experiments/retrieval.yaml --execute
python -m seqrank.cli report --config configs/experiments/retrieval.yaml
python -m seqrank.cli experiments --config configs/experiments/ablation.yaml --execute
python -m seqrank.cli report --config configs/experiments/ablation.yaml
# CPU 单种子计划/执行，建议为其选择独立 EXPERIMENT_ROOT
python -m seqrank.cli experiments --config configs/experiments/ablation.yaml \
  --set experiments.base=configs/datasets/kuairand_debug.yaml --set 'experiments.seeds=[42]'
```

矩阵包含：Popularity / ItemKNN / GRU4Rec-inspired / SASRec-inspired / popularity sampling；binary-only DNN、DIN、dwell-only、Shared-Bottom、MMoE-log、raw、normalized、bucket、CREAD-inspired、duration-conditioned；无 attention、无 ordinal、无 restoration。二分类为 click 时才是 CTR-only；默认 KuaiRand 二分类是 valid_play。

真正 CTR 矩阵：先另存一个 YAML `extends: configs/h200.yaml`（相对该文件位置修正路径），设置 click 适配/筛选与 `watchtime.condition: click`，重新准备新的数据目录；通过 `--set experiments.base=/path/to/ctr.yaml` 指向它。不要仅改报告标签。

单方法专项与任务权重：

```bash
python -m seqrank.cli train-ranking --config configs/h200.yaml \
  --set watchtime.mode=cread_inspired --set ranking.tasks.watch=0.3
python -m seqrank.cli diagnose-gradients --config configs/h200.yaml
python -m seqrank.cli evaluate-ranking --config configs/h200.yaml --split test
```

评估从 checkpoint 恢复训练时 watch artifact 与网络配置，避免因评估 YAML 不同而改变目标还原。融合取 checkpoint 中 ranking.fusion；若想比较新融合权重，应单独构建可追溯实验而不是悄悄覆盖。

`experiments` 不加 --execute 只写计划；加后按顺序执行所有种子，遇失败停止。不会自动跳过/覆盖已有训练目录；长矩阵中断后可用 resolved.json 对该 run 手工 train --resume，再独立评估，或缩小 variants 配置完成剩余。报告只接受实际 metrics.json。
