# 模型定义与论文差异

## 召回

共同神经接口：`user_vectors(history[B,L]) -> [B,D]`，`item_vectors(ids) -> [...,D]`，score=dot/√D；训练、精确检索、ANN 完全相同。Popularity/ItemKNN 通过 `score_block` 接入统一 `exact_topk`，没有虚构的稠密 embedding 或可训练 loss；单次统计拟合写入相同版本 checkpoint。全目录默认训练期出现物品，OOV 目标记 miss。

- Popularity：训练中已成熟 primary-positive 曝光次数，分数相同按较小映射 ID。
- ItemKNN：训练 unique-user/item 二元矩阵的物品 cosine，去自身后每物品保留 top neighbors；历史加和，重复历史可累积。空已知历史退回 popularity。这是 implicit cosine baseline，不是特定论文严格复现。
- GRU4Rec-inspired：共享 item embedding、可学习 BOS、GRU、最后非 PAD 状态。非 session-parallel 原始 GRU4Rec，不实现原论文 BPR-max 和特殊优化器；使用本项目共同目标，便于控制实验变量。
- SASRec-inspired：复用已有 pre-norm Transformer、BOS、位置 embedding、因果/padding masks，输出最后非 PAD 状态。每曝光一步预测，不对序列全部位置同时监督；不是原 SASRec 的逐项严格复现。

## 精排

所有模型共享特征 contract：用户、目标物品、类别 embedding、历史兴趣、目标×兴趣、8 维 dense。DNN 默认 mean pooling，DIN 使用 target attention；Shared-Bottom 一个共享 MLP + 任务塔；MMoE 多专家、每任务 gate/tower。模型名对应结构，DNN/DIN 配置默认单任务；用户也可显式启用多个头，但不把它冒充新的论文模型。

任务 `binary` 的实际含义写在 manifest.primary_label；仅当其为 click 才称 CTR。conversion 仅真实标签+click 主任务可启用。关闭任务时没有该头参数或预测值；融合引用关闭头会报错。

ESMM-style 使用 p_ctcvr=p_click·p_conversion_given_click，在完整有联合标签的曝光空间学习联合 BCE；不把仅点击集 CVR BCE 冒充 ESMM。模型未加入论文所有工业特征/训练策略，因此称 ESMM-style。

## 观看时长主线

原文：[CREAD, AAAI 2024](https://arxiv.org/html/2401.07521v1)；[CWM, KDD 2024](https://arxiv.org/html/2406.07932v1)、[CWM 官方代码](https://github.com/hyz20/CWM)。本文源码实现名称与严格论文复现区分如下。

| mode | 目标/输出 | 数值恢复 |
|---|---|---|
| raw | softplus 秒数的 Huber | 非负秒数 |
| log1p | softplus log-seconds 的 Huber | expm1 |
| normalized | watch / positive duration 的 Huber | ratio × duration，允许 >1 |
| bucket | train quantile 分桶 CE | softmax 加权训练桶内均值 |
| cread_inspired | EAD-inspired 分桶 + 多阈值 BCE + 恢复 Huber + 序约束 | Σ survival probability × bin width |
| duration_conditioned | train duration quartile 内标准化 log-watch 残差 Huber | residual × group sigma + group mu，然后 expm1 |

### CREAD-inspired 的实现对应

`adaptive_edges` 从训练 eligible watch reservoir 的经验 CDF 估计候选分位边界；alpha∈[0,5] 21 个点，使用 γ(z;α)=(1−exp(−αz))/(1−exp(−α))，alpha=0 时 γ=z。对候选计算论文启发的 within/between 离散误差代理 Aw+βAb 后选最小，重复边界去重。实现用归一化 [0,cap] 区间、有限样本平滑 bucket mass，而非论文完全相同的分布估计与超参数寻优。

`WatchHead` 的第 m 个 logit 表示 P(Y>t_m)，有 K 个上边界阈值；BCE 标签为 y>t_m；数值恢复 Σσ(logit_m)(t_m−t_{m−1})。这是 survival 积分的右端 Riemann 近似。最后 cap 阈值的标签通常为 0；离散误差不可忽略，需用秒数 MAE/RMSE 衡量。

恢复 Huber 在 cap 归一化秒数上计算，序约束 penalize ReLU(p_{m+1}−p_m)。没有硬单调投影；违反序关系受到软惩罚。主干是本项目 MMoE 而非完全复制论文工业结构。**经验 CDF/reservoir、质量平滑、归一化恢复项、网络与训练协议均是差异，因此不能把结果标为严格 CREAD 复现。** cap 是配置，不从 test 估计；分桶、代表值、duration groups 只读 train。

CWM 的目标涉及时长截断下的 counterfactual watch time、观看成本与对应 likelihood；简单 watch/duration 或按 duration 分组残差不等价。本仓库只提供明确命名的 duration-conditioned baseline，未实现/声称严格 CWM，也不把分组误差缩小解释成因果去偏成功。

预测上限适用于所有方法保证输出有限；训练目标按 cap 裁剪，评估使用原始未裁剪秒数并记录尾部样本数。归一化/conditioned 模式缺 duration 样本不监督；服务需要有效 duration 才有可信解释。

normalized / duration-conditioned 若融合依赖 watch，缺 duration 的候选被明确过滤（可少于 k 个返回），不把未定义预测当作可靠零秒。`/predict` 对这些模式缺有效 duration 返回校验错误。独立曝光时长评估也用相同 eligibility mask。
