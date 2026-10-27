# 损失、Mask 与分布式归一化

## 召回

score(u,i)=q(u)ᵀe(i)/√D。正样本为 primary label=1 且 item≥2；其他曝光不贡献召回损失。

二元采样：`L=softplus(-s_pos) + mean_j softplus(s_neg_j)`。uniform 或 `(train_popularity+1)^power` 概率，通过条件 CDF 精确排除当前正样本，允许采样重复，不排除历史消费物品。不是 sampled softmax，也不是全 softmax 无偏估计。未知目标安全占位后 mask=0；没有有效正样本的 batch 保留零梯度图，所有 rank 仍参加同步。

Full softmax：`L=-s_pos+logsumexp(s_all_training_items)`，对训练全目录精确 CE，内存随 B×I 增长，仅推荐小目录诊断。未实现 sampled-softmax，传入该名字会明确报错。

## 精排

`L_binary = BCEWithLogits(z_binary, label)`，mask=label≥0。

`p_joint=σ(z_click)σ(z_cvr)`；联合 loss 为 `-y_conv log p_joint -(1-y_conv)log(1-p_joint)`。代码在 log-sigmoid + expm1 域计算，避免极小联合概率下溢。conversion mask 要求真实 conversion 与 click 已观测；conversion=1 必须 click=1。条件 CVR 只在点击样本上评估，CTCVR 在观测联合标签曝光上评估。没有真实转换标签就关闭整个头。

Watch mask：`dwell≥0 && finite`，再按 condition=click / positive / observed 筛选。click 适用于条件观看时长；observed 适用于自动播放中所有观测曝光，非点击并不等于没有播放。normalized/conditioned 再要求 duration>0。模型中仅放历史已完成 watch 特征，当前 watch 只能作标签。

CREAD-inspired：设 p_m=σ(z_m)，b_m=1[y>t_m]，ŷ=Σp_mΔt_m。

`L_watch = mean_m BCE(p_m,b_m) + λrestore Huber(ŷ/cap,y/cap) + λordinal mean_m ReLU(p_{m+1}-p_m)`。

Bucket loss 为 CE，恢复 softmax 加权训练桶均值。其他模式在各自目标空间使用 smooth L1（Huber δ=1）。每种模式的单位不同，固定任务权重不可自动解释为等强度；应在验证集调权并观察共享梯度范数/余弦。最终选择标准为固定权重验证总 loss，因此跨不同目标变换比较时不能直接比较 selection_loss，应该比较统一秒数误差/二分类指标。

## DDP + accumulation

`components` 返回每任务有效样本 loss **sum** 和 count。对整个 accumulation window 先求各 rank、各 micro-batch 的有效 count 全局和 C_t。每个 rank/microbatch 反向 `Σ λ_t · world_size · numerator_t / max(C_t,1)`；DDP 默认梯度平均抵消 world_size，结果为全局有效样本均值。不能再除 accumulation，否则缩小更新；某任务全局无标签时梯度为零。

最后一个不足 accumulation 的窗口同样处理。训练 DistributedSampler 为等长补齐少量重复，分母包含这些实际参与训练的记录，日志明确标记。验证 sampler 不补齐，先全局 sum/count，再计算选模 loss。未用 test 选模型。

`diagnose-gradients` 用 autograd.grad 计算共享参数上的每任务未加权有效均值梯度，eval mode 关闭 dropout，不写参数 `.grad`、不执行 optimizer。若某任务该批没有标签，则省略其梯度余弦并保留 count=0。零范数 cosine 定义为 0，不能当作无冲突证据。
