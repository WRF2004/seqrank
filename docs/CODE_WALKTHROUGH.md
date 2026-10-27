# 初学者源码阅读路线

1. **README → DATASET**：先回答“预测哪个事件、标签什么时候可见、候选是什么”。effective play 不等于 click，缺标签不等于负样本。
2. **research/adapters.py**：从一行 KuaiRand CSV 到标准曝光。跟随 adapt_row，看单位、tab 语义、conversion=-1、clip 与 availability 的顺序。convert 的 content join 只接受不可变内容。
3. **research/dataset.py**：prepare_data 先磁盘排序和 train vocab，再逐用户用 pending heap 处理延迟反馈。history 记录按事件时间选最后 L 个；数组 right padding。train/valid/test 分界作用于全体用户。
4. **research/features.py**：dense_features 不读取当前标签；transform_batch 只用 train mean/std；candidate_batch 替换目标 item/duration/category，同时重复合法历史。理解 raw features 与 standardized features 不可二次变换。
5. **models.SequenceEncoder + research/networks.py**：BOS 解决空序列；causal/padding mask 如何叠加；[B,L,D] 到 [B,D]；DIN target attention 到 interest；expert [B,E,D] 与 gate [B,E] 如何相乘。
6. **research/watchtime.py**：先读 fit_watchtime，再读 WatchHead.forward/per_sample_loss。bucket logits 如何还原秒数？CREAD 阈值 sigmoid 和 softmax 分类有什么区别？duration groups 为什么只能 train 拟合？
7. **research/objectives.py**：mask 与 per-example loss；每任务 numerator/count；ESMM log domain；负采样为什么排除当前正例但允许重复历史。
8. **research/training.py**：累积窗口 count all_reduce，然后 forward/backward；DDP 的平均如何抵消 world_size；valid 不重复，best 不看 test。读取 checkpoint 状态和 RNG 保存顺序。
9. **research/metrics.py / evaluation.py**：手算一组 ties 的 AUC/GAUC 和 rank=2 的 NDCG；理解 precision-recall AP；区分单曝光预测与候选排名代理。rank bias 是描述性而非因果。
10. **research/serving.py / experiments.py**：raw ID→训练映射→同一特征→同一模型→同一融合；请求版本校验；resolved configs 如何追溯每次实验。

建议最后阅读 `tests/research/test_data.py`、`test_models.py`、`test_metrics_integration.py`：这是可以自行运行的断言，而非业务效果承诺。CPU fixture 是明确合成的接口测试；不要用其指标写真实论文结论。

旧文件可对照学习原型到研究工程的变化：legacy 仍保留研究起点；新的真实实验都走 research 命名空间。关键修复和取舍见 UPGRADE_AUDIT.md。历史 reports 保持原文，不把旧系统的测试状态复制为 v2 的状态。
