# 常见问题与明确限制

**缺数据/空 split**：schema 检查匹配文件；核对 RAW_ROOT、单位、带时区日期。不要把真实入口改成合成 fallback。输出目录已有部分失败结果时另选目录，保留错误日志后由你决定清理。

**没有 click 标签**：不要用 is_click 任意 tab 冒充 CTR。默认 valid_play 若缺 dwell/duration 会报 missing primary。确认不可变 metadata 的 duration 字段/单位，或明确 missing_primary=mask 并检查剩余样本数。没有转化标签关闭 conversion；不以全零训练。

**OOM**：先降低 batch_size、max_len、dim、layers、candidate_chunk、catalog_chunk；增加 accumulation 维持目标有效 batch。改变 max_len 必须重新准备数据。DDP 复制模型和 AdamW，不减每卡 embedding 内存；8 卡也不等于能容纳 8 倍词表。ItemKNN 减少 knn_block/neighbors；必要时先做全用户抽样的小目录实验。不要在共享服务器反复尝试占满所有卡。

**NaN/Inf**：检查 units、watch cap、missing sentinel、labels、lr；先 FP32、小 batch、低 lr 复现。normalized 要有 duration>0；极短视频可能有高 ratio，loss 为 Huber 但仍需分析异常。mask 不允许 NaN 标签透过；adapter 对非有限值拒绝。FP16 overflow 应由 scaler 跳过更新并降低 scale，BF16 通常无需 scaler但仍需验证。

**DDP 卡住**：每 rank 必须相同配置与共享 data，使用 torchrun；检查 rank0 之前是否已报缺数据/磁盘错误。CPU loader workers=0 可排除多进程磁盘句柄问题。训练 sampler 等长，验证 sampler 无补齐；validation forward 用 raw model 避免不等长同步。

**恢复拒绝**：确保同 stage、world size、training/model/task config、seed、数据 hash；仅 run/eval 输出可变。保持原总 epochs。迁移到新输出目录时检查 best.pt 是否存在；建议在原 run 用 last.pt 继续。恢复仅 epoch 边界，不承诺 mid-step 精确恢复。不同硬件/非确定性 kernel 仍可能产生浮点差异。

**模型/索引版本拒绝**：不要编辑 manifest 或复制半套 assets；运行 verify-data；使用同 prepared 和 checkpoint 创建 ANN。不同模型 checkpoint 或 index bytes 即使形状相同也拒绝。不要把 v1 weights 与 v2 CLI 混用。

**精确评估慢**：AUC/GAUC 使用 SQLite 外排序和磁盘预测缓存，避免 RAM 爆炸但需要磁盘空间；全目录检索总复杂度仍高。先在小数据验证，再测 batch/chunk；不要以随机负例采样指标冒充全目录指标。

**API 422**：版本、k≤candidates、正数限制、history 时间与反馈时间必须满足；frozen_train 不允许训练后历史。字段不认识直接拒绝。watch 缺失用 -1，禁止任意负值。接口不提供真实 raw label 推断，调用方须保证 positive 的任务语义。

**性能数字解释**：benchmark 是进程内召回+精排，含张量搬运，不含 HTTP/外部特征系统；ANN 输出的是序列化 index bytes，不是实际常驻峰值 RSS。不能称为公网 QPS 或工业容量。异步 CUDA 在计时边界 synchronize。

**当前边界**：未实现严格 CWM、在线随机实验、propensity/IPS、embedding 分片、动态内容 as-of join 服务、全流程自动失败重试、任意 step 恢复。默认 metadata snapshot 必须不可变；训练目录以训练出现的物品定义，不能处理未见新物品内容召回。Future work 可在实测瓶颈之后加入 TorchRec/FSDP、块级 shuffle、timestamp-aware content store、真实 propensity 研究，不以新目录替代核心正确性。
