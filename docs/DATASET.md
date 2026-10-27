# 数据契约与时间协议

官方资源：[KuaiRand](https://kuairand.com/) / [官方仓库](https://github.com/chongminggao/KuaiRand)。先读取你的实际 release 字段，不按文件规模猜语义。`schema` 检查必需列并记录 filename version hint；这是提示而非证明。未知版本仍按字段能力处理，缺列立即报错；标签无法推导时按 missing_primary=error 或显式 mask。

## 原始数据与标准曝光

| 标准字段 | KuaiRand 来源 | 类型/语义 |
|---|---|---|
| user_id / item_id | user_id / video_id | 原始字符串 ID |
| timestamp | time_ms | Unix 毫秒；不使用 date 列排序 |
| click | is_click，仅经核实的 click tab | 二元，缺失 -1 |
| valid_play | 已核实有效播放 tab 的 is_click，或由原始时长推导 | duration≤7s 时 watch≥duration；否则 watch>7s |
| dwell | play_time_ms / 1000 | 观测观看秒数，缺失 -1，不必小于视频长度 |
| duration | duration_ms / 1000 | 请求时已知内容长度；缺失 -1 |
| conversion | 无 | KuaiRand 固定为缺失 -1，绝非负标签 0 |
| category | 可选 category_column / immutable content tag | 完整字符串 token；多 tag 暂不拆多热 |
| available_at | 可配置；默认 event + 原始观看时长 + delay | 标签/行为反馈最早完成时间；严格晚于 event |
| tab / policy | tab / is_rand 或文件名 | 语义审计与过滤，不直接作为模型输入 |

原始 `is_click` 的含义依展示界面而变；官方材料不能推出所有版本通用 tab→界面映射。因此 **不提供猜测 tab 编号**。要做 CTR，配置 `adapter.primary_label=click`、`data.primary_label=click`、`adapter.tabs=[你核实的编号]` 和 `adapter.tab_semantics={编号: click}`。其他 tab 报错而非混入。默认 effective-play 可由原始 watch/duration 推导，但绝不被命名为 CTR。

用户抽样用 seed+raw user 的 SHA256 确定比例，保留所选用户完整输入时间线。Pure 的原始日志就不完整，采样不会修复缺失。文件名中的 27K/1K/Pure 仅作版本提示；不要求固定列全集。转换会记录源文件与配置 hash。

`adapter.content` 可加入官方 basic video metadata：指定 id/duration/category 字段和 duration_unit，并显式 `assert_immutable=true`。只允许内容属性；不自动接入日志全周期聚合的点赞数/热度等泄漏字段。缺失日志 duration 可回填；冲突内容快照拒绝。若静态属性假设不成立，应先在上游按生效时间做 as-of join，通用接口读取已经时点正确的曝光特征。

通用 CSV/Parquet 必需 `user_id,item_id,timestamp`，可选其他标准字段。`timestamp_unit` 为 ms/s/us；watch/duration 固定秒。Parquet 需要 `.[data]`。缺失二元标签用 -1，不能用 0 代替未知。conversion 为曝光级联合转化事件；conversion=1 必须 click=1。只有真实观测窗口已成熟时才允许记 conversion=0。数据源必须保证此定义。

## 清洗与协议

- NaN/空串/null 作为缺失；无穷值、非法标签、负时长（除 -1）、非法反馈时间报错。
- `max_watch_seconds` + `outlier_policy=error/drop/clip`；循环播放 `loop_policy=keep/clip/drop`。默认保留循环播放。不把归一化比值强行截到 1。
- 反馈可用时间用裁剪前播放时长，避免裁剪后提前消费反馈。有效播放语义也先依据原始 watch/duration；裁剪只是目标处理，不偷偷重定义 click。
- `duplicates=drop_exact` 对标准化后 user/item/time/labels/duration/category/availability 完全一致的记录去重；keep 则保留。无全局 impression ID 时无法区分同一毫秒真正重复曝光，需审视该取舍。
- 全局 `t < train_end` 为 train，`train_end ≤ t < valid_end` 为 valid，其余 test。支持带时区 ISO 或整数毫秒。默认 KuaiRand 日期只是实验起点，先确认输入覆盖，任何空 split 报错。
- 词表和数值标准化只拟合 train；PAD=0、OOV=1。all-user evaluation group ID 仅用于 GAUC，绝不入模。训练期未来出现的词表 token 是训练建模目录定义，不是历史行为特征。
- 标签 available_at≥train_end 的 train 曝光保留特征，但全部监督标签置缺失；不将未成熟反馈带入训练。验证/测试使用已观测日志作最终标签，数据源需保证评价窗口已成熟。
- 每个用户按 event 时间读取；仅在 available_at < 当前 t 时把已完成、primary-positive 行为加入历史。**同时间戳不可互看**；availability=t 也不允许。晚到旧行为按 event chronology 排序，不挤掉更新的事件。
- `rolling`：验证/测试允许消费严格过去、已经完成的真实行为，模型不更新；`frozen_train`：历史只用 train_end 之前已完成的训练行为。两种协议不可混表比较。

## 特征分层

原始曝光可用特征：user/item/category/duration/timestamp。历史：primary-positive item 序列与已完成 watch/duration/time。dense 八维见 `features.DENSE_NAMES`：log duration、missing flag、log history length、mean log watch/duration、log last gap、日内 sin/cos。均值方差只用 train，标准化后 clip [-10,10]。

仅标签/评估信息：label、click、conversion、dwell、group_user。当前 dwell 不能入预测特征；历史 dwell 只有完成后才可用。未实现曝光计数 CTR 统计特征，避免把当前标签混入聚合；提供的活动/流行度数组是训练期统计，用于基线/分组，不作为未来滚动特征。

## 输出与检查

`prepared/manifest.json`、`features.json`、`events.db`（仅词表，原始事件表已删除）、`catalog_*.npy`、`popularity.npy`、`activity.npy`、`train/valid/test-*.npy`。`quality` 输出 JSON、CSV、可选 PNG。分布为带 seed 的有限 reservoir；曝光数、缺失率、OOV 和全体 raw cardinality 精确统计；训练用户活动/物品流行度从完整训练数组计算。API 与 checkpoint 拒绝不匹配版本。

normalized 原始 ID 与 metadata 仍可能敏感；不提交数据或训练产物到 Git。词表保留 raw ID 只用于返回业务标识，不包含任何私密凭据。
