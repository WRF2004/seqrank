# v2 开发前审查（2026-10-27）

基线：4c3546171b90d507baf95a50cecf5bdc737b6481。已阅读全部 28 个跟踪文件；本地 tree 与远端基线一致。未执行项目代码或测试。

| 现有模块 | 可复用部分 | 缺陷与升级处理 |
|---|---|---|
| data.py | SQLite、同时间戳隔离、mmap | 缺少 duration/category/任务语义；v2 schema 独立，明确反馈可用时间 |
| models.py | BOS 因果 Transformer、DIN 注意力 | legacy CTR 缺失 mask、关闭头输出问题；v2 仅创建启用的头；修复 legacy mask |
| engine.py | 验证选模、原子保存 | 无 scheduler/RNG/accumulation；v2 trainer 完整保存；不混用旧 checkpoint |
| retrieval.py | 分块精确 top-k、HNSW | 旧训练评分除 sqrt(d)，检索没除（顺序相同，分值不同）；v2 共用 score；版本绑定扩展 |
| metrics.py | histogram AUC | 校准 L1 不是标准 ECE、GAUC 依赖用户连续；v2 磁盘 SQL 聚合支持任意顺序 |
| serve.py | raw ID 映射、校验 | 融合重复实现、没有特征 schema；v2 服务与评估共用特征/utility |
| CLI/config | 旧命令可保留 | v2 命令使用统一 YAML + overrides；显式区分真实数据/smoke |
| tests/reports | 旧回归测试/真实历史日志 | 原样保留历史报告；本次新增验证全部标记未执行 |

v2 放在 `seqrank.research`，避免把 data.py/models.py 重命名成同名包而破坏旧 import。旧流程为兼容入口，真实实验必须走 v2。
