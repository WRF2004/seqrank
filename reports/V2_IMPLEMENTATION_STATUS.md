# v2 implementation status — 2026-10-27

本次按用户要求完成代码、配置、测试与文档编写。没有下载数据/模型，没有执行项目代码、pytest、训练、服务或性能实验。现有历史报告原样保留，不能代表 v2。

交付覆盖：KuaiRand/通用适配，PIT 数据和质量统计，四类召回，四类精排结构，六类时长目标（含 CREAD-inspired），缺失标签/ESMM，CPU/DDP/AMP/accumulation/epoch 恢复，精确指标与端到端代理，ANN、batch API、梯度分析、多 seed 矩阵和报告。

待用户执行验证：实际 release schema/单位与 tab 语义、完整 CPU pytest、CUDA/BF16/FP16/DDP、模型恢复数值、FAISS 版本兼容、大目录内存和吞吐。没有“已通过”或预填效果数字。

已知限制：single-machine SQLite/I/O、dense embedding 副本、ItemKNN RAM、全目录计算成本；没有严格 CWM/线上系统/因果收益证明。CREAD-inspired 差异在 MODELS.md，不宣称论文严格复现。

真实实验表由 `seqrank.cli report` 从实际 metrics.json 生成；不要在执行前填写模型优劣结论。

静态检查：已用 Python AST 解析源码/测试/脚本，解析 YAML 并检查 extends 文件存在，核对文档 CLI 名称，执行 git diff --check。均未导入/执行项目模块。这些检查只能排除部分语法、引用与空白问题，不等价于测试通过。

后续修订：默认官方 release/data 路径；调参仅 valid，最终 test 要求显式 freeze-evaluation；冻结数据/权重/候选协议并拒绝漂移；检查 checkpoint 配置、特征统计和训练时长变换；补充对应回归测试。仍仅静态解析，测试与实验未执行。
