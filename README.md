# EvolveVLA — RoboTwin π0.5 混合数据 LoRA 实验

本仓库保存目前实际运行的 RoboTwin 实验代码、部署说明、数据来源与完整性清单。
入口是 [新服务器部署说明](docs/DEPLOYMENT.md)；交给另一台服务器上的 Codex 时，直接让它阅读根目录 `AGENTS.md` 和该说明。

当前训练配置：

| 项目 | 当前值 |
|---|---|
| 公开初始化模型 | [`Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k`](https://huggingface.co/Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k) |
| 模型 revision | `37e10d7f4f49149e2ce5e08a77b07f228a84322f` |
| 格式 / 公共 checkpoint 步数 | OpenPI JAX / Orbax；59999 |
| 微调 | LoRA only：语言模型 rank 16、动作专家 rank 32，其他参数冻结 |
| Batch size / 更新步数 | 64 / 24000；梯度累积 1 |
| 训练数据 | 本地鼠标垫 995 + 本地挂杯 995 + 官方 18 任务 × 50 = **2890 条** |
| 数据采样 | 本地 / 官方各 50%；本地两个任务各占总抽样 25% |
| 验证集 | 本地 18 任务 × 5 = 90 条；720 个固定验证锚点 |
| 归一化 | 初始化模型原始 `assets/aloha_agilex_sim/norm_stats.json`，不重算 |
| 保存与 rollout | 1000、4000、9000、18000、24000 步 |
| 实际评估 | Clean；鼠标垫与挂杯各 10 次有效配对，原模型和候选模型使用相同场景 |

这里的 Myungkyu checkpoint 是公开的独立复现模型，**不是 RoboTwin 团队排行榜权重**。
早期探索过的 `motus-robotics/pi0.5_robotwin2`、LIBERO checkpoint 都不是本轮初始化模型。
本轮还没有可以报告的闭环收益，历史结果见 [实验报告](outputs/experiment_progress_report_20261015.md)。

## 代码和资产放在哪里

- `scripts/`：训练、数据审计、转换、LoRA 导出、Clean 配对 rollout 和相关测试。
- `evolvevla/`、`tools/`、`tests/`：项目通用代码和早期实验工具。
- `deploy/`：新服务器部署、数据发布/恢复、环境快照和运行时补丁。
- `data/*/split_manifest.json`：不可变训练/验证划分。
- `outputs/`、`artifacts/`：精选数据来源与实验记录，不包含正在写入的训练日志。
- [GitHub Release `data-20261015-v1`](https://github.com/Zxy-MLlab/EvolveVLA/releases/tag/data-20261015-v1)：分卷数据。上传期间 release 是草稿，全部分卷校验后才公开。

数据清单共有 **2800 条已验收的本地原始轨迹**：两个重点任务各 1000 条，另外 16 个任务各 50 条。
另外 16 个任务的本地训练数据本轮不参与训练，但保留原始资产。
包含已转换的 2710 条本地训练轨迹、90 条本地验证轨迹、900 条官方 replay；训练加载器只选择两类本地任务，因此实际是 2890 条，不是 3610 条。

原始 HDF5、已转换数据与必要元数据合计约 **44.86 GiB**，53 个独立 tar 分卷，不进 Git 历史。
模型权重、旧 checkpoint、缓存、环境二进制及巨量历史日志均不上传到 Git。
公开模型、仿真资产和第三方源码在新机器按固定版本下载。

当前为经过实机运行验证的实验快照，保留了 `/root/autodl-tmp/EvolveVLA` 的绝对路径。
请按部署说明使用相同的物理路径；不要批量改路径后跳过 provenance 或 hash 检查。

2026-10-20 的闭环诊断、配对审计、路由决策和小样本纠偏数据保存在
`artifacts/gen1_control_repair_20261020/`、`artifacts/gen1_task_routing_20261020/`
和 `artifacts/gen1_recovery_smallfit_combined_train_20261020/`。总索引见
`artifacts/current_asset_manifest_20261021.json`。诊断结论是纠偏状态覆盖不足；24k
LoRA 候选未通过独立配对规则，因此路由仍固定到原始 Myungkyu checkpoint。
