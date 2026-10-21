# 纠偏后缀的小规模 LoRA 训练：待实现与验收方案

本文件只定义可审查方案和现有接口，没有启动训练，也没有修改活跃 runner、原始数据或旧训练门禁。当前接管采集器输出仍显式 `dataset_ready=false`；采集、重放与数据对齐全部通过之前，不把它们视为可用训练集。

**如果最后只有 3 个独立源 episode 的成功后缀，只能做小样本过拟合阳性对照。** 这不能估计泛化提升，不能将同一源 episode 的帧或多个接管点拆为 train/val 来制造验证集。

## 最小流水线

1. 冻结失败来源、专家接管策略和部署控制器。每个候选后缀必须满足：初始准入通过、模型前缀动作和已记录物理量精确匹配、dense 专家计划完成且最终原任务成功、独立 reset 后重放相同前缀并以指定部署控制器执行 sparse 标签仍成功。所有尝试保留；失败不混入成功示范。
2. 新建独立 suffix manifest 与 CPU builder。保存每帧原始 RGB、14D drive-target state，以及同一后缀下一捕获边界的 14D drive target 作为 action：`obs_t -> target_(t+1)`。记录真实 physics ticks，不能把 `save_freq=15` 当成 15 Hz；不把模型失败前缀贴成正确标签，不跨 episode 或接管断点拼 chunk。
3. 按源 episode 分组。分组至少绑定 source revision、task、原始 episode index/seed、原 HDF5 hash；该源的原专家轨迹、全部模型前缀及接管变体必须共用 split。先冻结分组再采更多后缀，官方 replay 也做源场景交叉检查。当前开发案例及已用来选参数的面板不能进入最终独立测试。
4. 新 loader 读取独立 manifest，统计任务、episode、帧数、有效 50 动作起点，完全由实际内容生成；不硬编码 1990、995 或 2890。先完成 CPU 全帧读取、图像颜色/尺寸、14D顺序、prompt、边界、下一帧 action 对齐与 norm roundtrip，再做 GPU 3 次更新 smoke。
5. 冻结训练参数、模型初始化和候选保存点。用未微调祖先及训练初始 step0 做离线对照；训练后导出完整 LoRA-aware checkpoint，重新核验 frozen base、adapter、原 norm 和生产恢复。
6. 固定部署控制器做配对 rollout。先检查训练输入/训练场景能否被学会，再检查完全未参与采集、开发、选模型的新源 episode。报告有效配对分母、专家拒绝、初始化异常、原模型独有成功与候选独有成功；禁止因离线 loss 最低就宣称闭环最好。

## 训练设置的最小选择

| 项目 | 首个小样本诊断 |
|---|---|
| 初始化 | 默认延续当前已验证 24k LoRA（单独记录精确身份）；原始 baseline 是对照 |
| 冻结 | 非 LoRA 全冻结；语言 rank16、动作专家 rank32 不变 |
| norm / 动作 | 原始 checkpoint quantile norm 不变；14D物理输入，经既有 delta/mask 转32D；horizon50 |
| 优化器 | 严格 typed restore 24k完整LoRA权重；新实验显式重启 Adam；保留适配器，**不调用 zero_b**；不假装恢复旧 step/moments |
| 学习率 | 预声明峰值1e-5、warmup20，保存step0/100/300/600；是诊断预算，不是最优值声明 |
| Batch | 最多16、无梯度累积；记录实际独立 source groups/有效起点/重复曝光，不能把batch增大视为新增数据 |
| 首版小样本 | 只用 verified suffix，源episode均衡再选完整有效起点；明确不含rehearsal，仅验证能否过拟合已知纠偏数据 |
| 量产候选混合 | 后续另立实验：25%suffix +25%原目标成功演示 +50%官方；不同时调整rank/norm/控制器 |
| 终止 | 非finite、冻结参数变化、norm/数据身份变化立即失败；600步仍不能拟合时转查实现/容量，不自动延长24000步 |

若选从原始 baseline 新建 LoRA，则随机A/零B用于严格 step0 等价；这是另一初始化条件，应独立记录。若恢复已有 optimizer，必须使用 typed restore 并核对全部 counters/moments，不能混用“重启优化器”和“原轨迹续训”的声明。

在仅3条后缀的诊断中，没有合法的恢复数据 held-out 指标。可用既有独立目标验证集与官方任务验证集检查误差退化，但不能叫“未见过的纠偏恢复能力”。每个checkpoint的采样误差按有效维/有效时域分别报告：活动臂关节、非活动臂、两夹爪、接近/打开窗口；不要仅看32D含padding的总loss。

已有离线指标说明模型选择必须改变：相同720个anchor（18任务×每任务5个episode×8个起点）上，24k的鼠标垫joint MAE约下降40%、挂杯下降52.8%，但全任务joint MAE约增加2.25%、gripper MAE增加1.0%；旧离线选择实际为9000步。loss按32维和50时域求平均，但没有分维贡献测量，不能将padding断言为主因。新方案不引入loss weighting或新的normalization；最终候选应按预先冻结的paired rollout规则决定，不能依最低loss宣布改善。

重放控制器和训练同时改变会混淆因果。首轮训练对比必须让step0与训练后模型使用同一个已冻结控制器；如果比较original与新控制器，单列2×2实验。恢复从中途状态继续是 privileged-state 诊断，完整从reset闭环是最终指标，两者分开统计。

## 现有接口与实际限制

| 组件 | 可复用接口 | 不可直接照搬的部分 |
|---|---|---|
| `gen1_recovery_takeover_20261020.py` | `prepare_cases`、前缀核验、`collect_suffix`、raw captures | 当前输出不是训练就绪；需新版本完成异常修复、dense复现与指定部署模式 sparse验收 |
| `gen1_multitask_data.py` | frame/episode完整性检查思路、索引审计 | `validate_split`要求至少3任务、正数train/val、旧expert字段；不能给新suffix伪造这些字段 |
| `gen1_replay_loader.py` | `JoinedDataset`、OpenPI原始/transform loader、norm equality检查 | expanded分支硬编码1990/每任务995；要求本地/官方覆盖与旧split绑定；需独立loader |
| `gen1_replay_sampler.py` | 分组范围校验、deterministic cycle结构 | 目前只支持official/local两源，源内按帧均匀；新suffix/episode均衡配比需新sampler |
| `gen1_replay_lora_train.py` | `make_config`、LoRA-only filter、冻结参数hash、`save_adapter` | CLI preflight硬编码2890；默认原始初始化后`zero_b`会清掉续训适配器 |
| `gen1_multitask_train.py` | `output_transform`、`validate_model`的显式噪声/物理误差框架 | `validate_inputs`依赖旧split；8个等距anchor未必覆盖短suffix关键阶段，需新的anchor manifest |
| `gen1_diagnose_pilot.py` | `checked_anchor`的50×32对齐、delta/abs/norm roundtrip | `source_vectors`按旧HDF5读取；suffix loader需实现对应reader且禁止跨界 |
| `gen1_replay_lora_extend.py` | `restore_payload`的typed Optax/NNX恢复核验 | main硬编码旧6000→24000实验，不作为新启动器 |
| `gen1_replay_lora_export.py` | 未合并LoRA导出、生产恢复与norm核验 | 新run manifest须符合source/model/frozen哈希语义；续训base不能误设为含LoRA的导出目录 |
| `gen1_policy_duration_repair_20261020.py` | 固定初始输入/专家准入、权重身份、真实physics计数的评估结构 | 当前活跃冻结runner不改；后续独立新panel/输出与候选身份 |
| `gen1_task_routing_20261020.py` | task_id在加载前选模型，非目标baseline/original | 目前目标pending；路由不构成50任务成功率不退化证据 |

建议新增文件为独立 suffix builder/validator、manifest-driven sampler/loader 和小样本 trainer；本轮先交付 builder/validator。训练入口与评估入口都应使用同一冻结数据清单，不能修改旧门禁使新数据冒充2890条训练集。

## 新 manifest 的必要内容

逐例记录原始source group与原split、采集/前缀/模型/控制器身份、接管边界、原指令、三路RGB文件hash、14D state/action、每帧物理时间、dense与sparse验收路径/hash、有效起点数、排除原因。失败、零长度、短于完整50动作窗口、缺摄像头、非finite、无法确定obs/action时序的案例保留为excluded记录，不能标为dataset_ready。

仅对通过全部门禁且拥有完整50动作起点的后缀产出ready条目。训练尾部不跨边界补其他episode；若采用重复末帧padding，必须另立明确mask语义实验，本次最小诊断直接只采完整窗口。
