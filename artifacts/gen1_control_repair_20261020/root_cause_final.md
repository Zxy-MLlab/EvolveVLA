# 闭环成功率问题：验证结论与处理

更新时间：2026-10-20

## 结论

本轮新建恢复数据的图像传输、确定性预处理、归一化及 LoRA 加载都通过核验；时长和重推理实验也没有改善结果。最符合现有证据的原因是：**专家轨迹上的动作预测误差下降，但策略在闭环偏离后的观测状态上缺少可靠纠偏能力，动作误差会随执行累积，最后影响放置、挂架和释放成功。** 这是当前证据支持的工作诊断，不能把其中单个因素说成已证明的唯一根因。新恢复数据审计不覆盖旧 2,890 条数据的每个文件，因此不能宣称历史训练集已被逐样本证明无误。

## 关键验证

- 原 24k LoRA 在固定新 seed 配对上：鼠标垫原模型 **1/10**、LoRA **3/10**；有 3 个候选独有成功，也有 1 个原模型独有成功，精确配对检验 `p=0.625`。挂杯双方 **2/9**；另有 1 个 seed 双方专家拒绝，按预案排除且不补样本。两个任务都未通过预先冻结的候选晋级规则。
- 执行时长/在线提前重推理都没有修好失败：`approach_replan` 的 4 个严格配对中，原方案和候选均 **1/4**，没有救回或丢失案例。Normal TOPP 时长改动的候选开发结果也为 **1/4 → 1/4**。
- 专家纠偏可行：挂杯 `87749` 的三次预先固定重放均通过部署控制器和 250 物理 tick 稳定保持，但三次来自同一 source episode，只准入一条训练样本。
- 纠偏训练诊断使用鼠标垫与挂杯各 1 条 source episode，共 **127 个转移、29 个高度重叠的 50 步窗口起点**，没有独立留出轨迹。图像传输逐像素一致，原始 normalization 与确定性训练变换对齐。
- 以 24k LoRA 初始化、batch 8 训练 600 步后，平均训练 batch loss 从前 100 步的 **0.00389** 降至最后 100 步的 **0.000457**。完整 LoRA 权重恢复、非零 adapter 保留、base 冻结和 step 600 导出均通过。
- 相同失败前缀的训练前后条件恢复结果仍是 **1/3 → 1/3**：鼠标垫都成功，两个挂杯都失败。三例的 RGB、14 维状态、指令、接管物理前缀和 source hashes 经独立审计一致；其中鼠标和 `87749` 还与训练 source 重叠，因此这个结果仅适用于训练场景诊断，不是泛化评估。
- 24k 专家轨迹动作误差在两个目标任务下降，但同一离线验证上的 18 任务关节 MAE 略升（`0.03317 → 0.03392 rad`）。历史 50 任务完整保留评估尚未完成。
- 依据当前实验记录，训练集总量 2,890 条（鼠标垫本地 995、挂杯本地 995、18 个公开任务共 900），但总条数不等于目标失败状态覆盖量。现有证据没有证明“995 条太少”或某个超参数本身是唯一原因；更具体的缺口是缺少多种模型偏离状态下、经部署执行器验证成功的纠偏轨迹。

## 当前处置

不把 24k LoRA 或 step 600 小样本模型投入路由。任务路由仍使用原始 Myungkyu checkpoint 和原控制器；所有 50 个任务 ID 已通过 CPU 路由选择检查，但这**不是**一次新的 50 任务 rollout，也不能证明基线在当前所有种子下的成功率。

## 下一轮有效修复

暂停对同一批静态专家轨迹继续加步数。为两目标任务收集多 seed、不同指令/物体、多个接管阶段的模型失败后纠偏轨迹，验证每条示范由原部署控制器重放成功并稳定保持。按 source episode 分组留出完整验证场景；训练时把纠偏数据与原任务回放混合，避免只拟合两条轨迹。只有独立新 seed 上的配对成功率增加且没有基线独有成功，才允许按任务开启候选；随后还要完成全 50 任务 Clean 保留评估。

本轮没有达到模型成功率提升目标。已完成的是定位闭环监督问题、排除部分数据/执行假设，并阻止未验证候选替换基线。

## 证据

- 新 seed 配对与路由建议：[route_confirmation/independent_summary.md](route_confirmation/independent_summary.md)
- 生效的基线专用路由配置：[baseline_only_config_v2.json](../gen1_task_routing_20261020/baseline_only_config_v2.json)；50 个任务的纯 CPU 选择检查：[baseline_only_route_selection.json](../gen1_task_routing_20261020/baseline_only_route_selection.json)
- 在线重规划配对审计：[approach_replan_dev4/paired_audit.json](approach_replan_dev4/paired_audit.json)
- 恢复示范重复与准入：[recovery_repro3/independent_selection.md](recovery_repro3/independent_selection.md)
- step 600 同前缀独立审计：[student_after600_independent_audit.json](student_after600_independent_audit.json)
- 训练与导出结果：[smallfit completed.json](../gen1_recovery_smallfit_combined_train_20261020/completed.json)
- combined 数据清单：[combined dataset manifest](../gen1_suffix_dataset_combined_original_20261020/manifest.json)
- 训练误差对照：[training_metric_evidence.md](training_metric_evidence.md)
