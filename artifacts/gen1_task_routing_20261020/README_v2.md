# 可执行的任务路由入口

当前使用 `scripts/gen1_task_routing_v2_20261020.py` 和 `baseline_only_config_v2.json`。固定新 seed 配对评估后，两个目标任务都未通过预先冻结的候选晋级规则；目标路由已明确禁用候选，所有任务选择原始 Myungkyu 权重和原执行器。保留原始 `pending_config_v2.json` 作为实验前快照。

```bash
python scripts/gen1_task_routing_v2_20261020.py run \
  --config outputs/gen1_task_routing_20261020/baseline_only_config_v2.json \
  --task beat_block_hammer --seed 100000 \
  --receipt outputs/gen1_task_routing_20261020/new_run_receipt.json \
  --output outputs/gen1_task_routing_20261020/new_run_result.json
```

每次使用新的输出路径。显式任务 ID 决定模型选择；未知或非目标任务不加载候选权重。候选启用必须绑定该任务的验证证据、完整 checkpoint 与执行器哈希。一个进程只加载一个模型。

v1 实际运行发现旧 public harness 的参数解析器没有提供新版模拟器要求的 `episode_index`。v2 在独立进程内补上历史默认值 `None`，并保持原控制器、专家筛选、成功判据和模型输入流程。没有修改冻结 harness。v1 的错误日志与 receipt 保留；它在任务初始化前失败，不是任务失败样本。

v2 真实 GPU 冒烟运行已完成：`beat_block_hammer`、seed100000，原模型成功 **1/1**，161 个动作。见 `hammer_run_result_v2.json`。独立兼容审查6项测试通过；历史50任务的全部任务 ID 也通过纯CPU路由选择检查，见 `all50_route_selection.json`。

这证明入口可执行且非目标任务选择原模型，不代表重新测过50任务成功率，也不是单个微调模型已经保持所有任务能力。此次新 seed 结果与 step600 同前缀审计均不支持启用候选；理由和结果 hash 记录在配置中。
