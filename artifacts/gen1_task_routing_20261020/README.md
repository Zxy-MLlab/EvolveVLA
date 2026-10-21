# 可选任务路由：候选仍待定

入口为 `scripts/gen1_task_routing_20261020.py`。现有 harness、默认 checkpoint、冻结实验 runner 均未修改，也未启动 GPU。

`pending_config.json` 的两个目标路由均为 `pending`，所以**当前所有任务都加载固定原始模型，并使用原执行器**。这不是已验证的改进配置。

| 显式任务 ID | 配置状态 | 实际模型与执行器 |
|---|---|---|
| place_mouse_pad / hanging_mug | pending 或 disabled | 固定 baseline + original |
| place_mouse_pad / hanging_mug | verified，且所有证据/哈希匹配 | 该任务唯一的候选模型及选定执行器 |
| 其他任务或未知 ID | 任意 | 固定 baseline + original |

未知 ID 的模型路由可回退，RoboTwin 不支持的任务名称仍会正常报错。路由不创建新任务，也不使用 seed、专家真值、物体状态或 rollout 成败选择权重。

## 运行接口

只做 CPU 核验并写入一份新的 receipt：

```bash
python scripts/gen1_task_routing_20261020.py plan \
  --config outputs/gen1_task_routing_20261020/pending_config.json \
  --task beat_block_hammer \
  --receipt outputs/gen1_task_routing_20261020/hammer_plan.json
```

实际执行接口如下，**本轮未执行此命令**：

```bash
python scripts/gen1_task_routing_20261020.py run \
  --config outputs/gen1_task_routing_20261020/pending_config.json \
  --task beat_block_hammer --seed 100000 \
  --receipt outputs/gen1_task_routing_20261020/hammer_run.json \
  --output outputs/gen1_task_routing_20261020/hammer_result.json
```

每个进程只跑一个任务的一次有效 episode，只加载一个 checkpoint。需要多次 rollout 时，用不同 seed 与新输出路径分别启动，以保持每次策略 RNG 从 0 开始。Clean、quantile normalization、无提示前缀、JAX noise 和完整 50 动作 chunk 固定。原始成功/专家准入流程沿用旧 harness。

baseline 或 original 控制器直接调用原 `robotwin_public_openpi_harness.py`。目标任务选中非 original 控制器时，新 wrapper 仅在模拟器进程导入处安装 `duration.create`，每次 `setup_demo` 重置门控；旧文件不写入。新控制器的同步双臂与局部规划时钟限制保持原声明。

## 将某个目标提升为 verified 的必要字段

候选条目须包含 `checkpoint`、`identity_sha256`、`identity_reference`、`identity_reference_sha256`、`controller_mode`、`deployment_evidence`、`deployment_evidence_sha256`。checkpoint 参考文件使用已有实验 manifest 的完整 `checkpoint.files_sha256`，启动前重新核验全部权重和 norm 文件；非目标任务不会读取候选权重。

deployment evidence 是独立决定文件，必须逐项绑定：

```json
{
  "status": "passed",
  "task_id": "place_mouse_pad",
  "checkpoint_identity_sha256": "待选定的完整身份",
  "controller_mode": "original 或 fixed15 或 opening15",
  "controller_source_sha256": "冻结控制器 SHA256",
  "validation_report": "/绝对路径/独立验证报告",
  "validation_report_sha256": "报告 SHA256"
}
```

该决定应由独立验证结果支持；权重恢复一致性验证不能替代闭环性能验证。代码校验证据绑定和完整性，不会自动判断报告中的统计结论。当前未创建 passed 决定。

## 核验与结论边界

7 项 CPU 测试通过，另在模拟器 Python 3.10 环境独立重跑通过。测试覆盖未知/非目标任务不核验或加载候选、pending 回退、禁止非目标覆盖、证据绑定、权重变更/新增文件检测、清除遗留干预环境变量、真实控制器 AST 经新模拟器 hook 执行并跨 setup 重置。

`baseline_plan_verified.json` 是最终代码下的真实纯 CPU plan：核验完整 baseline 文件后，未知任务选中 baseline/original，未加载 GPU。较早的 `baseline_plan_smoke.json` 仅保留过程记录，其源码哈希早于 Python 3.10 兼容性修正，不用于执行。

任务路由能保证非目标任务选择固定原始权重和执行器；**它本身没有证明成功率完全不变，也不能替代全部 50 任务回归评估**。当前目标候选尚未选定，不能宣称目标任务已经提高。
