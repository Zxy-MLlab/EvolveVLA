实际命令及结果以每个 run 的 manifest.json、逐案例 command/result 和日志为准。

开发对照：`gen1_policy_duration_repair_20261020.py run --panel panel_development4.json --timing-modes original fixed15 opening15`，分别 `--actor expert` 与 `--actor policy --checkpoint checkpoints/gen1_expanded_lora_step24000_export_20261015`。

控制器 9 项测试、IPC 与准入/真实计时 4 项测试通过。原始源码只读；进程内 AST hook 是唯一控制器注入点。
