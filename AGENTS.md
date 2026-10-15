# 新服务器上的 Codex 部署入口

用户希望在另一台服务器复现当前 RoboTwin 训练与 rollout。先完整阅读
`docs/DEPLOYMENT.md`、`docs/MODEL_AND_DATA.md`，再执行部署。

- 当前配置是 RoboTwin、LoRA only、batch 64、24000 步，初始化 Myungkyu 公共 JAX checkpoint。
- 不要误用历史脚本中的 LIBERO、motus 模型、batch 32 或早期 RoboCasa 环境。
- 保留本地鼠标垫 995 + 本地挂杯 995 + 官方 900 的训练范围及原模型 norm stats。
- 只做 Clean；两个目标任务各 10 次有效配对。保留现有专家准入、seed、终止条件与成功判定。
- `deploy/bootstrap.py` 只用于新服务器。发现正在训练的进程时不得重装环境、修改代码、删除数据或重启它。
- 首先检查 GitHub 数据 release 是否已公开且完整，再下载；草稿或缺卷不是数据已交付。
- 所有模型、数据、源码必须按清单验证。不要篡改 hash gate 来绕过下载/路径/版本错误。
- 新服务器部署未在当前训练机重演完整安装。逐阶段检查实际结果，遇到依赖安装问题先定位差异。
- 部署结束必须证明真实 batch-64 预检通过、训练 loss 已产生，以及 checkpoint 和实际 rollout 结果正常；仅有后台 PID 不算完整验收。
- `src/environment.md` 是早期 RoboCasa 环境记录，不适用于当前训练。当前环境快照在 `deploy/environments/`。
- 历史文件、已训练权重和原始数据不可因重新部署而删除。
