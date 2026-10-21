# 新服务器部署

此说明对应当前 batch-64 RoboTwin LoRA 实验。请先读 `MODEL_AND_DATA.md`。
当前训练机上的正在运行程序无需任何修改；下面操作只在另一台新服务器执行。

## 1. 机器、路径和网络

建议 Linux x86_64、NVIDIA GPU 显存至少 64 GiB、系统可用内存至少 64–96 GiB、磁盘预留约 300 GiB。
当前实测设备为 RTX PRO 6000 Blackwell 约 96 GiB 显存，driver 590.44.01；batch 64 时进程显存占用约 49 GiB。
系统应具备 NVIDIA 驱动、Vulkan、CUDA 构建工具（Curobo 需要）、git、编译器、ffmpeg、Miniconda、GitHub CLI。
硬件不同要先 smoke test；不要悄悄降低 batch 或替换模型来宣称复现完成。

为保持已验收 manifest 的绝对路径和哈希一致，代码必须位于实际目录 `/root/autodl-tmp/EvolveVLA`。
不要把该目录软链接到别处：部分校验会 resolve 路径。可把数据盘挂载到 `/root/autodl-tmp`。

```bash
mkdir -p /root/autodl-tmp
git clone https://github.com/Zxy-MLlab/EvolveVLA.git /root/autodl-tmp/EvolveVLA
cd /root/autodl-tmp/EvolveVLA
```

以下 `python` 指 Python 3.11.8 或更新版本，可用 Miniconda base 环境。
代理如果指向不存在的本地服务，需要仅在部署终端取消；不要更改其他训练进程的网络配置。
下载 Hugging Face 可按网络情况设置 `HF_ENDPOINT=https://hf-mirror.com`，revision 和 SHA256 必须保持不变。

## 2. 固定源码与环境

```bash
python deploy/bootstrap.py upstream
python deploy/bootstrap.py environments
```

第二步会在 `/root/autodl-tmp/conda_envs` 下新建三个环境：

| 环境 | Python | 用途 |
|---|---|---|
| `gen1_train_ready` | 3.11.16 | 真正的 JAX LoRA 训练和混合数据预检 |
| `scratch_orbax` | 3.11.16 | 策略推理、配对 rollout 的 policy 进程 |
| `robotwin_cu128` | 3.10.21 | RoboTwin / SAPIEN 仿真 |

训练环境的 Torch 是 CPU 版 `2.10.0+cpu`，GPU 训练由 **JAX 0.5.3** 完成；不要据此判断训练跑在 CPU。
仿真环境使用 Torch `2.11.0+cu128`。包版本完整记录在 `deploy/environments/`。
脚本按实际已运行环境快照安装并恢复已记录的 Transformers 改动，不让依赖解析器自动升级关键组件。
这些快照包含历史依赖冲突，`pip check` 不能替代实际 import、模型加载和数据 batch 检查。
如果单个包因 PyPI 或 CUDA wheel 可用性安装失败，Codex 应定位该包，保持核心版本，记录替代方案后重新实测。

默认不安装转换环境，因为发布包已有最终数据。需要重新转换时才使用 `environments --include-converter`（在新环境创建前决定）。
转换环境 `evolvevla_pi312` 是 Python 3.12 / LeRobot 0.6.1，**不包含 JAX**，不能拿来启动训练。
环境目录已存在时 bootstrap 会停止而非覆盖；重试前检查已完成阶段，不要删除有效环境。

## 3. 下载公开模型与仿真资产

```bash
python deploy/bootstrap.py model
python deploy/bootstrap.py simulation-assets
```

模型只下载公开 Myungkyu 固定 revision 的推理权重、metadata、norm stats，逐文件比对原校验单。
模型权重不在本 GitHub 仓库或数据 Release 内。

如需复现本机保存的 24,000 步诊断候选，可另行下载公开的
`model-gen1-lora-b64-24000-20261021` Release。该 Release 只有 450,846,720 字节的
Orbax adapter tar，SHA256 为
`857f919413cc3f79cb67a6c1db501e9ad1006171e53e220fb3a0bbc50c33eb6b`；解包后按
`artifacts/gen1_model_checkpoint_20261015/step24000_adapter_manifest.json` 校验。
它没有通过独立成功率门槛，默认路由仍使用原始 checkpoint。
仿真资产从固定 RoboTwin 数据 revision 下载 `objects.zip`、`embodiments.zip`、`background_texture.zip`，随后配置 embodiment 路径。
OIDN 与 Curobo 也按 `MODEL_AND_DATA.md` 固定版本准备。

## 4. 恢复数据资产

先确认 [data-20261015-v1](https://github.com/Zxy-MLlab/EvolveVLA/releases/tag/data-20261015-v1) 已公开。
Release 在 53 卷全部上传并校验后才公开，无法访问或缺卷时不能当作部署完成。
GitHub CLI 的 API 访问需要登录（公有仓库也可使用自己账号的登录状态）。

```bash
gh auth login
python deploy/data_release.py download --source /root/autodl-tmp/EvolveVLA \
  --cache /root/autodl-tmp/evolvevla-data-download
```

该工具逐卷下载、验证分卷 SHA256，再逐文件验证、解包。缓存只保留当前分卷，已恢复数据不删除。
已有完全相同文件会复用，发现同名但内容不同的文件就停止，不覆盖。
若网络中断留下 `.partial`，检查具体临时文件后再重试；不要删除原始 data 目录。
每卷是独立 tar，不能把 53 个 tar 拼成一个文件再解压。

全部资产约 44.86 GiB，不需要再下载官方 raw zip 来重做转换。
原始 train_inputs/val_inputs 软链接不是训练依赖，因此不随仓库上传；对应源 HDF5 全部恢复到原路径。

## 5. 预检、训练与自动 rollout

```bash
python deploy/bootstrap.py verify
python deploy/start_experiment.py
```

启动入口先执行 **9 个真实 batch 64（576 次抽样）**的混合预检，检查有限数值、数据范围、采样比例、720 个验证锚点和 norm stats。
通过后自动启动 LoRA，并在以下节点保存、导出并排队 rollout：

`1000 → 4000 → 9000 → 18000 → 24000`

模型基座冻结，仅更新约 4999 万 LoRA 参数。学习率 `1e-4`，预热 100 步，余弦衰减至 `1e-5`，batch 64，梯度累积 1。
数据范围固定为 2890 条；不要把另 16 个本地任务重新混入。
训练验证是离线动作误差，rollout 才是闭环成功率。

每个 checkpoint 用相同起始 seed `1500000`，两个目标任务各完成 10 次有效配对。
专家准入拒绝保留记录并按既定规则补充 seed；基础设施错误不能当成失败样本跳过。
此协议是两个任务的 Clean 配对诊断，不等同于官方 50 任务排行榜，不能与其均值直接比较。
`episode_steps=2000` 是桥接上限，任务自己的终止逻辑与上游成功判定保持原样。

## 6. 查看真实进度

```bash
cat outputs/gen1_expanded_train_rollouts_20261015.json
tail -n 3 outputs/gen1_replay_expanded_lora_b64_20261015/metrics.jsonl
tail -n 20 logs/gen1_expanded_lora_b64_train_20261015.log
```

- `metrics.jsonl` 中的 step 是参数更新次数，不是 epoch。
- checkpoint：`checkpoints/gen1_replay_expanded_lora_b64_20261015/<step>/adapter_state/`。
- 只有 `provenance.json` 已写出且 restore 验证通过，队列才允许导出。
- 闭环结果：`outputs/gen1_expanded_lora_step<step>_clean_pairs_20261015/comparison.json`。
- 队列状态为 completed 必须同时满足训练 24000 步和全部五个节点的真实 rollout 已完成。
- 队列失败时，训练可能仍在继续。先检查实际 PID、日志和已有 checkpoint，不要直接重启整轮。

## 7. 给另一台服务器 Codex 的提示词

> 请部署这个仓库的当前 RoboTwin 实验。完整阅读 AGENTS.md、docs/DEPLOYMENT.md 和 docs/MODEL_AND_DATA.md。
> 保留 Myungkyu 固定 revision、原 norm stats、LoRA only、batch 64、24000 步以及 2890 条训练范围。
> 数据从 data-20261015-v1 release 恢复并校验，不下载或复制这台旧服务器的模型权重。
> 不要修改正在运行的程序，不要跳过校验，不要把历史 motus/LIBERO/RoboCasa 配置当成当前方案。
> 完成真实数据预检、启动训练，并验证 1000/4000/9000/18000/24000 步保存和 Clean rollout。
> 每个目标任务 10 次有效配对，汇报真实成功率，不能用 loss 代替。

当前发布前验证包括源代码静态检查、数据分卷构建/恢复测试以及原训练机仍正常更新。
完整新服务器依赖安装尚未在本机执行，以免影响正在运行的训练；部署脚本不构成目标机器已跑通的证明。

## 8. 当前诊断资产

当前源码快照还包含 2026-10-20 的闭环诊断脚本和审计资产。入口报告是
`artifacts/gen1_control_repair_20261020/root_cause_final.md`，其中记录了固定新 seed
配对、执行时长对照、专家纠偏重放和 600 步小样本训练的结果。该报告明确指出 24k
LoRA 候选没有达到启用条件；`artifacts/gen1_task_routing_20261020/baseline_only_config_v2.json`
继续将两个目标任务路由到原始 checkpoint。小样本纠偏数据位于同一诊断目录的
`suffix_dataset/`，仅用于复现实验诊断，不是已验收的生产模型。
