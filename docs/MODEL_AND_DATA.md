# 模型和数据来源

## 当前公开模型

- Hugging Face：`Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k`
- 固定 revision：`37e10d7f4f49149e2ce5e08a77b07f228a84322f`
- 公共 checkpoint 自带 metadata：`config_name=rldx2_pi05jax_robotwin`、`asset_id=aloha_agilex_sim`、`step=59999`。
- 下载目录：`checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/`。
- 格式：OpenPI JAX / Orbax `params/`；不是 PyTorch `model.safetensors`。
- norm stats SHA256：`ec3f94d2763fb6e018cca42d6e0ee068eab53c041c522dbbc18ec0be9132ace8`。
- 原始模型文件校验单：`deploy/model_reference/SHA256SUMS`；发布仓库不携带权重。
- 59999 是公开 checkpoint 的记录步数。本轮本地 LoRA 从零更新到 24000，不是接着训练到 83999。

这是 Myungkyu 公开的独立复现，不能把它称为 RoboTwin 团队官方排行榜模型。
最初目标文件提到过 motus 的 π0.5，后续实验已切换到上述 Myungkyu 模型；LIBERO 也不在本轮范围内。

数据侧 14 维：左右臂各 6 个关节 + 1 个夹爪；三个 RGB 相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`。
数据频率 15 Hz。原始 state/action 是绝对关节目标，训练中沿用 pinned OpenPI 的 delta-joint 变换，夹爪不做 delta。
模型内部 action dim 32、horizon 50，额外维度按原变换补齐；输出再还原为 14 维绝对动作。
不要照早期目标文件中的 horizon 32 配置当前模型。

## 自采资产

`deploy/data_inventory.json` 列出每个文件的 SHA256、长度和分卷；原始轨迹还标注 task、seed 和 split。

| 类别 | 总数 | 参与本轮训练 | 验证 |
|---|---:|---:|---:|
| 本地 `place_mouse_pad` | 1000 | 995 | 5 |
| 本地 `hanging_mug` | 1000 | 995 | 5 |
| 本地其他 16 任务 | 800 | 0 | 80 |
| 官方对应 18 任务 | 900 | 900 | 0 |

本地共 2800 条经过成功、规划、重放验收的专家轨迹。原始划分为 2710 train + 90 val；本轮从 train 中选取两个重点任务。
保留全部 2800 条已验收原始资产，未验收尝试及重复转换缓存不进入发布清单。
原始文件位于 `data/gen1_task_batches_20261010/`，其中两条早期轨迹来自 `data/gen1_demo_clean/`。
成功采集是专家轨迹验收，不等于策略 rollout 成功。

## 官方 replay

- 数据源：[`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
- 固定 revision：`3dc3b798668feb99ac61cc9086d84cbcc3d79186`
- 对应原始文件：`dataset/<task>/demo_clean.zip`，18 个任务，各 50 条。
- 任务、来源、帧数和拆分绑定见 `outputs/gen1_official_replay_manifest_20261013.json`。
- 发布包包含已转换 replay，减少新服务器重复准备；来源与许可仍归对应上游项目，未声称它是自采数据。

转换数据：

| 目录 | 内容 |
|---|---|
| `data/gen1_multitask_coverage_expanded_train_20261014` | 全部本地 train 转换，2710 条 / 671178 帧 |
| `data/gen1_multitask_coverage_val_20261012` | 90 条独立本地验证轨迹 |
| `data/gen1_official_replay_train_20261013` | 官方 900 条 / 207952 帧 |

本轮选中的本地训练数据为 502624 帧，混合后总帧数 710576。训练按固定采样策略抽取起始帧，不能用轨迹数直接换算 epoch。
每轮总训练更新 24000 次 × batch 64 = 1536000 次起始帧抽样。

## 上游版本

- RoboTwin：`ba0fd21ad21c336bedc7bd8eacd3e514860910e2`
- XPolicyLab（含 OpenPI）：`c37109c500be67d0dea6b36bf7337bbd26e763cd`
- Curobo：`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，0.7.8
- OIDN：2.5.1
- Tokenizer：`gs://big_vision/paligemma_tokenizer.model`，SHA256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`

第三方仓库按原许可证下载；`deploy/runtime_overrides/` 保留了当前环境中实际改过的 Transformers 文件及原版权标注。
