# 控制修复开发集：独立配对审计

更新时间：2026-10-20T07:32:26.375009+00:00

| 执行者 | 模式 | 任务 | 有效配对 | original成功 | 新模式成功 | 改善 | 退化 |
|---|---|---|---:|---:|---:|---|---|
| expert_development | fixed15 | place_mouse_pad | 2 | 2 | 2 | — | — |
| expert_development | fixed15 | hanging_mug | 2 | 0 | 1 | hanging_mug_88038 | — |
| expert_development | fixed15 | all | 4 | 2 | 3 | hanging_mug_88038 | — |
| expert_development | opening15 | place_mouse_pad | 2 | 2 | 2 | — | — |
| expert_development | opening15 | hanging_mug | 2 | 0 | 1 | hanging_mug_88038 | — |
| expert_development | opening15 | all | 4 | 2 | 3 | hanging_mug_88038 | — |
| candidate_development | fixed15 | place_mouse_pad | 2 | 1 | 1 | — | — |
| candidate_development | fixed15 | hanging_mug | 2 | 0 | 0 | — | — |
| candidate_development | fixed15 | all | 4 | 1 | 1 | — | — |
| candidate_development | opening15 | place_mouse_pad | 2 | 1 | 1 | — | — |
| candidate_development | opening15 | hanging_mug | 2 | 0 | 0 | — | — |
| candidate_development | opening15 | all | 4 | 1 | 1 | — | — |

| 执行者 | 案例 | 模式 | 状态/成功 | 动作数 | 激活动作数 | resample动作数 | 实际ticks | 发出速度峰值(rad/s)* | 排除项 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| expert_development | place_mouse_pad_74150 | original | success/True | 128 | 0 | 0 | 10065 | 未测 | — |
| expert_development | place_mouse_pad_74150 | fixed15 | success/True | 128 | 128 | 81 | 1915 | 3.258058452748401 | — |
| expert_development | place_mouse_pad_74150 | opening15 | success/True | 128 | 11 | 0 | 9694 | 未测 | — |
| expert_development | place_mouse_pad_59008 | original | success/True | 141 | 0 | 0 | 11680 | 未测 | — |
| expert_development | place_mouse_pad_59008 | fixed15 | success/True | 141 | 141 | 94 | 2110 | 3.0925757221079455 | — |
| expert_development | place_mouse_pad_59008 | opening15 | success/True | 141 | 11 | 0 | 11309 | 未测 | — |
| expert_development | hanging_mug_88038 | original | failure/False | 358 | 0 | 0 | 28958 | 未测 | — |
| expert_development | hanging_mug_88038 | fixed15 | success/True | 330 | 330 | 223 | 4945 | 3.7223929317740265 | — |
| expert_development | hanging_mug_88038 | opening15 | success/True | 330 | 22 | 0 | 26305 | 未测 | — |
| expert_development | hanging_mug_87749 | original | failure/False | 357 | 0 | 0 | 29074 | 未测 | — |
| expert_development | hanging_mug_87749 | fixed15 | failure/False | 357 | 357 | 240 | 5355 | 3.6221516103101012 | — |
| expert_development | hanging_mug_87749 | opening15 | failure/False | 357 | 22 | 0 | 28304 | 未测 | — |
| candidate_development | place_mouse_pad_74150 | original | success/True | 126 | 0 | 0 | 8838 | 未测 | — |
| candidate_development | place_mouse_pad_74150 | fixed15 | success/True | 127 | 127 | 127 | 1896 | 3.3003369102936992 | — |
| candidate_development | place_mouse_pad_74150 | opening15 | success/True | 126 | 11 | 11 | 8679 | 0.23667460194346934 | — |
| candidate_development | place_mouse_pad_59008 | original | failure/False | 400 | 0 | 0 | 17941 | 未测 | — |
| candidate_development | place_mouse_pad_59008 | fixed15 | failure/False | 400 | 400 | 400 | 6000 | 3.376417438169141 | — |
| candidate_development | place_mouse_pad_59008 | opening15 | failure/False | 400 | 11 | 11 | 17583 | 0.20845313941182936 | — |
| candidate_development | hanging_mug_88038 | original | failure/False | 900 | 0 | 0 | 40410 | 未测 | — |
| candidate_development | hanging_mug_88038 | fixed15 | failure/False | 900 | 900 | 900 | 13500 | 3.670124110460099 | — |
| candidate_development | hanging_mug_88038 | opening15 | failure/False | 900 | 22 | 22 | 39829 | 0.22720236102940433 | — |
| candidate_development | hanging_mug_87749 | original | failure/False | 900 | 0 | 0 | 40791 | 未测 | — |
| candidate_development | hanging_mug_87749 | fixed15 | failure/False | 900 | 900 | 900 | 13500 | 3.581761806950067 | — |
| candidate_development | hanging_mug_87749 | opening15 | failure/False | 900 | 22 | 22 | 40500 | 0.3321128608224512 | — |

| 执行者 | 案例 | 模式 | 首激活动作 | 首物理差异边界 | 首目标差异动作 | 干预前一致 | 当前首次完整预测一致 |
|---|---|---|---:|---:|---:|---|---|
| expert_development | place_mouse_pad_74150 | fixed15 | 0 | 1 | None | True | 专家固定源目标 |
| expert_development | place_mouse_pad_59008 | fixed15 | 0 | 1 | None | True | 专家固定源目标 |
| expert_development | hanging_mug_88038 | fixed15 | 0 | 1 | 330 | True | 专家固定源目标 |
| expert_development | hanging_mug_87749 | fixed15 | 0 | 1 | None | True | 专家固定源目标 |
| expert_development | place_mouse_pad_74150 | opening15 | 117 | 118 | None | True | 专家固定源目标 |
| expert_development | place_mouse_pad_59008 | opening15 | 130 | 131 | None | True | 专家固定源目标 |
| expert_development | hanging_mug_88038 | opening15 | 130 | 131 | 330 | True | 专家固定源目标 |
| expert_development | hanging_mug_87749 | opening15 | 122 | 123 | None | True | 专家固定源目标 |
| candidate_development | place_mouse_pad_74150 | fixed15 | 0 | 1 | 50 | True | True |
| candidate_development | place_mouse_pad_59008 | fixed15 | 0 | 1 | 50 | True | True |
| candidate_development | hanging_mug_88038 | fixed15 | 0 | 1 | 50 | True | True |
| candidate_development | hanging_mug_87749 | fixed15 | 0 | 1 | 50 | True | True |
| candidate_development | place_mouse_pad_74150 | opening15 | 115 | 116 | None | True | True |
| candidate_development | place_mouse_pad_59008 | opening15 | 142 | 143 | 150 | True | True |
| candidate_development | hanging_mug_88038 | opening15 | 117 | 118 | 150 | True | True |
| candidate_development | hanging_mug_87749 | opening15 | 117 | 118 | 150 | True | True |

| 执行者 | 案例 | 模式 | 终末臂首次打开命令边界 | 窗口实际ticks | 窗口激活动作数 |
|---|---|---|---|---:|---:|
| expert_development | place_mouse_pad_74150 | original | 119→128 | 431 | 0 |
| expert_development | place_mouse_pad_74150 | fixed15 | 119→128 | 130 | 9 |
| expert_development | place_mouse_pad_74150 | opening15 | 119→128 | 130 | 9 |
| expert_development | place_mouse_pad_59008 | original | 132→141 | 431 | 0 |
| expert_development | place_mouse_pad_59008 | fixed15 | 132→141 | 130 | 9 |
| expert_development | place_mouse_pad_59008 | opening15 | 132→141 | 130 | 9 |
| expert_development | hanging_mug_88038 | original | 321→330 | 450 | 0 |
| expert_development | hanging_mug_88038 | fixed15 | 321→330 | 130 | 9 |
| expert_development | hanging_mug_88038 | opening15 | 321→330 | 130 | 9 |
| expert_development | hanging_mug_87749 | original | 324→333 | 450 | 0 |
| expert_development | hanging_mug_87749 | fixed15 | 324→333 | 135 | 9 |
| expert_development | hanging_mug_87749 | opening15 | 324→333 | 135 | 9 |
| candidate_development | place_mouse_pad_74150 | original | 117→126 | 262 | 0 |
| candidate_development | place_mouse_pad_74150 | fixed15 | 118→127 | 126 | 9 |
| candidate_development | place_mouse_pad_74150 | opening15 | 117→126 | 131 | 9 |
| candidate_development | place_mouse_pad_59008 | original | 144→153 | 291 | 0 |
| candidate_development | place_mouse_pad_59008 | fixed15 | 145→154 | 135 | 9 |
| candidate_development | place_mouse_pad_59008 | opening15 | 144→153 | 135 | 9 |
| candidate_development | hanging_mug_88038 | original | 309→318 | 281 | 0 |
| candidate_development | hanging_mug_88038 | fixed15 | 299→308 | 135 | 9 |
| candidate_development | hanging_mug_88038 | opening15 | 304→313 | 135 | 9 |
| candidate_development | hanging_mug_87749 | original | 317→326 | 288 | 0 |
| candidate_development | hanging_mug_87749 | fixed15 | 304→313 | 135 | 9 |
| candidate_development | hanging_mug_87749 | opening15 | 317→326 | 135 | 9 |

与旧轮original的可重复性核对（不是本轮处理效果）：

- expert_development/place_mouse_pad_74150：exact；首次预测最大差异 不适用。
- expert_development/place_mouse_pad_59008：exact；首次预测最大差异 不适用。
- expert_development/hanging_mug_88038：exact；首次预测最大差异 不适用。
- expert_development/hanging_mug_87749：exact；首次预测最大差异 不适用。
- candidate_development/place_mouse_pad_74150：cross_run_mismatch；首次预测最大差异 0.003740668296813965。
- candidate_development/place_mouse_pad_59008：cross_run_mismatch；首次预测最大差异 0.003740719170309603。
- candidate_development/hanging_mug_88038：cross_run_mismatch；首次预测最大差异 0.00518345832824707。
- candidate_development/hanging_mug_87749：cross_run_mismatch；首次预测最大差异 0.004442691802978516。

*速度峰值仅涵盖执行重采样的路径；未测不代表机器人速度为0。

详细JSON包含源state/JPEG独立核验、初始物理/输入/权重一致性、干预前前缀、首个物理/目标/时长差异、每臂打开命令窗口ticks和original透明回归。

本表只读开发集；拒绝、缺失、非零返回码与审计不通过不进入有效配对分母。确认集未读取。
