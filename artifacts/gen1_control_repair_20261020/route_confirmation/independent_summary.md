# 固定新seed：原模型与24k模型独立配对审计

更新时间：2026-10-20T08:07:22.445606+00:00

| 任务 | 有效配对 | 原模型成功 | 候选成功 | 原模型精确95%区间 | 候选精确95%区间 | 候选新增成功seed | 原模型独有成功seed | 预定路由建议 |
|---|---:|---:|---:|---|---|---|---|---|
| place_mouse_pad | 10/10 | 1 | 3 | 0.3%–44.5% | 6.7%–65.2% | [2102001, 2102008, 2102009] | [2102002] | keep_baseline_original |
| hanging_mug | 9/10 | 2 | 2 | 2.8%–60.0% | 2.8%–60.0% | [] | [] | keep_baseline_original |

| task/seed | 原模型 | 候选 | 有效配对 | 排除项 |
|---|---|---|---|---|
| place_mouse_pad/2102000 | failure | failure | True | — |
| place_mouse_pad/2102001 | failure | success | True | — |
| place_mouse_pad/2102002 | success | failure | True | — |
| place_mouse_pad/2102003 | failure | failure | True | — |
| place_mouse_pad/2102004 | failure | failure | True | — |
| place_mouse_pad/2102005 | failure | failure | True | — |
| place_mouse_pad/2102006 | failure | failure | True | — |
| place_mouse_pad/2102007 | failure | failure | True | — |
| place_mouse_pad/2102008 | failure | success | True | — |
| place_mouse_pad/2102009 | failure | success | True | — |
| hanging_mug/2103000 | failure | failure | True | — |
| hanging_mug/2103001 | expert_rejected | expert_rejected | False | expert_rejected |
| hanging_mug/2103002 | failure | failure | True | — |
| hanging_mug/2103003 | failure | failure | True | — |
| hanging_mug/2103004 | failure | failure | True | — |
| hanging_mug/2103005 | failure | failure | True | — |
| hanging_mug/2103006 | success | success | True | — |
| hanging_mug/2103007 | success | success | True | — |
| hanging_mug/2103008 | failure | failure | True | — |
| hanging_mug/2103009 | failure | failure | True | — |

预定规则：每任务有效配对≥8、候选成功数严格更多、原模型成功转候选失败为0；全部计划case结束后才给临时建议。拒绝与异常不当作失败、不补seed。

两模型均使用原控制器；新seed只证明不在已核对本地记录中，不能证明不在公开数据或物理场景完全独立。样本小，区间宽，暂时建议不代表统计显著。
