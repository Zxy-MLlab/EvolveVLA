# Myungkyu seed-0 59999 RoboTwin 2.0 Clean evaluation

This is an independently trained public reproduction, not the RoboTwin Team leaderboard artifact.

Checkpoint: `/root/autodl-tmp/EvolveVLA/checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k`
Source: `Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k@37e10d7f4f49149e2ce5e08a77b07f228a84322f`
RoboTwin: `ba0fd21ad21c336bedc7bd8eacd3e514860910e2`; XPolicyLab: `c37109c500be67d0dea6b36bf7337bbd26e763cd`.
Protocol: 50 Clean tasks x 10 accepted episodes; official expert prefilter, instructions, success predicates, and per-task step limits; 50-step OpenPI chunks; quantile normalization; JAX seed-0 noise; RT+OIDN; one GPU worker.
Randomized evaluation was paused at the user's request and is not included.

## Result

| This run | Official 100-trial result | Difference | 95% Wilson interval |
|---:|---:|---:|---:|
| 65.80% (329/500) | 70.66% | -4.86 pp | 61.54% to 69.82% |

The leaderboard displays 70.7%; the mean of its published per-task rates is 70.66%.
At 10 trials per task, task rates move in 10-point increments. Under the official per-task rates as fixed Bernoulli probabilities, the suite-mean sampling SD is 1.77 pp; the observed gap is -2.75 such SDs.

## Remaining provenance mismatch

The candidate declares checkpoint step `59999` and the official deploy config requests `59999`, so the nominal step matches.
The candidate normalization asset is `aloha_agilex_sim`, while the official deploy config requests `arx_x5_sim`. The assets do not match, and the leaderboard checkpoint is not the independently trained candidate evaluated here. This prevents attributing the full gap to 10-vs-100-trial sampling.
A 2026-10-07 rescan of 82 public `pi05 robotwin` model trees found no `arx_x5_sim` path. The official dataset still exposes only its step-29999 Pi_05 archive; the only public 59999 Orbax models are independent ShijieSKY seed-42 variants.

## Eliminated implementation mismatches

The pinned RoboTwin and XPolicyLab worktrees are clean. The actual OpenPI source path used by the runner is `/root/autodl-tmp/EvolveVLA/robotwin_repo/XPolicyLab/policy/Pi_05/openpi/src`. Its seven inference-critical policy, transform, configuration, model, and tokenizer files are byte-identical to the pinned XPolicyLab copy; their per-file SHA-256 values are recorded in `comparison.json`.
The sole OpenPI file difference is a TYPE_CHECKING-only lazy import in `training/checkpoints.py`; it does not alter checkpoint values or inference.
Official local evaluation starts a fresh Pi_05 server per task and OpenPI initializes its JAX key to 0. This sweep likewise loaded one policy per task with seed 0, so cross-task RNG reuse is not a mismatch.
Instruction selection also follows the same RNG order: expert `setup_demo`/`play_once`, environment close, second same-seed `setup_demo`, then the pinned official `build_instruction` call.

## Comparison with the public step-29999 artifact

The verified public step-29999 artifact scored 65.60% under the same Clean protocol. This independent 59999 reproduction changes the suite mean by +0.20 pp.
Across tasks, 15 improved, 18 regressed, and 17 were unchanged. Mean absolute task error changed from 15.50 to 14.54 pp; RMSE changed from 22.34 to 19.45 pp; per-task Pearson correlation changed from 0.627 to 0.677.

## Largest task deficits

| Task | This run | Official | Difference |
|---|---:|---:|---:|
| `place_dual_shoes` | 20% | 70% | -50 pp |
| `lift_pot` | 60% | 98% | -38 pp |
| `stack_blocks_three` | 40% | 77% | -37 pp |
| `place_bread_skillet` | 40% | 76% | -36 pp |
| `stamp_seal` | 20% | 53% | -33 pp |
| `place_object_scale` | 50% | 78% | -28 pp |
| `put_bottles_dustbin` | 50% | 78% | -28 pp |
| `place_mouse_pad` | 10% | 34% | -24 pp |
| `place_burger_fries` | 60% | 82% | -22 pp |
| `place_shoe` | 60% | 82% | -22 pp |

## Largest task gains

| Task | This run | Official | Difference |
|---|---:|---:|---:|
| `click_bell` | 80% | 31% | +49 pp |
| `blocks_ranking_size` | 80% | 45% | +35 pp |
| `place_cans_plasticbox` | 60% | 31% | +29 pp |
| `press_stapler` | 100% | 79% | +21 pp |
| `click_alarmclock` | 80% | 65% | +15 pp |
| `place_phone_stand` | 80% | 65% | +15 pp |
| `handover_block` | 40% | 30% | +10 pp |
| `move_stapler_pad` | 30% | 20% | +10 pp |
| `place_object_basket` | 80% | 70% | +10 pp |
| `place_a2b_left` | 80% | 71% | +9 pp |

## All tasks

| Task | Successes | This run | Official | Difference | Accepted seeds |
|---|---:|---:|---:|---:|---|
| `adjust_bottle` | 10/10 | 100% | 99% | +1 pp | 100001, 100002, 100005, 100006, 100008, 100010, 100012, 100013, 100015, 100016 |
| `beat_block_hammer` | 8/10 | 80% | 93% | -13 pp | 100000, 100001, 100002, 100003, 100006, 100007, 100008, 100009, 100010, 100011 |
| `blocks_ranking_rgb` | 7/10 | 70% | 72% | -2 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `blocks_ranking_size` | 8/10 | 80% | 45% | +35 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `click_alarmclock` | 8/10 | 80% | 65% | +15 pp | 100000, 100002, 100003, 100004, 100007, 100008, 100009, 100010, 100011, 100012 |
| `click_bell` | 8/10 | 80% | 31% | +49 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `dump_bin_bigbin` | 8/10 | 80% | 92% | -12 pp | 100001, 100002, 100003, 100004, 100005, 100010, 100011, 100015, 100016, 100017 |
| `grab_roller` | 10/10 | 100% | 99% | +1 pp | 100000, 100001, 100002, 100003, 100004, 100006, 100007, 100008, 100009, 100010 |
| `handover_block` | 4/10 | 40% | 30% | +10 pp | 100000, 100002, 100003, 100004, 100006, 100007, 100008, 100009, 100010, 100011 |
| `handover_mic` | 9/10 | 90% | 98% | -8 pp | 100000, 100001, 100003, 100005, 100006, 100007, 100008, 100009, 100011, 100012 |
| `hanging_mug` | 2/10 | 20% | 17% | +3 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100008, 100009, 100010 |
| `lift_pot` | 6/10 | 60% | 98% | -38 pp | 100000, 100002, 100003, 100005, 100009, 100012, 100014, 100019, 100020, 100022 |
| `move_can_pot` | 4/10 | 40% | 60% | -20 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `move_pillbottle_pad` | 5/10 | 50% | 46% | +4 pp | 100001, 100002, 100005, 100006, 100007, 100008, 100009, 100010, 100011, 100013 |
| `move_playingcard_away` | 8/10 | 80% | 84% | -4 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `move_stapler_pad` | 3/10 | 30% | 20% | +10 pp | 100000, 100001, 100003, 100004, 100005, 100006, 100007, 100008, 100009, 100010 |
| `open_laptop` | 9/10 | 90% | 93% | -3 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `open_microwave` | 9/10 | 90% | 86% | +4 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100009, 100010 |
| `pick_diverse_bottles` | 6/10 | 60% | 66% | -6 pp | 100000, 100002, 100007, 100009, 100012, 100013, 100014, 100015, 100016, 100022 |
| `pick_dual_bottles` | 7/10 | 70% | 72% | -2 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `place_a2b_left` | 8/10 | 80% | 71% | +9 pp | 100000, 100001, 100003, 100004, 100005, 100007, 100008, 100009, 100010, 100012 |
| `place_a2b_right` | 7/10 | 70% | 68% | +2 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `place_bread_basket` | 6/10 | 60% | 63% | -3 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `place_bread_skillet` | 4/10 | 40% | 76% | -36 pp | 100004, 100005, 100006, 100007, 100009, 100011, 100015, 100017, 100019, 100021 |
| `place_burger_fries` | 6/10 | 60% | 82% | -22 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `place_can_basket` | 4/10 | 40% | 56% | -16 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100007, 100008, 100009, 100010 |
| `place_cans_plasticbox` | 6/10 | 60% | 31% | +29 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `place_container_plate` | 9/10 | 90% | 92% | -2 pp | 100000, 100003, 100004, 100005, 100006, 100007, 100008, 100009, 100010, 100011 |
| `place_dual_shoes` | 2/10 | 20% | 70% | -50 pp | 100001, 100002, 100004, 100007, 100008, 100009, 100011, 100012, 100013, 100015 |
| `place_empty_cup` | 8/10 | 80% | 96% | -16 pp | 100000, 100001, 100002, 100004, 100005, 100007, 100008, 100009, 100010, 100011 |
| `place_fan` | 7/10 | 70% | 66% | +4 pp | 100002, 100004, 100005, 100006, 100007, 100008, 100009, 100010, 100011, 100012 |
| `place_mouse_pad` | 1/10 | 10% | 34% | -24 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `place_object_basket` | 8/10 | 80% | 70% | +10 pp | 100000, 100002, 100003, 100004, 100008, 100010, 100011, 100012, 100013, 100014 |
| `place_object_scale` | 5/10 | 50% | 78% | -28 pp | 100000, 100002, 100003, 100004, 100006, 100007, 100009, 100010, 100011, 100012 |
| `place_object_stand` | 9/10 | 90% | 85% | +5 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `place_phone_stand` | 8/10 | 80% | 65% | +15 pp | 100000, 100001, 100002, 100005, 100006, 100008, 100010, 100012, 100014, 100015 |
| `place_shoe` | 6/10 | 60% | 82% | -22 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100010, 100011 |
| `press_stapler` | 10/10 | 100% | 79% | +21 pp | 100000, 100002, 100003, 100004, 100005, 100006, 100007, 100009, 100010, 100011 |
| `put_bottles_dustbin` | 5/10 | 50% | 78% | -28 pp | 100000, 100002, 100004, 100005, 100006, 100007, 100008, 100009, 100010, 100013 |
| `put_object_cabinet` | 3/10 | 30% | 39% | -9 pp | 100009, 100027, 100044, 100045, 100048, 100050, 100051, 100053, 100054, 100066 |
| `rotate_qrcode` | 7/10 | 70% | 90% | -20 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `scan_object` | 4/10 | 40% | 49% | -9 pp | 100002, 100006, 100013, 100014, 100017, 100019, 100029, 100037, 100038, 100040 |
| `shake_bottle` | 10/10 | 100% | 100% | +0 pp | 100000, 100001, 100003, 100004, 100005, 100006, 100009, 100013, 100015, 100018 |
| `shake_bottle_horizontally` | 10/10 | 100% | 99% | +1 pp | 100000, 100001, 100003, 100004, 100005, 100006, 100009, 100013, 100015, 100018 |
| `stack_blocks_three` | 4/10 | 40% | 77% | -37 pp | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100008, 100009, 100010 |
| `stack_blocks_two` | 8/10 | 80% | 91% | -11 pp | 100000, 100001, 100003, 100004, 100005, 100006, 100007, 100008, 100009, 100010 |
| `stack_bowls_three` | 7/10 | 70% | 81% | -11 pp | 100000, 100001, 100004, 100006, 100009, 100011, 100015, 100017, 100020, 100022 |
| `stack_bowls_two` | 10/10 | 100% | 94% | +6 pp | 100000, 100001, 100002, 100003, 100004, 100006, 100008, 100011, 100014, 100015 |
| `stamp_seal` | 2/10 | 20% | 53% | -33 pp | 100000, 100001, 100004, 100007, 100008, 100010, 100011, 100013, 100015, 100016 |
| `turn_switch` | 6/10 | 60% | 52% | +8 pp | 100000, 100001, 100002, 100004, 100005, 100006, 100007, 100008, 100009, 100010 |
