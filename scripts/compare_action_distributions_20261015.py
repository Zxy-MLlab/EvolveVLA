"""Read-only numeric comparison; no video decoding or training changes."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path('/root/autodl-tmp/EvolveVLA')
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def load(name, rows):
    root = ROOT / 'data' / name
    table = pd.concat([pd.read_parquet(p, columns=['index', 'episode_index', 'frame_index',
                      'action', 'observation.state']) for p in sorted((root/'data').rglob('*.parquet'))])
    table = table.sort_values('index')
    assert np.array_equal(table['index'], np.arange(len(table)))
    assert len(table) == sum(r['frames'] for r in rows)
    actions = np.stack(table.action).astype(np.float64)
    states = np.stack(table['observation.state']).astype(np.float64)
    assert actions.shape == states.shape == (len(table), 14)
    assert np.isfinite(actions).all() and np.isfinite(states).all()
    result = {}
    offset = 0
    for episode, row in enumerate(rows):
        end = offset + row['frames']
        part = table.iloc[offset:end]
        assert (part.episode_index == episode).all()
        assert np.array_equal(part.frame_index, np.arange(row['frames']))
        a, s = actions[offset:end], states[offset:end]
        assert np.array_equal(a[:-1], s[1:])
        result.setdefault(row['task'], []).append((a, a-s))
        offset = end
    return result


def summary(a, delta, episodes):
    return dict(episodes=episodes, frames=len(a),
        action_mean=a.mean(0).tolist(), action_std=a.std(0).tolist(),
        action_quantiles=np.quantile(a, [.01, .5, .99], axis=0).tolist(),
        mean_abs_joint_delta=float(np.abs(delta[:,JOINTS]).mean()),
        abs_joint_delta_q99=float(np.quantile(np.abs(delta[:,JOINTS]), .99)),
        grippers=[dict(mean=float(a[:,i].mean()), below_01=float((a[:,i]<.1).mean()),
                       above_09=float((a[:,i]>.9).mean()),
                       q01_q50_q99=np.quantile(a[:,i], [.01,.5,.99]).tolist()) for i in (6,13)])


def arm_activity(episodes):
    counts = Counter()
    for _, delta in episodes:
        left = np.abs(delta[:,:6]).sum()
        right = np.abs(delta[:,7:13]).sum()
        counts['left_dominant' if left > 2*right else
               'right_dominant' if right > 2*left else 'both_or_similar'] += 1
    return dict(counts)


def main():
    local_manifest = json.loads((ROOT/'data/gen1_multitask_coverage_expanded_20261014/split_manifest.json').read_text())
    official_manifest = json.loads((ROOT/'outputs/gen1_official_replay_manifest_20261013.json').read_text())
    local_rows = sorted((r for r in local_manifest['episodes'] if r['split']=='train'), key=lambda r:r['input_link'])
    official_rows = official_manifest['episodes']
    assert set(local_manifest['tasks']) == set(official_manifest['tasks'])
    local_ids = {(r['task'],r['seed']) for r in local_manifest['episodes']}
    assert not local_ids.intersection((r['task'],r['seed']) for r in official_rows)
    local = load('gen1_multitask_coverage_expanded_train_20261014', local_rows)
    official = load('gen1_official_replay_train_20261013', official_rows)
    report = dict(scope='All finalized numeric parquet frames; expanded RGB audit still pending',
        weighting='Frame weighted within each task; compare tasks separately',
        delta_definition='action[t] minus state[t], one-step diagnostic; not full 50-step training chunks',
        task_seed_overlap=0, tasks={})
    for task in sorted(local):
        la = np.concatenate([x[0] for x in local[task]])
        ld = np.concatenate([x[1] for x in local[task]])
        oa = np.concatenate([x[0] for x in official[task]])
        od = np.concatenate([x[1] for x in official[task]])
        w = [float(wasserstein_distance(la[:,i],oa[:,i])) for i in range(14)]
        scale = oa.std(0)
        # Exclude almost-constant official dimensions from standardized summaries.
        active = [i for i in JOINTS if scale[i]>.01]
        normalized = [w[i]/scale[i] for i in active]
        eq_local = float(np.mean([np.abs(x[1][:,JOINTS]).mean() for x in local[task]]))
        eq_off = float(np.mean([np.abs(x[1][:,JOINTS]).mean() for x in official[task]]))
        entry = dict(local=summary(la,ld,len(local[task])), official=summary(oa,od,len(official[task])),
            local_arm_activity=arm_activity(local[task]), official_arm_activity=arm_activity(official[task]),
            action_wasserstein_by_dim=w, active_joint_dims=active,
            mean_active_joint_wasserstein_in_official_std=float(np.mean(normalized)),
            max_active_joint_wasserstein_in_official_std=float(max(normalized)),
            episode_weighted_mean_abs_joint_delta_ratio=eq_local/eq_off,
            local_joint_outside_official_minmax_fraction=float(((la[:,JOINTS]<oa[:,JOINTS].min(0)-1e-6)|
                (la[:,JOINTS]>oa[:,JOINTS].max(0)+1e-6)).mean()))
        report['tasks'][task]=entry
        print(task, json.dumps(dict(local_frames=len(la), official_frames=len(oa),
            standardized_w1=entry['mean_active_joint_wasserstein_in_official_std'],
            delta_ratio=entry['episode_weighted_mean_abs_joint_delta_ratio'],
            local_arm_activity=entry['local_arm_activity'], official_arm_activity=entry['official_arm_activity'],
            local_gripper_mean=[x['mean'] for x in entry['local']['grippers']],
            official_gripper_mean=[x['mean'] for x in entry['official']['grippers']])), flush=True)
    output=ROOT/'outputs/action_distribution_comparison_with_arm_activity_20261015.json'
    with output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print('REPORT',output,flush=True)


if __name__=='__main__':
    main()
