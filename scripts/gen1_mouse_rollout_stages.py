"""CPU-only descriptive stages for completed mouse rollouts; never selects models."""
import argparse
import hashlib
import json
from pathlib import Path


def stages(rows, result):
    if result['status'] != 'completed' or result['task'] != 'place_mouse_pad':
        raise ValueError('Requires completed mouse rollout')
    if [r['step'] for r in rows] != list(range(result['steps'] + 1)):
        raise ValueError('Incomplete or noncontiguous physical trace')
    arm = result['binding']['fields']['{a}']
    if arm not in ('left', 'right'):
        raise ValueError('Unknown active arm')
    def command(row):
        return row['physical'][arm]['drive_targets'][-1]
    def position(row):
        return row['physical']['mouse']['position']
    initial_z = position(rows[0])[2]
    closures = [i for i in range(1, len(rows))
                if command(rows[i-1]) >= 0.2 and command(rows[i]) < 0.2]
    openings = [i for i in range(1, len(rows))
                if command(rows[i-1]) <= 0.8 and command(rows[i]) > 0.8]
    events = []
    for i in closures:
        end = min(i + 50, len(rows)-1)
        z = position(rows[i])[2]
        events.append(dict(step=i, observed_followup_steps=end-i,
            height_gain_at_50_or_terminal=position(rows[end])[2]-z,
            max_height_gain_next_50=max(position(r)[2]-z for r in rows[i:end+1])))
    releases = []
    for i in openings:
        p = position(rows[i])
        target = rows[i]['physical']['target']['position']
        releases.append(dict(step=i, preceded_by_closure=any(c < i for c in closures),
            height_above_initial=p[2]-initial_z, xy_error=[p[k]-target[k] for k in (0, 1)]))
    terminal = rows[-1]['physical']
    p, q = terminal['mouse']['position'], terminal['mouse']['quaternion']
    t = terminal['target']['position']
    xy = [p[k]-t[k] for k in (0, 1)]
    predicates = dict(position=abs(xy[0]) < 0.015 and abs(xy[1]) < 0.012,
        orientation=abs(abs(q[2]*q[3])-0.49) < 0.015 or abs(abs(q[0]*q[1])-0.49) < 0.015,
        left_open=terminal['left']['drive_targets'][-1] > 0.8,
        right_open=terminal['right']['drive_targets'][-1] > 0.8)
    return dict(seed=result['seed'], checkpoint=result['checkpoint'], success=result['success'],
        active_arm=arm, closures=events, openings=releases,
        max_height_above_initial=max(position(r)[2] for r in rows)-initial_z,
        terminal_xy_error=xy, terminal_predicates=predicates,
        reconstructed_terminal_success=all(predicates.values()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, sources = [], {}
    for path in sorted(args.panel.glob('place_mouse_pad_*/result.json')):
        result = json.loads(path.read_text())
        if result['status'] != 'completed':
            continue
        trace = path.with_name('physical.jsonl')
        raw = trace.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        records.append(stages(rows, result))
        sources[str(trace)] = hashlib.sha256(raw).hexdigest()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(records=records, source_sha256=sources,
        limitations=['Command thresholds do not measure contact or prove a grasp.',
            'Height and release coordinates describe behavior, not causal failure.',
            'Partial panel snapshot; not paired success-rate evidence.',
            'Terminal predicate reconstruction is diagnostic only; recorded success remains authoritative.'])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
