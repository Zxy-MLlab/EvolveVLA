#!/usr/bin/env python3
"""CPU-only source phase anchors for a later fixed-noise offline policy audit.

Gripper-command events are observable labels, not proof of grasp/contact.
Hanging mug has an intermediate LEFT release and a terminal RIGHT release.
Frame offsets are sample indices; the collection intervals are not uniform.
"""
import argparse
import hashlib
import json
from pathlib import Path


FIELDS = ('left_arm_joint_states', 'left_ee_joint_states',
          'right_arm_joint_states', 'right_ee_joint_states')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def crossing_events(values, side, epsilon=1e-5):
    """Match each opening through .2 to its preceding closure through .2."""
    import numpy as np
    values = np.asarray(values)
    changes = np.diff(values)
    closures = np.flatnonzero((values[:-1] > .2) & (values[1:] <= .2)) + 1
    openings = np.flatnonzero((values[:-1] <= .2) & (values[1:] > .2)) + 1
    events = []
    previous_opening = -1
    for opening in openings:
        before = [int(frame) for frame in closures if previous_opening < frame < opening]
        previous_opening = int(opening)
        if not before:
            continue
        closure = before[-1]
        close_start = closure
        while close_start > 1 and changes[close_start - 2] < -epsilon:
            close_start -= 1
        close_end = closure
        while close_end < len(values) - 1 and changes[close_end] < -epsilon:
            close_end += 1
        open_start = int(opening)
        while open_start > 1 and changes[open_start - 2] > epsilon:
            open_start -= 1
        open_end = int(opening)
        while open_end < len(values) - 1 and changes[open_end] > epsilon:
            open_end += 1
        events.append(dict(side=side, closure_start=close_start, closure_crossing_02=closure,
            closure_end=close_end, release_start=open_start, release_crossing_02=int(opening),
            release_end=open_end,
            release_crossing_05=next((i for i in range(open_start, open_end + 1) if values[i] >= .5), None),
            release_crossing_08=next((i for i in range(open_start, open_end + 1) if values[i] >= .8), None)))
    return events


def annotate_case(case, offsets=(20, 10, 5)):
    import h5py
    import numpy as np
    source = Path(case['source_hdf5'])
    source_hash = digest(source)
    if source_hash != case.get('source_hdf5_sha256', case.get('source_sha256')):
        raise ValueError('Source identity changed')
    with h5py.File(source) as stream:
        state, action = [np.concatenate([stream[f'{group}/{key}'][:] for key in FIELDS], axis=1)
                         for group in ('state', 'action')]
    if state.shape != action.shape or action.shape[1] != 14:
        raise ValueError('Unexpected source shape')
    np.testing.assert_array_equal(state[1:], action[:-1])
    events = crossing_events(action[:, 6], 'left') + crossing_events(action[:, 13], 'right')
    events.sort(key=lambda row: row['release_start'])
    if not events:
        raise ValueError(f'No close-then-open cycle for {case["id"]}')
    terminal = events[-1]
    if case['task'] == 'hanging_mug' and terminal['side'] != 'right':
        raise ValueError('Hanging mug terminal release is not right arm')
    if case['task'] == 'place_mouse_pad' and len(events) != 1:
        raise ValueError('Expected one mouse grasp/release cycle')
    annotations = []
    anchors = []
    for index, event in enumerate(events):
        side = event['side']
        axes = list(range(6)) if side == 'left' else list(range(7, 13))
        grip = 6 if side == 'left' else 13
        phase = 'terminal_placement_release' if event is terminal else 'intermediate_placement_release'
        event = dict(event, event_id=f'{side}_release_{index}', phase=phase,
            active_joint_axes=axes, gripper_axis=grip,
            carrying_command_window=[event['closure_end'], event['release_start']],
            terminal_approach_proxy_window=[max(event['closure_end'], event['release_start'] - 50),
                                            event['release_start']],
            approach_proxy_note='Last 50 recorded labels while command remains closed; no geometric/contact inference.')
        annotations.append(event)
        for offset in offsets:
            frame = event['release_start'] - offset
            if frame < 0:
                continue
            available = min(50, len(action) - frame)
            expected = action[frame:frame + available]
            reference = {}
            for horizon in (5, 10, 50):
                count = min(horizon, available)
                reference[str(horizon)] = dict(valid_actions=count,
                    active_joint_displacement_from_anchor_mean=float(
                        np.abs(expected[:count, axes] - state[frame, axes]).mean()),
                    gripper_min=float(expected[:count, grip].min()),
                    gripper_max=float(expected[:count, grip].max()))
            anchors.append(dict(id=f'{case["id"]}:{event["event_id"]}:minus{offset}',
                event_id=event['event_id'], phase=phase, side=side, frame=frame,
                frames_before_release=offset, available_horizon=available,
                active_joint_axes=axes, gripper_axis=grip,
                source_state=state[frame].tolist(),
                source_action=action[frame].tolist(),
                source_release_start_offset=offset,
                source_release_crossing_02_offset=event['release_crossing_02'] - frame,
                source_release_crossing_05_offset=event['release_crossing_05'] - frame
                    if event['release_crossing_05'] is not None else None,
                source_release_crossing_08_offset=event['release_crossing_08'] - frame
                    if event['release_crossing_08'] is not None else None,
                source_window_statistics=reference))
    return dict(case, source_frames=len(action), events=annotations, anchors=anchors,
        terminal_release_side=terminal['side'], terminal_release_start=terminal['release_start'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = [annotate_case(case) for case in manifest['cases']]
    artifact = dict(schema=1, gpu_used=False, source_manifest=str(args.manifest.resolve()),
        source_manifest_sha256=digest(args.manifest), script_sha256=digest(__file__),
        cases=cases, total_anchors=sum(len(case['anchors']) for case in cases),
        terminal_anchors=sum(sum(row['phase'] == 'terminal_placement_release' for row in case['anchors']) for case in cases),
        planned_policy_audit=dict(
            source_inputs='Decode source frame RGB using existing audited camera convention; source 14D drive-target state; exact source first seen prompt.',
            models='Compare initializer and 24k LoRA on identical anchors; keep all source-held observation states.',
            noise='Common explicit JAX noise seeds 0, 1, 2, reset each anchor/model; model horizon50 and Euler10 unchanged.',
            metrics=['First5/10/50 valid-horizon active-arm six-joint MAE in radians',
                     'Same active gripper MAE in native0..1 units; inactive-arm MAE separately',
                     'Signed active joint error at source release start and .2/.5/.8 crossings',
                     'Predicted first opening .2/.5/.8 crossing minus expert offset, with no crossing marked censored',
                     'Per-anchor error and episode/task/active-arm summaries, each episode weighted equally'],
            interpretation='Expert-conditioned action error only. Cannot establish on-policy recovery ability or Cartesian placement error; threshold frame shifts are not physical time shifts.'),
        limitations=['Opening is command-labelled; it does not prove object release or grasp success.',
            'Mug intermediate left placement must not be pooled with terminal right hanging release.',
            'No model inference has been run by this artifact; no model-error conclusion yet.',
            'Source sample intervals vary at dense segment boundaries; no conversion from frames to seconds.'])
    with args.output.open('x') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(cases=len(cases), total_anchors=artifact['total_anchors'],
                          terminal_anchors=artifact['terminal_anchors'], output=str(args.output))))


if __name__ == '__main__':
    main()
