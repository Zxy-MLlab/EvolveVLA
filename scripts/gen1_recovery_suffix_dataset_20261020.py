#!/usr/bin/env python3
"""CPU-only preflight/build for independently replay-verified recovery suffixes.

This creates an independent lossless NPZ dataset, not a legacy LeRobot manifest.
Only expert suffix observations are supervised: obs[t] -> drive_target[t+1].
Full 50-action windows stay inside one source suffix. Short or failed suffixes
remain exclusions; no historical training gate or source data is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
NORM = ROOT/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json'
CAMERAS = {'cam_high': 'head_camera', 'cam_left_wrist': 'left_camera', 'cam_right_wrist': 'right_camera'}
MODES = ('original', 'fixed15', 'opening15')
HORIZON, CAPTURE_TICKS = 50, 15


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_path(value, expected, base):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (base/path).resolve()
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f'Artifact missing or hash mismatch: {path}')
    return path


def parent_identity(case):
    identity = {key: case[key] for key in ('source_revision', 'task', 'seed', 'episode_index', 'source_hdf5_sha256')}
    if case['task'] not in ('place_mouse_pad', 'hanging_mug'):
        raise ValueError('Unsupported recovery target task')
    if not all(type(identity[k]) is int for k in ('seed', 'episode_index')):
        raise ValueError('Source episode seed/index must be integers')
    uid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return uid, identity


def source_split(case, group, split_groups):
    if split_groups is None:
        return 'diagnostic_train'
    value = split_groups.get(group)
    if value not in ('train', 'val', 'test', 'diagnostic_train'):
        raise ValueError('Every source episode group needs one explicit split')
    original = case.get('source_split')
    if original in ('train', 'val', 'test') and original != value:
        raise ValueError('Recovery split conflicts with recorded source split')
    return value


def load_uniform_captures(path, timestep):
    """Read trusted local simulator pickles only after each frozen hash passes."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No captured observations')
    if [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Capture indices are not contiguous')
    retained, dropped = [], []
    for i, row in enumerate(rows):
        tick = row['physics_tick']
        if type(tick) is not int or tick < 0:
            raise ValueError('Invalid capture physics tick')
        expected = len(retained)*CAPTURE_TICKS
        if tick != expected:
            # A single last short interval may be explicitly retained by the
            # replay recorder for diagnostics; never turn it into a 15-tick label.
            if (i == len(rows)-1 and retained and retained[-1]['physics_tick'] < tick < expected
                    and row.get('partial_final_interval') is True):
                dropped.append(dict(index=i, physics_tick=tick, reason='partial_final_interval'))
                continue
            raise ValueError('Captures are not at exact 0,15,30,... physics ticks')
        if 'simulation_seconds' in row and not np.isclose(row['simulation_seconds'], tick*timestep, atol=1e-8, rtol=1e-7):
            raise ValueError('Capture time does not match physics ticks')
        retained.append(row)
    states, images, hashes = [], {k: [] for k in CAMERAS}, {}
    for row in retained:
        raw = checked_path(row['path'], row['sha256'], path.parent)
        hashes[str(raw)] = row['sha256']
        with raw.open('rb') as stream:
            observation = pickle.load(stream)
        state = np.asarray(observation['joint_action']['vector'])
        if state.shape != (14,) or not np.issubdtype(state.dtype, np.number) or not np.isfinite(state).all():
            raise ValueError('Suffix drive-target state must be finite 14D')
        if np.any(state[[6, 13]] < -1e-6) or np.any(state[[6, 13]] > 1+1e-6):
            raise ValueError('Gripper command outside normalized range')
        states.append(state)
        for destination, source in CAMERAS.items():
            image = np.asarray(observation['observation'][source]['rgb'])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError('RGB must be native uint8 HWC with three channels')
            if images[destination] and image.shape != images[destination][0].shape:
                raise ValueError('Camera shape changes inside suffix')
            images[destination].append(image)
    states = np.stack(states)
    canonical = states.astype(np.float32)
    if not np.isfinite(canonical).all():
        raise ValueError('Float32 cast produced nonfinite labels')
    ticks = np.array([r['physics_tick'] for r in retained], dtype=np.int64)
    arrays = dict(observation_state=canonical[:-1], action=canonical[1:],
        observation_physics_tick=ticks[:-1], action_physics_tick=ticks[1:],
        observation_time_seconds=ticks[:-1]*timestep,
        action_time_seconds=ticks[1:]*timestep,
        action_is_terminal_hold=np.array([r.get('phase')=='explicit_terminal_hold' for r in retained[1:]],dtype=bool))
    arrays.update({k: np.stack(values[:-1]) for k, values in images.items()} if len(retained)>1 else {})
    return arrays, dict(retained_captures=len(retained), transitions=max(0,len(retained)-1),
        dropped_capture_records=dropped, raw_sha256=hashes,
        max_float32_state_cast_error=float(np.max(np.abs(states-canonical.astype(states.dtype)))),
        image_shapes={k: list(v[0].shape) for k, v in images.items()},
        terminal_hold_transition_count=int(arrays['action_is_terminal_hold'].sum()),
        expert_transition_count=int((~arrays['action_is_terminal_hold']).sum()),
        capture_interval_physics_ticks=CAPTURE_TICKS, timestep=timestep,
        seconds_per_transition=CAPTURE_TICKS*timestep,
        actual_uniform_hz=1/(CAPTURE_TICKS*timestep))


def require_replay_result(case, result, label):
    """Readiness summaries cannot override rejected or unstable raw outcomes."""
    expected = 'dense_replay_completed' if label == 'dense' else 'sparse_replay_completed'
    if (result.get('status') != expected or result.get('simulator_returncode') != 0
            or any(result.get(key) is not True for key in ('success', 'final_success', 'exact_prefix', 'stable_success'))):
        raise ValueError(f'Raw {label} replay did not complete with exact prefix and stable success')
    if any(result.get(key) != case[key] for key in ('id', 'task', 'seed', 'episode_index', 'takeover_boundary_step')):
        raise ValueError(f'Raw {label} replay case identity differs')
    gate = result.get('expert_gate') or {}
    if gate.get('status') != 'passed' or gate.get('bypassed') is not False:
        raise ValueError(f'Raw {label} replay expert gate did not pass')
    if label == 'dense':
        if result.get('dense_source_match') is not True or result.get('source_mismatches') != []:
            raise ValueError('Raw dense replay did not exactly reproduce source captures')
    elif result.get('native_eval_success') is not True or result.get('official_rollout_success') is not True:
        raise ValueError('Raw sparse replay did not pass the official rollout success condition')
    hold = result.get('terminal_hold') or {}
    if (case.get('terminal_hold_ticks') != 250 or hold.get('requested_ticks') != 250
            or hold.get('physics_ticks') != 250
            or any(hold.get(key) is not True for key in ('success', 'initial_success', 'all_ticks_success',
                                                       'final_success', 'positions_preserved'))
            or hold.get('measured_state_reset') is not False
            or type(hold.get('start_tick')) is not int or hold['start_tick'] < 0
            or result.get('physics_ticks') != hold['start_tick'] + 250):
        raise ValueError(f'Raw {label} replay lacks a complete stable250tick hold without measured-state reset')


def validate_case(readiness_path, mode, split_groups=None):
    readiness_path = Path(readiness_path).resolve()
    base, status = readiness_path.parent, read(readiness_path)
    if status.get('schema_version') != 1:
        raise ValueError('Unsupported recovery readiness schema')
    if not all(status.get(k) is True for k in ('exact_prefix', 'dense_success', 'dense_source_match')):
        raise ValueError('Dense/exact-prefix/source-replay prerequisites did not pass')
    selected = status.get('sparse_modes', {}).get(mode)
    if not selected or selected.get('exact_prefix') is not True or selected.get('success') is not True:
        raise ValueError('Selected deployment mode did not pass sparse replay with exact prefix')
    case_path = checked_path(status['case_path'], status['case_sha256'], base)
    dense_path = checked_path(status['dense_result_path'], status['dense_result_sha256'], base)
    captures_path = checked_path(status['dense_captures_path'], status['dense_captures_sha256'], base)
    sparse_path = checked_path(selected['result_path'], selected['sha256'], base)
    case, dense, sparse = read(case_path), read(dense_path), read(sparse_path)
    require_replay_result(case, dense, 'dense')
    require_replay_result(case, sparse, 'sparse')
    if sparse.get('timing_mode', sparse.get('deployment_mode')) != mode:
        raise ValueError('Sparse replay controller mode mismatch')
    if case['id'] != status['id'] or case['task'] != status['task']:
        raise ValueError('Readiness and source episode identity differ')
    if not isinstance(case.get('prompt'), str) or not case['prompt'].strip():
        raise ValueError('Original source instruction is missing')
    original_source = checked_path(case['source_hdf5'], case['source_hdf5_sha256'], base)
    group, identity = parent_identity(case)
    split = source_split(case, group, split_groups)
    timestep = float(status.get('timestep', dense.get('timestep', 0.)))
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError('A measured positive simulator timestep is required')
    arrays, audit = load_uniform_captures(captures_path, timestep)
    if audit['transitions'] < HORIZON:
        # Recovery may still be successful; it simply cannot supply this
        # diagnostic's complete real 50-action training window.
        return None, dict(id=case['id'], task=case['task'], parent_episode_uid=group,
            status='excluded_short_suffix', recovery_verified=True, dataset_ready=False,
            transitions=audit['transitions'], minimum_real_transitions=HORIZON,
            reason='No full real 50-action chunk; no padding/cross-episode joining allowed')
    row = dict(id=case['id'], task=case['task'], parent_episode_uid=group,
        parent_episode_identity=identity, split=split, prompt=case['prompt'],
        takeover_boundary_step=case['takeover_boundary_step'],
        source_checkpoint_identity_sha256=case['source_checkpoint_identity_sha256'],
        deployment_mode=mode, status='passed', recovery_verified=True, dataset_ready=True,
        terminal_hold_provenance=dict(requested_ticks=case.get('terminal_hold_ticks'),
            dense=dense.get('terminal_hold'),sparse=sparse.get('terminal_hold')),
        frames=audit['transitions'], valid_start_count=audit['transitions']-HORIZON+1,
        valid_start_range=[0,audit['transitions']-HORIZON+1], horizon=HORIZON,
        action_alignment='observation at capture t; label is next capture drive target t+1',
        original_source_hdf5=str(original_source), suffix_supervision_only=True,
        privileged_fields_not_in_model_inputs=True, capture_audit=audit,
        evidence_sha256={str(p): digest(p) for p in (readiness_path,case_path,dense_path,captures_path,sparse_path)},
        source_readiness=str(readiness_path))
    return arrays, row


def process(input_root, output, mode, operation='preflight', split_config=None):
    if mode not in MODES or operation not in ('preflight', 'build'):
        raise ValueError('Unsupported dataset operation/mode')
    input_root, output = Path(input_root).resolve(), Path(output).resolve()
    if not input_root.is_dir() or output.exists():
        raise ValueError('Existing input root and fresh output directory required')
    split_groups = None
    if split_config:
        split_groups = read(split_config)['parent_episode_splits']
    norm_sha = digest(NORM)
    paths = sorted(input_root.glob('*/readiness.json'))
    rows, excluded, data = [], [], []
    source_groups = {}
    for path in paths:
        try:
            arrays, row = validate_case(path, mode, split_groups)
            if arrays is None:
                excluded.append(row)
                continue
            group = row['parent_episode_uid']
            if group in source_groups:
                raise ValueError('Duplicate parent source episode; select one verified suffix per source group')
            if any(r['id'] == row['id'] for r in rows):
                raise ValueError('Duplicate recovery case ID')
            source_groups[group] = row['split']
            rows.append(row)
            data.append(arrays)
        except (ValueError, KeyError, OSError, TypeError, EOFError, pickle.UnpicklingError) as error:
            excluded.append(dict(id=path.parent.name, status='excluded_unverified',
                dataset_ready=False, readiness=str(path), reason=f'{type(error).__name__}: {error}'))
    output.mkdir(parents=True)
    if operation == 'build':
        for index, (row, arrays) in enumerate(zip(rows, data, strict=True)):
            target = output/f'episode_{index:06d}.npz'
            np.savez_compressed(target, **arrays)
            with np.load(target, allow_pickle=False) as restored:
                if set(restored.files) != set(arrays):
                    raise ValueError('Dataset archive keys changed')
                for key, value in arrays.items():
                    np.testing.assert_array_equal(restored[key], value)
            row.update(dataset_path=str(target), dataset_sha256=digest(target), archive_roundtrip_exact=True)
        shutil.copyfile(NORM, output/'norm_stats.json')
        if digest(output/'norm_stats.json') != norm_sha:
            raise ValueError('Normalization copy changed')
    manifest = dict(schema_version=1, format='recovery_suffix_npz_v1', operation=operation,
        status='passed' if rows else 'no_eligible_suffixes', dataset_ready=bool(rows) and operation=='build',
        deployment_mode=mode, horizon=HORIZON, padding_policy='sample full real windows only',
        input_root=str(input_root), attempted_readiness_files=len(paths),
        accepted_suffixes=len(rows), independent_source_episodes=len(source_groups),
        accepted_frames=sum(r['frames'] for r in rows),
        valid_training_starts=sum(r['valid_start_count'] for r in rows if r['split'] in ('train','diagnostic_train')),
        episodes=rows, excluded=excluded, source_episode_splits=source_groups,
        normalization=dict(path=str(NORM), sha256=norm_sha, recomputed=False,
            copied_path=str(output/'norm_stats.json') if operation=='build' else None),
        builder_sha256=digest(SCRIPT), split_config_sha256=digest(split_config) if split_config else None,
        gpu_used=False, training_started=False,
        use_scope='smallfit_diagnostic_only' if split_groups is None or len(source_groups)<=3 else 'group_split_data_requires_independent_rollout_validation',
        limitations=['Successful dense recovery alone is insufficient; the selected sparse deployment replay must also pass.',
            'Only RGB, drive-target state and next drive-target action are student inputs; expert geometry is provenance only.',
            'Three or fewer source episodes support a fitting positive control, not a generalization claim.',
            'This independent NPZ format needs a dedicated manifest-driven training loader; it is not the historical LeRobot split.',
            'dataset_ready denotes data gate passage, not a trained or improved policy.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('operation', choices=('preflight','build'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment-mode', choices=MODES, required=True)
    parser.add_argument('--split-config', type=Path)
    args = parser.parse_args()
    result = process(args.input,args.output,args.deployment_mode,args.operation,args.split_config)
    print(json.dumps({k: result[k] for k in ('status','dataset_ready','accepted_suffixes',
        'independent_source_episodes','accepted_frames','valid_training_starts','use_scope')},indent=2))


if __name__ == '__main__':
    main()
