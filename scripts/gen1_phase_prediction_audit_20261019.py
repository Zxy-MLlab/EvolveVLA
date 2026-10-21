#!/usr/bin/env python3
"""Optional source-observation phase audit; prepare is CPU-only, run uses one GPU.

One explicit checkpoint per invocation permits serialized baseline/candidate
comparison. No rollout occurs: every prediction sees its original expert image,
drive-target state and literal instruction. Source sample offsets are not time.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_execution_horizon_ablation_20261019 as horizon
from gen1_phase_anchor_metadata_20261019 import FIELDS


def chunk_metrics(prediction, expected, anchor):
    import numpy as np
    prediction = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if prediction.shape != (50, 14) or expected.ndim != 2 or expected.shape[1] != 14:
        raise ValueError('Expected 50 x 14 prediction and valid source horizon')
    if not np.isfinite(prediction).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite actions')
    valid = min(50, len(expected))
    if valid == 0:
        raise ValueError('Empty source horizon')
    active = anchor['active_joint_axes']
    inactive = [i for i in range(14) if i not in (*active, 6, 13)]
    grip = anchor['gripper_axis']
    difference = prediction[:valid] - expected[:valid]
    metrics = {}
    for window in (5, 10, 50):
        count = min(window, valid)
        error = np.abs(difference[:count])
        metrics[str(window)] = dict(valid_actions=count,
            active_joint_mae_radians=float(error[:, active].mean()),
            active_joint_max_abs_radians=float(error[:, active].max()),
            inactive_joint_mae_radians=float(error[:, inactive].mean()),
            active_gripper_mae=float(error[:, grip].mean()),
            inactive_gripper_mae=float(error[:, 13 if grip == 6 else 6].mean()))
    crossing = {}
    for threshold, suffix in ((.2, '02'), (.5, '05'), (.8, '08')):
        target_crossings = np.flatnonzero(expected[:valid, grip] > threshold)
        predicted_crossings = np.flatnonzero(prediction[:valid, grip] > threshold)
        actual = int(target_crossings[0]) if len(target_crossings) else None
        predicted = int(predicted_crossings[0]) if len(predicted_crossings) else None
        crossing[suffix] = dict(source_offset=actual, predicted_offset=predicted,
            signed_sample_shift=predicted - actual if actual is not None and predicted is not None else None,
            source_censored=actual is None, prediction_censored=predicted is None,
            predicted_gripper_at_source_crossing=float(prediction[actual, grip]) if actual is not None else None)
    offset = anchor['source_release_start_offset']
    at_release = None
    if 0 <= offset < valid:
        at_release = dict(active_joint_signed_error_radians=difference[offset, active].tolist(),
            active_joint_mae_radians=float(np.abs(difference[offset, active]).mean()),
            gripper_signed_error=float(difference[offset, grip]),
            predicted=prediction[offset].tolist(), source=expected[offset].tolist())
    return dict(windows=metrics, opening_crossings=crossing, at_source_release_start=at_release)


def select_anchors(payload, case_ids=None, include_intermediate=False):
    lookup = {case['id']: case for case in payload['cases']}
    if case_ids and (len(set(case_ids)) != len(case_ids) or not set(case_ids) <= lookup.keys()):
        raise ValueError('Unknown or duplicate case ID')
    cases = [lookup[name] for name in case_ids] if case_ids else payload['cases']
    selected = [(case, anchor) for case in cases for anchor in case['anchors']
                if include_intermediate or anchor['phase'] == 'terminal_placement_release']
    if not selected:
        raise ValueError('No requested anchors')
    return selected


def source_inputs(case, anchor):
    import cv2
    import h5py
    import numpy as np
    frame = anchor['frame']
    with h5py.File(case['source_hdf5']) as stream:
        state = np.concatenate([stream[f'state/{key}'][frame] for key in FIELDS]).astype(np.float32)
        expected = np.concatenate([stream[f'action/{key}'][frame:frame + 50] for key in FIELDS], axis=1).astype(np.float32)
        images = {}
        for target, source in (('cam_high', 'cam_head'), ('cam_left_wrist', 'cam_left_wrist'),
                               ('cam_right_wrist', 'cam_right_wrist')):
            encoded = bytes(stream[f'vision/{source}/colors'][frame])
            # Collection encodes its RGB array directly through OpenCV. Native
            # conversion uses this same decode without a BGR/RGB channel swap.
            images[target] = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if images[target] is None or images[target].shape != (240, 320, 3):
                raise ValueError('Source camera decode failed')
    np.testing.assert_array_equal(state, np.asarray(anchor['source_state'], dtype=np.float32))
    return images, state, expected


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--noise-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--include-intermediate', action='store_true')
    parser.add_argument('--device-memory-fraction', type=float, default=.14)
    parser.add_argument('--min-free-gib', type=float, default=20.)
    args = parser.parse_args()
    if not 0 < args.device_memory_fraction <= .25 or any(s < 0 for s in args.noise_seeds):
        parser.error('Invalid GPU allocation or noise seeds')
    args.checkpoint = args.checkpoint.resolve()
    payload = json.loads(args.anchors.read_text())
    selected = select_anchors(payload, args.case_ids, args.include_intermediate)
    verified = set()
    for case, anchor in selected:
        if case['id'] not in verified:
            if horizon.digest(case['source_hdf5']) != case['source_hdf5_sha256']:
                raise ValueError('Source HDF5 changed')
            verified.add(case['id'])
        source_inputs(case, anchor)
    files = sorted(p for sub in ('params', 'assets') for p in (args.checkpoint / sub).rglob('*') if p.is_file())
    if not files or not (args.checkpoint / 'params/_METADATA').is_file():
        raise ValueError('Orbax checkpoint missing')
    for name in ('model_config.json', 'verification.json'):
        if (args.checkpoint / name).is_file():
            files.append(args.checkpoint / name)
    identity = {str(p.relative_to(args.checkpoint)): horizon.digest(p) for p in files}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(mode=args.mode, gpu_used=args.mode == 'run', checkpoint=str(args.checkpoint),
        checkpoint_files_sha256=identity, anchors=str(args.anchors.resolve()),
        anchors_sha256=horizon.digest(args.anchors), runner_sha256=horizon.digest(__file__),
        anchor_ids=[a['id'] for _, a in selected], cases=sorted(verified), noise_seeds=args.noise_seeds,
        output_horizon=50, euler_steps=10, device_memory_fraction=args.device_memory_fraction,
        interpretation='Source expert-observation audit only; frame timing and native gripper units; no Cartesian position or closed-loop success inference.')
    horizon.write_json(args.output / 'manifest.json', manifest)
    if args.mode == 'prepare':
        print('CPU_PHASE_PREFLIGHT_PASSED', len(selected), flush=True)
        return
    gpu = horizon.gpu_status()
    if gpu['free_gib'] < args.min_free_gib:
        raise RuntimeError(f'Insufficient GPU memory: {gpu}; no inference launched')
    env = horizon.evaluation.binding.runtime_environment()
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.device_memory_fraction),
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    os.environ.update(env)
    sys.path[:0] = [str(horizon.evaluation.binding.OPENPI / 'src'),
                    str(horizon.evaluation.binding.OPENPI / 'packages/openpi-client/src')]
    import jax
    import numpy as np
    from pi05_openpi_adapter import PublicPi05Adapter
    policy = PublicPi05Adapter(str(args.checkpoint), device='cuda:0', prompt_prefix='none',
                               normalization='quantile', policy_seed=0, noise_source='jax')
    with (args.output / 'predictions.jsonl').open('x', buffering=1) as stream:
        for case, anchor in selected:
            images, state, expected = source_inputs(case, anchor)
            for seed in args.noise_seeds:
                policy.policy._rng = jax.random.key(seed)
                policy._jax_rng = jax.random.key(seed)
                started = time.perf_counter()
                prediction = policy.infer_chunk(images=images, state=state, prompt=case['prompt'])
                duration = time.perf_counter() - started
                row = dict(case_id=case['id'], task=case['task'], anchor=anchor,
                    noise_seed=seed, inference_seconds=duration,
                    metrics=chunk_metrics(prediction, expected, anchor),
                    predicted_actions=np.asarray(prediction).tolist(), source_actions=expected.tolist())
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                print(json.dumps(dict(case=case['id'], anchor=anchor['id'], seed=seed,
                    first10=row['metrics']['windows']['10'])), flush=True)
    horizon.write_json(args.output / 'completed.json', dict(status='completed',
        predictions=len(selected) * len(args.noise_seeds), cases=len(verified)))


if __name__ == '__main__':
    main()
