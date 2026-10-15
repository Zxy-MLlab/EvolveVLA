#!/usr/bin/env python3
"""Measure sampled physical and padded actions on fixed held-out anchors.

Read-only inference with production bfloat16 weights and the unchanged sampler.
This diagnostic does not alter the benchmark, train, or select a checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT/'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true',
        help='Validate all held-out inputs on CPU without loading model weights')
    args = p.parse_args()
    os.environ.update(HF_LEROBOT_HOME=str(ROOT/'data'), HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
        OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
        JAX_PLATFORMS='cpu' if args.prepare_only else 'cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        CUDA_VISIBLE_DEVICES='' if args.prepare_only else '0')
    sys.path[:0] = [str(ROOT/'scripts'), str(OPENPI/'src'), str(OPENPI/'packages/openpi-client/src')]
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_lib
    import gen1_diagnose_pilot as diagnostic
    import gen1_train_pilot as pilot
    import gen1_video_cache
    import gen1_recovery_eval as recovery
    import gen1_recovery_compare as comparison
    args.output.mkdir(exist_ok=False)
    if not args.prepare_only:
        recovery.release_checkpoint_cache()
    cfg = pilot.build_config(diagnostic.defaults())
    cache_report = gen1_video_cache.install([ROOT/'data'/diagnostic.defaults().val_repo_id])
    bundle = diagnostic.dataset_bundle(cfg, diagnostic.defaults().val_repo_id)
    _, inverse, _, _ = diagnostic.transforms_for(bundle[0])
    manifest_path = ROOT/'outputs/gen1_diagnostic_prepare_cpu_v3_20261011/manifest.json'
    anchors = [r for r in json.loads(manifest_path.read_text())['anchors'] if r['split'] == 'val']
    assert len(anchors) == 80
    validation = []
    for anchor in anchors:
        _, transformed, expected, _, _ = diagnostic.checked_anchor(bundle, anchor)
        assert np.array_equal(transformed['actions'][..., 14:], np.zeros((50, 18)))
        validation.append((anchor, transformed, expected))
    if args.prepare_only:
        comparison.write(args.output/'prepared.json', dict(status='passed', anchors=len(validation),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            anchor_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            training_inference_inputs_equal=True, physical_action_roundtrip=True,
            padded_targets_zero=True, cache_report=cache_report,
            no_model_loaded=True, no_gpu_used=True))
        print('PREPARED', len(validation), 'held-out anchors; padded targets exactly zero', flush=True)
        return
    weights = model_lib.restore_params(args.checkpoint/'params', dtype=jnp.bfloat16)
    loaded = cfg.model.load(weights)
    loaded.eval()
    graph, params = nnx.split(loaded)
    del weights, loaded
    def sample(state, obs, noise):
        model = nnx.merge(graph, state)
        return model.sample_actions(jax.random.key(0), obs, noise=noise, num_steps=10)
    sample_fn = jax.jit(sample)
    noise = jax.random.normal(jax.random.key(1729), (8, 50, 32))
    # Hold every physical noise coordinate fixed and vary only padding. This
    # measures coupling through the native sampler, without changing inference.
    padding_noise_seeds = [1730, 1731]
    alternative_noise = [noise.at[..., 14:].set(
        jax.random.normal(jax.random.key(seed), (8, 50, 18)))
        for seed in padding_noise_seeds]
    for alternative in alternative_noise:
        if not np.array_equal(np.asarray(noise[..., :14]), np.asarray(alternative[..., :14])):
            raise ValueError('Physical noise changed in padding sensitivity diagnostic')
        if np.array_equal(np.asarray(noise[..., 14:]), np.asarray(alternative[..., 14:])):
            raise ValueError('Padding noise did not change')
    rows = []
    input_hash = hashlib.sha256()
    for start in range(0, len(validation), 8):
        selected = validation[start:start+8]
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *[r[1] for r in selected])
        obs = model_lib.Observation.from_dict(batch)
        for leaf in jax.tree.leaves(obs):
            input_hash.update(np.asarray(leaf).tobytes())
        predicted = np.asarray(sample_fn(params, obs, noise))
        if not np.isfinite(predicted).all():
            raise ValueError('Nonfinite sampled actions')
        alternatives = [np.asarray(sample_fn(params, obs, alternative))
                        for alternative in alternative_noise]
        if not all(np.isfinite(value).all() for value in alternatives):
            raise ValueError('Nonfinite padding sensitivity actions')
        for index, ((anchor, transformed, expected), prediction) in enumerate(zip(selected, predicted)):
            actual = inverse(dict(state=np.array(transformed['state']), actions=prediction.copy()))['actions']
            error = np.abs(actual - expected)
            valid = np.asarray(anchor['valid'])
            joints = [i for i in range(14) if i not in (6, 13)]
            padding = np.abs(prediction[:, 14:])
            changes = np.stack([np.abs(inverse(dict(state=np.array(transformed['state']),
                actions=value[index].copy()))['actions'] - actual) for value in alternatives])
            rows.append(dict(anchor_id=anchor['anchor_id'], task=anchor['task'],
                joint_mae=float(error[valid][:, joints].mean()),
                gripper_mae=float(error[valid][:, [6, 13]].mean()),
                first10_joint_mae=float(error[:10, joints].mean()),
                first10_gripper_mae=float(error[:10, [6, 13]].mean()),
                normalized_padded_mae=float(padding.mean()),
                normalized_padded_max=float(padding.max()),
                normalized_padded_rms=float(np.sqrt(np.square(padding).mean())),
                padding_noise_joint_mae=float(changes[:, valid][:, :, joints].mean()),
                padding_noise_gripper_mae=float(changes[:, valid][:, :, [6, 13]].mean()),
                padding_noise_first10_joint_mae=float(changes[:, :10, joints].mean()),
                padding_noise_first10_gripper_mae=float(changes[:, :10, [6, 13]].mean())))
        print('ANCHORS', start+len(selected), flush=True)
    keys = list(comparison.KEYS) + ['normalized_padded_mae', 'normalized_padded_rms',
        'padding_noise_joint_mae', 'padding_noise_gripper_mae',
        'padding_noise_first10_joint_mae', 'padding_noise_first10_gripper_mae']
    metrics = {}
    for task in ('hanging_mug', 'place_mouse_pad', 'all'):
        subset = [r for r in rows if task == 'all' or r['task'] == task]
        metrics[task] = {key: float(np.mean([r[key] for r in subset])) for key in keys}
        metrics[task]['normalized_padded_max'] = max(r['normalized_padded_max'] for r in subset)
    comparison.write(args.output/'report.json', dict(checkpoint=str(args.checkpoint),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        anchor_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        observation_sha256=input_hash.hexdigest(), noise_seed=1729, num_steps=10,
        padding_noise_seeds=padding_noise_seeds, physical_noise_unchanged=True,
        dtype='bfloat16', batch_size=8, unchanged_sampler=True,
        limitation='Padded error is diagnostic evidence, not proof of a causal rollout failure.',
        metrics=metrics, records=rows))
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
