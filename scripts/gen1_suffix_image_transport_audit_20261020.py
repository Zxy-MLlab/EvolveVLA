#!/usr/bin/env python3
"""CPU-only audit of raw suffix RGB versus the actual deployment input path.

Uses the real simulator observation/IPC/adapter input functions. Model weight
loading and inference alone are replaced with an input recorder. No renderer,
simulation, model forward, or training is run. JPEG roundtrip is a separately
labelled counterfactual, never inserted into the training or deployment path.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
import gen1_recovery_smallfit_20261020 as smallfit


class Difference:
    def __init__(self):
        self.elements = self.abs_sum = self.squared_sum = self.changed = self.maximum = self.frames = 0

    def add(self, actual, expected):
        if actual.shape != expected.shape:
            raise ValueError('Image shapes differ')
        d = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        self.elements += d.size
        self.abs_sum += float(d.sum())
        self.squared_sum += float(np.square(d).sum())
        self.changed += int(np.count_nonzero(d))
        self.maximum = max(self.maximum, float(d.max()))
        self.frames += 1

    def report(self):
        return dict(frames=self.frames, channel_values=self.elements,
                    mae=self.abs_sum/self.elements, rmse=(self.squared_sum/self.elements)**.5,
                    max_abs=self.maximum, changed_channel_values=self.changed,
                    changed_channel_value_fraction=self.changed/self.elements,
                    exact=self.changed == 0)


class InputRecorder:
    """Native policy transform construction, with model computation omitted."""
    def __init__(self, model, *, transforms, **kwargs):
        from openpi import transforms as t
        self.forward = t.compose(transforms)
        self.last = None

    def infer(self, observation, *, noise=None):
        self.last = self.forward(observation)
        return dict(actions=np.zeros((50, 14), dtype=np.float32))


def compare_tree(actual, expected, prefix=''):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f'Input tree keys differ at {prefix}')
        for key in expected:
            compare_tree(actual[key], expected[key], prefix+'/'+key)
        return
    a, b = np.asarray(actual), np.asarray(expected)
    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError(f'Input differs at {prefix}: {a.shape}/{a.dtype} versus {b.shape}/{b.dtype}')


def audit(dataset_path, prepare_path):
    smallfit.environment(False)
    os.environ['ROBOTWIN_OPENPI_SRC'] = str(smallfit.OPENPI/'src')
    import cv2
    import torch
    import robotwin_eval_harness as harness
    import robotwin_public_openpi_harness as public
    import pi05_openpi_adapter as adapter
    from openpi.models import model, pi0_config
    from openpi.policies import policy_config
    import openpi.transforms as transforms
    import jax

    if Path(model.__file__).resolve() != (smallfit.OPENPI/'src/openpi/models/model.py').resolve():
        raise ValueError('Audit imported a different OpenPI revision')

    if (harness.OBS_HEIGHT, harness.OBS_WIDTH) != (240, 320):
        raise ValueError('Audit must use the controlled deployment default camera dimensions')
    ramp = np.arange(240*320*3, dtype=np.uint32).astype(np.uint8).reshape(240, 320, 3)
    probe = dict(joint_action=dict(vector=np.zeros(14, np.float32)),
                 observation={native:dict(rgb=ramp) for native in smallfit.suffix.CAMERAS.values()})
    with contextlib.redirect_stdout(io.StringIO()):
        ramp_observation = harness.simulator_observation(probe)
    stream = io.BytesIO()
    harness.send_message(stream, ramp_observation)
    stream.seek(0)
    ramp_images, _ = public._observation_for_openpi(harness.receive_message(stream))
    for image in ramp_images.values():
        np.testing.assert_array_equal(image, ramp)
    dataset = smallfit.SuffixDataset(dataset_path)
    prepared = smallfit.suffix.read(prepare_path)
    if prepared['dataset_sha256'] != smallfit.suffix.digest(dataset_path):
        raise ValueError('Prepare does not refer to this suffix dataset')
    for path, sha in prepared['code_sha256'].items():
        if smallfit.suffix.digest(path) != sha:
            raise ValueError(f'Prepared source changed: {path}')
    args = argparse.Namespace(checkpoint_dir=smallfit.ROOT/'checkpoints/cpu_transport_audit', batch_size=8)
    cfg = smallfit.make_config(args, smallfit.StrictFullLoraLoader(smallfit.INITIAL))
    training_forward, _ = smallfit.transformations(cfg)
    # The actual constructor and create_trained_policy assemble deployment
    # transforms and load its normalization assets. Only model operations are
    # patched; no full parameters are restored and no forward is evaluated.
    with patch.object(model, 'restore_params', return_value={}), \
         patch.object(pi0_config.Pi0Config, 'load', return_value=object()), \
         patch.object(policy_config._policy, 'Policy', InputRecorder):
        policy = adapter.PublicPi05Adapter(smallfit.INITIAL, device='cpu', noise_source='internal')

    deployment = {camera: Difference() for camera in smallfit.suffix.CAMERAS}
    jpeg = {camera: Difference() for camera in smallfit.suffix.CAMERAS}
    transformed_windows = []
    for episode, (row, arrays) in enumerate(zip(dataset.rows, dataset.arrays, strict=True)):
        for start in range(len(arrays['action'])):
            raw = dict(joint_action=dict(vector=arrays['observation_state'][start]),
                       observation={native: dict(rgb=arrays[camera][start])
                                    for camera, native in smallfit.suffix.CAMERAS.items()})
            with contextlib.redirect_stdout(io.StringIO()):
                observation = harness.simulator_observation(raw)
            stream = io.BytesIO()
            harness.send_message(stream, observation)
            stream.seek(0)
            transported = harness.receive_message(stream)
            images, state = public._observation_for_openpi(transported)
            np.testing.assert_array_equal(state, arrays['observation_state'][start])
            for camera in images:
                native = arrays[camera][start]
                deployment[camera].add(images[camera], native)
                ok, encoded = cv2.imencode('.jpg', native)
                if not ok:
                    raise ValueError('Counterfactual JPEG encoding failed')
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                jpeg[camera].add(decoded, native)
            if start < row['valid_start_count']:
                policy.infer_chunk(images=images, state=state, prompt=row['prompt'])
                deployed = policy.policy.last
                trained = training_forward(dataset.raw(episode, start))
                trained.pop('actions')
                compare_tree(trained, deployed)
                # Also cover Observation.from_dict's uint8 -> float32 [-1,1]
                # conversion, before training-only stochastic augmentation.
                a = model.Observation.from_dict(jax.tree.map(lambda x:np.asarray(x)[None], trained))
                b = model.Observation.from_dict(jax.tree.map(lambda x:np.asarray(x)[None], deployed))
                compare_tree(a.images, b.images)
                transformed_windows.append(dict(id=row['id'], start=start,
                    deterministic_input_tree_exact=True, normalized_images_exact=True))

    paths = [Path(__file__), Path(smallfit.__file__), Path(smallfit.suffix.__file__),
             Path(harness.__file__), Path(public.__file__), Path(adapter.__file__),
             Path(model.__file__), Path(policy_config.__file__), Path(transforms.__file__),
             smallfit.SCRIPT.with_name('robotwin_obs_bridge.py'),
             smallfit.SCRIPT.with_name('gen1_raw_paired_eval.py'),
             smallfit.SCRIPT.with_name('gen1_execution_horizon_ablation_20261019.py'),
             smallfit.OPENPI/'src/openpi/models/pi0.py', smallfit.OPENPI/'scripts/train.py',
             smallfit.OPENPI/'src/openpi/policies/aloha_policy.py',
             smallfit.OPENPI/'src/openpi/training/config.py']
    return dict(status='passed', dataset=str(Path(dataset_path).resolve()),
        dataset_sha256=smallfit.suffix.digest(dataset_path),
        prepare=str(Path(prepare_path).resolve()), prepare_sha256=smallfit.suffix.digest(prepare_path),
        scope='Same captured observations through CPU deployment transport and deterministic policy transforms',
        versions=dict(opencv=cv2.__version__, torch=torch.__version__),
        transport=dict(jpeg_used=False, method='native uint8 HWC -> spatial bridge -> torch CHW float32/255 -> length-prefixed pickle -> HWC *255 uint8',
            default_resize=[240, 320], native_to_deployed_by_camera={k:v.report() for k,v in deployment.items()},
            all_256_uint8_values_roundtrip_exact=True,
            all_native_pixels_exact=all(v.changed == 0 for v in deployment.values())),
        counterfactual_jpeg=dict(applied_to_training=False, applied_to_deployment=False,
            method='OpenCV default .jpg encode then IMREAD_COLOR decode, matching collection channel convention',
            native_to_decoded_by_camera={k:v.report() for k,v in jpeg.items()}),
        deterministic_model_input_comparisons=transformed_windows,
        episode_inventory=[dict(id=r['id'], source_group=r['parent_episode_uid'], frames=len(a['action']),
            valid_training_starts=r['valid_start_count']) for r,a in zip(dataset.rows,dataset.arrays,strict=True)],
        all_valid_start_inputs_exact=True,
        augmentation=dict(changed=False, enabled_in_native_train=True, enabled_in_inference=False,
            head='95 percent random crop, resize, rotation[-5,+5]deg, color jitter',
            wrists='color jitter', exact_full_training_forward_inputs_claimed=False),
        gpu_used=False, model_weights_loaded=False, model_forward_run=False,
        transport_transform_required=False, raw_npz_modified=False,
        source_sha256={str(p.resolve()):smallfit.suffix.digest(p) for p in paths},
        limitations=['Equality is for the same captured native observations, not separately replayed physical states.',
                    'Every retained observation frame is checked for transport; full deterministic input trees are checked on every valid 50-action start.',
                    'Training-only stochastic augmentation remains enabled; exact full model forward inputs are not claimed.',
                    'No checkpoint restore, policy inference, GPU work or training was performed.'])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    result = audit(args.dataset, args.prepared)
    args.output.mkdir(parents=True, exist_ok=False)
    smallfit.write(args.output/'audit.json', result)
    print(json.dumps({k:result[k] for k in ('status','transport','counterfactual_jpeg',
        'all_valid_start_inputs_exact','transport_transform_required','gpu_used')}, indent=2))


if __name__ == '__main__':
    main()
