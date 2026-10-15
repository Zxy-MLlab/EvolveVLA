#!/usr/bin/env python3
"""Export selected LoRA adapters with their frozen base for LoRA-aware inference.

Never load this artifact with the plain Gemma variants: model_config.json is
mandatory. Adapters stay explicit rather than silently disappearing on load.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--step', type=int, help='Otherwise use held-out selection.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    training, target = a.training.resolve(), a.output.resolve()
    if target.exists() or not target.is_relative_to(ROOT/'checkpoints'):
        raise ValueError('Fresh project checkpoint target required')
    config = json.loads((training/'run_config.json').read_text())
    if a.step is None:
        selected = json.loads((training/'selection.json').read_text())['selected']
        checkpoint = Path(selected['checkpoint'])
    else:
        checkpoint = ROOT/'checkpoints'/training.name/str(a.step)
    provenance = json.loads((checkpoint/'provenance.json').read_text())
    if not provenance['frozen_parameters_unchanged'] or not provenance['adapter_restore_exact']:
        raise ValueError('Adapter verification missing')
    if shutil.disk_usage(ROOT).free < 17*2**30:
        raise ValueError('Need 7 GiB export space and 10 GiB reserve')
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', HF_HUB_OFFLINE='1',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.path.insert(0,str(ROOT/'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi/src'))
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    import jax
    import numpy as np
    import ml_dtypes
    import orbax.checkpoint as ocp
    from openpi.models.model import restore_params
    from openpi.models.pi0_config import Pi0Config
    source = Path(config['source'])
    norm = source/'assets/aloha_agilex_sim/norm_stats.json'
    if hashlib.sha256(norm.read_bytes()).hexdigest() != config['norm_sha256']:
        raise ValueError('Source normalization changed')
    frozen = json.loads((training/'frozen_initial_sha256.json').read_text())
    params = flatten_dict(restore_params(source/'params',restore_type=np.ndarray),sep='/')
    if params.keys() != frozen.keys():
        raise ValueError('Frozen base structure changed')
    for key in params:
        value = params[key].astype(ml_dtypes.bfloat16)
        if hashlib.sha256(value.tobytes()).hexdigest() != frozen[key]:
            raise ValueError(f'Frozen base mismatch: {key}')
        params[key] = value
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as loader:
        adapter = loader.restore(checkpoint/'adapter_state')['adapters']
    adapters = flatten_dict(adapter,sep='/')
    if not adapters or any('lora' not in key or key in params for key in adapters):
        raise ValueError('Invalid adapter keys')
    params.update({k:np.asarray(v).astype(ml_dtypes.bfloat16) for k,v in adapters.items()})
    model_config = Pi0Config(**config['model'])
    abstract = flatten_dict(nnx.state(nnx.eval_shape(model_config.create,jax.random.key(0))).to_pure_dict(),sep='/')
    if abstract.keys() != params.keys():
        raise ValueError('Export does not match LoRA inference model')
    for key,value in params.items():
        if value.shape != abstract[key].shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid parameter: {key}')
    target.mkdir(parents=True)
    shutil.copytree(source/'assets',target/'assets')
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler(save_concurrent_gb=2)) as saver:
        saver.save(target/'params',{'params':unflatten_dict(params,sep='/')})
    restored = flatten_dict(restore_params(target/'params',restore_type=np.ndarray),sep='/')
    if params.keys() != restored.keys():
        raise ValueError('Production loader dropped parameters')
    for key in params:
        np.testing.assert_array_equal(params[key],restored[key])
    (target/'model_config.json').write_text(json.dumps(config['model'],indent=2))
    (target/'verification.json').write_text(json.dumps(dict(status='passed',
        training=str(training), adapter_checkpoint=str(checkpoint),
        kind='full_lora_model_unmerged', requires_lora_model_config=True,
        frozen_backbone_exact=True, normalization_exact=True,
        all_finite=True, production_restore_exact=True,
        adapter_leaves=len(adapters), dtype='bfloat16'),indent=2))
    print('LORA_EXPORT_VERIFIED',target,flush=True)


if __name__ == '__main__':
    main()
