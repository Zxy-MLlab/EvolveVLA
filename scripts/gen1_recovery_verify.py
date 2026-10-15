#!/usr/bin/env python3
"""Independent CPU production-loader verification of an inference checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True)
    a=p.parse_args()
    root=Path('/root/autodl-tmp/EvolveVLA')
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1')
    sys.path.insert(0,str(root/'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi/src'))
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.models.model import restore_params
    params=restore_params(a.checkpoint/'params',restore_type=np.ndarray)
    flat={'/'.join(k):v for k,v in ocp.tree.to_flat_dict(params).items()}
    expected=json.loads((a.training/'frozen_initial_sha256.json').read_text())
    for key,value in flat.items():
        assert str(value.dtype)=='bfloat16',(key,value.dtype)
        for start in range(0,value.size,1_000_000):
            assert np.isfinite(value.reshape(-1)[start:start+1_000_000]).all(),key
        if key in expected:assert hashlib.sha256(value.tobytes()).hexdigest()==expected[key],key
    assert set(expected)<=set(flat)
    source=root/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json'
    norm=a.checkpoint/'assets/aloha_agilex_sim/norm_stats.json'
    assert source.read_bytes()==norm.read_bytes()
    report=dict(status='passed',leaves=len(flat),frozen_leaves=len(expected),
        frozen_backbone_exact=True,normalization_exact=True,all_finite=True,dtype='bfloat16',
        loader='openpi.models.model.restore_params',checkpoint=str(a.checkpoint))
    with (a.checkpoint/'verification.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
