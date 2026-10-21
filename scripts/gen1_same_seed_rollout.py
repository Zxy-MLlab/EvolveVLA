#!/usr/bin/env python3
"""Roll out the reference and 24k LoRA policy on recorded training seeds.

The simulator receives the original successful episode index and seed.  This
keeps reset selection aligned with data collection while reusing the audited
Clean expert gate and policy adapter.
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT / 'scripts'))
import gen1_recovery_eval as evaluation

CASES = {
    'place_mouse_pad_59000': dict(task='place_mouse_pad', seed=59000, episode_index=2),
    'hanging_mug_38002': dict(task='hanging_mug', seed=38002, episode_index=0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'checkpoints/gen1_expanded_lora_step24000_export_20261015')
    parser.add_argument('--case', choices=sorted(CASES), action='append')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    output.mkdir(parents=False)
    selected = [CASES[k] | {'id': k} for k in (args.case or sorted(CASES))]
    reference = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
    candidate = args.checkpoint.resolve()
    records = []
    env = evaluation.binding.runtime_environment()
    os_env = dict(env)
    import os
    os.environ.update(os_env)
    sys.path[:0] = [str(evaluation.binding.OPENPI / 'src'),
                    str(evaluation.binding.OPENPI / 'packages/openpi-client/src')]
    for case in selected:
        for label, checkpoint in (('reference', reference), ('candidate', candidate)):
            directory = output / label / case['id']
            directory.mkdir(parents=True)
            # The recorded prompt pool is metadata evidence; the reset state is
            # independent of which compatible seen sentence is selected.
            evaluation.worker(Namespace(task=case['task'], seed=case['seed'],
                                        episode_index=case['episode_index'],
                                        checkpoint=str(checkpoint), output=directory),
                              policy_cache=None)
            result = json.loads((directory / 'result.json').read_text())
            records.append(dict(case=case['id'], label=label, **result))
            print(label, case['id'], result['status'], result.get('success'),
                  result.get('steps'), flush=True)
    by_case = {}
    for row in records:
        by_case.setdefault(row['case'], {})[row['label']] = row
    pairs = {}
    for name, row in by_case.items():
        ref, cand = row.get('reference'), row.get('candidate')
        pairs[name] = dict(
            reference_status=ref.get('status'), candidate_status=cand.get('status'),
            reference_success=ref.get('success'), candidate_success=cand.get('success'),
            initial_equal=(ref.get('hashes') == cand.get('hashes'))
            if ref and cand and ref.get('status') == cand.get('status') == 'completed' else False,
            reference_initial=ref.get('hashes') if ref else None,
            candidate_initial=cand.get('hashes') if cand else None,
        )
    (output / 'summary.json').write_text(json.dumps(dict(
        protocol='same recorded training task seed and collection episode index; Clean; no retries',
        checkpoint=str(candidate), cases=selected, records=records, pairs=pairs,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
