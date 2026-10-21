#!/usr/bin/env python3
"""Roll out on exact task/seed/index/prompt combinations from local training data."""
import argparse
import json
import os
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT / 'scripts'))
import gen1_recovery_eval as evaluation

ORIGINAL_TEMPLATE_SPEC = evaluation.binding.template_spec
ORIGINAL_BIND_PROMPT = evaluation.binding.bind_prompt


def literal_template_spec(task, seed, description_root):
    prompt = os.environ.get('GEN1_LITERAL_PROMPT')
    if not prompt:
        return ORIGINAL_TEMPLATE_SPEC(task, seed, description_root)
    return dict(task=task, seed=seed, pool='training_seen_literal', index=-1,
                template=prompt, literal_prompt=True,
                source='local_training_instruction_json')


def literal_bind_prompt(spec, episode_info, description_root):
    if not spec.get('literal_prompt'):
        return ORIGINAL_BIND_PROMPT(spec, episode_info, description_root)
    # Reuse the frozen binding logic solely to obtain the task fields and
    # object identity; replace only the sentence with the recorded training
    # instruction. No scene or object selection is changed.
    original = ORIGINAL_TEMPLATE_SPEC(spec['task'], spec['seed'], description_root)
    # ORIGINAL_BIND_PROMPT validates the global template_spec function. Keep
    # that check on the frozen official selector while it computes fields.
    evaluation.binding.template_spec = ORIGINAL_TEMPLATE_SPEC
    try:
        bound = ORIGINAL_BIND_PROMPT(original, episode_info, description_root)
    finally:
        evaluation.binding.template_spec = literal_template_spec
    return dict(prompt=spec['template'], fields=bound['fields'],
                objects=bound['objects'], template=spec)


evaluation.binding.template_spec = literal_template_spec
evaluation.binding.bind_prompt = literal_bind_prompt
evaluation.SCRIPT = Path(__file__).resolve()

CASES = {
    'place_mouse_pad_59000': dict(task='place_mouse_pad', seed=59000,
                                  episode_index=2,
                                  episode='episode_0000002'),
    'hanging_mug_38002': dict(task='hanging_mug', seed=38002,
                              episode_index=0,
                              episode='episode_0000000'),
}


def prompt_for(case):
    path = (ROOT / 'data/gen1_task_batches_20261010' / case['task'] /
            'episodes' / case['episode'] / 'instruction' /
            f"{case['episode']}.json")
    return json.loads(path.read_text())['seen'][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'checkpoints/gen1_expanded_lora_step24000_export_20261015')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    output.mkdir(parents=False)
    reference = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
    candidate = args.checkpoint.resolve()
    selected = []
    for name, base in CASES.items():
        case = dict(base, id=name, prompt=prompt_for(base))
        selected.append(case)
        for label, checkpoint in (('reference', reference), ('candidate', candidate)):
            directory = output / label / name
            directory.mkdir(parents=True)
            os.environ['GEN1_LITERAL_PROMPT'] = case['prompt']
            env = evaluation.binding.runtime_environment()
            os.environ.update(env)
            sys.path[:0] = [str(evaluation.binding.OPENPI / 'src'),
                            str(evaluation.binding.OPENPI / 'packages/openpi-client/src')]
            evaluation.worker(Namespace(task=case['task'], seed=case['seed'],
                                        episode_index=case['episode_index'],
                                        checkpoint=str(checkpoint), output=directory),
                              policy_cache=None)
            result = json.loads((directory / 'result.json').read_text())
            print(label, name, result['status'], result.get('success'),
                  result.get('steps'), flush=True)
    pairs = {}
    for case in selected:
        ref = json.loads((output / 'reference' / case['id'] / 'result.json').read_text())
        cand = json.loads((output / 'candidate' / case['id'] / 'result.json').read_text())
        pairs[case['id']] = dict(
            task=case['task'], seed=case['seed'], episode_index=case['episode_index'],
            training_prompt=case['prompt'], initial_equal=ref.get('hashes') == cand.get('hashes'),
            reference_success=ref.get('success'), candidate_success=cand.get('success'),
            reference_status=ref.get('status'), candidate_status=cand.get('status'),
            reference_hashes=ref.get('hashes'), candidate_hashes=cand.get('hashes'))
    (output / 'summary.json').write_text(json.dumps(dict(
        protocol='exact local training sample: task + seed + collection episode index + first seen instruction; Clean; no retries',
        checkpoint=str(candidate), pairs=pairs), indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        evaluation.simulator()
    else:
        main()
