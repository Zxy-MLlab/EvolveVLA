#!/usr/bin/env python3
"""Evaluate the 24k LoRA policy on every local training episode seed.

Each case reuses the recorded task, seed, collection episode index, and first
seen instruction. The runner is resumable and writes one compact result per
case while retaining the detailed diagnostic files produced by the audited
Clean evaluator.
"""
import argparse
import contextlib
import hashlib
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
    original = ORIGINAL_TEMPLATE_SPEC(spec['task'], spec['seed'], description_root)
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(split_path):
    split = json.loads(Path(split_path).read_text())
    rows = [row for row in split['episodes']
            if row['split'] == 'train' and row['task'] in ('place_mouse_pad', 'hanging_mug')]
    rows.sort(key=lambda x: (x['task'], int(x['seed']), x['episode_directory']))
    if len(rows) != 1990:
        raise ValueError(f'Expected 1990 local train episodes, found {len(rows)}')
    cases = []
    for row in rows:
        episode_dir = Path(row['episode_directory'])
        instruction_files = sorted((episode_dir / 'instruction').glob('*.json'))
        if len(instruction_files) != 1:
            raise ValueError(f'Expected one instruction file: {episode_dir}')
        instructions = json.loads(instruction_files[0].read_text())['seen']
        if not instructions or not isinstance(instructions[0], str):
            raise ValueError(f'Invalid training instruction: {episode_dir}')
        name = f"{row['task']}_{int(row['seed'])}"
        episode_index = int(episode_dir.name.split('_')[-1])
        cases.append(dict(id=name, task=row['task'], seed=int(row['seed']),
                          episode_index=episode_index, prompt=instructions[0],
                          source_episode_directory=str(episode_dir),
                          source_hdf5=row['path'], source_hdf5_sha256=row['sha256']))
    if len({x['id'] for x in cases}) != len(cases):
        raise ValueError('Duplicate task/seed in local training set')
    return cases


def compact(result, case):
    return dict(id=case['id'], task=case['task'], seed=case['seed'],
                episode_index=case['episode_index'], status=result.get('status'),
                success=result.get('success'), steps=result.get('steps'),
                hashes=result.get('hashes'), error=result.get('error') or result.get('message'))


@contextlib.contextmanager
def quiet_case(directory):
    """Keep simulator/native diagnostics per case instead of flooding stdout."""
    log = (directory / 'worker_stderr.log').open('w')
    saved = os.dup(2)
    try:
        os.dup2(log.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        log.close()


def summary(cases, results, checkpoint):
    totals = {}
    for task in ('place_mouse_pad', 'hanging_mug'):
        rows = [r for r in results if r['task'] == task]
        completed = [r for r in rows if r['status'] == 'completed']
        totals[task] = dict(requested=sum(c['task'] == task for c in cases),
                            attempted=len(rows), completed=len(completed),
                            successes=sum(bool(r.get('success')) for r in completed),
                            success_rate=(sum(bool(r.get('success')) for r in completed) / len(completed)
                                          if completed else None),
                            statuses={s: sum(r['status'] == s for r in rows)
                                      for s in sorted({r['status'] for r in rows})})
    completed = [r for r in results if r['status'] == 'completed']
    return dict(protocol='all local train episodes; exact task + seed + collection episode index + first seen instruction; Clean; no retries',
                checkpoint=str(checkpoint), total_requested=len(cases), total_attempted=len(results),
                total_completed=len(completed), total_successes=sum(bool(r.get('success')) for r in completed),
                total_success_rate=(sum(bool(r.get('success')) for r in completed) / len(completed)
                                    if completed else None), tasks=totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'checkpoints/gen1_expanded_lora_step24000_export_20261015')
    parser.add_argument('--split-manifest', type=Path,
                        default=ROOT / 'data/gen1_multitask_coverage_expanded_20261014/split_manifest.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    cases = load_cases(args.split_manifest)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    identity = dict(checkpoint=str(args.checkpoint.resolve()), split_manifest=str(args.split_manifest.resolve()),
                    split_manifest_sha256=digest(args.split_manifest), cases=len(cases),
                    runner=str(Path(__file__).resolve()), runner_sha256=digest(__file__))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != identity:
            raise ValueError('Existing output identity differs; choose a new output directory')
    else:
        manifest_path.write_text(json.dumps(identity, indent=2) + '\n')
    results_path = output / 'results.jsonl'
    results = []
    done = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                results.append(row); done.add(row['id'])
    elif results_path.exists():
        raise ValueError('Output exists; pass --resume to continue')
    env = evaluation.binding.runtime_environment()
    os.environ.update(env)
    sys.path[:0] = [str(evaluation.binding.OPENPI / 'src'),
                    str(evaluation.binding.OPENPI / 'packages/openpi-client/src')]
    cache = []
    with results_path.open('a', buffering=1) as stream:
        for index, case in enumerate(cases, 1):
            if case['id'] in done:
                continue
            os.environ['GEN1_LITERAL_PROMPT'] = case['prompt']
            directory = output / case['id']
            directory.mkdir()
            try:
                with quiet_case(directory):
                    evaluation.worker(Namespace(task=case['task'], seed=case['seed'],
                                                episode_index=case['episode_index'],
                                                checkpoint=str(args.checkpoint.resolve()), output=directory),
                                      cache)
                result = json.loads((directory / 'result.json').read_text())
                row = compact(result, case)
            except Exception as exc:
                row = dict(id=case['id'], task=case['task'], seed=case['seed'],
                           episode_index=case['episode_index'], status='runner_error',
                           success=None, steps=None, hashes=None,
                           error=f'{type(exc).__name__}: {exc}')
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            results.append(row); done.add(case['id'])
            (output / 'progress.json').write_text(json.dumps(
                dict(completed_cases=len(results), next_case=index + 1,
                     summary=summary(cases, results, args.checkpoint)), indent=2) + '\n')
            print(json.dumps(dict(index=index, total=len(cases), **row),
                             allow_nan=False), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary(cases, results, args.checkpoint),
                                                     indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        evaluation.simulator()
    else:
        main()
