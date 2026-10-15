#!/usr/bin/env python3
"""Complete ten valid Clean pairs with an outcome-independent seed schedule.

Expert admission exclusions remain in the audit trail. Only admission status,
never policy success, determines supplemental seeds. Initial input mismatches
and infrastructure errors stop the experiment rather than being excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_recovery_eval as evaluation
import gen1_recovery_compare as comparison

ROOT = evaluation.ROOT
SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(path.read_text())


def next_cases(original, candidate, seed_start, target=10):
    key_sets = []
    for panel in (original, candidate):
        keys = [(r['task'], r['seed']) for r in panel['records']]
        if len(set(keys)) != len(keys):
            raise RuntimeError('Duplicate scheduled case')
        key_sets.append(set(keys))
    if key_sets[0] != key_sets[1]:
        raise RuntimeError('Missing scheduled case; finish the existing round first')
    result = comparison.summarize(original, candidate)
    cases = []
    for task in evaluation.TASKS:
        rows = [r for panel in (original, candidate) for r in panel['records'] if r['task'] == task]
        if any(r['status'] not in ('completed', 'expert_rejected') for r in rows):
            raise RuntimeError('Infrastructure error must be resolved before continuing')
        for excluded in result['tasks'][task]['unpaired']:
            if excluded['original_status'] == excluded['candidate_status'] == 'completed':
                raise RuntimeError('Completed pair has different initial inputs or prompt binding')
        deficit = target - result['tasks'][task]['paired_completed']
        if deficit < 0:
            raise RuntimeError('More valid pairs than prespecified')
        next_seed = max([seed_start - 1] + [r['seed'] for r in rows]) + 1
        if next_seed + deficit > seed_start + 50:
            raise RuntimeError('Admission exclusions exceeded the prespecified 50-seed limit')
        cases.extend(dict(task=task, seed=seed) for seed in range(next_seed, next_seed + deficit))
    return cases


def panel(args):
    os.environ.update(evaluation.binding.runtime_environment())
    openpi = evaluation.binding.OPENPI
    sys.path[:0] = [str(openpi/'src'), str(openpi/'packages/openpi-client/src')]
    cache = []
    records = []
    for row in read(args.cases):
        directory = args.output / f"{row['task']}_{row['seed']}"
        directory.mkdir()
        case = argparse.Namespace(**row, output=directory, checkpoint=str(args.checkpoint))
        evaluation.worker(case, cache)
        record = read(directory/'result.json')
        records.append(record)
        comparison.write(args.output/f'progress_{len(records):02d}.json', {'records': records})
        print('RESULT', row['task'], row['seed'], record['status'], record.get('success'), flush=True)
        if record['status'] not in ('completed', 'expert_rejected'):
            raise RuntimeError(str(record))
    comparison.write(args.output/'summary.json', {'records': records})


def run_panel(directory, checkpoint, cases_path):
    directory.mkdir()
    evaluation.release_checkpoint_cache()
    command = [evaluation.c.CONFIG['inference_python'], '-I', '-u', '-B', str(SCRIPT),
               'panel', '--output', str(directory), '--checkpoint', str(checkpoint),
               '--cases', str(cases_path)]
    with (directory/'panel.log').open('x') as log:
        process = subprocess.Popen(command, env=evaluation.binding.runtime_environment(),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        baseline = evaluation.memory_status()
        started = time.monotonic()
        try:
            while process.poll() is None:
                current = evaluation.memory_status()
                if any(current['events'][k] > baseline['events'][k] for k in ('oom', 'oom_kill')):
                    raise RuntimeError('New cgroup OOM event')
                if current['estimated_available'] is not None and current['estimated_available'] < 4 * 2**30:
                    raise RuntimeError('Less than 4 GiB estimated available memory')
                if time.monotonic() - started > 7200:
                    raise TimeoutError('Panel exceeded two hours')
                time.sleep(2)
        except BaseException:
            evaluation.binding.kill_group(process)
            raise
    if process.returncode:
        raise RuntimeError(f'Panel failed: {directory}')
    return read(directory/'summary.json')['records']


def run(args):
    if bool(args.original_reference) != bool(args.candidate_reference):
        raise ValueError('Both reference panels are required together')
    original_checkpoint = ROOT/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
    original, candidate = {'records': []}, {'records': []}
    if args.original_reference:
        for reference, checkpoint, records in (
            (args.original_reference, original_checkpoint, original),
            (args.candidate_reference, args.checkpoint, candidate),
        ):
            manifest = read(reference/'manifest.json')
            if (manifest['checkpoint'] != str(checkpoint) or manifest['seed_start'] != args.seed
                    or manifest['config'] != evaluation.c.CONFIG
                    or not manifest['reuse_policy']):
                raise ValueError('Reference checkpoint or seed mismatch')
            records['records'].extend(read(reference/'summary.json')['records'])
    args.output.mkdir(exist_ok=False)
    dependencies = [SCRIPT, Path(evaluation.__file__), Path(comparison.__file__),
                    Path(evaluation.binding.__file__), Path(evaluation.c.__file__),
                    ROOT/'scripts/robotwin_eval_harness.py', ROOT/'scripts/pi05_openpi_adapter.py']
    comparison.write(args.output/'manifest.json', dict(
        checkpoint=str(args.checkpoint), seed_start=args.seed, target_pairs_per_task=10,
        max_scheduled_seeds_per_task=50, original_reference=str(args.original_reference),
        candidate_reference=str(args.candidate_reference), config=evaluation.c.CONFIG,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        protocol='Clean controlled diagnostic; ascending seeds; supplemental count uses only admission and input equality; no outcome retries; fixed bound seen prompts; policy RNG reset each episode',
        development_panel_reused=bool(args.original_reference)))
    round_number = 0
    while True:
        cases = next_cases(original, candidate, args.seed)
        if not cases:
            break
        cases_path = args.output/f'round_{round_number:02d}_cases.json'
        comparison.write(cases_path, cases)
        for label, checkpoint, records in [('original', original_checkpoint, original),
                                           ('candidate', args.checkpoint, candidate)]:
            directory = args.output/f'round_{round_number:02d}_{label}'
            records['records'].extend(run_panel(directory, checkpoint, cases_path))
            print('FINISHED', round_number, label, flush=True)
        comparison.write(args.output/f'round_{round_number:02d}_comparison.json',
                         comparison.summarize(original, candidate))
        round_number += 1
    comparison.write(args.output/'original.json', original)
    comparison.write(args.output/'candidate.json', candidate)
    result = comparison.summarize(original, candidate)
    result['development_panel_reused'] = bool(args.original_reference)
    result['checkpoint'] = str(args.checkpoint)
    result['attempted_per_model'] = {
        label: {task: sum(r['task'] == task for r in panel['records']) for task in evaluation.TASKS}
        for label, panel in [('original', original), ('candidate', candidate)]}
    result['outcome_assessment'] = assess(result)
    comparison.write(args.output/'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


def assess(result):
    """Descriptive acceptance and exact two-sided paired sign/McNemar test."""
    from math import comb
    tasks = result['tasks'].values()
    original_only = sum(t['original_only'] for t in tasks)
    candidate_only = sum(t['candidate_only'] for t in tasks)
    discordant = original_only + candidate_only
    p_value = min(1.0, 2 * sum(comb(discordant, k)
                              for k in range(min(original_only, candidate_only) + 1)) / 2**discordant)
    return dict(original_successes=sum(t['original_successes'] for t in tasks),
                candidate_successes=sum(t['candidate_successes'] for t in tasks),
                total_pairs=sum(t['paired_completed'] for t in tasks),
                observed_gain_without_task_regression=(candidate_only > original_only and
                    all(t['candidate_successes'] >= t['original_successes'] for t in tasks)),
                exact_two_sided_paired_p=p_value,
                limitation='Two-task sample; gain is descriptive, not proof augmentation alone caused it. P-value treats independent seed pairs as Bernoulli discordances.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['run', 'panel'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=400000)
    parser.add_argument('--original-reference', type=Path)
    parser.add_argument('--candidate-reference', type=Path)
    parser.add_argument('--cases', type=Path)
    args = parser.parse_args()
    (panel if args.mode == 'panel' else run)(args)


if __name__ == '__main__':
    main()
