#!/usr/bin/env python3
"""Separate broad Clean panel entry; never patches active files or GPU jobs."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_recovery_eval as evaluation
import gen1_recovery_confirm as confirmation
from gen1_multitask_binding import bind, read_pool

SCRIPT = Path(__file__).resolve()
ROOT = evaluation.ROOT
SPLIT = ROOT / 'data/gen1_multitask_coverage_20261012/split_manifest.json'


def task_names():
    return tuple(json.loads(SPLIT.read_text())['tasks'])


def scene_spec(task, seed, description_root):
    if task not in task_names() or type(seed) is not int or seed < 0:
        raise ValueError('Foreign task or invalid seed')
    path = Path(description_root).resolve() / 'task_instruction' / (task + '.json')
    _, sha = read_pool(path)
    # This transport marker is intercepted by the existing bound_task hook.
    # It is never sent to the policy or used as the task's instruction.
    return dict(task=task, seed=seed, template='__GEN1_SCENE_BOUND_SEEN__',
                source=str(path), source_sha256=sha)


def scene_bind(spec, episode_info, description_root):
    if spec != scene_spec(spec['task'], spec['seed'], description_root):
        raise ValueError('Scene specification changed')
    return bind(spec['task'], spec['seed'], episode_info, description_root)


@contextlib.contextmanager
def installed():
    b = evaluation.binding
    with b.replaced(evaluation, 'SCRIPT', SCRIPT), \
         b.replaced(b, 'template_spec', scene_spec), \
         b.replaced(b, 'bind_prompt', scene_bind):
        yield


def summarize(original, candidate, tasks):
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('Invalid task panel')
    panels = []
    for panel in (original, candidate):
        result = {}
        for row in panel['records']:
            key = (row['task'], row['seed'])
            if row['task'] not in tasks or key in result:
                raise ValueError('Foreign or duplicate case')
            if row['status'] not in ('completed', 'expert_rejected'):
                raise ValueError('Infrastructure error')
            if row['status'] == 'completed':
                if type(row.get('success')) is not bool or not row.get('hashes') or not row.get('binding'):
                    raise ValueError('Incomplete policy result')
            result[key] = row
        panels.append(result)
    aa, bb = panels
    if aa.keys() != bb.keys():
        raise ValueError('Missing scheduled case')
    results = {}
    for task in tasks:
        pairs, excluded = [], []
        for key in sorted(k for k in aa if k[0] == task):
            a, b = aa[key], bb[key]
            if a['status'] == b['status'] == 'completed':
                if a['hashes'] != b['hashes'] or a['binding'] != b['binding']:
                    raise ValueError('Initial input or prompt mismatch')
                pairs.append(dict(seed=key[1], original=a['success'], candidate=b['success']))
            else:
                excluded.append(dict(seed=key[1], original_status=a['status'], candidate_status=b['status']))
        results[task] = dict(pairs=pairs, unpaired=excluded, paired_completed=len(pairs),
            original_successes=sum(p['original'] for p in pairs),
            candidate_successes=sum(p['candidate'] for p in pairs),
            original_only=sum(p['original'] and not p['candidate'] for p in pairs),
            candidate_only=sum(p['candidate'] and not p['original'] for p in pairs))
    return dict(tasks=results, limitation='Matched observed initial inputs and bound seen prompts; not complete physics state equality or the official 50-task benchmark.')


def next_cases(original, candidate, tasks, seed_start):
    if type(seed_start) is not int or seed_start < 100000:
        raise ValueError('Evaluation seed must be outside collection range')
    report = summarize(original, candidate, tasks)
    cases = []
    for task in tasks:
        prior = [r['seed'] for r in original['records'] if r['task'] == task]
        if sorted(prior) != list(range(seed_start, seed_start + len(prior))):
            raise ValueError('Noncontiguous seed schedule')
        deficit = 10 - report['tasks'][task]['paired_completed']
        if deficit < 0 or len(prior) + deficit > 50:
            raise ValueError('Prespecified trial or admission limit exceeded')
        cases.extend(dict(task=task, seed=seed) for seed in
                     range(seed_start + len(prior), seed_start + len(prior) + deficit))
    return cases


def gpu_free():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError('GPU occupied; broad evaluation must follow training and confirmation')


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(args):
    tasks = tuple(args.tasks or task_names())
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(task_names()):
        raise ValueError('Invalid broad panel tasks')
    if args.seed < 900000:
        raise ValueError('Broad regression uses fresh seeds starting at 900000 or later')
    output = args.output.resolve()
    if output.parent != ROOT / 'outputs' or output.exists():
        raise ValueError('Require new direct child of project outputs')
    candidate = args.checkpoint.resolve()
    verification = json.loads((candidate / 'verification.json').read_text())
    if (verification.get('status') != 'passed'
            or Path(verification['checkpoint']).resolve() != candidate
            or verification.get('normalization_exact') is not True
            or verification.get('all_finite') is not True):
        raise ValueError('Candidate production verification required')
    original = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
    if candidate == original or not (candidate / 'params/_METADATA').is_file():
        raise ValueError('Missing or invalid candidate checkpoint')
    gpu_free()
    output.mkdir()
    paths = [SCRIPT, SPLIT, Path(__file__).with_name('gen1_multitask_binding.py'),
             Path(evaluation.__file__), Path(confirmation.__file__), Path(evaluation.binding.__file__),
             Path(evaluation.c.__file__), ROOT / 'scripts/robotwin_eval_harness.py',
             ROOT / 'scripts/robotwin_public_openpi_harness.py', ROOT / 'scripts/pi05_openpi_adapter.py']
    repo = Path(evaluation.c.CONFIG['robotwin_root'])
    paths += [repo / 'envs' / (task + '.py') for task in tasks]
    paths += [repo / 'description/task_instruction' / (task + '.json') for task in tasks]
    paths += sorted((repo / 'description/objects_description').rglob('*.json'))
    paths += evaluation.binding.manifest_code_paths()
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write(output / 'manifest.json', dict(tasks=tasks, seed_start=args.seed, pairs_per_task=10,
        max_seeds_per_task=50, candidate=str(candidate), original=str(original),
        config=evaluation.c.CONFIG, code_and_inputs_sha256=hashes,
        goal_achieved=False, protocol='Clean; scene-bound official seen prompts; common initial observed inputs; reset policy RNG; admission-only supplementation'))
    all_original, all_candidate = {'records': []}, {'records': []}
    try:
        for task in tasks:
            panels = {'original': {'records': []}, 'candidate': {'records': []}}
            round_number = 0
            while True:
                cases = next_cases(panels['original'], panels['candidate'], [task], args.seed)
                if not cases:
                    break
                cases_path = output / f'{task}_round_{round_number:02d}_cases.json'
                write(cases_path, cases)
                for label, checkpoint in (('original', original), ('candidate', candidate)):
                    for name, sha in hashes.items():
                        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                            raise ValueError(f'Frozen evaluation input changed: {name}')
                    gpu_free()
                    directory = output / f'{task}_round_{round_number:02d}_{label}'
                    # The existing supervisor enforces serial children, memory
                    # checks and a two-hour deadline for each <=10-case panel.
                    with evaluation.binding.replaced(confirmation, 'SCRIPT', SCRIPT):
                        rows = confirmation.run_panel(directory, checkpoint, cases_path)
                    expected = {(r['task'], r['seed']) for r in cases}
                    if (len(rows) != len(cases)
                            or {(r['task'], r['seed']) for r in rows} != expected
                            or any(Path(r['checkpoint']).resolve() != checkpoint for r in rows)):
                        raise ValueError('Panel output differs from requested cases or checkpoint')
                    panels[label]['records'].extend(rows)
                    print('FINISHED', task, round_number, label, flush=True)
                write(output / f'{task}_round_{round_number:02d}_comparison.json',
                      summarize(panels['original'], panels['candidate'], [task]))
                round_number += 1
            all_original['records'].extend(panels['original']['records'])
            all_candidate['records'].extend(panels['candidate']['records'])
            result = summarize(panels['original'], panels['candidate'], [task])
            if result['tasks'][task]['paired_completed'] != 10:
                raise ValueError('Incomplete task')
            write(output / f'{task}_comparison.json', result)
        write(output / 'original.json', all_original)
        write(output / 'candidate.json', all_candidate)
        result = summarize(all_original, all_candidate, tasks)
        result['outcome_assessment'] = confirmation.assess(result)
        result['goal_achieved'] = False
        write(output / 'comparison.json', result)
    except BaseException as exc:
        write(output / 'failed.json', dict(error=f'{type(exc).__name__}: {exc}', goal_achieved=False))
        raise


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        with installed():
            return evaluation.simulator()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['panel', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--seed', type=int, default=900000)
    args = parser.parse_args()
    if args.mode == 'run':
        return run(args)
    if args.cases is None:
        parser.error('panel requires --cases')
    rows = json.loads(args.cases.read_text())
    if not rows or len(rows) > 10 or len({r['task'] for r in rows}) != 1:
        raise ValueError('One task and at most ten cases per supervised panel')
    if len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Duplicate panel seed')
    for row in rows:
        scene_spec(row['task'], row['seed'], ROOT / 'robotwin_repo_ba0fd21/description')
    with installed():
        confirmation.panel(args)


if __name__ == '__main__':
    main()
