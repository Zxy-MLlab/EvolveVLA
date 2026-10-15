#!/usr/bin/env python3
"""Wait for this recovery run and extend replicated gains to broad Clean checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_multitask_eval as broad
from gen1_multitask_data import digest
from gen1_recovery_next_after_audit import process_identity, checked_result

ROOT = broad.ROOT
MANAGER = ROOT / 'outputs/gen1_multitask_manager_v2_20261012'
TRAINING = ROOT / 'outputs/gen1_multitask_balanced_20261012'
PYTHON = '/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python'


def profile(run_kind):
    if run_kind == 'balanced':
        return dict(manager=MANAGER, training=TRAINING,
            confirmation='gen1_multitask_confirmation{}_20261012',
            seeds=(700000, 800000), broad_seed=900000,
            destination=ROOT / 'outputs/gen1_multitask_broad_clean_20261012')
    if run_kind == 'phaseaware':
        return dict(manager=ROOT / 'outputs/gen1_phaseaware_after_20261013',
            training=ROOT / 'outputs/gen1_multitask_phaseaware_v3_20261013',
            confirmation='gen1_phaseaware_confirmation{}_20261013',
            seeds=(1100000, 1200000), broad_seed=1300000,
            destination=ROOT / 'outputs/gen1_phaseaware_broad_clean_20261013')
    raise ValueError('Unknown run kind')


def decision(run_kind='balanced'):
    config = profile(run_kind)
    manager, training_dir = config['manager'], config['training']
    if (manager / 'failed.json').exists():
        raise RuntimeError('Upstream manager failed; preserve its evidence before continuing')
    completed = json.loads((manager / 'completed.json').read_text())
    training = json.loads((training_dir / 'completed.json').read_text())
    selected = json.loads((training_dir / 'selection.json').read_text())['selected']
    # The phase-aware coordinator writes completed.json only on success but
    # has no status field; require its independently checked selection too.
    if run_kind == 'phaseaware':
        if json.loads((manager / 'checked_selection.json').read_text()) != selected:
            raise ValueError('Checked selection differs from training selection')
    elif completed['status'] != 'completed':
        raise ValueError('Incomplete manager')
    if (training['status'] != 'completed'
            or training['steps'] != 24000 or completed['selection'] != selected):
        raise ValueError('Incomplete training or inconsistent selection')
    confirmations = completed['confirmations']
    if not 1 <= len(confirmations) <= 2:
        raise ValueError('Invalid confirmation count')
    observed = []
    for number, entry in enumerate(confirmations, start=1):
        directory = ROOT / 'outputs' / config['confirmation'].format(number)
        expected_path = directory if run_kind == 'phaseaware' else directory / 'comparison.json'
        if Path(entry['path']).resolve() != expected_path:
            raise ValueError('Unexpected confirmation path')
        result = checked_result(directory / 'comparison.json')
        manifest = json.loads((directory / 'manifest.json').read_text())
        if (result['checkpoint'] != selected['checkpoint']
                or manifest['checkpoint'] != selected['checkpoint']
                or manifest['seed_start'] != config['seeds'][number - 1]
                or result['outcome_assessment'] != entry['assessment']):
            raise ValueError('Confirmation checkpoint, seed or assessment mismatch')
        if set(result['tasks']) != {'place_mouse_pad', 'hanging_mug'}:
            raise ValueError('Unexpected confirmation task set')
        for task in result['tasks'].values():
            if any(type(p['seed']) is not int or not
                   config['seeds'][number - 1] <= p['seed'] < config['seeds'][number - 1] + 50
                   for p in task['pairs']):
                raise ValueError('Confirmation pair seed outside scheduled range')
        observed.append(result['outcome_assessment'])
    gain = len(observed) == 2 and all(r['observed_gain_without_task_regression'] for r in observed)
    if len(observed) == 1 and observed[0]['observed_gain_without_task_regression']:
        raise ValueError('Second independent confirmation missing')
    return dict(action='broad_clean_regression' if gain else 'needs_failure_analysis',
                checkpoint=selected['checkpoint'], confirmations=observed, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manager-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-kind', choices=('balanced', 'phaseaware'), default='balanced')
    args = parser.parse_args()
    config = profile(args.run_kind)
    manager = config['manager']
    identity = process_identity(args.manager_pid)
    if identity is None:
        raise ValueError('Queue requires the live known manager')
    cmdline = Path(f'/proc/{args.manager_pid}/cmdline').read_bytes().split(b'\0')
    if args.run_kind == 'phaseaware':
        plan = json.loads((manager / 'plan.json').read_text())
        scripts = [Path(arg.decode()).name for arg in cmdline if arg]
        if ('gen1_phaseaware_after.py' not in scripts
                or b'--training-pid' not in cmdline
                or cmdline[cmdline.index(b'--training-pid') + 1] != str(plan['training_pid']).encode()
                or plan['training'] != str(config['training'])):
            raise ValueError('PID is not this phase-aware coordinator')
    else:
        recorded = json.loads((manager / 'training_process.json').read_text())
        stat = Path(f'/proc/{recorded["pid"]}/stat')
        if stat.exists() and int(stat.read_text().rsplit(')',1)[1].split()[1]) != args.manager_pid:
            raise ValueError('PID does not own recorded training child')
        if str(manager).encode() not in cmdline:
            raise ValueError('PID is not this run manager')
    output = args.output.resolve()
    if output.parent != ROOT / 'outputs':
        raise ValueError('Expected project output child')
    output.mkdir(exist_ok=False)
    paths = [Path(__file__), broad.SCRIPT, ROOT / 'scripts/gen1_multitask_binding.py',
             ROOT / 'scripts/gen1_recovery_next_after_audit.py']
    paths += broad.evaluation.binding.manifest_code_paths()
    paths += [Path(broad.evaluation.__file__), Path(broad.confirmation.__file__), broad.SPLIT]
    if args.run_kind == 'phaseaware':
        paths += [ROOT / 'scripts/gen1_phaseaware_after.py']
    hashes = {str(p): digest(p) for p in paths}
    destination = config['destination']
    broad.write(output / 'plan.json', dict(manager_pid=args.manager_pid,
        process_start_time=identity, manager=str(manager), sources=hashes,
        run_kind=args.run_kind, training=str(config['training']),
        tasks=broad.task_names(), seed_start=config['broad_seed'], target_pairs_per_task=10,
        destination=str(destination), clean_only=True,
        condition='Both fresh two-task confirmations have descriptive gain without target-task regression; otherwise preserve failure evidence and do not launch broad evaluation'))
    print('WAITING_FOR_MANAGER', args.manager_pid, flush=True)
    try:
        while process_identity(args.manager_pid) == identity:
            time.sleep(15)
        for name, sha in hashes.items():
            if digest(name) != sha:
                raise ValueError(f'Queued source changed: {name}')
        choice = decision(args.run_kind)
        broad.write(output / 'decision.json', choice)
        if choice['action'] == 'needs_failure_analysis':
            print('NEEDS_FAILURE_ANALYSIS', json.dumps(choice), flush=True)
            return
        broad.gpu_free()
        command = [PYTHON, '-u', str(broad.SCRIPT), 'run', '--checkpoint', choice['checkpoint'],
                   '--output', str(destination), '--seed', str(config['broad_seed'])]
        with (output / 'regression.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            broad.write(output / 'regression_process.json', dict(pid=child.pid,
                process_start_time=process_identity(child.pid), command=command))
            code = child.wait()
        broad.write(output / 'regression_exit.json', dict(returncode=code))
        if code:
            raise RuntimeError('Broad regression failed; inspect regression.log')
        result = json.loads((destination / 'comparison.json').read_text())
        if set(result['tasks']) != set(broad.task_names()) or any(
                row['paired_completed'] != 10 for row in result['tasks'].values()):
            raise ValueError('Incomplete broad regression')
        broad.write(output / 'completed.json', dict(comparison=str(destination / 'comparison.json'),
            assessment=result['outcome_assessment'], goal_achieved=False))
    except BaseException as exc:
        broad.write(output / 'failed.json', dict(error=f'{type(exc).__name__}: {exc}', goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
