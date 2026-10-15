"""Wait for the existing training run, verify selection, then run Clean pairs."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from gen1_multitask_data import digest, require
from gen1_multitask_run import gpu_free, write, PYTHON, ROOT
from gen1_recovery_next_after_audit import process_identity, checked_result

TRAINING = ROOT / 'outputs/gen1_multitask_phaseaware_v3_20261013'
CHECKPOINTS = ROOT / 'checkpoints/gen1_multitask_phaseaware_v3_20261013'
QUEUE = ROOT / 'outputs/gen1_phaseaware_after_20261013'
SEEDS = (1100000, 1200000)
METRICS = ('joint_mae', 'gripper_mae', 'first10_joint_mae', 'first10_gripper_mae')


def read(path):
    return json.loads(path.read_text())


def checked_selection(training, checkpoints):
    config = read(training / 'run_config.json')
    complete = read(training / 'completed.json')
    require(complete['status'] == 'completed' and complete['steps'] == 24000,
            'Training is incomplete')
    steps = [1000, 4000, 12000, 24000]
    require(config['save_steps'] == complete['save_steps'] == steps, 'Changed save schedule')
    selection = read(training / 'selection.json')
    initial = read(training / 'validation_0000.json')
    tasks = config['tasks']
    require(len(tasks) == 18 and len(set(tasks)) == 18, 'Changed task panel')
    identities = [(r['task'], r['episode'], r['frame']) for r in initial['records']]
    require(len(identities) == len(set(identities)) == 720, 'Changed validation panel')
    require([r['step'] for r in selection['candidates']] == steps, 'Missing candidates')
    for candidate in selection['candidates']:
        step = candidate['step']
        expected = checkpoints / str(step)
        require(Path(candidate['checkpoint']).resolve() == expected.resolve(), 'Foreign checkpoint')
        require((expected / 'params/_METADATA').is_file(), 'Missing checkpoint metadata')
        require(read(expected / 'provenance.json')['step'] == step, 'Wrong checkpoint step')
        validation = read(training / f'validation_{step:04d}.json')
        require(validation['step'] == step, 'Wrong validation step')
        require([(r['task'], r['episode'], r['frame']) for r in validation['records']]
                == identities, 'Validation panel mismatch')
        # Recompute summaries from records before trusting the selected score.
        for panel in (initial, validation):
            for task in tasks:
                rows = [r for r in panel['records'] if r['task'] == task]
                require(len(rows) == 40, 'Wrong per-task anchor count')
                for key in METRICS:
                    values = [r[key] for r in rows]
                    require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid metric')
                    require(math.isclose(sum(values)/len(values), panel['metrics'][task][key],
                                         rel_tol=1e-10, abs_tol=1e-12), 'Summary mismatch')
        score = sum(validation['metrics'][t][k] / max(initial['metrics'][t][k], 1e-8)
                    for t in tasks for k in METRICS) / (4 * len(tasks))
        require(math.isclose(score, candidate['score'], rel_tol=1e-10, abs_tol=1e-12),
                'Selection score mismatch')
    require(selection['selected'] == min(selection['candidates'],
            key=lambda r: (r['score'], r['step'])), 'Wrong selected checkpoint')
    return selection['selected']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    args = parser.parse_args()
    identity = process_identity(args.training_pid)
    require(identity is not None, 'Expected live training process')
    command = Path(f'/proc/{args.training_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    require('gen1_multitask_train.py' in command and TRAINING.name in command, 'Wrong process')
    QUEUE.mkdir(exist_ok=False)
    dependencies = [Path(__file__)] + [ROOT / 'scripts' / name for name in (
        'gen1_recovery_verify.py', 'gen1_recovery_confirm.py', 'gen1_recovery_eval.py',
        'gen1_recovery_compare.py', 'gen1_recovery_next_after_audit.py',
        'gen1_multitask_run.py', 'gen1_multitask_data.py',
        'gen1_raw_paired_eval.py', 'controlled_paired_eval.py',
        'robotwin_eval_harness.py', 'robotwin_public_openpi_harness.py', 'pi05_openpi_adapter.py')]
    hashes = {str(p): digest(p) for p in dependencies}
    write(QUEUE / 'plan.json', dict(training_pid=args.training_pid, process_identity=identity,
        training=str(TRAINING), source_sha256=hashes, clean_only=True,
        seeds=SEEDS, pairs_per_task=10, retraining=False,
        rule='Verify completed offline selection, CPU restore, then fresh paired Clean panel; second independent panel only if first has gain without task regression'))
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', WANDB_MODE='disabled')

    def run(stage, command):
        require(all(digest(p) == h for p, h in hashes.items()), 'Queued dependency changed')
        gpu_free()
        with (QUEUE / f'{stage}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(QUEUE / f'{stage}_process.json', dict(pid=child.pid,
                identity=process_identity(child.pid), command=command))
            code = child.wait()
        write(QUEUE / f'{stage}_exit.json', dict(returncode=code))
        require(code == 0, f'{stage} failed; no automatic retry')

    try:
        print('WAITING_FOR_TRAINING', args.training_pid, identity, flush=True)
        while process_identity(args.training_pid) == identity:
            time.sleep(15)
        choice = checked_selection(TRAINING, CHECKPOINTS)
        write(QUEUE / 'checked_selection.json', choice)
        checkpoint = Path(choice['checkpoint'])
        run('verify', [PYTHON, '-u', str(ROOT / 'scripts/gen1_recovery_verify.py'),
                       '--checkpoint', str(checkpoint), '--training', str(TRAINING)])
        verification = read(checkpoint / 'verification.json')
        require(verification['status'] == 'passed' and verification['all_finite']
                and verification['normalization_exact'], 'Restore verification failed')
        results = []
        for number, seed in enumerate(SEEDS, 1):
            output = ROOT / f'outputs/gen1_phaseaware_confirmation{number}_20261013'
            require(not output.exists(), 'Refusing to overwrite or retry an existing panel')
            run(f'confirmation{number}', [PYTHON, '-u', str(ROOT / 'scripts/gen1_recovery_confirm.py'),
                'run', '--checkpoint', str(checkpoint), '--seed', str(seed), '--output', str(output)])
            result = checked_result(output / 'comparison.json')
            require(result['checkpoint'] == str(checkpoint), 'Wrong evaluation checkpoint')
            results.append(dict(path=str(output), assessment=result['outcome_assessment']))
            if not result['outcome_assessment']['observed_gain_without_task_regression']:
                break
        write(QUEUE / 'completed.json', dict(selection=choice, confirmations=results,
            goal_achieved=False, next_action='Review paired failures or broader Clean regression; no automatic retraining'))
    except BaseException as exc:
        write(QUEUE / 'failed.json', dict(error=f'{type(exc).__name__}: {exc}', goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
