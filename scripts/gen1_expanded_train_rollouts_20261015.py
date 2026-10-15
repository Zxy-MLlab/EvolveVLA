#!/usr/bin/env python3
"""Start the audited LoRA run and evaluate each committed checkpoint on Clean."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from gen1_after_collection_20261014 import (
    DATA, LOGS, OUTPUTS, ROOT, RUN, SCRIPTS, SPLIT_DIR, TRAIN_PY, TRAIN_REPO, record,
)
from gen1_multitask_data import digest, require

STEPS = [1000, 4000, 9000, 18000, 24000]
PREFLIGHT = OUTPUTS / 'gen1_replay_expanded_loader_preflight_20261014.json'
STATE = OUTPUTS / 'gen1_expanded_train_rollouts_20261015.json'


def read(path):
    return json.loads(Path(path).read_text())


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def checkpoint_ready(step):
    path = ROOT / 'checkpoints' / RUN / str(step)
    if not (path / 'provenance.json').is_file():
        return False
    value = read(path / 'provenance.json')
    require(value['step'] == step and value['frozen_parameters_unchanged']
            and value['adapter_restore_exact'] and value['run'] == str(OUTPUTS / RUN),
            f'Invalid checkpoint provenance: {path}')
    return True


def checked_result(path):
    result = read(path)
    require(set(result['tasks']) == {'place_mouse_pad', 'hanging_mug'}, 'Wrong rollout tasks')
    for task in result['tasks'].values():
        require(task['paired_completed'] == 10, 'Rollout has fewer than ten valid pairs')
    return result['outcome_assessment']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--preflight-pid', type=int, required=True)
    args = parser.parse_args()
    with (OUTPUTS / 'gen1_expanded_train_rollouts_20261015.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not STATE.exists(), 'Queue already exists; inspect rather than duplicate it')
        state = dict(stage='waiting_for_preflight', training_run=RUN, batch_size=64, save_steps=STEPS,
                     rollout_steps=STEPS, setting='Clean', tasks=['place_mouse_pad', 'hanging_mug'],
                     valid_pairs_per_task=10, seed_start=1500000, evaluations=[],
                     training_python=str(TRAIN_PY), preflight_pid=args.preflight_pid)

        def save(**fields):
            state.update(fields, updated_at_unix=time.time())
            temporary = STATE.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(STATE)
            print(json.dumps(state), flush=True)

        def launch(command, log):
            with log.open('x') as output:
                return subprocess.Popen([str(v) for v in command], cwd=ROOT,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)

        save()
        trainer = None
        try:
            prior = identity(args.preflight_pid)
            if prior is not None:
                command = Path(f'/proc/{args.preflight_pid}/cmdline').read_bytes()
                require(b'gen1_replay_expanded_preflight_20261014.py' in command,
                        'Wrong preflight process')
                record('mixed_preflight_retry', pid=args.preflight_pid, python=str(TRAIN_PY))
                while identity(args.preflight_pid) == prior:
                    time.sleep(10)
            preflight = read(PREFLIGHT)
            require(preflight['status'] == 'passed', 'Mixed preflight did not pass')
            require(preflight['batch_size'] == 64, 'Batch-64 mixed preflight required')
            require(preflight['configuration']['total_train_episodes'] == 2890
                    and preflight['configuration']['local_train_tasks'] == ['hanging_mug', 'place_mouse_pad'],
                    'Training data scope changed')
            for key, name in [('loader_sha256', 'gen1_replay_loader.py'),
                              ('sampler_sha256', 'gen1_replay_sampler.py')]:
                require(preflight['configuration'][key] == digest(SCRIPTS / name), 'Preflight code changed')
            require(shutil.disk_usage(ROOT).free > 60 * 2**30,
                    'Need 60 GiB for training, five exports, and evaluation reserve')
            require(not (OUTPUTS / RUN).exists() and not (ROOT / 'checkpoints' / RUN).exists(),
                    'Training run already exists')
            trainer = launch([TRAIN_PY, '-B', '-u', SCRIPTS / 'gen1_replay_lora_train.py',
                '--mode', 'train', '--run-name', RUN, '--batch-size', '64', '--steps', '24000',
                '--save-steps', *map(str, STEPS), '--split-manifest', SPLIT_DIR / 'split_manifest.json',
                '--train-repo-id', TRAIN_REPO, '--val-repo-id', 'gen1_multitask_coverage_val_20261012'],
                LOGS / 'gen1_expanded_lora_b64_train_20261015.log')
            save(stage='training', training_pid=trainer.pid, preflight_sha256=digest(PREFLIGHT))
            record('train', pid=trainer.pid, log=str(LOGS / 'gen1_expanded_lora_b64_train_20261015.log'))
            for step in STEPS:
                save(stage='waiting_for_checkpoint', next_rollout_step=step)
                while not checkpoint_ready(step):
                    require(trainer.poll() is None, f'Trainer exited ({trainer.returncode}) before step {step}')
                    time.sleep(15)
                export = ROOT / 'checkpoints' / f'gen1_expanded_lora_step{step}_export_20261015'
                output = OUTPUTS / f'gen1_expanded_lora_step{step}_clean_pairs_20261015'
                save(stage='exporting', current_step=step)
                child = launch([TRAIN_PY, '-B', '-u', SCRIPTS / 'gen1_replay_lora_export.py',
                    '--training', OUTPUTS / RUN, '--step', str(step), '--output', export],
                    LOGS / f'gen1_expanded_export_step{step}_20261015.log')
                save(child_pid=child.pid)
                require(child.wait() == 0, f'Export failed at step {step}')
                save(stage='rollout', current_step=step, rollout_output=str(output))
                child = launch([TRAIN_PY, '-B', '-u', SCRIPTS / 'gen1_replay_lora_rollout.py',
                    'run', '--checkpoint', export, '--output', output, '--seed', '1500000'],
                    LOGS / f'gen1_expanded_rollout_step{step}_20261015.log')
                save(child_pid=child.pid)
                require(child.wait() == 0, f'Rollout failed at step {step}')
                assessment = checked_result(output / 'comparison.json')
                state['evaluations'].append(dict(step=step, output=str(output), **assessment))
                save(stage='rollout_completed', child_pid=None)
            require(trainer.wait() == 0, 'Training process failed')
            completed = read(OUTPUTS / RUN / 'completed.json')
            require(completed['status'] == 'completed' and completed['steps'] == 24000,
                    'Final training completion missing')
            save(stage='completed')
            record('completed', training_output=str(OUTPUTS / RUN), rollout_state=str(STATE))
        except BaseException as error:
            save(stage='failed', error=str(error), traceback=traceback.format_exc(),
                 training_still_running=trainer is not None and trainer.poll() is None)
            record('pipeline_failed', error=str(error), rollout_state=str(STATE),
                   training_pid=None if trainer is None else trainer.pid,
                   training_still_running=state['training_still_running'])
            raise


if __name__ == '__main__':
    main()
