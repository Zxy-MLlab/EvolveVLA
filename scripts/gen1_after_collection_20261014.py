#!/usr/bin/env python3
"""Wait for the two-task collection, then audit, convert and start LoRA training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/root/autodl-tmp/EvolveVLA')
SCRIPTS = ROOT / 'scripts'
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
LOGS = ROOT / 'logs'
COLLECTOR = 457255
ROBOTWIN_PY = Path('/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python')
DATA_PY = Path('/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python')
TRAIN_PY = Path('/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python')
SPLIT_DIR = DATA / 'gen1_multitask_coverage_expanded_20261014'
TRAIN_REPO = 'gen1_multitask_coverage_expanded_train_20261014'
RUN = 'gen1_replay_expanded_lora_b64_20261015'
STATE = OUTPUTS / 'gen1_after_collection_20261014.json'


def record(stage, **fields):
    value = dict(stage=stage, updated_at_unix=time.time(), **fields)
    temporary = STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(value), flush=True)


def collector_running():
    try:
        command = Path(f'/proc/{COLLECTOR}/cmdline').read_bytes().split(b'\0')
        stat = Path(f'/proc/{COLLECTOR}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[0] != 'Z' and any(b'gen1_collect_two_tasks_1000_20261014.py' in arg
                                      for arg in command)
    except FileNotFoundError:
        return False


def run(stage, command):
    record(stage, command=[str(x) for x in command])
    log = LOGS / f'gen1_after_collection_{stage}_20261014.log'
    with log.open('x', buffering=1) as stream:
        completed = subprocess.run([str(x) for x in command], cwd=ROOT,
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f'{stage} failed ({completed.returncode}); see {log}')
    record(stage + '_passed', log=str(log))


def main():
    resuming_split = False
    resuming_conversion = False
    if STATE.exists():
        previous = json.loads(STATE.read_text())
        audit = OUTPUTS / 'gen1_two_task_1000_collection_final_audit_20261014.json'
        split_error = previous.get('error', '').startswith('split failed (1)') and not SPLIT_DIR.exists()
        conversion_error = (previous.get('error') in (
                                'Converter exited without a finalized conversion audit',
                                "[Errno 17] File exists: '/root/autodl-tmp/EvolveVLA/logs/gen1_after_collection_collection_audit_20261014.log'")
                            and (SPLIT_DIR / 'split_manifest.json').exists()
                            and not (DATA / TRAIN_REPO).exists()
                            and not (DATA / f'{TRAIN_REPO}.conversion.lock').exists())
        if previous.get('stage') != 'failed' or not audit.exists() or not (split_error or conversion_error):
            raise RuntimeError('Queue state already exists; refusing duplicate pipeline')
        audited = json.loads(audit.read_text())
        if not audited.get('complete') or any(audited['tasks'][task]['accepted'] != 1000
                                              for task in ('place_mouse_pad', 'hanging_mug')):
            raise RuntimeError('Final collection audit is not complete')
        resuming_split, resuming_conversion = split_error, conversion_error
    else:
        record('waiting_for_collection', collector_pid=COLLECTOR)
        while collector_running():
            time.sleep(30)
    manifests = [json.loads((DATA/'gen1_task_batches_20261010'/task/'manifest.json').read_text())
                 for task in ('place_mouse_pad', 'hanging_mug')]
    if any(m['status'] != 'complete' or len(m['credited_episodes']) + len(m['episodes']) != 1000
           for m in manifests):
        raise RuntimeError('Collector exited without two complete 1000-episode manifests')
    if not (resuming_split or resuming_conversion):
        run('collection_audit', [ROBOTWIN_PY, '-B', SCRIPTS/'audit_two_task_collection_20261014.py',
            '--output', OUTPUTS/'gen1_two_task_1000_collection_final_audit_20261014.json', '--require-complete'])
    if shutil.disk_usage(ROOT).free < 35 * 2**30:
        raise RuntimeError('Less than 35 GiB free before conversion; training not started')
    if not resuming_conversion:
        run('split_retry' if resuming_split else 'split', [ROBOTWIN_PY, '-B', SCRIPTS/'gen1_prepare_expanded_20261014.py',
            '--output', SPLIT_DIR])
    else:
        from gen1_multitask_data import validate_split
        split = json.loads((SPLIT_DIR / 'split_manifest.json').read_text())
        validate_split(split)
        if split['counts'] != {'train': 2710, 'val': 90}:
            raise RuntimeError('Expanded split changed before conversion restart')
    run('convert_retry' if resuming_conversion else 'convert', [DATA_PY, '-B', SCRIPTS/'gen1_convert_ready.py',
        '--input-dir', SPLIT_DIR/'train_inputs', '--lerobot-home', DATA,
        '--repo-id', TRAIN_REPO, '--min-free-gib', '25'])
    run('conversion_audit', [DATA_PY, '-B', SCRIPTS/'gen1_multitask_audit.py',
        '--dataset', DATA/TRAIN_REPO, '--split-manifest', SPLIT_DIR/'split_manifest.json',
        '--split', 'train', '--output', OUTPUTS/'gen1_expanded_conversion_independent_audit_20261014.json'])
    run('mixed_preflight', [TRAIN_PY, '-B', SCRIPTS/'gen1_replay_expanded_preflight_20261014.py',
        '--output', OUTPUTS/'gen1_replay_expanded_loader_preflight_20261014.json'])
    if shutil.disk_usage(ROOT).free < 25 * 2**30:
        raise RuntimeError('Less than 25 GiB free before training; training not started')
    run('train', [TRAIN_PY, '-B', '-u', SCRIPTS/'gen1_replay_lora_train.py',
        '--mode', 'train', '--run-name', RUN, '--batch-size', '64', '--steps', '24000',
        '--save-steps', '1000', '4000', '9000', '18000', '24000',
        '--split-manifest', SPLIT_DIR/'split_manifest.json', '--train-repo-id', TRAIN_REPO,
        '--val-repo-id', 'gen1_multitask_coverage_val_20261012'])
    record('completed', training_output=str(OUTPUTS/RUN))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record('failed', error=str(error), traceback=traceback.format_exc())
        raise
