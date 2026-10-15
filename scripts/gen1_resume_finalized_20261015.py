#!/usr/bin/env python3
"""Resume finalized expanded data with one full sequential audit, then train."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import traceback

from gen1_after_collection_20261014 import (
    DATA, DATA_PY, LOGS, OUTPUTS, ROOT, RUN, SCRIPTS, SPLIT_DIR, STATE, TRAIN_PY,
    TRAIN_REPO, record, run,
)
from gen1_multitask_data import digest, require, validate_split

AUDIT = OUTPUTS / 'gen1_expanded_conversion_independent_audit_20261014.json'
PROGRESS = OUTPUTS / 'gen1_expanded_sequential_audit_progress_20261015.json'


def main():
    # This lock lasts through training; a second resume must never launch jobs.
    with (OUTPUTS / 'gen1_resume_finalized_20261015.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
                names = {Path(arg.decode()).name for arg in argv if arg}
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            require(not names.intersection({'gen1_after_collection_20261014.py',
                    'gen1_convert_ready.py', 'gen1_replay_lora_train.py'}),
                    f'Previous pipeline still active: PID {proc.name}')
        previous = json.loads(STATE.read_text())
        require(previous['stage'] == 'convert_retry', 'Unexpected pipeline state')
        dataset = DATA / TRAIN_REPO
        split = json.loads((SPLIT_DIR / 'split_manifest.json').read_text())
        validate_split(split)
        require(split['counts'] == {'train': 2710, 'val': 90}, 'Expanded split changed')
        info = json.loads((dataset / 'meta/info.json').read_text())
        require((info['total_episodes'], info['total_frames']) == (2710, 671178),
                'Finalized dataset counts changed')
        require(not list(dataset.glob('tmp*')), 'Temporary encoding files remain')
        for path in (AUDIT, dataset / 'conversion_audit.json', OUTPUTS / RUN,
                     ROOT / 'checkpoints' / RUN,
                     OUTPUTS / 'gen1_replay_expanded_loader_preflight_20261014.json'):
            require(not path.exists(), f'Refusing existing pipeline output: {path}')
        archive = OUTPUTS / 'gen1_before_sequential_resume_20261015.json'
        with archive.open('x') as stream:
            json.dump(previous, stream, indent=2)
        conversion_lock = DATA / f'{TRAIN_REPO}.conversion.lock'
        if conversion_lock.exists():
            archived_lock = OUTPUTS / 'gen1_stopped_converter_20261015.lock'
            require(not archived_lock.exists(), 'Converter lock archive already exists')
            conversion_lock.rename(archived_lock)
        try:
            run('sequential_audit_resume', [DATA_PY, '-B', '-u', SCRIPTS / 'gen1_multitask_audit.py',
                '--dataset', dataset, '--split-manifest', SPLIT_DIR / 'split_manifest.json',
                '--split', 'train', '--output', AUDIT, '--progress', PROGRESS])
            report = json.loads(AUDIT.read_text())
            require(report['status'] == 'passed' and report['all_frames_decoded']
                    and report['state_action_all_cells_exact']
                    and (report['episodes'], report['frames']) == (2710, 671178),
                    'Full sequential audit did not pass')
            require(report['split_sha256'] == digest(SPLIT_DIR / 'split_manifest.json'),
                    'Split changed during audit')
            # One complete independent audit is the conversion certificate too;
            # never run a second redundant all-frame video pass.
            with (dataset / 'conversion_audit.json').open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write('\n')
            run('mixed_preflight', [TRAIN_PY, '-B', SCRIPTS / 'gen1_replay_expanded_preflight_20261014.py',
                '--output', OUTPUTS / 'gen1_replay_expanded_loader_preflight_20261014.json'])
            require(shutil.disk_usage(ROOT).free >= 25 * 2**30, 'Training disk reserve exhausted')
            run('train', [TRAIN_PY, '-B', '-u', SCRIPTS / 'gen1_replay_lora_train.py',
                '--mode', 'train', '--run-name', RUN, '--batch-size', '64', '--steps', '24000',
                '--save-steps', '1000', '4000', '9000', '18000', '24000',
                '--split-manifest', SPLIT_DIR / 'split_manifest.json', '--train-repo-id', TRAIN_REPO,
                '--val-repo-id', 'gen1_multitask_coverage_val_20261012'])
            record('completed', training_output=str(OUTPUTS / RUN))
        except BaseException as error:
            record('failed', error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
