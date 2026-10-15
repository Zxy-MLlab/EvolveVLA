#!/usr/bin/env python3
"""Fresh-server entry point: real batch-64 preflight, training, checkpoint rollouts."""
import json
from pathlib import Path
import subprocess
import sys

from bootstrap import ROOT, ENVS, require_fresh_server


def main():
    require_fresh_server()
    run_name = 'gen1_replay_expanded_lora_b64_20261015'
    for path in (ROOT / 'outputs' / run_name, ROOT / 'checkpoints' / run_name,
                 ROOT / 'outputs/gen1_expanded_train_rollouts_20261015.json',
                 ROOT / 'outputs/gen1_replay_expanded_loader_preflight_20261014.json'):
        if path.exists():
            raise ValueError(f'Existing experiment preserved; inspect it before restarting: {path}')
    python = ENVS / 'gen1_train_ready/bin/python'
    (ROOT / 'logs').mkdir(exist_ok=True)
    preflight_log = ROOT / 'logs/deployment_preflight.log'
    with preflight_log.open('x') as output:
        preflight = subprocess.Popen([str(python), '-B', '-u',
            str(ROOT / 'scripts/gen1_replay_expanded_preflight_20261014.py'), '--output',
            str(ROOT / 'outputs/gen1_replay_expanded_loader_preflight_20261014.json')],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    with (ROOT / 'logs/deployment_train_rollouts.log').open('x') as output:
        queue = subprocess.Popen([str(python), '-B', '-u',
            str(ROOT / 'scripts/gen1_expanded_train_rollouts_20261015.py'),
            '--preflight-pid', str(preflight.pid)], cwd=ROOT,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(preflight_pid=preflight.pid, queue_pid=queue.pid,
        batch_size=64, total_steps=24000, save_and_rollout_steps=[1000,4000,9000,18000,24000],
        preflight_log=str(preflight_log),
        state=str(ROOT / 'outputs/gen1_expanded_train_rollouts_20261015.json')), indent=2))


if __name__ == '__main__':
    main()
