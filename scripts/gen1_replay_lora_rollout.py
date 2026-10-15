#!/usr/bin/env python3
"""Run the unchanged Clean paired protocol with bounded inference GPU memory."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT/'scripts'))
import gen1_recovery_confirm as confirmation
import gen1_recovery_eval as evaluation


def limited_runtime():
    env = _original_runtime()
    # The same single GPU is training. Disable inference preallocation and
    # leave room for the training process and RoboTwin's renderer.
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.35'
    return env


_original_runtime = evaluation.binding.runtime_environment
evaluation.binding.runtime_environment = limited_runtime
confirmation.SCRIPT = Path(__file__).resolve()


def main():
    if len(sys.argv) < 2:
        raise ValueError('Specify worker, run, or panel')
    mode = sys.argv[1]
    if mode == 'worker':
        evaluation.main()
    elif mode in ('run', 'panel'):
        if mode == 'run':
            import argparse
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('mode')
            parser.add_argument('--output', type=Path, required=True)
            args, _ = parser.parse_known_args()
            checkpoint = next((sys.argv[i+1] for i,x in enumerate(sys.argv[:-1])
                               if x == '--checkpoint'), None)
            if checkpoint is None:
                raise ValueError('Checkpoint required')
            verification = json.loads((Path(checkpoint)/'verification.json').read_text())
            if (verification.get('status') != 'passed'
                    or verification.get('kind') != 'full_lora_model_unmerged'
                    or not verification.get('production_restore_exact')):
                raise ValueError('Verified LoRA export required')
            record = args.output.with_suffix('.runner.json')
            with record.open('x') as stream:
                json.dump(dict(checkpoint=str(Path(checkpoint).resolve()),
                    runner=str(Path(__file__).resolve()),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    inference_preallocate=False, inference_memory_fraction=0.35,
                    training_concurrent=True, protocol='existing gen1_recovery_confirm'),stream,indent=2)
        confirmation.main()
    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    main()
