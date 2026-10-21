#!/usr/bin/env python3
"""Run a fixed novel-seed panel with the unchanged original controller.

This is an isolated resource wrapper around the established controlled worker.
No replenishment of expert-rejected cases, outcome retries, or source edits.
"""
import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_recovery_confirm as confirmation
import gen1_execution_horizon_ablation_20261019 as horizon
import gen1_policy_duration_repair_20261020 as duration


def resource_environment(original):
    env = {k: v for k, v in original().items()
           if not k.startswith(('GEN1_', 'EVOLVEVLA_ROUTING_'))}
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.12',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0')
    return env


def inventory(checkpoint):
    return {str(p.relative_to(checkpoint)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in checkpoint.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-timeout', type=int, default=1800)
    args = parser.parse_args()
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    cases = json.loads(args.cases.read_text())
    keys = [(row['task'], row['seed']) for row in cases]
    if (len(keys) != 20 or len(set(keys)) != 20 or not 0 < args.case_timeout <= 1800
            or any(row['task'] not in horizon.evaluation.TASKS or row.get('episode_index') != 0
                   or type(row['seed']) is not int or row['seed'] < 0 for row in cases)
            or any(sum(row['task'] == task for row in cases) != 10 for task in horizon.evaluation.TASKS)):
        raise ValueError('Expected exactly ten frozen seeds per target task with reset episode index zero')
    identity = duration.checkpoint_identity(args.checkpoint)
    protocol = json.loads(args.protocol.read_text())
    if horizon.digest(args.cases) != protocol['cases_sha256']:
        raise ValueError('Cases differ from the frozen routing protocol')
    allowed_identities = protocol['checkpoint_identities']
    if isinstance(allowed_identities, dict):
        allowed_identities = list(allowed_identities.values())
    if identity['identity_sha256'] not in allowed_identities:
        raise ValueError('Checkpoint not authorized by the frozen routing protocol')
    frozen_inventory = inventory(args.checkpoint)
    sources = set(horizon.evaluation.binding.manifest_code_paths())
    sources.update([SCRIPT, Path(confirmation.__file__), Path(horizon.__file__),
                    Path(duration.__file__), Path(horizon.evaluation.__file__),
                    Path(confirmation.comparison.__file__), args.cases.resolve(), args.protocol.resolve()])
    hashes = {str(p): horizon.digest(p) for p in sorted(sources)}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol=__doc__, cases=cases, checkpoint=identity, source_sha256=hashes,
                    checkpoint_inventory=frozen_inventory, controller='original',
                    reset_episode_index=0, execute_prefix=50, policy_seed=0,
                    case_timeout=args.case_timeout, same_case_retries=0,
                    expert_rejection_replenishment=False, resource_only_override=True,
                    device_memory_fraction=.12, preallocate=False)
    horizon.write_json(args.output/'manifest.json', manifest)
    if args.mode == 'prepare':
        horizon.write_json(args.output/'preflight.json', dict(status='passed', gpu_used=False))
        print('PREPARED', args.output, flush=True)
        return
    if horizon.gpu_status()['free_gib'] < 18:
        raise RuntimeError('Less than 18 GiB free GPU memory before model load')
    for name in list(os.environ):
        if name.startswith(('GEN1_', 'EVOLVEVLA_ROUTING_')):
            os.environ.pop(name)
    original_environment = horizon.evaluation.binding.runtime_environment
    original_worker = horizon.evaluation.worker

    def worker(case, cache):
        for path, expected in hashes.items():
            if horizon.digest(path) != expected:
                raise ValueError(f'Frozen evaluation source changed: {path}')
        if inventory(args.checkpoint) != frozen_inventory:
            raise ValueError('Checkpoint inventory or file metadata changed during run')
        horizon.write_json(case.output/'requested_case.json',
                           dict(task=case.task, seed=case.seed, episode_index=case.episode_index,
                                checkpoint_identity_sha256=identity['identity_sha256']))
        with horizon.case_timeout(args.case_timeout):
            original_worker(case, cache)
        result = json.loads((case.output/'result.json').read_text())
        if (result.get('task') != case.task or result.get('seed') != case.seed
                or result.get('checkpoint') != str(args.checkpoint)):
            raise ValueError('Returned case identity mismatch')
        if result['status'] == 'completed' and (type(result.get('success')) is not bool
                                               or result.get('simulator_returncode') != 0):
            raise ValueError('Invalid terminal outcome or simulator returncode')
        horizon.write_json(case.output/'wrapper_audit.json',
                           dict(source_unchanged=True, checkpoint_inventory_unchanged=True,
                                requested_identity_matches=True,
                                checkpoint_identity_sha256=identity['identity_sha256']))

    with horizon.evaluation.binding.replaced(horizon.evaluation.binding, 'runtime_environment',
                                             lambda: resource_environment(original_environment)), \
         horizon.evaluation.binding.replaced(horizon.evaluation, 'worker', worker):
        confirmation.panel(args)


if __name__ == '__main__':
    main()
