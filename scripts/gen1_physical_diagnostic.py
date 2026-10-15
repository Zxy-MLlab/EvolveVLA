#!/usr/bin/env python3
"""Bounded physical diagnostic coordinator; execution requires an identity contract.

A separate identity contract records the approved historical delta without
refreshing the frozen inventory. Seeds are deliberately reused, not a fresh
benchmark. Samples describe endpoints, not contact or elapsed simulation time.
"""
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

FROZEN = Path(__file__).with_name('gen1_raw_paired_eval.py').resolve()
WRAPPER = Path(__file__).resolve()
SCHEDULE = ((200000, False), (200000, True), (200000, True), (200001, True))
MAX_ROWS = 901
MAX_BYTES = 4 * 2**20
EXECUTION_ENABLED = True
ORIGINAL_SHA = '468dc63f4519a7ecaea559a5dae8447d0f4328067e8f74bfd4838e2db86da293'
FROZEN_SHA = '2aef7fe2ee8042c1a65ccf6598073419713a2023044c40a171fa2a1f81abb0bb'
KNOWN_ROOTS = ('outputs/gen1_raw_paired_run_v1_20261011',
               'outputs/gen1_expert_gate_diagnostic_v1_20261011')
LIMITATION = 'Diagnostic deliberately reuses historical seeds; not a fresh benchmark.'
DEPENDENCIES = {'jax': '0.5.3', 'jaxlib': '0.5.3', 'orbax-checkpoint': '0.11.13',
                'flax': '0.10.2', 'numpy': '1.26.4', 'tensorstore': '0.1.74',
                'torch': '2.7.1', 'lerobot': '0.3.3', 'transformers': '4.53.2'}


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or symlinked identity: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class HashlibCompat:
    """Private frozen-module adapter; never mutate the shared hashlib module."""
    def __getattr__(self, name):
        return getattr(hashlib, name)

    @staticmethod
    def file_digest(stream, algorithm):
        if algorithm != 'sha256':
            raise ValueError('Unexpected frozen digest algorithm')
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest


def load_frozen():
    if sha(FROZEN) != FROZEN_SHA:
        raise ValueError('Frozen runner identity changed')
    spec = importlib.util.spec_from_file_location('_physical_frozen', FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(hashlib, 'file_digest', None)):
        module.hashlib = HashlibCompat()
    return module


def inventory(runner):
    """Keep frozen metadata membership and hash ALL files in the approved delta."""
    paths = set(runner.historical_paths())
    for name in KNOWN_ROOTS:
        root = runner.ROOT / name
        if root.is_symlink():
            raise ValueError('Symlinked historical root')
        if root.exists():
            for path in root.rglob('*'):
                if path.is_symlink():
                    raise ValueError('Symlink in historical delta')
                if path.is_file():
                    paths.add(str(path.relative_to(runner.ROOT)))
    return {name: sha(runner.ROOT / name) for name in sorted(paths)}


def output_paths(runner, manifest, prefix):
    if not prefix or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in prefix):
        raise ValueError('Invalid diagnostic prefix')
    paths = set()
    for trial, (_, case, enabled) in enumerate(schedule(manifest)):
        root = f'outputs/{prefix}_{trial}'
        paths.add(root + '/summary.json')
        for label in ('original', 'raw'):
            directory = root + '/' + case['id'] + '_' + label
            names = ['worker.log', 'supervision.json', 'result.json', 'trace.jsonl', 'faulthandler_sim.txt',
                     'initial/cam_high.npy', 'initial/cam_left_wrist.npy',
                     'initial/cam_right_wrist.npy', 'initial/state.npy', 'initial/hashes.json']
            if label == 'raw': names.append('raw_restore.json')
            if enabled: names.append('physical.jsonl')
            paths.update(directory + '/' + name for name in names)
    return sorted(paths)


def prepare_contract(runner, manifest, destination, prefix):
    manifest, destination = Path(manifest), Path(destination)
    if sha(manifest) != ORIGINAL_SHA:
        raise ValueError('Original manifest identity changed')
    current = inventory(runner)
    original = str(manifest.relative_to(runner.ROOT))
    expected = {p for p in runner.HISTORICAL_FILES if p.split('/')[0] in ('outputs', 'results', 'artifacts')}
    actual = set(current) - {original}
    added, removed = actual - expected, expected - actual
    if removed or any(not any(p.startswith(root + '/') for root in KNOWN_ROOTS) for p in added):
        raise ValueError('Unexpected historical inventory delta')
    m = runner.verified_manifest(manifest, ORIGINAL_SHA)
    allowed = output_paths(runner, m, prefix)
    if destination.parent != runner.OUTPUTS or destination.exists() or destination.is_symlink():
        raise ValueError('Fresh direct-child contract required')
    for trial in range(4):
        path = runner.OUTPUTS / f'{prefix}_{trial}'
        if path.exists() or path.is_symlink(): raise ValueError('Fresh trial roots required')
    contract = dict(version=1, limitation=LIMITATION, original=str(manifest), original_sha=ORIGINAL_SHA,
                    wrapper_sha=sha(WRAPPER), frozen_sha=FROZEN_SHA, dependencies=DEPENDENCIES,
                    schedule=[list(x) for x in SCHEDULE], prefix=prefix, snapshot=current,
                    added=sorted(added), allowed=allowed, contract=str(destination))
    runner.checked_write(destination, contract, runner.OUTPUTS / f'{prefix}_0')
    return contract


def validate_contract(runner, contract_path, digest):
    contract_path = Path(contract_path)
    if sha(contract_path) != digest: raise ValueError('Diagnostic contract identity changed')
    c = json.loads(contract_path.read_text())
    if (c['version'] != 1 or c['contract'] != str(contract_path) or
            c['original_sha'] != ORIGINAL_SHA or c['frozen_sha'] != FROZEN_SHA or
            sha(FROZEN) != FROZEN_SHA or c['wrapper_sha'] != sha(WRAPPER) or
            c['dependencies'] != DEPENDENCIES or c['schedule'] != [list(x) for x in SCHEDULE] or
            c['limitation'] != LIMITATION):
        raise ValueError('Diagnostic identity changed')
    current = inventory(runner)
    for name, identity in c['snapshot'].items():
        if current.get(name) != identity: raise ValueError('Snapshot changed: ' + name)
    extra = set(current) - set(c['snapshot']) - {str(contract_path.relative_to(runner.ROOT))}
    if extra - set(c['allowed']): raise ValueError('Unexpected later inventory addition')
    # Inspect every entry in new roots, including binary files and symlink directories.
    for trial in range(4):
        root = runner.OUTPUTS / f"{c['prefix']}_{trial}"
        if root.is_symlink(): raise ValueError('Symlinked diagnostic root')
        if root.exists():
            for p in root.rglob('*'):
                name = str(p.relative_to(runner.ROOT))
                if p.is_symlink() or (p.is_file() and name not in c['allowed']):
                    raise ValueError('Unexpected diagnostic output')
                if p.is_dir() and not any(a.startswith(name + '/') for a in c['allowed']):
                    raise ValueError('Unexpected diagnostic directory')
    m = runner.verified_manifest(c['original'], ORIGINAL_SHA)
    if c['allowed'] != output_paths(runner, m, c['prefix']):
        raise ValueError('Output prescription changed')
    return c, m


def schedule(manifest):
    result = []
    for seed, enabled in SCHEDULE:
        matches = [(i, c) for i, c in enumerate(manifest['cases'])
                   if c['task'] == 'hanging_mug' and c['seed'] == seed]
        if len(matches) != 1:
            raise ValueError('Fixed diagnostic case absent or ambiguous')
        index, case = matches[0]
        result.append((index, case, enabled))
    return tuple(result)


def run_schedule(manifest, execute_pair, pair_gate, compare_control):
    """Fixed four-pair schedule; abort on every exclusion, error or control mismatch."""
    records = []
    for trial, (index, case, enabled) in enumerate(schedule(manifest)):
        pair = execute_pair(trial, index, case, enabled)
        if set(pair) != {'original', 'raw'}:
            raise ValueError('Incomplete pair')
        if any(v.get('status') != 'completed' for v in pair.values()):
            raise ValueError('Admission exclusion or worker error; no replacement')
        if pair_gate(pair['original'], pair['raw']) != 'valid':
            raise ValueError('Pair gate failed')
        records.append(pair)
        if trial in (1, 2):
            compare_control(records[trial - 1], pair, trial == 2)
    return records


def vector(values, length):
    values = [float(v) for v in values]
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid physical vector')
    return values


def sample(task):
    robot = task.robot
    result = {}
    for side in ('left', 'right'):
        entity = getattr(robot, side + '_entity')
        active = list(entity.get_active_joints())
        qpos = entity.get_qpos()
        fingers = {}
        for joint, _, _ in getattr(robot, side + '_gripper'):
            name = joint.get_name()
            if not isinstance(name, str) or not name or name in fingers:
                raise ValueError('Invalid finger joint names')
            fingers[name] = vector([qpos[active.index(joint)]], 1)[0]
        if not fingers:
            raise ValueError('Missing measured fingers')
        result[side] = dict(
            measured_arm_qpos=vector(getattr(robot, 'get_' + side + '_arm_real_jointState')()[:6], 6),
            measured_finger_qpos=fingers,
            drive_targets_and_gripper_command=vector(getattr(robot, 'get_' + side + '_arm_jointState')(), 7),
            tcp_world_xyz=vector(getattr(robot, 'get_' + side + '_tcp_pose')()[:3], 3))
    pose = task.mug.get_pose()
    result.update(mug_world_xyz=vector(pose.p, 3), mug_world_quaternion=vector(pose.q, 4),
                  mug_function_world_xyz=vector(task.mug.get_functional_point(0)[:3], 3),
                  rack_world_xyz=vector(task.rack.get_pose().p, 3),
                  rack_function_world_xyz=vector(task.rack.get_functional_point(0)[:3], 3),
                  eval_success=bool(task.eval_success))
    return result


class Observer:
    def __init__(self, runner, directory, enabled, seed):
        self.runner, self.directory = runner, Path(directory)
        self.enabled, self.seed = enabled, seed
        self.stream = None
        self.rows = self.bytes = 0
        self.started = self.ended = False

    def capture(self, task, payload):
        kind = payload.get('type')
        if kind == 'case_status':
            raise ValueError('Expert admission/error status; no replacement')
        if kind not in ('episode_start', 'step'):
            return
        if kind == 'episode_start':
            if self.started or payload.get('seed') != self.seed:
                raise ValueError('Unexpected episode start')
            self.started = True
            index = 0
        else:
            index = payload.get('info', {}).get('step')
            if not self.started or self.ended or type(index) is not int or index != self.rows:
                raise ValueError('Unexpected action boundary')
        if self.rows >= MAX_ROWS:
            raise ValueError('Physical row cap exceeded')
        if index == 900 and not (payload.get('terminated') or payload.get('truncated')):
            raise ValueError('Missing terminal flag at action cap')
        if self.enabled:
            row = dict(seed=self.seed, step=index, physical=sample(task),
                       terminated=bool(payload.get('terminated')),
                       truncated=bool(payload.get('truncated')),
                       success=bool(payload.get('info', {}).get('is_success')))
            data = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
            if self.bytes + len(data) > MAX_BYTES:
                raise ValueError('Physical byte cap exceeded')
            self.runner.guard(self.directory.parent, len(data))
            if self.stream is None:
                path = self.directory / 'physical.jsonl'
                self.stream = path.open('xb')
            self.stream.write(data)
            self.bytes += len(data)
        self.rows += 1
        self.ended = bool(payload.get('terminated') or payload.get('truncated'))

    def close(self):
        if self.stream is not None:
            self.stream.close()


@contextlib.contextmanager
def observed_shim(runner, original_shim, evaluator, harness, spec, descriptions, observer):
    """Compose the real shim; no native imports before frozen deferred import."""
    task = None
    construct = evaluator.class_decorator
    send = harness.send_message
    with contextlib.ExitStack() as stack:
        def make(*args, **kwargs):
            nonlocal task
            if task is not None:
                raise ValueError('Multiple task construction')
            task = construct(*args, **kwargs)
            return task

        def transmit(stream, payload):
            observer.capture(task, payload)
            return send(stream, payload)

        stack.callback(observer.close)
        stack.enter_context(runner.replaced(evaluator, 'class_decorator', make))
        stack.enter_context(runner.replaced(harness, 'send_message', transmit))
        stack.enter_context(original_shim(evaluator, harness, spec, descriptions))
        yield
        if not observer.started or not observer.ended:
            raise ValueError('Incomplete diagnostic episode')


def simulator_command(runner, config, case):
    return [config['sim_python'], '-I', '-B', str(Path(runner.__file__).resolve()), 'sim',
            '--role', 'sim', '--controlled-case', '--episodes', '1', '--task', case['task'],
            '--seed', str(case['seed']), '--episode-steps', str(config['episode_steps']),
            '--robotwin-root', config['robotwin_root'], '--task-config', config['task_config'],
            '--first-instruction', case['template']['template'], '--first-instruction-seed', str(case['seed'])]


class SimulatorProxy:
    """Replace only runner.subprocess, never the global subprocess module."""
    def __init__(self, original, expected, cwd, enabled, identity=None):
        self.original, self.expected, self.cwd = original, expected, cwd
        self.enabled, self.calls = enabled, 0
        self.identity = identity

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Popen(self, command, **kwargs):
        if self.calls or command != self.expected or set(kwargs) != {'stdin', 'stdout', 'env', 'cwd'}:
            raise ValueError('Unexpected simulator launch')
        if kwargs['cwd'] != self.cwd or kwargs['stdin'] != self.original.PIPE or kwargs['stdout'] != self.original.PIPE:
            raise ValueError('Unexpected simulator launch options')
        if not isinstance(kwargs['env'], dict):
            raise ValueError('Missing simulator environment')
        self.calls += 1
        redirected = list(command)
        redirected[3:5] = [str(WRAPPER), 'sim-on' if self.enabled else 'sim-off']
        if self.identity is not None:
            kwargs['env'] = dict(kwargs['env'], GEN1_PHYSICAL_IDENTITY=json.dumps(self.identity))
        return self.original.Popen(redirected, **kwargs)


def disabled():
    raise RuntimeError('Physical execution disabled pending main review and explicit diagnostic identity contract')


def compare_control(before, after, repeated):
    for label in ('original', 'raw'):
        a, b = before[label], after[label]
        keys = ('status', 'success', 'steps', 'hashes', 'binding', 'noise_hashes', 'chunks')
        if any(k not in a or k not in b or a[k] != b[k] for k in keys):
            raise ValueError('Control result mismatch')
        # Exact serialized numeric records include actions, explicit noise, state,
        # their hashes, chunk boundaries, and terminal flags. No common-prefix gate.
        left, right = Path(a['_directory']), Path(b['_directory'])
        if not (left/'trace.jsonl').stat().st_size or sha(left/'trace.jsonl') != sha(right/'trace.jsonl'):
            raise ValueError('Control trace mismatch')
        if repeated and sha(left/'physical.jsonl') != sha(right/'physical.jsonl'):
            raise ValueError('Repeated physical trace mismatch')


def run(runner, contract_path, digest):
    c, m = validate_contract(runner, contract_path, digest)
    import controlled_paired_eval as config
    import gen1_raw_diagnostic as resources
    for trial in range(4):
        root = runner.OUTPUTS / f"{c['prefix']}_{trial}"
        if root.exists() or root.is_symlink(): raise ValueError('Fresh trial roots required')

    def execute(trial, index, case, enabled):
        validate_contract(runner, contract_path, digest)
        output = runner.OUTPUTS / f"{c['prefix']}_{trial}"
        runner.guard(output, 64*2**20)
        output.mkdir()
        pair = {}
        for label in ('original', 'raw'):
            directory = output / (case['id'] + '_' + label)
            directory.mkdir()
            command = [config.CONFIG['inference_python'], '-I', '-B', str(WRAPPER), 'worker',
                       '--contract', str(contract_path), '--digest', digest,
                       '--trial', str(trial), '--label', label]
            with (directory/'worker.log').open('x') as log:
                evidence = runner.supervise(command, runner.runtime_environment(), log, output, resources)
            runner.checked_write(directory/'supervision.json', evidence, output)
            if evidence['status'] != 'passed': raise RuntimeError('Worker supervision failed')
            result = json.loads((directory/'result.json').read_text())
            if result.get('status') != 'completed': raise ValueError('Admission exclusion or worker error')
            result['_directory'] = str(directory)
            pair[label] = result
            validate_contract(runner, contract_path, digest)
        runner.checked_write(output/'summary.json', dict(limitation=LIMITATION, pair=pair), output)
        return pair
    return run_schedule(m, execute, runner.pair_gate, compare_control)


def dispatch_worker(runner, contract_path, digest, trial, label):
    c, m = validate_contract(runner, contract_path, digest)
    if type(trial) is not int or not 0 <= trial < 4 or label not in ('original', 'raw'):
        raise ValueError('Unexpected worker selection')
    _, case, enabled = schedule(m)[trial]
    output = runner.OUTPUTS / f"{c['prefix']}_{trial}"
    runner.check_dependency_versions()
    os.environ.update(runner.runtime_environment())
    sys.path[:0] = [str(runner.OPENPI/'src'), str(runner.OPENPI/'packages/openpi-client/src')]
    import controlled_paired_eval as config
    proxy = SimulatorProxy(runner.subprocess, simulator_command(runner, config.CONFIG, case),
                           config.CONFIG['robotwin_root'], enabled,
                           dict(contract=str(contract_path), digest=digest, trial=trial, label=label))
    with runner.replaced(runner, 'subprocess', proxy):
        return runner.worker(m, label, case, output)


def dispatch_simulator(runner, mode):
    identity = json.loads(os.environ['GEN1_PHYSICAL_IDENTITY'])
    c, m = validate_contract(runner, identity['contract'], identity['digest'])
    trial, label = identity['trial'], identity['label']
    if type(trial) is not int or not 0 <= trial < 4 or label not in ('original', 'raw'):
        raise ValueError('Unexpected simulator selection')
    _, case, enabled = schedule(m)[trial]
    if mode != ('sim-on' if enabled else 'sim-off'): raise ValueError('Observer mode mismatch')
    directory = runner.OUTPUTS / f"{c['prefix']}_{trial}" / (case['id'] + '_' + label)
    if os.environ.get('ROBOTWIN_CONTROLLED_OUTPUT') != str(directory):
        raise ValueError('Simulator output mismatch')
    import controlled_paired_eval as config
    expected = simulator_command(runner, config.CONFIG, case)
    if sys.argv[2:] != expected[5:]:
        raise ValueError('Simulator arguments changed')
    if os.environ.get('GEN1_RAW_TEMPLATE_SPEC') != json.dumps(case['template']):
        raise ValueError('Simulator template changed')
    original = runner.simulator_shim
    observer = Observer(runner, directory, enabled, case['seed'])
    def shim(evaluator, harness, spec, descriptions):
        return observed_shim(runner, original, evaluator, harness, spec, descriptions, observer)
    del sys.argv[1]
    with runner.replaced(runner, 'simulator_shim', shim):
        return runner.simulator()


def main():
    # Changing this source invalidates any previously prepared identity contract.
    # No CLI switch or environment variable can enable execution.
    if not EXECUTION_ENABLED: disabled()
    import argparse
    sys.path.insert(0, str(WRAPPER.parent))
    runner = load_frozen()
    if len(sys.argv) > 1 and sys.argv[1] in ('sim-on', 'sim-off'):
        return dispatch_simulator(runner, sys.argv[1])
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--digest')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prefix')
    parser.add_argument('--trial', type=int)
    parser.add_argument('--label', choices=('original', 'raw'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare_contract(runner, args.manifest, args.contract, args.prefix)
        return 0
    if args.mode == 'run':
        run(runner, args.contract, args.digest)
        return 0
    return dispatch_worker(runner, args.contract, args.digest, args.trial, args.label)


if __name__ == '__main__':
    raise SystemExit(main())
