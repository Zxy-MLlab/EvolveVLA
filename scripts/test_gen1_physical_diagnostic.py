"""CPU-only tests: real frozen shim/worker/simulator interfaces, fake physics/IPC."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import pytest

PATH = Path(__file__).with_name('gen1_physical_diagnostic.py')
spec = importlib.util.spec_from_file_location('physical', PATH)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
r = d.load_frozen()


def task():
    joint = types.SimpleNamespace(get_name=lambda: 'finger')
    entity = types.SimpleNamespace(get_active_joints=lambda: [joint], get_qpos=lambda: [.03])
    robot = types.SimpleNamespace()
    for side in ('left', 'right'):
        setattr(robot, side + '_entity', entity)
        setattr(robot, side + '_gripper', [(joint, 1, 0)])
        setattr(robot, 'get_' + side + '_arm_real_jointState', lambda: [1]*6 + [999])
        setattr(robot, 'get_' + side + '_arm_jointState', lambda: [2]*6 + [.9])
        setattr(robot, 'get_' + side + '_tcp_pose', lambda: [.1, .2, .3, 1, 0, 0, 0])
    actor = types.SimpleNamespace(get_pose=lambda: types.SimpleNamespace(p=[.1,.2,.9], q=[1,0,0,0]),
                                 get_functional_point=lambda i: [.1,.2,.9])
    return types.SimpleNamespace(robot=robot, mug=actor, rack=actor, eval_success=False,
                                 play_once=lambda: {'info': {}}, set_instruction=lambda **kw: None)


def start():
    return dict(type='episode_start', seed=200000, instruction='template')


def step(i=1, terminal=True):
    return dict(type='step', observation=None, info=dict(step=i, is_success=False),
                terminated=False, truncated=terminal)


def observer(tmp_path, enabled=True):
    guards = []
    runner = types.SimpleNamespace(guard=lambda *args: guards.append(args))
    return d.Observer(runner, tmp_path, enabled, 200000), guards


def test_hash_without_file_digest(tmp_path):
    path = tmp_path/'identity.bin'
    data = b'physical-identity' * 100000
    path.write_bytes(data)
    with patch.object(hashlib, 'file_digest', None, create=True):
        assert d.sha(path) == hashlib.sha256(data).hexdigest()
        loaded = d.load_frozen()
        assert loaded.__file__ == str(d.FROZEN)
        assert hashlib.file_digest is None
        subtree = tmp_path/'params'
        subtree.mkdir()
        model_file = subtree/'weights.bin'
        model_file.write_bytes(data)
        assert loaded.file_identity(tmp_path, ['params']) == {
            str(model_file): hashlib.sha256(data).hexdigest()}
        assert loaded.hashlib.sha256(b'unchanged').digest() == hashlib.sha256(b'unchanged').digest()
        model_file.write_bytes(b'changed')
        assert loaded.file_identity(tmp_path, ['params'])[str(model_file)] != hashlib.sha256(data).hexdigest()
        with model_file.open('rb') as stream, pytest.raises(ValueError):
            loaded.hashlib.file_digest(stream, 'md5')


def test_measured_semantics():
    s = d.sample(task())
    assert s['left']['measured_arm_qpos'] == [1]*6
    assert s['left']['drive_targets_and_gripper_command'] == [2]*6 + [.9]
    assert s['left']['measured_finger_qpos'] == {'finger': .03}


@pytest.mark.parametrize('enabled', [False, True])
def test_real_shim_composition_terminal_and_restore(tmp_path, enabled):
    t = task()
    ev = types.SimpleNamespace(class_decorator=lambda name: t)
    messages = []
    harness = types.SimpleNamespace(send_message=lambda stream, payload: messages.append(payload))
    old_construct, old_send = ev.class_decorator, harness.send_message
    o, guards = observer(tmp_path, enabled)
    with patch.object(r, 'bind_prompt', return_value={'prompt': 'bound'}):
        with d.observed_shim(r, r.simulator_shim, ev, harness, {'template':'template'}, tmp_path, o):
            assert ev.class_decorator('hanging_mug') is t
            t.play_once()
            t.set_instruction('template')
            harness.send_message(None, start())
            harness.send_message(None, step())
    assert ev.class_decorator is old_construct and harness.send_message is old_send
    assert messages[1]['diagnostic_state'] is None
    assert 'physical' not in messages[0]
    assert o.rows == 2 and o.ended
    assert bool(guards) == enabled
    if enabled:
        rows = [json.loads(line) for line in (tmp_path/'physical.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == [0, 1]
        assert rows[-1]['physical']['mug_world_xyz'] == [.1,.2,.9]
    else:
        assert not list(tmp_path.iterdir())


def test_off_does_not_read_physics(tmp_path):
    o, _ = observer(tmp_path, False)
    o.capture(None, start())
    o.capture(None, step())


@pytest.mark.parametrize('failure', ['nan', 'infinite', 'missing', 'getter'])
def test_physical_errors_fail_closed(tmp_path, failure):
    t = task()
    if failure == 'nan':
        t.robot.get_left_tcp_pose = lambda: [float('nan'),0,0]
    elif failure == 'infinite':
        t.robot.left_entity.get_qpos = lambda: [float('inf')]
    elif failure == 'missing':
        t.mug.get_functional_point = lambda i: None
    else:
        def broken(): raise RuntimeError('read error')
        t.robot.get_left_arm_real_jointState = broken
    o, _ = observer(tmp_path)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        o.capture(t, start())
    assert not list(tmp_path.iterdir())


def test_caps_and_protocol(tmp_path):
    o, _ = observer(tmp_path, False)
    with pytest.raises(ValueError): o.capture(None, step())
    o.capture(None, start())
    for i in range(1, 901): o.capture(None, step(i, i == 900))
    assert o.rows == 901
    with pytest.raises(ValueError): o.capture(None, step(901))
    with pytest.raises(ValueError): o.capture(None, start())
    with pytest.raises(ValueError): o.capture(None, {'type':'case_status'})
    o, _ = observer(tmp_path)
    with patch.object(d, 'MAX_BYTES', 1), pytest.raises(ValueError): o.capture(task(), start())
    assert not list(tmp_path.iterdir())


def test_guard_failure(tmp_path):
    o, _ = observer(tmp_path)
    def fail(*args): raise RuntimeError('storage')
    o.runner.guard = fail
    with pytest.raises(RuntimeError): o.capture(task(), start())
    assert not list(tmp_path.iterdir())


def test_incomplete_shim(tmp_path):
    ev = types.SimpleNamespace(class_decorator=lambda: task())
    h = types.SimpleNamespace(send_message=lambda *args: None)
    o, _ = observer(tmp_path, False)
    with pytest.raises(ValueError, match='Incomplete'):
        with d.observed_shim(r, r.simulator_shim, ev, h, {}, tmp_path, o): pass


def test_proxy_against_real_worker(tmp_path):
    case = dict(id='hanging_mug_200000', task='hanging_mug', seed=200000, template={'template':'hello'})
    config = dict(sim_python='/sim/python', episode_steps=900, robotwin_root='/repo', task_config='clean')
    calls = []
    child = types.SimpleNamespace(stdin=types.SimpleNamespace(close=lambda: None), stdout=object(),
                                  wait=lambda **kw: 0, returncode=0)
    native = types.SimpleNamespace(PIPE=-1, Popen=lambda cmd, **kw: calls.append((cmd,kw)) or child)
    proxy = d.SimulatorProxy(native, d.simulator_command(r, config, case), '/repo', True)
    modules = {
        'controlled_paired_eval':types.SimpleNamespace(CONFIG=config),
        'robotwin_public_openpi_harness':types.SimpleNamespace(_configure_sim_renderer_env=lambda env: None),
        'robotwin_eval_harness':types.SimpleNamespace(receive_message=lambda stream: {}, send_message=lambda *a: None)}
    identity = r.__file__
    before = hashlib.sha256(d.FROZEN.read_bytes()).hexdigest()
    with patch.dict(sys.modules, modules), contextlib.ExitStack() as stack:
        for name, value in dict(subprocess=proxy, runtime_environment=lambda: {}, restrict_worker_files=lambda:None,
                                paired_episode=lambda *a: {'status':'completed'}, checked_write=lambda *a:None).items():
            stack.enter_context(patch.object(r, name, value))
        assert r.worker({}, 'original', case, tmp_path) == 0
    assert calls[0][0][3:5] == [str(d.WRAPPER), 'sim-on']
    assert calls[0][1]['env']['GEN1_RAW_TEMPLATE_SPEC'] == json.dumps(case['template'])
    assert r.__file__ == identity
    assert hashlib.sha256(d.FROZEN.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError): proxy.Popen(proxy.expected, **calls[0][1])
    for mutation in ('argv', 'cwd', 'extra'):
        proxy = d.SimulatorProxy(native, ['python','-I','-B','frozen','sim'], '/repo', False)
        cmd = list(proxy.expected)
        kw = dict(stdin=-1, stdout=-1, env={}, cwd='/repo')
        if mutation == 'argv': cmd.append('--unexpected')
        elif mutation == 'cwd': kw['cwd'] = '/other'
        else: kw['shell'] = True
        with pytest.raises(ValueError): proxy.Popen(cmd, **kw)


def test_frozen_deferred_import(tmp_path):
    import importlib
    events = []
    evaluator = types.SimpleNamespace(__file__=r.ROOT/'robotwin_repo_ba0fd21/scripts/eval_policy_xpolicylab.py')
    @contextlib.contextmanager
    def shim(*args):
        events.append('attach')
        yield
        events.append('detach')
    def main():
        events.append('native_stdout_redirect')
        importlib.import_module('eval_policy_xpolicylab')
        importlib.import_module('eval_policy_xpolicylab')
        events.append('run')
        return 0
    h = types.SimpleNamespace(main=main)
    old_path = list(sys.path)
    try:
        with patch.dict(sys.modules, {'robotwin_eval_harness':h}), patch.dict(r.os.environ, GEN1_RAW_TEMPLATE_SPEC='{}'), \
             patch.object(r.os, 'chdir'), patch.object(importlib, 'import_module', return_value=evaluator), \
             patch.object(r, 'simulator_shim', shim):
            assert r.simulator() == 0
    finally:
        sys.path[:] = old_path
    assert events == ['native_stdout_redirect','attach','run','detach']


def test_schedule_and_stop():
    m = {'cases':[{'task':'hanging_mug', 'seed':s} for s in (200000,200001)]}
    calls, comparisons = [], []
    def execute(*args):
        calls.append(args)
        return {k:{'status':'completed'} for k in ('original','raw')}
    d.run_schedule(m, execute, lambda *a:'valid', lambda *a:comparisons.append(a))
    assert [(c[2]['seed'], c[3]) for c in calls] == list(d.SCHEDULE)
    assert len(comparisons) == 2 and comparisons[1][2] is True
    for status in ('expert_rejected','worker_error'):
        calls.clear()
        def fail(*args):
            calls.append(args)
            return {k:{'status':status} for k in ('original','raw')}
        with pytest.raises(ValueError): d.run_schedule(m, fail, r.pair_gate, lambda *a:None)
        assert len(calls) == 1
    calls.clear()
    def mismatch(*args): raise ValueError('control mismatch')
    with pytest.raises(ValueError): d.run_schedule(m, execute, lambda *a:'valid', mismatch)
    assert len(calls) == 2


@pytest.fixture
def contract_fixture(tmp_path):
    outputs = tmp_path/'outputs'
    outputs.mkdir()
    historical = outputs/'historical.json'
    historical.write_text('{}')
    manifest = outputs/'original.json'
    manifest.write_text('{}')
    known = tmp_path/d.KNOWN_ROOTS[0]
    known.mkdir()
    (known/'old.json').write_text('{}')
    (known/'image.npy').write_bytes(b'binary')
    m = {'cases':[dict(task='hanging_mug', seed=s, id=f'hanging_mug_{s}') for s in (200000,200001)]}
    calls = []
    def verify(*args):
        calls.append(args)
        assert d.sha(historical) == hashlib.sha256(b'{}').hexdigest()
        return m
    def write(path, data, output): path.write_text(json.dumps(data))
    fake = types.SimpleNamespace(ROOT=tmp_path, OUTPUTS=outputs,
        HISTORICAL_FILES={'outputs/historical.json':'unused'},
        historical_paths=lambda: {str(p.relative_to(tmp_path)) for p in outputs.rglob('*.json')},
        verified_manifest=verify, checked_write=write)
    path = outputs/'contract.json'
    with patch.object(d, 'ORIGINAL_SHA', d.sha(manifest)):
        c = d.prepare_contract(fake, manifest, path, 'physical')
        yield fake, path, d.sha(path), c, calls


def test_contract_known_delta_and_later_mutations(contract_fixture):
    fake, path, digest, c, calls = contract_fixture
    assert any(p.endswith('image.npy') for p in c['added'])
    assert d.validate_contract(fake, path, digest)[0] == c
    assert len(calls) == 2
    faults = [name for name in c['allowed'] if name.endswith('/faulthandler_sim.txt')]
    assert len(faults) == 8
    for name in faults:
        fault = fake.ROOT/name
        fault.parent.mkdir(parents=True, exist_ok=True)
        fault.write_text('simulator diagnostic')
    d.validate_contract(fake, path, digest)
    allowed = fake.ROOT/c['allowed'][0]
    allowed.parent.mkdir(parents=True, exist_ok=True)
    allowed.write_text('{}')
    d.validate_contract(fake, path, digest)
    unexpected = allowed.parent/'unexpected.npy'
    unexpected.write_bytes(b'x')
    with pytest.raises(ValueError, match='Unexpected diagnostic'): d.validate_contract(fake, path, digest)
    unexpected.unlink()
    historical = fake.ROOT/d.KNOWN_ROOTS[0]/'image.npy'
    historical.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Snapshot changed'): d.validate_contract(fake, path, digest)
    historical.unlink()
    with pytest.raises(ValueError, match='Snapshot changed'): d.validate_contract(fake, path, digest)


def test_contract_rejects_unrelated_addition(contract_fixture):
    fake, path, digest, c, calls = contract_fixture
    (fake.OUTPUTS/'unrelated.json').write_text('{}')
    with pytest.raises(ValueError, match='Unexpected later'): d.validate_contract(fake, path, digest)
    with pytest.raises(ValueError, match='Unexpected historical'):
        d.prepare_contract(fake, Path(c['original']), fake.OUTPUTS/'other.json', 'other')


def test_exact_control_trace_and_physical(tmp_path):
    import numpy as np
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir(); b.mkdir()
    noise = r.array_record(np.zeros((50,32)), [50,32])
    actions = r.array_record(np.zeros((50,14)), [50,14])
    rows = [dict(kind='chunk', index=0, before_step=0, noise=noise, actions=actions),
            dict(kind='step', index=1, state=None, terminated=False, truncated=True, success=False)]
    for directory in (a,b):
        (directory/'trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (directory/'physical.jsonl').write_text('{"step":1}\n')
    result = dict(status='completed', success=False, steps=1, hashes={'state':'abc'},
                  binding={'prompt':'bound'}, noise_hashes=[noise['sha256']], chunks=1)
    before = {label:dict(result, _directory=str(a)) for label in ('original','raw')}
    after = {label:dict(result, _directory=str(b)) for label in ('original','raw')}
    d.compare_control(before, after, True)
    rows[0]['actions']['values'][0][0] = 1
    (b/'trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='trace mismatch'): d.compare_control(before, after, False)


def test_real_supervisor_log_and_failure(tmp_path):
    resources = types.SimpleNamespace(discover_cgroups=lambda: {}, memory_snapshot=lambda *a: {},
                                     check_memory=lambda *a, **kw: None)
    with patch.object(r, 'guard'), patch.object(r, 'kill_group') if hasattr(r, 'kill_group') else contextlib.nullcontext():
        with (tmp_path/'worker.log').open('x') as log:
            result = r.supervise([sys.executable, '-I', '-B', '-c', 'print("cpu log")'],
                                 {}, log, tmp_path, resources)
        assert result['status'] == 'passed'
        assert 'cpu log' in (tmp_path/'worker.log').read_text()
        with (tmp_path/'failed.log').open('x') as log:
            result = r.supervise([sys.executable, '-I', '-B', '-c', 'raise SystemExit(7)'],
                                 {}, log, tmp_path, resources)
        assert result['status'] == 'failed'


def test_coordinator_stops_on_supervision_failure(contract_fixture):
    fake, path, digest, c, calls = contract_fixture
    fake.guard = lambda *a: None
    fake.runtime_environment = lambda: {}
    launched = []
    def supervise(command, env, log, output, resources):
        assert hasattr(log, 'buffer')
        launched.append(command)
        return {'status':'failed'}
    fake.supervise = supervise
    fake.pair_gate = r.pair_gate
    modules = {'controlled_paired_eval': types.SimpleNamespace(CONFIG={'inference_python':sys.executable}),
               'gen1_raw_diagnostic': types.SimpleNamespace()}
    with patch.dict(sys.modules, modules), pytest.raises(RuntimeError, match='supervision failed'):
        d.run(fake, path, digest)
    assert len(launched) == 1
    assert '--trial' in launched[0] and launched[0][-1] == 'original'


@pytest.mark.parametrize('trial', [0, 1])
def test_simulator_dispatch_contract_and_restore(tmp_path, trial):
    case = dict(id='hanging_mug_200000', task='hanging_mug', seed=200000,
                template={'template': 'hello'})
    manifest = {'cases': [case, dict(case, seed=200001)]}
    config = dict(sim_python='/sim/python', episode_steps=900,
                  robotwin_root='/repo', task_config='clean')
    mode = 'sim-on' if trial else 'sim-off'
    directory = tmp_path/f'physical_{trial}'/(case['id'] + '_original')
    directory.mkdir(parents=True)
    identity = dict(contract='contract', digest='digest', trial=trial, label='original')
    env = dict(GEN1_PHYSICAL_IDENTITY=json.dumps(identity),
               ROBOTWIN_CONTROLLED_OUTPUT=str(directory),
               GEN1_RAW_TEMPLATE_SPEC=json.dumps(case['template']))
    original = r.simulator_shim
    def simulator():
        assert sys.argv[1:] == d.simulator_command(r, config, case)[5:]
        evaluator = types.SimpleNamespace(class_decorator=lambda name: task())
        harness = types.SimpleNamespace(send_message=lambda *a: None)
        with r.simulator_shim(evaluator, harness, case['template'], tmp_path):
            t = evaluator.class_decorator('hanging_mug')
            t.play_once()
            t.set_instruction('hello')
            harness.send_message(None, dict(start(), instruction='hello'))
            harness.send_message(None, step())
        return 0
    argv = [str(PATH), mode] + d.simulator_command(r, config, case)[5:]
    with patch.object(d, 'validate_contract', return_value=({'prefix': 'physical'}, manifest)) as validate, \
         patch.object(r, 'OUTPUTS', tmp_path), patch.object(r, 'guard'), \
         patch.object(r, 'bind_prompt', return_value={'prompt': 'bound'}), \
         patch.object(r, 'simulator', simulator), patch.object(sys, 'argv', argv), \
         patch.dict(r.os.environ, env), \
         patch.dict(sys.modules, {'controlled_paired_eval': types.SimpleNamespace(CONFIG=config)}):
        assert d.dispatch_simulator(r, mode) == 0
        validate.assert_called_once_with(r, 'contract', 'digest')
    assert r.simulator_shim is original
    assert (directory/'physical.jsonl').exists() == bool(trial)


@pytest.mark.parametrize('mutation', ['mode', 'output', 'argv', 'template', 'trial', 'label'])
def test_simulator_dispatch_rejects_mismatch(tmp_path, mutation):
    case = dict(id='hanging_mug_200000', task='hanging_mug', seed=200000,
                template={'template': 'hello'})
    manifest = {'cases': [case, dict(case, seed=200001)]}
    config = dict(sim_python='/sim/python', episode_steps=900,
                  robotwin_root='/repo', task_config='clean')
    identity = dict(contract='contract', digest='digest', trial=0, label='original')
    env = dict(ROBOTWIN_CONTROLLED_OUTPUT=str(tmp_path/'physical_0'/ (case['id']+'_original')),
               GEN1_RAW_TEMPLATE_SPEC=json.dumps(case['template']))
    argv = [str(PATH), 'sim-off'] + d.simulator_command(r, config, case)[5:]
    if mutation == 'output': env['ROBOTWIN_CONTROLLED_OUTPUT'] = '/wrong'
    if mutation == 'template': env['GEN1_RAW_TEMPLATE_SPEC'] = '{}'
    if mutation == 'argv': argv.append('--unexpected')
    if mutation == 'trial': identity['trial'] = True
    if mutation == 'label': identity['label'] = 'ema'
    env['GEN1_PHYSICAL_IDENTITY'] = json.dumps(identity)
    with patch.object(d, 'validate_contract', return_value=({'prefix': 'physical'}, manifest)), \
         patch.object(r, 'OUTPUTS', tmp_path), patch.object(r, 'simulator') as simulator, \
         patch.object(sys, 'argv', argv), patch.dict(r.os.environ, env), \
         patch.dict(sys.modules, {'controlled_paired_eval': types.SimpleNamespace(CONFIG=config)}):
        with pytest.raises(ValueError):
            d.dispatch_simulator(r, 'sim-on' if mutation == 'mode' else 'sim-off')
        simulator.assert_not_called()


def test_worker_dispatch_identity_and_restore(tmp_path):
    case = dict(id='hanging_mug_200000', task='hanging_mug', seed=200000,
                template={'template': 'hello'})
    manifest = {'cases': [case, dict(case, seed=200001)]}
    config = dict(sim_python='/sim/python', episode_steps=900,
                  robotwin_root='/repo', task_config='clean')
    native = r.subprocess
    def worker(m, label, selected, output):
        assert m is manifest and selected is case and label == 'raw'
        assert output == tmp_path/'physical_1'
        assert isinstance(r.subprocess, d.SimulatorProxy)
        assert r.subprocess.original is native and r.subprocess.enabled
        assert r.subprocess.identity == dict(contract='contract', digest='digest', trial=1, label='raw')
        return 0
    old_path = list(sys.path)
    try:
        with patch.object(d, 'validate_contract', return_value=({'prefix': 'physical'}, manifest)), \
             patch.object(r, 'OUTPUTS', tmp_path), patch.object(r, 'check_dependency_versions') as check, \
             patch.object(r, 'runtime_environment', return_value={}), patch.object(r, 'worker', worker), \
             patch.dict(sys.modules, {'controlled_paired_eval': types.SimpleNamespace(CONFIG=config)}):
            assert d.dispatch_worker(r, 'contract', 'digest', 1, 'raw') == 0
            check.assert_called_once_with()
    finally:
        sys.path[:] = old_path
    assert r.subprocess is native


def test_disabled_cli_exits_nonzero():
    import subprocess
    code = ("import importlib.util; "
            f"s=importlib.util.spec_from_file_location('physical', {str(PATH)!r}); "
            "d=importlib.util.module_from_spec(s); s.loader.exec_module(d); "
            "d.EXECUTION_ENABLED=False; d.main()")
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'execution disabled' in result.stderr


@pytest.mark.parametrize('mode', ['prepare','run','worker','sim-on','sim-off'])
def test_execution_disabled(mode):
    with patch.object(d, 'EXECUTION_ENABLED', False), \
         patch.object(sys, 'argv', [str(PATH), mode]), pytest.raises(RuntimeError, match='disabled'):
        d.main()
