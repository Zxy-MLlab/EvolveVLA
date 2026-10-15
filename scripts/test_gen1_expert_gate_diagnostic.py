import importlib.util
from pathlib import Path
import types
import json
import pytest

PATH = Path(__file__).with_name('gen1_expert_gate_diagnostic.py')
spec = importlib.util.spec_from_file_location('diag', PATH)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class Pose:
    def __init__(self, p, q=(1, 0, 0, 0)):
        self.p, self.q = p, q


class Actor:
    def __init__(self, pose, functional):
        self.pose, self.functional = pose, functional
        self.calls = []
    def get_pose(self):
        self.calls.append('pose')
        return self.pose
    def get_functional_point(self, idx):
        self.calls.append(('functional', idx))
        return self.functional


class Task:
    def __init__(self, result=True, plan=True):
        self.mug = Actor(Pose([0.5, 0.5, 0.9]), [0.5, 0.5, 0.9])
        self.rack = Actor(Pose([0.5, 0.5, 0.8]), [0.5, 0.5, 0.8])
        self.result, self.plan_success = result, plan
        self.calls = []
        self.closed = 0
    def setup_demo(self, *args, **kwargs): self.calls.append('setup')
    def play_once(self): self.calls.append('play'); return {'info': {}}
    def check_success(self):
        self.calls.append('check')
        m = self.mug.get_functional_point(0)[:3]
        r = self.rack.get_pose().p
        f = self.rack.get_functional_point(0)[:3]
        return self.result and all(abs(m[i] - (r[i] + f[i]) / 2) < .02 for i in (0, 1)) and self.is_right_gripper_open() and m[2] > .86
    def is_right_gripper_open(self): self.calls.append('gripper'); return True
    def close_env(self): self.closed += 1


def test_tapped_check_preserves_result_and_calls():
    t = Task(result=True)
    record = {}
    with diag.instrument_task(t, record):
        t.setup_demo(seed=1)
        t.play_once()
        assert t.check_success() is True
    assert t.calls == ['setup', 'play', 'check', 'gripper']
    assert record['calls'] == {'setup': 1, 'play': 1, 'check': 1}
    assert record['check_success'] is True
    assert record['predicate_inputs']['right_gripper_open'] is True


def test_short_circuit_preserved():
    t = Task(result=False)
    record = {}
    with diag.instrument_task(t, record):
        t.setup_demo(seed=1)
        t.play_once()
        assert t.check_success() is False
    assert t.calls == ['setup', 'play', 'check']
    assert 'right_gripper_open' not in record['predicate_inputs']
    assert record['predicate_margins']['gripper_check_evaluated'] is False


def test_context_restores_methods_and_cleanup():
    t = Task()
    original = t.check_success
    record = {}
    with diag.instrument_task(t, record):
        t.setup_demo(seed=1)
        t.play_once()
        t.check_success()
    assert t.check_success.__func__ is original.__func__
    t.close_env()
    assert t.closed == 1


def test_execution_disabled_and_exact_schedule(monkeypatch):
    monkeypatch.setattr(diag, 'EXECUTION_ENABLED', False)
    assert diag.REPETITIONS == (0, 1, 2, 3)
    try:
        diag.require_enabled()
    except RuntimeError as exc:
        assert 'disabled' in str(exc)
    else:
        raise AssertionError('diagnostic unexpectedly enabled')


@pytest.mark.parametrize('result,plan', [(True, True), (False, True), (False, False)])
def test_harness_stops_before_policy(result, plan):
    task = Task(result, plan)
    evaluator = types.SimpleNamespace(class_decorator=lambda name: task)
    harness = types.SimpleNamespace(safe_close_task=lambda t: t.close_env())
    def run(args):
        t = evaluator.class_decorator(diag.TASK)
        try:
            t.setup_demo()
            t.play_once()
            if t.plan_success:
                bool(t.check_success())
            harness.safe_close_task(t)
            raise AssertionError('Policy path reached')
        finally:
            harness.safe_close_task(t)
    harness.run_simulator = run
    record = {}
    diag.run_harness(harness, evaluator, None, record)
    assert record['status'] == 'expert_only_finished'
    assert task.closed == 1
    assert record['calls'] == {'setup': 1, 'play': 1, 'check': int(plan)}
    assert not any(name in task.__dict__ for name in ('setup_demo', 'play_once', 'check_success'))


@pytest.mark.parametrize('stage', ['setup_demo', 'play_once', 'check_success'])
def test_errors_are_not_successfully_stopped(stage):
    task = Task()
    def fail(*args, **kwargs):
        raise ValueError(stage)
    setattr(task, stage, fail)
    evaluator = types.SimpleNamespace(class_decorator=lambda name: task)
    harness = types.SimpleNamespace(safe_close_task=lambda t: t.close_env())
    def run(args):
        t = evaluator.class_decorator(diag.TASK)
        try:
            t.setup_demo()
            t.play_once()
            t.check_success()
        finally:
            harness.safe_close_task(t)
    harness.run_simulator = run
    record = {}
    with pytest.raises(ValueError, match=stage):
        diag.run_harness(harness, evaluator, None, record)
    assert 'status' not in record
    assert task.closed == 1
    assert getattr(task, stage) is fail


def test_internal_checks_preserved():
    task = Task()
    def play():
        task.calls.append('play')
        return task.check_success()
    task.play_once = play
    record = {}
    with diag.instrument_task(task, record):
        task.setup_demo()
        assert task.play_once() is True
        assert task.check_success() is True
    assert record['checks_during_play'] == 1
    assert record['calls']['check'] == 1
    assert task.calls.count('check') == 2


def test_disabled_entrypoints(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, 'EXECUTION_ENABLED', False)
    with pytest.raises(RuntimeError, match='disabled'):
        diag.worker(tmp_path, 0)
    with pytest.raises(RuntimeError, match='disabled'):
        diag.run(tmp_path)


def test_getter_counts_and_boolean_identity():
    task = Task()
    record = {}
    with diag.instrument_task(task, record):
        task.setup_demo()
        task.play_once()
        assert task.check_success() is True
    assert task.mug.calls == ['pose', ('functional', 0)]
    assert task.rack.calls == ['pose', 'pose', ('functional', 0)]
    class TruthBomb:
        def __bool__(self):
            raise AssertionError('Unexpected truth evaluation')
    with pytest.raises(diag.DiagnosticFailure):
        diag.scalar_bool(TruthBomb())


def test_worker_runtime_wiring(monkeypatch, tmp_path):
    import importlib
    events = []
    monkeypatch.setattr(diag, 'EXECUTION_ENABLED', True)
    monkeypatch.setattr(diag, 'verify_frozen_sources', lambda: events.append('verify'))
    monkeypatch.setattr(diag.frozen, 'restrict_worker_files', lambda: events.append('restrict'))
    monkeypatch.setattr(diag.os, 'chdir', lambda path: events.append('cwd'))
    monkeypatch.setattr(diag.sys, 'path', list(diag.sys.path))
    monkeypatch.setattr(diag.frozen, 'checked_write', lambda p, v, o: p.write_text(json.dumps(v)))
    task = Task()
    evaluator = types.SimpleNamespace(
        __file__=str(diag.REPO / 'scripts/eval_policy_xpolicylab.py'),
        class_decorator=lambda name: task)
    harness = types.SimpleNamespace(safe_close_task=lambda t: t.close_env())
    def run(args):
        assert events == ['verify', 'restrict', 'cwd']
        events.append('harness')
        module = importlib.import_module('eval_policy_xpolicylab')
        t = module.class_decorator(args.task)
        try:
            t.setup_demo(seed=args.seed)
            t.play_once()
            t.check_success()
            harness.safe_close_task(t)
            raise AssertionError('Policy path reached')
        finally:
            harness.safe_close_task(t)
    harness.run_simulator = run
    monkeypatch.setitem(diag.sys.modules, 'robotwin_eval_harness', harness)
    monkeypatch.setitem(diag.sys.modules, 'eval_policy_xpolicylab', evaluator)
    original_import = importlib.import_module
    diag.worker(tmp_path, 0)
    assert importlib.import_module is original_import
    result = json.loads((tmp_path / 'diagnostic.json').read_text())
    assert result['status'] == 'expert_only_finished'
    assert task.closed == 1


@pytest.mark.parametrize('failure_at', [None, 1])
def test_four_fresh_commands_no_replacement(monkeypatch, tmp_path, failure_at):
    monkeypatch.setattr(diag, 'EXECUTION_ENABLED', True)
    monkeypatch.setattr(diag, 'verify_frozen_sources', lambda: {'config': {'sim_python': '/mock/python'}})
    monkeypatch.setattr(diag.frozen, 'guard', lambda *args: None)
    renderer = types.SimpleNamespace(_renderer_metadata=lambda: {})
    def configure(env):
        assert renderer._renderer_metadata()['render_backend'] == 'rt'
        env['LD_PRELOAD'] = 'mock-renderer'
    renderer._configure_sim_renderer_env = configure
    monkeypatch.setitem(diag.sys.modules, 'robotwin_public_openpi_harness', renderer)
    monkeypatch.setattr(diag.frozen, 'runtime_environment', lambda: {
        'ROBOTWIN_RENDER_BACKEND': 'rt', 'ROBOTWIN_RT_DENOISER': 'oidn',
        'ROBOTWIN_OIDN_RUNTIME': '/mock/oidn'})
    monkeypatch.setattr(diag.frozen, 'checked_write', lambda p, v, o: p.write_text(json.dumps(v)))
    commands = []
    handles = []
    def supervise(command, env, log, output, resources, timeout):
        assert hasattr(log, 'buffer') and callable(log.flush)
        handles.append(log)
        index = len(commands)
        commands.append(command)
        assert timeout == 1800
        assert env['LD_PRELOAD'] == 'mock-renderer'
        # Exercise the exact frozen writer with the coordinator's handle.
        import os
        import selectors
        reader, writer = os.pipe()
        try:
            os.write(writer, b'cpu-log-check\n')
            with selectors.DefaultSelector() as selector:
                selector.register(reader, selectors.EVENT_READ)
                state = {'bytes': 0, 'truncated': False}
                diag.frozen.drain_log(None, log, output, state, selector)
                assert state['bytes'] == len(b'cpu-log-check\n')
        finally:
            os.close(reader)
            os.close(writer)
        directory = Path(env['ROBOTWIN_CONTROLLED_OUTPUT'])
        assert command[-1] == str(directory)
        assert command[-3] == str(index)
        (directory / 'diagnostic.json').write_text(json.dumps({
            'status': 'expert_only_finished', 'repetition': index,
            'seed': diag.SEED, 'policy_loaded': False}))
        return {'status': 'failed' if index == failure_at else 'passed'}
    output = tmp_path / 'gen1_expert_gate_diagnostic_mock'
    diag.run(output, supervisor=supervise)
    assert len(commands) == (4 if failure_at is None else 2)
    assert len({command[-1] for command in commands}) == len(commands)
    assert all(handle.closed for handle in handles)
    assert all(Path(handle.name).read_bytes() == b'cpu-log-check\n' for handle in handles)
    summary = json.loads((output / 'diagnostic_summary.json').read_text())
    assert summary['status'] == ('complete' if failure_at is None else 'aborted')


@pytest.mark.parametrize('status', ['complete', 'aborted'])
def test_cli_exit_status(monkeypatch, tmp_path, status):
    monkeypatch.setattr(diag, 'EXECUTION_ENABLED', True)
    monkeypatch.setattr(diag.sys, 'argv', ['diagnostic', '--output', str(tmp_path)])
    monkeypatch.setattr(diag, 'run', lambda output: {'status': status})
    if status == 'aborted':
        with pytest.raises(SystemExit) as exc:
            diag.main()
        assert exc.value.code == 1
    else:
        diag.main()
