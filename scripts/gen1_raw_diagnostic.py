#!/usr/bin/env python3
"""External raw-weight offline diagnostic; prepare is standard-library only.

No training, rollout, optimizer restore, checkpoint rewrite, or automatic retry.
GPU execution is deliberately a separate, explicitly authorized operation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
SCRIPT = Path(__file__).resolve()
FIXED = ROOT / 'outputs/gen1_diagnostic_prepare_cpu_v3_20261011/manifest.json'
FIXED_SHA = 'cfac4604b412c1da93a938996634ca19c351aa54ad853e8d5434e64162473dc0'
PREVIOUS = {mode: ROOT / f'outputs/gen1_diagnostic_gpu_{name}_v3_20261011'
            for mode, name in [('smoke', 'smoke'), ('evaluate', 'full')]}
RAW = ROOT / 'checkpoints/gen1_pilot_2000_20261010/1999/train_state'
GIB = 2**30
RESERVE = 12 * GIB


def frozen():
    spec = importlib.util.spec_from_file_location('raw_frozen_diagnostic', ROOT / 'scripts/gen1_diagnose_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d = frozen()  # This module's imports are standard-library only.
require = d.require


def limitations():
    return [s for s in d.LIMITATIONS if not s.startswith('Raw TrainState weights intentionally omitted:')] + [
        'Raw TrainState params only; optimizer is never selected. Logical leaf tracing does not bound OCDBT physical block reads.',
        'NumPy Orbax deserialization ignores restore_concurrent_gb; no hard memory bound is claimed.',
        'Visible cgroup ancestors only: ancestors hidden by a cgroup namespace cannot be measured; host availability is supplemental.',
        'Polling every 0.25 seconds and conservative pre-read headroom cannot prevent all transient allocation spikes.']


def extract_params(flat):
    result = {}
    for key, value in flat.items():
        require(isinstance(key, tuple) and all(isinstance(k, str) for k in key), 'Expected full string tuple keys')
        if not key or key[0] != 'params':
            continue
        require(len(key) >= 3 and key[-1] == 'value', f'Expected final value: {key!r}')
        target = key[1:-1]
        require(target not in result, 'Duplicate parameter')
        result[target] = value
    return result


def validate_reference(values, reference):
    require(set(values) == set(reference), 'Reference keys mismatch')
    for key, (shape, dtype) in reference.items():
        require(tuple(values[key].shape) == tuple(shape), f'Reference shape mismatch: {key!r}')
        require(str(values[key].dtype) == dtype, f'Reference dtype mismatch: {key!r}')


def validate_array(value, shape, dtype):
    import numpy as np
    require(type(value) is np.ndarray, 'Expected native NumPy ndarray')
    require(value.shape == tuple(shape) and str(value.dtype) == dtype, 'Restored shape/dtype mismatch')
    flat = value.reshape(-1)
    for start in range(0, flat.size, 1_000_000):
        require(bool(np.isfinite(flat[start:start + 1_000_000]).all()), 'Nonfinite raw parameter')


def discover_cgroups():
    """Resolve the process's cgroup v2 directory and every visible ancestor."""
    membership = [line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if line.startswith('0::')]
    require(len(membership) == 1, 'Cannot establish cgroup v2 membership')
    candidates = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        if after.split()[0] != 'cgroup2':
            continue
        fields = before.split()
        root, mount = Path(fields[3]), Path(fields[4])
        group = Path(membership[0])
        if group.is_relative_to(root):
            leaf = mount / group.relative_to(root)
        elif group == Path('/'):
            leaf = mount
        else:
            continue
        if not (leaf / 'memory.current').is_file():
            continue
        current = leaf
        while True:
            if (current / 'memory.current').is_file():
                candidates.append(current)
            if current == mount:
                break
            current = current.parent
    require(bool(candidates), 'No visible memory cgroup; refusing host-only fallback')
    return sorted(set(candidates))


def memory_snapshot(paths):
    groups = {}
    for path in paths:
        current = int((path / 'memory.current').read_text())
        limits = []
        for name in ('memory.high', 'memory.max'):
            value = (path / name).read_text().strip()
            if value != 'max':
                limits.append(int(value))
        events = dict((k, int(v)) for k, v in (line.split() for line in (path / 'memory.events').read_text().splitlines()))
        groups[str(path)] = dict(current=current, limit=min(limits) if limits else None,
                                 headroom=min(limits) - current if limits else None, events=events)
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    return dict(groups=groups, host_available=available)


def check_memory(snapshot, needed=RESERVE, baseline=None):
    bounded = [g['headroom'] for g in snapshot['groups'].values() if g['headroom'] is not None]
    require(bool(bounded), 'No finite visible cgroup high/max; refusing host-only fallback')
    require(min(bounded) >= needed, f'Cgroup reserve/headroom below {needed} bytes')
    if snapshot['host_available'] is not None:
        require(snapshot['host_available'] >= needed, 'Supplemental host available memory too low')
    if baseline is not None:
        require(snapshot['groups'].keys() == baseline['groups'].keys(), 'Cgroup membership changed')
        for key, group in snapshot['groups'].items():
            for event in ('oom', 'oom_kill'):
                require(event in group['events'] and event in baseline['groups'][key]['events'], 'Missing OOM counters')
                require(group['events'][event] == baseline['groups'][key]['events'][event], f'New or reset {event} counter: {key}')


def pre_read(paths, leafbytes, baseline):
    check_memory(memory_snapshot(paths), max(24 * GIB, 3 * leafbytes + 8 * GIB), baseline)


@contextmanager
def trace_leaf(checkpoint, key, ocp, evidence):
    from orbax.checkpoint import type_handlers
    original = type_handlers.NumpyHandler.deserialize
    names = []

    async def traced(handler, infos, args=None):
        for info in infos:
            require(Path(str(info.path)).resolve().is_relative_to(checkpoint.resolve()), 'Unexpected checkpoint read')
            require(info.name == '.'.join(key), f'Unselected leaf read: {info.name}')
            names.append(info.name)
        return await original(handler, infos, args)

    type_handlers.NumpyHandler.deserialize = traced
    try:
        yield
        require(Counter(names) == Counter(['.'.join(key)]), 'Unexpected logical read count')
        evidence.append(list(key))
    finally:
        type_handlers.NumpyHandler.deserialize = original


def restore_leaf(reader, checkpoint, key, ocp, evidence):
    import numpy as np
    require(key == ('step',) or (len(key) >= 3 and key[0] == 'params' and key[-1] == 'value'), 'Forbidden selection')
    item = ocp.tree.from_flat_dict({key: 0})
    args = ocp.tree.from_flat_dict({key: ocp.RestoreArgs(restore_type=np.ndarray)})
    with trace_leaf(checkpoint, key, ocp, evidence):
        restored = reader.restore(checkpoint, args=ocp.args.PyTreeRestore(item=item, restore_args=args, transforms={}))
    flat = ocp.tree.to_flat_dict(restored)
    require(set(flat) == {key}, 'Restore returned unselected leaves')
    return flat[key]


def raw_load(cfg, checkpoint, evidence_path):
    import importlib.metadata
    require(checkpoint == d.PILOT / '1999', 'Wrong raw checkpoint identity')
    require(importlib.metadata.version('orbax-checkpoint') == '0.11.13', 'Requires Orbax 0.11.13')
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.shared import normalize
    require(all(device.platform == 'gpu' for device in jax.devices()), 'GPU required')
    paths = discover_cgroups()
    baseline = memory_snapshot(paths)
    check_memory(baseline)
    # Independent architecture template, not inferred from checkpoint metadata.
    abstract = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(0))).to_pure_dict()
    reference = {k: (tuple(v.shape), str(v.dtype)) for k, v in ocp.tree.to_flat_dict(abstract).items()}
    stats = normalize.load(checkpoint / 'assets/aloha_agilex_sim')
    original = normalize.load(d.SOURCE / 'assets/aloha_agilex_sim')
    require(stats.keys() == original.keys(), 'Normalization keys changed')
    for key in stats:
        for field in ('mean', 'std', 'q01', 'q99'):
            x, y = getattr(stats[key], field), getattr(original[key], field)
            require(x is not None and y is not None and np.array_equal(d.finite(x), d.finite(y)), 'Normalization changed')
    evidence = dict(kind='raw', checkpoint=str(RAW), internal_step=None, logical_reads=[], leaves=[], memory_baseline=baseline)
    params = {}
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as reader:
        metadata = ocp.tree.to_flat_dict(reader.metadata(RAW))
        validate_reference(extract_params(metadata), reference)  # All keys/shapes/dtypes before any tensor read.
        step_meta = metadata.get(('step',))
        require(step_meta is not None and tuple(step_meta.shape) == () and str(step_meta.dtype) in ('int32', 'int64'), 'Invalid step metadata')
        pre_read(paths, 8, baseline)
        step = restore_leaf(reader, RAW, ('step',), ocp, evidence['logical_reads'])
        validate_array(step, (), str(step_meta.dtype))
        require(int(step) == 2000, 'Raw internal step must be 2000')
        evidence['internal_step'] = int(step)
        del step
        for key, (shape, dtype) in sorted(reference.items()):
            full = ('params',) + key + ('value',)
            pre_read(paths, math.prod(shape) * np.dtype(dtype).itemsize, baseline)
            value = restore_leaf(reader, RAW, full, ocp, evidence['logical_reads'])
            validate_array(value, shape, dtype)
            device_value = jnp.asarray(value, dtype=jnp.bfloat16)
            device_value.block_until_ready()  # CPU temporary remains alive until transfer/cast completes.
            require(str(device_value.dtype) == 'bfloat16', 'Production cast failed')
            params[key] = device_value
            evidence['leaves'].append(dict(key=list(full), shape=list(shape), native_dtype=str(value.dtype), inference_dtype=str(device_value.dtype)))
            del value, device_value
            gc.collect()
            check_memory(memory_snapshot(paths), baseline=baseline)
    evidence['memory_final'] = memory_snapshot(paths)
    check_memory(evidence['memory_final'], baseline=baseline)
    d.guard(evidence_path.parent.parent, 64 * 2**20)
    d.write_json(evidence_path, evidence)
    loaded = cfg.model.load(ocp.tree.from_flat_dict(params), remove_extra_params=False)
    return loaded, dict(kind='raw', checkpoint=str(RAW), internal_step=2000,
                        native=sorted({dtype for _, dtype in reference.values()}), inference='bfloat16', explicit_production_cast=True)


def check_records(records, completed, manifest, model, smoke):
    anchors = manifest['anchors'][:1] if smoke else manifest['anchors']
    require(completed['status'] == 'passed' and completed['model'] == model and
            completed['anchors'] == len(anchors) and completed['records'] == len(anchors) * 3 and
            completed['check_inputs_only'] is False and completed['production_parameter_dtype'] == 'bfloat16' and
            completed['model_compute_dtype'] == 'bfloat16', 'Saved worker identity mismatch')
    dtype = completed['diagnostic_parameter_dtypes']
    require(dtype['native'] == ['float32'] and dtype['inference'] == 'bfloat16' and dtype['explicit_production_cast'] is True, 'Saved dtype identity mismatch')
    if model == 'raw':
        require(dtype.get('kind') == 'raw' and dtype.get('checkpoint') == str(RAW) and
                dtype.get('internal_step') == 2000, 'Saved raw identity mismatch')
    require(len(records) == len(anchors) * 3, 'Saved record count mismatch')
    for row, (a, draw, seed) in zip(records, ((a, i, s) for a in anchors for i, s in enumerate(d.SEEDS)), strict=True):
        for key in ('anchor_id', 'split', 'task', 'episode', 'frame', 'valid'):
            require(row[key] == a[key], f'Saved record identity mismatch: {key}')
        require(row['model'] == model and row['draw'] == draw and row['seed'] == seed, 'Saved RNG/model mismatch')
        for field, width in [('native_loss', None), ('abs_error', 14), ('squared_error', 14), ('outside', 14)]:
            require(len(row[field]) == 50, 'Saved horizon mismatch')
            if width:
                require(all(len(v) == width for v in row[field]), 'Saved action dimension mismatch')
        for _, value in d.flatten_numbers(row):
            require(math.isfinite(value), 'Nonfinite saved record')


def fixed_inputs():
    require(d.digest(FIXED) == FIXED_SHA, 'Fixed manifest changed')
    manifest = d.read_json(FIXED)
    require(manifest['original'] == str(d.SOURCE) and manifest['ema'] == str(d.PILOT / '1999') and
            manifest['seeds'] == list(d.SEEDS) and len(manifest['anchors']) == 160 and manifest['num_steps'] == 10, 'Fixed panel identity mismatch')
    return manifest


def previous_inputs(manifest):
    paths = []
    for mode, directory in PREVIOUS.items():
        report_path = directory / 'report.json'
        report = d.read_json(report_path)
        require(report['status'] == 'passed' and report['mode'] == mode and report['manifest_sha256'] == FIXED_SHA and report['check_inputs_only'] is False, 'Previous report identity mismatch')
        paths.append(report_path)
        for model in ('original', 'ema'):
            records, completed = [directory / model / name for name in ('records.json', 'completed.json')]
            check_records(d.read_json(records), d.read_json(completed), manifest, model, mode == 'smoke')
            paths.extend([records, completed])
    return paths


def prepare(out):
    require('jax' not in sys.modules and 'torch' not in sys.modules, 'Prepare must not import a backend')
    manifest = fixed_inputs()
    previous = previous_inputs(manifest)
    tracked = set(map(Path, manifest['sha256'])) | set(previous) | {FIXED, SCRIPT,
        ROOT / 'scripts/test_gen1_raw_diagnostic.py', ROOT / 'scripts/test_gen1_selective_raw_restore.py'}
    raw_files = d.files_under(RAW)
    require(bool(raw_files), 'Missing raw checkpoint')
    tracked.update(raw_files)
    hashes = {}
    for path in sorted(tracked):
        d.guard(out, 64 * 2**20)
        hashes[str(path)] = d.digest(path)
        if str(path) in manifest['sha256']:
            require(hashes[str(path)] == manifest['sha256'][str(path)], f'Frozen input changed: {path}')
    d.write_json(out / 'manifest.json', dict(schema=1, kind='raw', status='prepared', raw=str(RAW),
        fixed_manifest=str(FIXED), fixed_sha256=FIXED_SHA, sha256=hashes,
        raw_files=[str(p) for p in raw_files], previous={k: str(v) for k, v in PREVIOUS.items()}, limitations=limitations()))


def validate_manifest(path):
    manifest = d.read_json(path)
    require(manifest.get('schema') == 1 and manifest.get('kind') == 'raw' and manifest.get('status') == 'prepared' and
            manifest['raw'] == str(RAW) and manifest['fixed_manifest'] == str(FIXED) and manifest['fixed_sha256'] == FIXED_SHA and
            manifest['previous'] == {k: str(v) for k, v in PREVIOUS.items()}, 'Invalid raw manifest')
    fixed = fixed_inputs()
    required = set(fixed['sha256']) | {str(FIXED), str(SCRIPT), str(ROOT / 'scripts/test_gen1_raw_diagnostic.py'),
        str(ROOT / 'scripts/test_gen1_selective_raw_restore.py')} | {str(p) for p in previous_inputs(fixed)}
    raw_files = [str(p) for p in d.files_under(RAW)]
    require(raw_files == manifest['raw_files'] and bool(raw_files), 'Raw file inventory changed')
    require(set(manifest['sha256']) == required | set(raw_files), 'Incomplete frozen identities')
    for file, sha in manifest['sha256'].items():
        require(d.digest(file) == sha, f'Frozen input changed: {file}')
    return manifest


def supervise(command, timeout, out):
    paths = discover_cgroups()
    baseline = memory_snapshot(paths)
    check_memory(baseline, max(RESERVE, 24 * GIB))
    evidence = dict(baseline=baseline, poll_seconds=0.25, reserve_bytes=RESERVE, samples=0,
                    minimum_headroom=None, status='failed', limitations=limitations()[-2:])
    proc = None
    try:
        proc = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + timeout
        while True:
            current = memory_snapshot(paths)
            evidence['last'] = current
            evidence['samples'] += 1
            headroom = min(g['headroom'] for g in current['groups'].values() if g['headroom'] is not None)
            evidence['minimum_headroom'] = min(evidence['minimum_headroom'] or headroom, headroom)
            check_memory(current, baseline=baseline)
            d.guard(out, 64 * 2**20)
            require(time.monotonic() < deadline, 'Worker timeout')
            code = proc.poll()
            if code is not None:
                require(code == 0, f'Worker exited {code}')
                break
            time.sleep(0.25)
        evidence['status'] = 'passed'
    except BaseException as exc:
        evidence['error'] = repr(exc)
        raise
    finally:
        if proc is not None:
            # Always kill the entire group, including descendants after leader exit.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
        d.write_json(out / 'supervision.json', evidence)


def worker(a):
    require(a.allow_inference, 'Requires --allow-inference')
    require(sys.flags.isolated and sys.flags.dont_write_bytecode, 'Worker requires -I -B')
    require(a.output.parent.parent.resolve() == ROOT / 'outputs' and a.output.is_dir() and not any(a.output.iterdir()), 'Unsafe worker output')
    validate_manifest(a.manifest)
    original = d.strict_load
    d.strict_load = lambda cfg, checkpoint: raw_load(cfg, checkpoint, a.output / 'raw_restore.json')
    try:
        # Frozen worker selects EMA path for any non-original label. Only loading is overridden;
        # records/completion retain the explicit raw label. No module file is modified.
        d.worker(argparse.Namespace(manifest=FIXED, output=a.output, worker_model='raw', smoke=a.smoke, check_inputs_only=False))
    finally:
        d.strict_load = original


def run(a):
    require(a.allow_inference, 'Requires --allow-inference')
    validate_manifest(a.manifest)
    if a.command == 'evaluate':
        require(a.smoke_report is not None, 'Full160 requires a passed raw smoke report')
        smoke = d.read_json(a.smoke_report)
        require(smoke['status'] == 'passed' and smoke['mode'] == 'smoke' and smoke['kind'] == 'raw' and
                smoke['manifest_sha256'] == d.digest(a.manifest), 'Unmatched raw smoke report')
        smoke_dir = a.smoke_report.parent
        expected_evidence = {str(smoke_dir / 'raw' / name) for name in
                             ('records.json', 'completed.json', 'raw_restore.json')} | {str(smoke_dir / 'supervision.json')}
        require(set(smoke['worker_sha256']) == expected_evidence, 'Incomplete smoke evidence')
        for file, sha in smoke['worker_sha256'].items():
            require(d.digest(file) == sha, 'Smoke evidence changed')
        check_records(d.read_json(smoke_dir / 'raw/records.json'), d.read_json(smoke_dir / 'raw/completed.json'), fixed_inputs(), 'raw', True)
        require(d.read_json(smoke_dir / 'supervision.json')['status'] == 'passed', 'Smoke supervision failed')
    sub = a.output / 'raw'
    sub.mkdir(exist_ok=False)
    command = [sys.executable, '-I', '-B', str(SCRIPT), '_worker', '--manifest', str(a.manifest),
               '--output', str(sub), '--allow-inference']
    if a.command == 'smoke':
        command.append('--smoke')
    supervise(command, 1800 if a.command == 'smoke' else 21600, a.output)
    records = d.read_json(sub / 'records.json')
    check_records(records, d.read_json(sub / 'completed.json'), fixed_inputs(), 'raw', a.command == 'smoke')
    raw = d.summarize(records)
    summaries = {}
    for name in ('original', 'ema'):
        summaries[name] = d.summarize(d.read_json(PREVIOUS[a.command] / name / 'records.json'))
    comparisons = {}
    for name, summary in summaries.items():
        paired = d.paired_summary(summary, raw)
        for episode in paired['episodes']:
            episode['raw_minus_' + name] = episode.pop('ema_minus_original')
        comparisons['raw_minus_' + name] = paired
    evidence_files = [sub / name for name in ('records.json', 'completed.json', 'raw_restore.json')] + [a.output / 'supervision.json']
    d.guard(a.output, 64 * 2**20)
    d.write_json(a.output / 'report.json', dict(status='passed', mode=a.command, kind='raw',
        manifest_sha256=d.digest(a.manifest), fixed_manifest_sha256=FIXED_SHA,
        reused_results=str(PREVIOUS[a.command]), smoke_report_sha256=d.digest(a.smoke_report) if a.smoke_report else None,
        worker_sha256={str(p): d.digest(p) for p in evidence_files}, raw=raw, **summaries,
        comparisons=comparisons, limitations=limitations()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('prepare', 'smoke', 'evaluate', '_worker'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--smoke-report', type=Path)
    parser.add_argument('--allow-inference', action='store_true')
    parser.add_argument('--smoke', action='store_true', help=argparse.SUPPRESS)
    a = parser.parse_args()
    require(sys.flags.isolated and sys.flags.dont_write_bytecode, 'Run with -I -B')
    require(a.command == 'prepare' or (a.allow_inference and a.manifest is not None), 'GPU modes require --allow-inference and --manifest')
    if a.command == '_worker':
        worker(a)
        return
    a.output = d.fresh_output(a.output)
    try:
        prepare(a.output) if a.command == 'prepare' else run(a)
    except BaseException as exc:
        d.write_json(a.output / 'failed.json', dict(status='failed', error=repr(exc)))
        raise
    print(a.output, flush=True)


if __name__ == '__main__':
    main()
