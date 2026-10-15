#!/usr/bin/env python3
"""External one-chunk A/A diagnostic; prepare is CPU-only, run requires opt-in.

Never imports a simulator. Instrumentation materializes arrays and synchronizes
accelerator work: numerical boundaries are preserved, timing is not comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'outputs/gen1_diagnostic_aa_manifest_20261011_v2/manifest.json'
FROZEN_SHA = '0ea3ee098b970361fc2700469ff879b904db7e3d0fee8e85959ec2e1796b4e28'
INITIAL = ROOT / 'outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r1/initial'
KEYS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist', 'state')
LIMITS = dict(output_budget_bytes=64 * 1024**2, min_free_bytes=70 * 1024**3, timeout_seconds=1800)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def resource_gate(output, limits=LIMITS):
    output = Path(output)
    existing = output if output.exists() else output.parent
    if shutil.disk_usage(existing).free < limits['min_free_bytes']:
        raise RuntimeError('free-space gate failed')
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0
    if size > limits['output_budget_bytes']:
        raise RuntimeError('output-budget gate failed')


def load_inputs(directory=INITIAL):
    directory = Path(directory)
    hashes = json.loads((directory / 'hashes.json').read_text())
    values = {}
    for key in KEYS:
        a = np.load(directory / f'{key}.npy', allow_pickle=False)
        expected_shape = (14,) if key == 'state' else (240, 320, 3)
        expected_dtype = np.dtype('float32') if key == 'state' else np.dtype('uint8')
        if a.shape != expected_shape or a.dtype != expected_dtype or not np.isfinite(a).all():
            raise ValueError(f'invalid input: {key}')
        expected = hashes[key] if key == 'state' else hashes[key]['sha256']
        if hashlib.sha256(a.tobytes()).hexdigest() != expected:
            raise ValueError(f'input hash mismatch: {key}')
        values[key] = a
    return values


def verify_identities(identities):
    for name, digest in identities.items():
        if sha(name) != digest:
            raise ValueError(f'identity changed: {name}')


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    resource_gate(output)
    if sha(FROZEN) != FROZEN_SHA:
        raise ValueError('frozen manifest changed')
    frozen = json.loads(FROZEN.read_text())
    identities = {str(FROZEN): FROZEN_SHA}
    for group in ('identities', 'sources', 'code'):
        identities.update(frozen[group])
    verify_identities(identities)
    load_inputs()
    case = next(c for c in frozen['cases'] if c['id'] == 'hanging_mug_100000_r1')
    if hashlib.sha256(case['prompt'].encode()).hexdigest() != json.loads((INITIAL / 'hashes.json').read_text())['prompt']:
        raise ValueError('prompt hash mismatch')
    source = Path(frozen['config']['robotwin_root']) / 'XPolicyLab/policy/Pi_05/openpi/src'
    # Freeze actual model/transform implementation too, not just adapter files.
    for path in sorted(source.rglob('*.py')):
        identities[str(path)] = sha(path)
    tokenizer = Path('/root/autodl-tmp/openpi_cache/big_vision/paligemma_tokenizer.model')
    for path in [Path(__file__).resolve(), tokenizer, INITIAL / 'hashes.json', *(INITIAL / f'{k}.npy' for k in KEYS)]:
        identities[str(path)] = sha(path)
    manifest = dict(version=1, frozen_manifest=str(FROZEN), initial=str(INITIAL),
                    model=frozen['model'], prompt=case['prompt'], policy_seed=0,
                    normalization='quantile', prompt_prefix='none', noise_source='jax',
                    source=str(source), tokenizer=str(tokenizer), inference_python=frozen['config']['inference_python'],
                    repetitions=['r1', 'r2'], limits=LIMITS, identities=identities,
                    protocol='Two fresh sequential processes, one infer_chunk each; no simulator, retries or warmup.')
    output.mkdir(exist_ok=False)
    write(output / 'manifest.json', manifest)
    (output / 'manifest.json').chmod(0o444)
    write(output / 'manifest.sha256.json', {'sha256': sha(output / 'manifest.json')})
    (output / 'manifest.sha256.json').chmod(0o444)
    return manifest


def manifest_at(output):
    output = Path(output)
    digest = json.loads((output / 'manifest.sha256.json').read_text())['sha256']
    if sha(output / 'manifest.json') != digest:
        raise ValueError('manifest changed')
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['repetitions'] != ['r1', 'r2']:
        raise ValueError('requires two predeclared workers')
    verify_identities(manifest['identities'])
    return manifest


def snapshot(value):
    """Copy leaves immediately, so later in-place transforms cannot alter capture."""
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    a = np.asarray(value)
    if a.dtype.kind not in 'biufUS' or a.nbytes > 8 * 1024**2:
        raise ValueError('unsupported or oversized captured leaf')
    if a.dtype.kind in 'f' and not np.isfinite(a).all():
        raise ValueError('nonfinite captured leaf')
    return a.copy()


def capture_once(adapter, **kwargs):
    """Delegate once at every boundary; never substitute outputs or RNG values."""
    policy = adapter.policy
    names = ('infer', '_input_transform', '_sample_actions', '_output_transform')
    originals = {name: getattr(policy, name) for name in names}
    owned = {name: name in vars(policy) for name in names}
    captured, counts = {}, {name: 0 for name in names}

    def wrap(name):
        def call(*args, **kw):
            counts[name] += 1
            if counts[name] != 1:
                raise ValueError(f'boundary called more than once: {name}')
            if name == 'infer':
                if kw.get('noise') is None:
                    raise ValueError('explicit noise required')
                captured['noise'] = snapshot(kw['noise'])
            if name == '_output_transform':
                captured['model_space'] = snapshot(args[0])
            result = originals[name](*args, **kw)
            if name == '_input_transform':
                captured['processed_inputs'] = snapshot(result)
            elif name == '_sample_actions':
                captured['sample_actions'] = snapshot(result)
            elif name == 'infer':
                captured['physical_output'] = snapshot(result['actions'])
            return result
        return call

    try:
        for name in names:
            setattr(policy, name, wrap(name))
        result = adapter.infer_chunk(**kwargs)
        captured['adapter_output'] = snapshot(result)
        if any(n != 1 for n in counts.values()):
            raise ValueError(f'expected one call per boundary: {counts}')
        if result.shape != (50, 14):
            raise ValueError('expected full 50 x 14 physical action chunk')
        return result, captured, counts
    finally:
        for name, original in originals.items():
            if owned[name]:
                setattr(policy, name, original)
            else:
                delattr(policy, name)


def save_leaves(directory, tree):
    directory = Path(directory)
    records = {}
    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + [str(key)])
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path + [str(i)])
        else:
            filename = f'leaf_{len(records):04d}.npy'
            with (directory / filename).open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            records[json.dumps(path)] = dict(file=filename, shape=list(value.shape), dtype=value.dtype.str,
                                             sha256=sha(directory / filename))
    visit(tree, [])
    write(directory / 'leaves.json', records)
    return records


def metrics(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('shape/dtype mismatch')
    result = dict(exact=bool(np.array_equal(a, b)), byte_exact=a.tobytes() == b.tobytes())
    if a.dtype.kind not in 'biuf':
        return result
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite comparison')
    zero = (a == 0) & (b == 0)
    result['signed_zero_mismatches'] = int(np.count_nonzero(zero & (np.signbit(a) != np.signbit(b)))) if a.dtype.kind == 'f' else 0
    delta = a.astype(np.float64) - b.astype(np.float64)
    def stats(d, axis=None):
        return dict(max=np.max(np.abs(d), axis=axis).tolist(), mae=np.mean(np.abs(d), axis=axis).tolist(),
                    rmse=np.sqrt(np.mean(d * d, axis=axis)).tolist())
    result.update(stats(delta))
    if delta.ndim == 2:
        result['per_horizon'] = stats(delta, 1)
        result['per_dimension'] = stats(delta, 0)
    return result


def compare(output):
    output = Path(output)
    maps = [json.loads((output / rep / 'leaves.json').read_text()) for rep in ('r1', 'r2')]
    if maps[0].keys() != maps[1].keys():
        raise ValueError('captured leaf paths differ')
    report = {}
    for key in maps[0]:
        arrays = []
        for rep, mapping in zip(('r1', 'r2'), maps):
            path = output / rep / mapping[key]['file']
            if sha(path) != mapping[key]['sha256']:
                raise ValueError('captured array changed')
            arrays.append(np.load(path, allow_pickle=False))
        report[key] = metrics(*arrays)
        if json.loads(key) in (['physical_output'], ['adapter_output']):
            a, b = arrays
            joints = [i for i in range(14) if i not in (6, 13)]
            report[key]['joints'] = metrics(a[:, joints], b[:, joints])
            report[key]['grippers'] = metrics(a[:, [6, 13]], b[:, [6, 13]])
            report[key]['first_action'] = metrics(a[:1], b[:1])
    write(output / 'comparison.json', report)
    return report


def worker(output, rep):
    if not sys.flags.isolated:
        raise RuntimeError('worker requires python -I')
    output = Path(output)
    m = manifest_at(output)
    if rep not in m['repetitions']:
        raise ValueError('unplanned repetition')
    resource_gate(output, m['limits'])
    # Bound even explicitly invoked workers; normal runs also supervise the group.
    signal.alarm(int(m['limits']['timeout_seconds']))
    destination = output / rep
    destination.mkdir(exist_ok=False)
    os.environ.setdefault('OPENPI_DATA_HOME', '/root/autodl-tmp/openpi_cache')
    sys.path.insert(0, m['source'])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pi05_openpi_adapter import PublicPi05Adapter
    from openpi.shared import download
    original_download = download.maybe_download
    def local_tokenizer(url, *args, **kwargs):
        if str(url) == 'gs://big_vision/paligemma_tokenizer.model':
            return Path(m['tokenizer'])
        path = Path(url)
        if path.exists():
            return path
        raise ValueError(f'replay prohibits network downloads: {url}')
    download.maybe_download = local_tokenizer
    try:
        adapter = PublicPi05Adapter(m['model'], policy_seed=m['policy_seed'], noise_source='jax',
                                   normalization=m['normalization'], prompt_prefix=m['prompt_prefix'])
    finally:
        download.maybe_download = original_download
    if adapter.checkpoint_format != 'openpi_orbax':
        raise ValueError('requires original Orbax checkpoint')
    values = load_inputs(m['initial'])
    _, captured, counts = capture_once(adapter, images={k: values[k] for k in KEYS[:-1]},
                                       state=values['state'], prompt=m['prompt'])
    save_leaves(destination, captured)
    import jax
    packages = {}
    for package in ('numpy', 'jax', 'jaxlib', 'orbax-checkpoint', 'torch', 'sentencepiece'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    write(destination / 'runtime.json', dict(pid=os.getpid(), python=sys.version, executable=sys.executable,
          platform=platform.platform(), isolated=sys.flags.isolated, packages=packages,
          devices=[str(d) for d in jax.devices()], counts=counts,
          environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'XLA_FLAGS', 'JAX_PLATFORM_NAME', 'OPENPI_DATA_HOME')},
          manifest_sha256=sha(output / 'manifest.json')))
    resource_gate(output, m['limits'])


def supervise(command, log, output, limits):
    with Path(log).open('xb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        try:
            while process.poll() is None:
                resource_gate(output, limits)
                if time.monotonic() - start > limits['timeout_seconds']:
                    raise TimeoutError('worker deadline exceeded')
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(f'worker exited {process.returncode}')
            resource_gate(output, limits)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def run(output):
    output = Path(output).resolve()
    m = manifest_at(output)
    resource_gate(output, m['limits'])
    # An exclusive marker prevents retries or continuation of a partial run.
    write(output / 'started.json', {'time': time.time(), 'pid': os.getpid()})
    for rep in m['repetitions']:
        command = [m['inference_python'], '-I', '-B', str(Path(__file__).resolve()),
                   'worker', '--output', str(output), '--rep', rep, '--allow-inference']
        supervise(command, output / f'{rep}.log', output, m['limits'])
    compare(output)
    resource_gate(output, m['limits'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'worker', 'compare'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rep', choices=('r1', 'r2'))
    parser.add_argument('--allow-inference', action='store_true')
    args = parser.parse_args()
    if args.mode in ('run', 'worker') and not args.allow_inference:
        parser.error('inference requires explicit --allow-inference; prepare/compare are CPU-only')
    if args.mode == 'worker':
        worker(args.output, args.rep)
    else:
        {'prepare': prepare, 'run': run, 'compare': compare}[args.mode](args.output)


if __name__ == '__main__':
    main()
