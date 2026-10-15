"""Delete an explicit set of obsolete training artifacts after live-use checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
REPORT = ROOT / 'outputs/checkpoint_cleanup_20261014.json'
TARGETS = (
    ('gen1_pilot_2000_20261010/1999/train_state',
     'Completed old pilot: raw training/optimizer state; inference params retained'),
    ('gen1_myungkyu_smoke_cap8_20261009T161534/1/params',
     'One-update smoke weights; collector verification audit retained'),
    ('gen1_recovery_action_expert_v2_20261012/500/params.orbax-checkpoint-tmp-0',
     'Incomplete checkpoint from abandoned save'),
)
PROTECTED = (
    'checkpoints/gen1_pilot_2000_20261010/1999/params/_METADATA',
    'checkpoints/gen1_myungkyu_smoke_cap8_20261009T161534/verification_audit_20261009T162251Z_c66175d3cf11.json',
    'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json',
    'data/gen1_multitask_coverage_20261012/split_manifest.json',
    'outputs/gen1_pilot_2000_20261010/completed.json',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(report, exclusive=False):
    with REPORT.open('x' if exclusive else 'w') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert not REPORT.exists(), 'Cleanup already recorded; inspect report instead'
    done = json.loads((ROOT / 'outputs/gen1_pilot_2000_20261010/completed.json').read_text())
    assert done['status'] == 'completed' and done['final_step'] == 1999
    protected = {p: digest(ROOT / p) for p in PROTECTED}
    assert json.loads((ROOT / PROTECTED[1]).read_text())['status'] == 'passed'
    targets = []
    for relative, reason in TARGETS:
        path = ROOT / 'checkpoints' / relative
        assert path.is_dir() and not path.is_symlink() and path.resolve() == path
        entries = list(path.rglob('*'))
        assert entries and not any(p.is_symlink() for p in entries)
        assert not any((ROOT / p).is_relative_to(path) for p in PROTECTED)
        targets.append({'path': str(path), 'reason': reason,
                        'allocated_bytes': sum(p.stat().st_blocks * 512 for p in entries if p.is_file())})
    # Refuse to remove a checkpoint referenced or held open by any live process.
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            for target in targets:
                assert str(Path(target['path']).parent) not in command, command
            for fd in (process / 'fd').iterdir():
                try:
                    opened = os.readlink(fd)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                assert not any(opened == t['path'] or opened.startswith(t['path'] + '/')
                               for t in targets), opened
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    report = {'status': 'validated', 'targets': targets, 'protected_sha256': protected,
              'free_before_bytes': shutil.disk_usage(ROOT).free, 'removed': [],
              'recoverability': 'Permanently deleted; reconstruct weights/state by rerunning old training.'}
    print(json.dumps(report, indent=2), flush=True)
    if not args.apply:
        return
    save(report, exclusive=True)
    for target in targets:
        shutil.rmtree(target['path'])
        assert not Path(target['path']).exists()
        report['removed'].append(target['path'])
        save(report)
    assert protected == {p: digest(ROOT / p) for p in PROTECTED}
    report.update(status='completed', finished_unix=time.time(),
                  free_after_bytes=shutil.disk_usage(ROOT).free)
    report['net_freed_bytes'] = report['free_after_bytes'] - report['free_before_bytes']
    save(report)
    print('FREED_GIB', round(report['net_freed_bytes'] / 2**30, 2),
          'FREE_GIB', round(report['free_after_bytes'] / 2**30, 2), flush=True)


if __name__ == '__main__':
    main()
