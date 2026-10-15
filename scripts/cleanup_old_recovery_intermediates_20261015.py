"""Remove four superseded intermediate weights from the completed recovery run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


ROOT = Path('/root/autodl-tmp/EvolveVLA')
RUN = ROOT / 'checkpoints/gen1_recovery_action_expert_v3_20261012'
REPORT = ROOT / 'outputs/checkpoint_cleanup_20261015.json'
STEPS = (500, 1000, 1500, 2500)
PROTECTED = (
    ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json',
    RUN / '2000/params/_METADATA',
    RUN / '3000/params/_METADATA',
    ROOT / 'outputs/gen1_recovery_fresh_compare_v3_20261012/selection.json',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(report, first=False):
    with REPORT.open('x' if first else 'w') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert not REPORT.exists(), f'Report already exists: {REPORT}'
    completed = json.loads((ROOT / 'outputs/gen1_recovery_action_expert_v3_20261012/completed.json').read_text())
    selection = json.loads(PROTECTED[-1].read_text())
    assert completed['status'] == 'completed' and completed['steps'] == 3000
    assert selection['selected']['step'] == 3000
    assert {2000, 3000}.isdisjoint(STEPS)
    protected = {str(path): sha256(path) for path in PROTECTED}
    targets = []
    for step in STEPS:
        path = RUN / str(step)
        assert path.is_dir() and not path.is_symlink() and path.resolve() == path
        assert str(path) in completed['checkpoints']
        entries = list(path.rglob('*'))
        assert entries and not any(entry.is_symlink() for entry in entries)
        assert (path / 'params/_METADATA').is_file()
        targets.append({'path': str(path), 'allocated_bytes': sum(
            entry.stat().st_blocks * 512 for entry in entries if entry.is_file())})

    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(process / 'cwd')
            for target in targets:
                path = target['path']
                assert path not in command and cwd != path and not cwd.startswith(path + '/')
            for fd in (process / 'fd').iterdir():
                try:
                    opened = os.readlink(fd)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                assert not any(opened == target['path'] or opened.startswith(target['path'] + '/')
                               for target in targets)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue

    report = {'status': 'validated', 'targets': targets, 'protected_sha256': protected,
              'free_before_bytes': shutil.disk_usage(ROOT).free, 'removed': [],
              'recoverability': 'Permanently deleted; recover only by rerunning old training.'}
    print(json.dumps(report, indent=2), flush=True)
    if not args.apply:
        return
    write_report(report, first=True)
    for target in targets:
        shutil.rmtree(target['path'])
        assert not Path(target['path']).exists()
        report['removed'].append(target['path'])
        write_report(report)
    assert protected == {str(path): sha256(path) for path in PROTECTED}
    report.update(status='completed', finished_unix=time.time(),
                  free_after_bytes=shutil.disk_usage(ROOT).free)
    report['net_freed_bytes'] = report['free_after_bytes'] - report['free_before_bytes']
    write_report(report)
    print('FREED_GIB', round(report['net_freed_bytes'] / 2**30, 2), flush=True)


if __name__ == '__main__':
    main()
