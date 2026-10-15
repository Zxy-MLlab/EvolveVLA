"""Remove exactly six obsolete full-model intermediates, retaining run evidence."""
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
RUNS = ('gen1_multitask_balanced_20261012', 'gen1_multitask_phaseaware_v3_20261013')
STEPS = (1000, 4000, 12000)
REPORT = ROOT/'outputs/checkpoint_cleanup_20261013.json'


def main():
    if REPORT.exists():
        raise ValueError('Cleanup already recorded; refusing to repeat')
    targets = []
    for run in RUNS:
        output = ROOT/'outputs'/run
        done = json.loads((output/'completed.json').read_text())
        selected = json.loads((output/'selection.json').read_text())['selected']
        assert done['status'] == 'completed' and done['steps'] == 24000
        assert selected['step'] == 24000
        assert Path(selected['checkpoint']).resolve() == ROOT/'checkpoints'/run/'24000'
        assert (Path(selected['checkpoint'])/'params/_METADATA').is_file()
        for step in STEPS:
            target = ROOT/'checkpoints'/run/str(step)/'params'
            assert target.is_dir() and not target.is_symlink()
            assert target.resolve() == target
            files = list(target.rglob('*'))
            assert files and not any(p.is_symlink() for p in files)
            provenance = json.loads((target.parent/'provenance.json').read_text())
            assert provenance['step'] == step
            targets.append(dict(path=str(target), run=run, step=step,
                logical_bytes=sum(p.stat().st_size for p in files if p.is_file()),
                allocated_bytes=sum(p.stat().st_blocks*512 for p in files if p.is_file())))
    # Refuse deletion of anything named by a live process or held open.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            assert not any(str(Path(t['path']).parent) in command for t in targets), command
            for fd in (process/'fd').iterdir():
                try:
                    path = os.readlink(fd)
                except (FileNotFoundError, PermissionError):
                    continue
                assert not any(path == t['path'] or path.startswith(t['path']+'/') for t in targets), path
        except (FileNotFoundError, PermissionError):
            continue
    report = dict(status='validated', user_authorized=True, targets=targets,
        reason='Completed runs; unselected intermediate weights. Selected final models retained.',
        retained='Original model, all current LoRA checkpoints, final old models, logs/metrics/provenance/videos',
        recoverability='Deleted weight files cannot be restored from trash; exact intermediate models require rerunning training.',
        free_before_bytes=shutil.disk_usage(ROOT).free, removed=[])
    with REPORT.open('x') as stream:
        json.dump(report, stream, indent=2)
    for target in targets:
        shutil.rmtree(target['path'])
        assert not Path(target['path']).exists()
        report['removed'].append(target['path'])
        print('DELETED_OBSOLETE_PARAMS', target['path'], flush=True)
        REPORT.write_text(json.dumps(report, indent=2))
    report.update(status='completed', finished_unix=time.time(),
                  free_after_bytes=shutil.disk_usage(ROOT).free)
    report['net_freed_bytes'] = report['free_after_bytes']-report['free_before_bytes']
    REPORT.write_text(json.dumps(report, indent=2))
    print('FREED_GIB', round(report['net_freed_bytes']/2**30, 2),
          'FREE_GIB', round(report['free_after_bytes']/2**30, 2), flush=True)


if __name__ == '__main__':
    main()
