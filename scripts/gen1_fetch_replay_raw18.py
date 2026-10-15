"""Prepare official raw replay for the existing 18-task scope, with disk reserve."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from gen1_fetch_replay_raw_probe import ROOT, REV

DEST = ROOT / 'data/gen1_official_raw18_20261013'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
    DEST.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {'User-Agent': 'curl/8.5.0'}
    url = f'https://hf-mirror.com/api/datasets/TianxingChen/RoboTwin2.0/tree/{REV}?recursive=true&limit=1000'
    with opener.open(urllib.request.Request(url, headers=headers), timeout=45) as source:
        listing = json.load(source)
    split = json.loads((ROOT / 'data/gen1_multitask_coverage_20261012/split_manifest.json').read_text())
    tasks = split['tasks']
    entries = [r for r in listing if r['path'] in {f'dataset/{t}/demo_clean.zip' for t in tasks}]
    if len(entries) != len(tasks) or len(tasks) != 18:
        raise ValueError('Incomplete pinned raw source listing')
    write(DEST / 'plan.json', dict(revision=REV, tasks=tasks, entries=entries,
        reserve_gib=30, purpose='Original Clean replay for mixing with local trajectories',
        normalization='unchanged initializer assets', starts_training=False, extracts=False))
    completed = []
    try:
        for entry in entries:
            task = entry['path'].split('/')[1]
            size, sha = entry['size'], entry['lfs']['oid']
            probe = ROOT / f'data/gen1_official_raw_probe_20261013/{task}.zip'
            if probe.exists():
                path = probe
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != sha or path.stat().st_size != size:
                    raise ValueError('Reused archive mismatch')
            else:
                if shutil.disk_usage(DEST).free < size + 30 * 2**30:
                    raise RuntimeError('Preserve 30 GiB for existing training and evaluation')
                partial = DEST / f'{task}.zip.partial'
                url = f'https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/{entry["path"]}?download=true'
                h, total = hashlib.sha256(), 0
                with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as source, partial.open('xb') as target:
                    while block := source.read(4 * 1024**2):
                        total += len(block)
                        if total > size or shutil.disk_usage(DEST).free < 30 * 2**30:
                            raise RuntimeError('Pinned size or disk reserve exceeded')
                        target.write(block)
                        h.update(block)
                if total != size or h.hexdigest() != sha:
                    raise ValueError('Raw archive size/SHA256 mismatch')
                path = DEST / f'{task}.zip'
                partial.rename(path)
            row = dict(task=task, archive=str(path), bytes=size, sha256=sha, revision=REV)
            write(DEST / f'{task}.json', row)
            completed.append(row)
            print('VERIFIED',task,len(completed),'/ 18',flush=True)
        write(DEST / 'completed.json', dict(status='downloads_verified', sources=completed,
            ready_for_training=False, next='Raw source audit, split and mixed-loader preflight'))
    except BaseException as exc:
        write(DEST / 'failed.json', dict(error=f'{type(exc).__name__}: {exc}', verified=completed,
                                       partial_retained=True, ready_for_training=False))
        raise


if __name__ == '__main__':
    main()
