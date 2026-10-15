"""Fetch two pinned raw Clean archives for export parity; CPU/network only."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path('/root/autodl-tmp/EvolveVLA')
DEST = ROOT / 'data/gen1_official_raw_probe_20261013'
REV = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
FILES = [
    ('place_mouse_pad', 257569474, '32830874f06d31f3a56c6694ca1e93344aecc525c9c32c8c774bdb896d39f210'),
    ('hanging_mug', 726876729, 'bfdbe40d9e7323e8aa763684cd2861f0278aeac456b25c477239fd86566ef45a'),
]


def main():
    DEST.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for task, size, sha in FILES:
        if shutil.disk_usage(DEST).free < size + 30 * 2**30:
            raise RuntimeError('Preserve 30 GiB training reserve')
        url = f'https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/dataset/{task}/demo_clean.zip?download=true'
        request = urllib.request.Request(url, headers={'User-Agent': 'curl/8.5.0'})
        part = DEST / f'{task}.zip.partial'
        digest, total = hashlib.sha256(), 0
        with opener.open(request, timeout=60) as source, part.open('xb') as target:
            while block := source.read(4 * 1024**2):
                total += len(block)
                if total > size:
                    raise ValueError('Pinned size exceeded')
                digest.update(block)
                target.write(block)
        if total != size or digest.hexdigest() != sha:
            raise ValueError('Pinned archive digest mismatch')
        final = DEST / f'{task}.zip'
        part.rename(final)
        with zipfile.ZipFile(final) as archive:
            inventory = [dict(path=e.filename, bytes=e.file_size) for e in archive.infolist()]
        with (DEST / f'{task}.json').open('x') as report:
            json.dump(dict(task=task, revision=REV, url=url, sha256=sha, bytes=total,
                           status='verified', inventory=inventory, starts_training=False), report, indent=2)
        print('VERIFIED', task, total, flush=True)


if __name__ == '__main__':
    main()
