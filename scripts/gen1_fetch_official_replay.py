"""Fetch the pinned official Clean LeRobot archive for replay preparation.

Does not alter training, extract files, or assert exact initializer dataset
identity. Verifies the Hub LFS digest before exposing a completed archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile


ROOT = Path('/root/autodl-tmp/EvolveVLA')
DEST = ROOT / 'data/gen1_official_replay_source_20261013'
REVISION = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
SIZE = 4100134542
SHA = 'ed5948c2bb693e098e0edc9c693614f8621948b3fc5953748932b34ee5bf0419'
URL = ('https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/'
       + REVISION + '/lerobot_dataset/RoboTwin_lerobot_v30.zip?download=true')


def write(name, value):
    with (DEST / name).open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
    global DEST
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    args = parser.parse_args()
    DEST = args.output.resolve()
    if DEST.parent != ROOT / 'data':
        raise ValueError('Expected a new direct child of project data')
    DEST.mkdir(exist_ok=False)
    write('plan.json', dict(repository='TianxingChen/RoboTwin2.0', revision=REVISION,
        path='lerobot_dataset/RoboTwin_lerobot_v30.zip', bytes=SIZE, sha256=SHA,
        purpose='Original RoboTwin Clean replay mixed with locally collected data',
        normalization='Keep current Myungkyu initializer norm_stats byte-identical',
        exact_initializer_dataset_identity='Not yet established',
        extraction=False, starts_training=False))
    try:
        if shutil.disk_usage(DEST).free < SIZE + 30 * 2**30:
            raise RuntimeError('Insufficient free space with 30 GiB training reserve')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        card_url = ('https://hf-mirror.com/Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k/'
                    'raw/37e10d7f4f49149e2ce5e08a77b07f228a84322f/README.md')
        headers = {'User-Agent': 'curl/8.5.0'}
        with opener.open(urllib.request.Request(card_url, headers=headers), timeout=30) as source:
            card = source.read(1024 * 1024)
        with (DEST / 'initializer_model_card.md').open('xb') as stream:
            stream.write(card)
        partial = DEST / 'RoboTwin_lerobot_v30.zip.partial'
        total, digest, last = 0, hashlib.sha256(), time.monotonic()
        with opener.open(urllib.request.Request(URL, headers=headers), timeout=60) as source, partial.open('xb') as stream:
            while block := source.read(4 * 1024 * 1024):
                total += len(block)
                if total > SIZE:
                    raise ValueError('Archive exceeds pinned size')
                if shutil.disk_usage(DEST).free < 30 * 2**30:
                    raise RuntimeError('Reached reserved free-space threshold')
                stream.write(block)
                digest.update(block)
                if time.monotonic() - last >= 20:
                    print(json.dumps(dict(downloaded=total, expected=SIZE)), flush=True)
                    last = time.monotonic()
        if total != SIZE or digest.hexdigest() != SHA:
            raise ValueError('Archive size or SHA256 mismatch')
        with zipfile.ZipFile(partial) as archive:
            entries = archive.infolist()
            if any(Path(e.filename).is_absolute() or '..' in Path(e.filename).parts
                   for e in entries):
                raise ValueError('Unsafe archive path')
            inventory = [dict(path=e.filename, bytes=e.file_size,
                              compressed_bytes=e.compress_size) for e in entries]
            write('archive_inventory.json', inventory)
            metadata = {}
            for entry in entries:
                if entry.filename.endswith('/meta/info.json') and entry.file_size < 1024**2:
                    metadata[entry.filename] = json.loads(archive.read(entry))
        final = DEST / 'RoboTwin_lerobot_v30.zip'
        partial.rename(final)
        write('completed.json', dict(status='download_verified', archive=str(final),
            sha256=SHA, bytes=total, entries=len(entries),
            unpacked_bytes=sum(e.file_size for e in entries), metadata=metadata,
            ready_for_training=False,
            remaining='Validate action semantics, prompts, video alignment and split separation'))
        print('DOWNLOAD_VERIFIED', final, flush=True)
    except BaseException as exc:
        write('failed.json', dict(error=f'{type(exc).__name__}: {exc}',
                                 partial_retained=True, ready_for_training=False))
        raise


if __name__ == '__main__':
    main()
