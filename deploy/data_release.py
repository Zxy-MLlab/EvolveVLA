#!/usr/bin/env python3
"""Bounded-space GitHub Release publication and verified data restoration."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
REPOSITORY = 'Zxy-MLlab/EvolveVLA'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gh(*args, json_output=False):
    binary = shutil.which('gh') or str(Path.home() / '.local/bin/gh')
    result = subprocess.run([binary, *map(str, args)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if json_output else result.stdout


class HashingReader:
    def __init__(self, stream):
        self.stream, self.hasher = stream, hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.hasher.update(block)
        return block


def safe_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'data':
        raise ValueError(f'Unsafe data path: {name}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Data path escapes root: {name}')
    return path


def build_shard(root, package, output):
    if output.exists():
        raise ValueError(f'Refusing existing archive: {output}')
    partial = output.with_suffix('.tar.partial')
    with partial.open('xb') as stream, tarfile.open(fileobj=stream, mode='w|', format=tarfile.PAX_FORMAT) as archive:
        for row in package['files']:
            path = safe_path(root, row['path'])
            if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']:
                raise ValueError(f'Source file changed: {path}')
            info = tarfile.TarInfo(row['path'])
            info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
            with path.open('rb') as source:
                reader = HashingReader(source)
                archive.addfile(info, reader)
                if reader.hasher.hexdigest() != row['sha256']:
                    raise ValueError(f'Source SHA256 mismatch: {path}')
    partial.rename(output)
    if output.stat().st_size >= 2 * 2**30:
        raise ValueError('GitHub release per-file size limit exceeded')
    return dict(name=package['name'], bytes=output.stat().st_size, sha256=digest(output))


def restore_shard(archive_path, package, root):
    expected = {row['path']: row for row in package['files']}
    seen = set()
    with tarfile.open(archive_path, 'r:') as archive:
        for member in archive:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f'Unexpected archive member: {member.name}')
            seen.add(member.name)
            row = expected[member.name]
            path = safe_path(root, member.name)
            if member.size != row['bytes']:
                raise ValueError('Wrong archived file length')
            if path.exists():
                if path.is_symlink() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                    raise ValueError(f'Refusing to overwrite different file: {path}')
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.restore.partial')
            hasher = hashlib.sha256()
            with archive.extractfile(member) as source, temporary.open('xb') as target:
                while block := source.read(4 * 2**20):
                    hasher.update(block)
                    target.write(block)
            if hasher.hexdigest() != row['sha256']:
                raise ValueError(f'Restored hash mismatch: {path}')
            temporary.rename(path)
    if seen != set(expected):
        raise ValueError('Archive omitted required files')


def publish(args, plan):
    directory = args.cache.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / 'release_index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else dict(
        schema=1, inventory_sha256=digest(args.inventory), release_tag=plan['release_tag'], assets=[])
    if index['inventory_sha256'] != digest(args.inventory):
        raise ValueError('Inventory changed since publication started')
    selected = [p for p in plan['packages'] if args.package is None or p['name'] == args.package]
    if not selected:
        raise ValueError('Unknown data package')
    release = None
    if not args.local_only:
        gh('auth', 'status')
        existing = gh('api', f'repos/{REPOSITORY}/releases?per_page=100', json_output=True)
        release = next((v for v in existing if v['tag_name'] == plan['release_tag']), None)
        if release is None:
            gh('release', 'create', plan['release_tag'], '--repo', REPOSITORY, '--draft',
               '--target', args.target, '--title', 'RoboTwin collected data and finalized training datasets (2026-10-15)',
               '--notes', '2800 accepted local trajectories, finalized local train/validation and 900 official replay episodes. '
               'SHA256 inventories and restoration instructions are in deploy/. No model weights.')
            release = gh('api', f'repos/{REPOSITORY}/releases/tags/{plan["release_tag"]}', json_output=True)
        if not release['draft']:
            raise ValueError('Release already public; do not mutate a published data release')
    for package in selected:
        if shutil.disk_usage(directory).free < (args.reserve_gib + 2) * 2**30:
            raise RuntimeError('Preserve training disk reserve before creating a shard')
        path = directory / package['name']
        # Resume only from our completed index; never trust an unverified archive.
        previous = next((r for r in index['assets'] if r['name'] == path.name), None)
        if path.exists():
            if previous is None or path.stat().st_size != previous['bytes'] or digest(path) != previous['sha256']:
                raise ValueError(f'Unverified local archive exists: {path}')
            entry = previous
        else:
            entry = build_shard(args.source.resolve(), package, path)
        if not args.local_only:
            assets = gh('api', f'repos/{REPOSITORY}/releases/{release["id"]}/assets?per_page=100', json_output=True)
            remote = next((v for v in assets if v['name'] == entry['name']), None)
            if remote is None:
                gh('release', 'upload', plan['release_tag'], path, '--repo', REPOSITORY)
                assets = gh('api', f'repos/{REPOSITORY}/releases/{release["id"]}/assets?per_page=100', json_output=True)
                remote = next(v for v in assets if v['name'] == entry['name'])
            if remote['size'] != entry['bytes']:
                raise ValueError('Remote asset length differs; never overwrite it automatically')
            if remote.get('digest'):
                if remote['digest'] != 'sha256:' + entry['sha256']:
                    raise ValueError('Remote asset SHA256 mismatch')
            else:
                with tempfile.TemporaryDirectory(prefix='verify-upload-', dir=directory) as temporary:
                    gh('release', 'download', plan['release_tag'], '--repo', REPOSITORY,
                       '--pattern', entry['name'], '--dir', temporary)
                    if digest(Path(temporary) / entry['name']) != entry['sha256']:
                        raise ValueError('Downloaded asset SHA256 mismatch')
            entry = dict(entry, github_asset_id=remote['id'], remote_verified=True)
        index['assets'] = [r for r in index['assets'] if r['name'] != entry['name']] + [entry]
        temporary = index_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(index, indent=2) + '\n')
        temporary.replace(index_path)
        print(json.dumps(dict(stage='local_verified' if args.local_only else 'uploaded_verified', **entry)), flush=True)
        if not args.local_only:
            path.unlink()  # Only the verified, regenerable temporary shard, never source data.
    if not args.local_only and args.package is None:
        required = {p['name'] for p in plan['packages']}
        if {r['name'] for r in index['assets'] if r.get('remote_verified')} != required:
            raise ValueError('Release incomplete')
        gh('release', 'upload', plan['release_tag'], index_path, args.inventory, '--repo', REPOSITORY)
        gh('release', 'edit', plan['release_tag'], '--repo', REPOSITORY, '--draft=false')
        print('DATA_RELEASE_PUBLISHED', plan['release_tag'], flush=True)


def download(args, plan):
    root = args.source.resolve()
    if str(root) != plan['canonical_root']:
        raise ValueError('This audited snapshot requires the documented canonical project path')
    args.cache.mkdir(parents=True, exist_ok=True)
    index_path = args.cache / 'release_index.json'
    if not index_path.exists():
        gh('release', 'download', plan['release_tag'], '--repo', REPOSITORY,
           '--pattern', 'release_index.json', '--dir', args.cache)
    index = json.loads(index_path.read_text())
    if index['inventory_sha256'] != digest(args.inventory):
        raise ValueError('Code and data release inventory differ')
    entries = {r['name']: r for r in index['assets']}
    for package in plan['packages']:
        if args.package and args.package != package['name']:
            continue
        row = entries[package['name']]
        path = args.cache / row['name']
        if not path.exists():
            gh('release', 'download', plan['release_tag'], '--repo', REPOSITORY,
               '--pattern', path.name, '--dir', args.cache)
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Bad downloaded shard: {path}')
        restore_shard(path, package, root)
        path.unlink()  # Re-downloadable cache only; extracted files stay.
        print('DATA_RESTORED', row['name'], flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['publish', 'download'])
    parser.add_argument('--inventory', type=Path, default=HERE / 'data_inventory.json')
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/EvolveVLA'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--package')
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--reserve-gib', type=int, default=60)
    parser.add_argument('--target', default='main')
    args = parser.parse_args()
    if args.reserve_gib < 25:
        parser.error('Reserve must be at least 25 GiB')
    plan = json.loads(args.inventory.read_text())
    (publish if args.mode == 'publish' else download)(args, plan)


if __name__ == '__main__':
    main()
