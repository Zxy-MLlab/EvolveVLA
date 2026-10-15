#!/usr/bin/env python3
"""Read-only source snapshot; never edit the running experiment."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def snapshot(source, target):
    source, target = source.resolve(), target.resolve()
    if source == target or source in target.parents:
        raise ValueError('Publishing checkout must be outside the live project')
    copied = []

    def copy(relative):
        path = source / relative
        if not path.is_file() or path.is_symlink():
            return
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=digest(path)))

    for directory in ('scripts', 'evolvevla', 'tools', 'tests', 'src', 'pi05_eval', 'configs'):
        for path in sorted((source / directory).rglob('*')):
            if (path.suffix in {'.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml', '.toml'}
                    and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts
                    and not any(part.startswith('.') for part in path.relative_to(source).parts)):
                copy(path.relative_to(source))
    for name in ('install_final.sh', 'install_final_v2.sh', 'constraints.txt',
                 'download_pi05_official.sh', 'download_pi05_selective.sh'):
        copy(Path(name))
    for name in ('experiment_progress_report_20261015.md',
                 'gen1_official_replay_manifest_20261013.json',
                 'gen1_two_task_1000_collection_final_audit_20261014.json',
                 'two_local_training_scope_20261015.json'):
        copy(Path('outputs') / name)
    for path in (source / 'artifacts/gen0_myungkyu_59999_clean').glob('*'):
        if path.suffix in ('.json', '.md', '.jsonl'):
            copy(path.relative_to(source))
    for split in ('gen1_multitask_coverage_20261012', 'gen1_multitask_coverage_expanded_20261014'):
        copy(Path('data') / split / 'split_manifest.json')
    model = source / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
    for name in ('SHA256SUMS', 'rldx2_openpi.json'):
        dest = target / 'deploy/model_reference' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model / name, dest)
    write(target / 'deploy/source_inventory.json', dict(source=str(source), files=copied))

    # Raw hashes already certified by the completed collection and full-frame audit.
    # Avoid reading 40+ GiB again while training is active.
    split = json.loads((source / 'data/gen1_multitask_coverage_expanded_20261014/split_manifest.json').read_text())
    groups = {'local_raw': {}, 'training_ready': {}}
    for row in split['episodes']:
        path = Path(row['path'])
        relative = str(path.relative_to(source))
        groups['local_raw'][relative] = dict(path=relative, bytes=path.stat().st_size,
            sha256=row['sha256'], task=row['task'], split=row['split'], seed=row['seed'],
            hash_source='completed source/full-frame audit; verified again while packaging')
    for path in (source / 'data/gen1_task_batches_20261010').glob('*/manifest.json'):
        relative = str(path.relative_to(source))
        groups['local_raw'][relative] = dict(path=relative, bytes=path.stat().st_size, sha256=digest(path))
    for directory in ('gen1_multitask_coverage_expanded_train_20261014',
                      'gen1_multitask_coverage_val_20261012', 'gen1_official_replay_train_20261013'):
        for path in sorted((source / 'data' / directory).rglob('*')):
            if path.is_file() and not path.is_symlink():
                relative = str(path.relative_to(source))
                groups['training_ready'][relative] = dict(path=relative, bytes=path.stat().st_size,
                                                        sha256=digest(path))
    packages = []
    for group, items in groups.items():
        files = list(items.values())
        # Independent tar shards, each < 1 GiB including headers; no full duplicate.
        current, size, number = [], 0, 0
        for item in files:
            estimated = ((item['bytes'] + 511) // 512 + 8) * 512
            if estimated > 900 * 2**20:
                raise ValueError('Individual asset too large for release shard')
            if current and size + estimated > 900 * 2**20:
                packages.append(dict(name=f'{group}-{number:03d}.tar', group=group, files=current))
                current, size, number = [], 0, number + 1
            current.append(item)
            size += estimated
        if current:
            packages.append(dict(name=f'{group}-{number:03d}.tar', group=group, files=current))
    write(target / 'deploy/data_inventory.json', dict(schema=1, canonical_root=str(source),
        release_tag='data-20261015-v1', accepted_local_episodes=2800,
        selected_local_training_episodes=1990, official_training_episodes=900,
        validation_episodes=90, packages=packages,
        total_file_bytes=sum(f['bytes'] for p in packages for f in p['files']),
        note='All 2800 accepted local trajectories retained; other 16 local tasks excluded from current training. '
             'Includes finalized official replay, with upstream provenance. No model weights.'))

    probe = r'''
import base64,csv,hashlib,importlib.metadata as m,json,sys
packages={}; modified=[]
for d in m.distributions():
 name=d.metadata['Name']; key=name.lower().replace('_','-')
 packages[key]=dict(name=name,version=m.version(name))
 if key not in ('sapien','mplib','lerobot','transformers','datasets','orbax-checkpoint','jax','flax'):
  continue
 record=d.read_text('RECORD')
 if not record: continue
 for rel,expected,size in csv.reader(record.splitlines()):
  if not rel.endswith('.py') or not expected.startswith('sha256='): continue
  path=d.locate_file(rel)
  if not path.is_file(): continue
  actual=hashlib.sha256(path.read_bytes()).digest()
  encoded=base64.urlsafe_b64encode(actual).decode().rstrip('=')
  if encoded!=expected.split('=',1)[1]:
   modified.append(dict(package=name,version=m.version(name),path=str(path),relative=rel,
                        sha256=actual.hex(),original_record_sha256=expected))
print(json.dumps(dict(python=sys.version,packages=packages,modified=modified)))
'''
    for name in ('gen1_train_ready', 'scratch_orbax', 'robotwin_cu128', 'evolvevla_pi312'):
        python = source.parent / 'conda_envs' / name / 'bin/python'
        value = json.loads(subprocess.check_output([str(python), '-B', '-c', probe], text=True))
        write(target / 'deploy/environments' / f'{name}.json', value)
        excluded = {'openpi', 'openpi-client', 'nvidia-curobo', 'pip', 'setuptools', 'wheel'}
        lock = '\n'.join(f"{v['name']}=={v['version']}" for k, v in sorted(value['packages'].items())
                         if k not in excluded) + '\n'
        (target / 'deploy/environments' / f'{name}.txt').write_text(lock)
        for row in value['modified']:
            dest = target / 'deploy/runtime_overrides' / name / row['relative']
            if '..' in Path(row['relative']).parts:
                raise ValueError('Unsafe installed package path')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row['path'], dest)
    print(json.dumps(dict(code_files=len(copied), data_shards=len(packages),
                         asset_gib=round(sum(f['bytes'] for p in packages for f in p['files'])/2**30, 2))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.source, args.target)
