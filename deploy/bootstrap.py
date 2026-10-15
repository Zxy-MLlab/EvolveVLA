#!/usr/bin/env python3
"""Deploy on a fresh server. Refuses to modify a running training server."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path('/root/autodl-tmp/EvolveVLA')
HERE = Path(__file__).resolve().parent
ENVS = ROOT.parent / 'conda_envs'
ROBOTWIN = ROOT / 'robotwin_repo_ba0fd21'
OPENPI = ROBOTWIN / 'XPolicyLab/policy/Pi_05/openpi'
MODEL = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
ROBOTWIN_REV = 'ba0fd21ad21c336bedc7bd8eacd3e514860910e2'
XPOLICY_REV = 'c37109c500be67d0dea6b36bf7337bbd26e763cd'
MODEL_REV = '37e10d7f4f49149e2ce5e08a77b07f228a84322f'
DATA_REV = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
CUROBO_REV = 'd64c4b005459db10c5dd867d8b30a87d5bda9bdb'


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_fresh_server():
    if HERE.parent.resolve() != ROOT:
        raise ValueError(f'Clone physically at {ROOT}; this snapshot retains audited absolute paths')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(a.decode(errors='replace')).name in
               ('gen1_replay_lora_train.py', 'gen1_replay_lora_extend.py') for a in args if a):
            raise RuntimeError('Training is active on this machine; bootstrap refuses to touch its environment')


def checkout(url, target, revision):
    if target.exists():
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise ValueError(f'Existing checkout differs: {target}; preserving it')
        return
    run('git', 'clone', '--filter=blob:none', '--no-checkout', url, target)
    run('git', '-C', target, 'checkout', '--detach', revision)


def upstream():
    checkout('https://github.com/RoboTwin-Platform/RoboTwin.git', ROBOTWIN, ROBOTWIN_REV)
    child = ROBOTWIN / 'XPolicyLab'
    # Git leaves an empty directory for the uninitialized submodule.
    if child.is_dir() and not any(child.iterdir()):
        child.rmdir()
    checkout('https://github.com/XPolicyLab/XPolicyLab.git', child, XPOLICY_REV)
    for name in ('data', 'outputs', 'logs', 'checkpoints', 'tmp/gen1_training/jax', 'third_party'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def environments(args):
    conda = shutil.which('conda') or str(Path('/root/miniconda3/bin/conda'))
    names = ['gen1_train_ready', 'scratch_orbax', 'robotwin_cu128']
    if args.include_converter:
        names.append('evolvevla_pi312')
    for name in names:
        environment = ENVS / name
        if environment.exists():
            raise ValueError(f'Existing environment preserved: {environment}; inspect before resuming')
        snapshot = json.loads((HERE / 'environments' / f'{name}.json').read_text())
        version = snapshot['python'].split()[0]
        run(conda, 'create', '-y', '-p', environment, f'python={version}', 'pip')
        python = environment / 'bin/python'
        index = 'https://download.pytorch.org/whl/cpu' if name == 'gen1_train_ready' else 'https://download.pytorch.org/whl/cu128'
        # Exact observed packages; historical environments contain dependency
        # conflicts, so do not allow a fresh resolver to silently upgrade JAX.
        run(python, '-m', 'pip', 'install', '--no-deps', '--extra-index-url', index,
            '-r', HERE / 'environments' / f'{name}.txt')
        if name in ('gen1_train_ready', 'scratch_orbax'):
            run(python, '-m', 'pip', 'install', '--no-deps', '-e', OPENPI,
                '-e', OPENPI / 'packages/openpi-client')
        for row in snapshot['modified']:
            library = subprocess.check_output([str(python), '-c',
                'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
            target = Path(library) / row['relative']
            expected = row['original_record_sha256'].split('=', 1)[1]
            original = base64.urlsafe_b64encode(bytes.fromhex(sha(target))).decode().rstrip('=')
            if original != expected:
                raise ValueError(f'Installed upstream file differs before override: {target}')
            replacement = HERE / 'runtime_overrides' / name / row['relative']
            if sha(replacement) != row['sha256']:
                raise ValueError('Snapshot override hash changed')
            shutil.copy2(replacement, target)
        print('ENVIRONMENT_INSTALLED', name, flush=True)
    # Required by the RoboTwin expert admission path. Keep this independent of
    # model inference and use the exact source revision observed on this server.
    curobo = ROBOTWIN / 'envs/curobo'
    checkout('https://github.com/NVlabs/curobo.git', curobo, CUROBO_REV)
    run(ENVS / 'robotwin_cu128/bin/python', '-m', 'pip', 'install', '--no-deps',
        '--no-build-isolation', '-e', curobo)


def model():
    python = ENVS / 'gen1_train_ready/bin/python'
    script = """from huggingface_hub import snapshot_download
snapshot_download(repo_id='Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k',
 revision='37e10d7f4f49149e2ce5e08a77b07f228a84322f', local_dir=%r,
 allow_patterns=['params/**','assets/**','rldx2_openpi.json','README.md','SHA256SUMS'])
""" % str(MODEL)
    run(python, '-c', script)
    for line in (HERE / 'model_reference/SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(MODEL / name.strip().lstrip('*')) != expected:
            raise ValueError(f'Model checksum mismatch: {name}')
    token = ROOT.parent / 'openpi_cache/big_vision/paligemma_tokenizer.model'
    token.parent.mkdir(parents=True, exist_ok=True)
    if not token.exists():
        urllib.request.urlretrieve('https://storage.googleapis.com/big_vision/paligemma_tokenizer.model', token)
    if sha(token) != '8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6':
        raise ValueError('Tokenizer SHA256 mismatch')
    print('PUBLIC_MODEL_AND_TOKENIZER_VERIFIED', flush=True)


def simulation_assets():
    assets = ROBOTWIN / 'assets'
    python = ENVS / 'gen1_train_ready/bin/python'
    script = """from huggingface_hub import snapshot_download
snapshot_download(repo_id='TianxingChen/RoboTwin2.0', repo_type='dataset',
 revision=%r, allow_patterns=['background_texture.zip','embodiments.zip','objects.zip'],local_dir=%r)
""" % (DATA_REV, str(assets))
    run(python, '-c', script)
    for name in ('background_texture', 'embodiments', 'objects'):
        if (assets / name).exists():
            raise ValueError(f'Existing asset directory preserved: {name}')
        with zipfile.ZipFile(assets / (name + '.zip')) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe asset ZIP path')
            archive.extractall(assets)
    run(ENVS / 'robotwin_cu128/bin/python', ROBOTWIN / 'scripts/update_embodiment_config_path.py', cwd=ROBOTWIN)
    oidn = ROOT / 'third_party/oidn-2.5.1'
    if not oidn.exists():
        bundle = ROOT / 'third_party/oidn-2.5.1.x86_64.linux.tar.gz'
        urllib.request.urlretrieve('https://github.com/RenderKit/oidn/releases/download/v2.5.1/oidn-2.5.1.x86_64.linux.tar.gz', bundle)
        with tarfile.open(bundle) as archive:
            archive.extractall(ROOT / 'third_party', filter='data')
    if not (oidn / 'lib/libOpenImageDenoise.so.2').is_file():
        raise ValueError('OIDN runtime missing after extraction')


def verify():
    expected = json.loads((HERE / 'data_inventory.json').read_text())
    missing = [r['path'] for p in expected['packages'] for r in p['files']
               if not (ROOT / r['path']).is_file() or (ROOT / r['path']).stat().st_size != r['bytes']]
    if missing:
        raise ValueError(f'{len(missing)} data files missing or wrong size; first: {missing[:3]}')
    for name in ('gen1_train_ready', 'scratch_orbax'):
        run(ENVS / name / 'bin/python', '-B', '-c',
            'import jax,flax,orbax.checkpoint; print(jax.__version__,jax.devices())')
    run(ENVS / 'robotwin_cu128/bin/python', '-B', '-c', 'import sapien,curobo,mplib; print("SIM_IMPORTS_OK")')
    norm = MODEL / 'assets/aloha_agilex_sim/norm_stats.json'
    if sha(norm) != 'ec3f94d2763fb6e018cca42d6e0ee068eab53c041c522dbbc18ec0be9132ace8':
        raise ValueError('Wrong model normalization assets')
    print('INSTALLATION_CHECK_PASSED; next run deploy/start_experiment.py', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['upstream', 'environments', 'model', 'simulation-assets', 'verify'])
    parser.add_argument('--include-converter', action='store_true')
    args = parser.parse_args()
    require_fresh_server()
    if args.stage == 'environments':
        environments(args)
    else:
        {'upstream': upstream, 'model': model, 'simulation-assets': simulation_assets, 'verify': verify}[args.stage]()


if __name__ == '__main__':
    main()
