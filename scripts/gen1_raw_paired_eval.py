#!/usr/bin/env python3
"""External raw-policy paired evaluator; panel execution remains disabled.

Official template bindings come from existing expert episode_info at runtime;
CPU preparation freezes templates, not invented bindings. Integration, tracing,
manifest and supervision code exists; execution remains disabled for primary
review. Shared-storage external writes cannot be bounded by this process. This
module loads no tensors at import.
"""
import contextlib
import hashlib
import json
import string
import signal
import time
import os
from pathlib import Path
import re
import subprocess
import selectors
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OUTPUTS = ROOT / 'outputs'
OPENPI = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
PILOT = ROOT / 'checkpoints/gen1_pilot_2000_20261010/1999'
GIB = 2**30
LIMIT = 2 * GIB
MIN_FREE = 70 * GIB


def df_available(path):
    """Byte-precision quota-aware reading; every malformed result fails closed."""
    env = dict(os.environ, LC_ALL='C')
    result = subprocess.run(['/usr/bin/df', '-B1', '--output=avail', '--', str(path)],
                            check=True, capture_output=True, text=True, env=env,
                            timeout=10)
    lines = result.stdout.strip().splitlines()
    if (len(lines) != 2 or lines[0].strip() != 'Avail' or
            re.fullmatch(r'[0-9]+', lines[1].strip()) is None):
        raise RuntimeError('Cannot parse df available bytes')
    return int(lines[1].strip())


def guard(out, pending=0):
    """Preserve raw diagnostic byte/reserve constraints and bound output scope."""
    out = Path(out)
    if type(pending) is not int or pending < 0:
        raise ValueError('pending must be nonnegative integer bytes')
    if out.is_symlink() or out.resolve().parent != OUTPUTS.resolve():
        raise ValueError('Output must be a direct child of established outputs root')
    size = 0
    if out.exists():
        if not out.is_dir():
            raise ValueError('Output must be a directory')
        # scandir errors propagate; Path.rglob can suppress unreadable directories.
        def visit(directory):
            total = 0
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise ValueError('Symlink inside output tree')
                    if entry.is_dir(follow_symlinks=False):
                        total += visit(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    else:
                        raise ValueError('Unexpected output file type')
            return total
        size = visit(out)
    if size + pending > LIMIT:
        raise RuntimeError('Diagnostic output exceeds 2 GiB including pending writes')
    if df_available(out if out.exists() else out.parent) - pending < MIN_FREE:
        raise RuntimeError('Diagnostic requires 70 GiB free after planned write')


@contextlib.contextmanager
def replaced(obj, attribute, value):
    """Restore exact original attribute ownership, including on exceptions."""
    owned = attribute in vars(obj)
    original = getattr(obj, attribute)
    setattr(obj, attribute, value)
    try:
        yield
    finally:
        if owned:
            setattr(obj, attribute, original)
        else:
            delattr(obj, attribute)


def raw_factory(policy_config, raw, evidence_path):
    """Build the exact production transform pipeline around selective raw_load.

    Dependencies are passed explicitly for CPU mocks. The guard replacement is
    scoped only around raw_load; neither frozen module is changed on disk.
    """
    evidence_path = Path(evidence_path)

    def create(train_config, checkpoint_dir, *, repack_transforms=None,
               sample_kwargs=None, default_prompt=None, norm_stats=None,
               pytorch_device=None):
        checkpoint_dir = Path(checkpoint_dir).resolve()
        if checkpoint_dir != PILOT.resolve():
            raise ValueError('Raw factory requires exact pilot checkpoint')
        if norm_stats is not None:
            raise ValueError('Custom normalization is forbidden')
        if sample_kwargs is not None:
            raise ValueError('Custom sampling arguments are forbidden')
        if getattr(train_config.model, 'action_horizon', None) != 50:
            raise ValueError('Raw policy requires full 50-action horizon')
        if getattr(train_config.model, 'action_dim', None) != 32:
            raise ValueError('Raw policy requires model action dimension 32')
        transforms = policy_config.transforms
        repack = repack_transforms or transforms.Group()
        with contextlib.ExitStack() as stack:
            stack.enter_context(replaced(raw.d, 'guard', guard))
            if hasattr(raw.d, 'write_json'):
                stack.enter_context(replaced(raw.d, 'write_json',
                    lambda path, value: checked_write(path, value, evidence_path.parent.parent)))
            model, report = raw.raw_load(train_config, checkpoint_dir, evidence_path)
        if report.get('kind') != 'raw' or report.get('internal_step') != 2000:
            raise ValueError('Unexpected raw restore evidence')
        dc = train_config.data.create(train_config.assets_dirs, train_config.model)
        if dc.asset_id is None or dc.use_quantile_norm is not True:
            raise ValueError('Production quantile normalization required')
        stats = policy_config._checkpoints.load_norm_stats(checkpoint_dir / 'assets', dc.asset_id)
        return policy_config._policy.Policy(
            model,
            transforms=[*repack.inputs, transforms.InjectDefaultPrompt(default_prompt),
                        *dc.data_transforms.inputs,
                        transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm),
                        *dc.model_transforms.inputs],
            output_transforms=[*dc.model_transforms.outputs,
                               transforms.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
                               *dc.data_transforms.outputs, *repack.outputs],
            sample_kwargs=sample_kwargs, metadata=train_config.policy_metadata,
            is_pytorch=False, pytorch_device=None)
    return create


def build_adapter(label, directory):
    """Explicit runtime API only; not called by preflight or module import."""
    if label not in ('original', 'raw'):
        raise ValueError('Expected original or raw')
    import controlled_paired_eval as c
    from pi05_openpi_adapter import PublicPi05Adapter
    from openpi.policies import policy_config
    kwargs = dict(device='cuda:0', prompt_prefix='none', normalization='quantile',
                  policy_seed=0, noise_source='jax')
    if label == 'original':
        return PublicPi05Adapter(c.MODELS['reference'], **kwargs)
    import gen1_raw_diagnostic as raw
    with replaced(policy_config, 'create_trained_policy',
                  raw_factory(policy_config, raw, Path(directory) / 'raw_restore.json')):
        return PublicPi05Adapter(str(PILOT), **kwargs)


def stable_index(task, seed, field, count):
    if type(seed) is not int or count < 1:
        raise ValueError('Invalid deterministic selection inputs')
    digest = hashlib.sha256(f'gen1-raw-paired-v1:{task}:{seed}:{field}'.encode()).digest()
    return int.from_bytes(digest, 'big') % count


def template_spec(task, seed, description_root):
    if task not in ('hanging_mug', 'place_mouse_pad'):
        raise ValueError('Unexpected task')
    path = Path(description_root) / 'task_instruction' / (task + '.json')
    data = path.read_bytes()
    pool = json.loads(data)['seen']
    if not isinstance(pool, list) or not pool or not all(isinstance(x, str) and x for x in pool):
        raise ValueError('Invalid official seen pool')
    index = stable_index(task, seed, 'template', len(pool))
    return dict(task=task, seed=seed, pool='seen', index=index, template=pool[index],
                source=str(path.resolve()), source_sha256=hashlib.sha256(data).hexdigest(),
                binding_rule='sha256-gen1-raw-paired-v1-modulo-seen-v1')


def bind_prompt(spec, episode_info, description_root):
    """Bind only existing expert fields; deterministic official seen descriptions.

    Never call the official random.choice binder, infer a color/arm, or fall back
    to another template. Return auditable raw fields and description identities.
    """
    if spec != template_spec(spec['task'], spec['seed'], description_root):
        raise ValueError('Frozen template identity changed')
    params = episode_info.get('info') if isinstance(episode_info, dict) else None
    if not isinstance(params, dict):
        raise ValueError('Missing expert binding payload')
    required = {'{A}', '{B}'} | ({'{a}'} if spec['task'] == 'place_mouse_pad' else set())
    if not required.issubset(params):
        raise ValueError('Missing expert binding fields')
    if set(params) != required:
        raise ValueError('Unexpected expert binding fields')
    fields = {}
    for key in sorted(required):
        value = params[key]
        if not isinstance(value, str) or not value or '{' in value or '}' in value:
            raise ValueError('Invalid expert binding value')
        fields[key] = value
    prefix = '039_mug/base' if spec['task'] == 'hanging_mug' else '047_mouse/base'
    if re.fullmatch(re.escape(prefix) + r'\d+', fields['{A}']) is None:
        raise ValueError('Expert object identity outside task pool')
    if spec['task'] == 'hanging_mug':
        if fields['{B}'] != '040_rack/base0':
            raise ValueError('Expert rack identity changed')
    elif fields['{B}'] not in ('Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta', 'Black', 'Gray'):
        raise ValueError('Unknown expert mat color')
    substitutions, objects = {}, {}
    object_root = (Path(description_root) / 'objects_description').resolve()
    for key, value in fields.items():
        if key == '{a}':
            if value not in ('left', 'right'):
                raise ValueError('Unknown expert arm')
            substitutions[key[1:-1]] = f'the {value} arm'
        elif '/' in value or '\\' in value:
            if '\\' in value:
                raise ValueError('Invalid object description path')
            path = (object_root / (value + '.json')).resolve()
            if not path.is_relative_to(object_root):
                raise ValueError('Object description path escapes source tree')
            data = path.read_bytes()
            pool = json.loads(data)['seen']
            if not isinstance(pool, list) or not pool or not all(isinstance(x, str) and x for x in pool):
                raise ValueError('Missing seen object descriptions')
            index = stable_index(spec['task'], spec['seed'], key, len(pool))
            substitutions[key[1:-1]] = 'the ' + pool[index]
            objects[key] = dict(source=str(path), source_sha256=hashlib.sha256(data).hexdigest(),
                                index=index, description=pool[index])
        else:
            substitutions[key[1:-1]] = value
    for _, key, format_spec, conversion in string.Formatter().parse(spec['template']):
        if key is not None and (key not in substitutions or format_spec or conversion):
            raise ValueError('Unsupported or missing placeholder')
    prompt = spec['template'].format_map(substitutions)
    if '{' in prompt or '}' in prompt:
        raise ValueError('Unbound placeholder')
    return dict(prompt=prompt, fields=fields, objects=objects, template=spec)


# Generated from a read-only audit; replaced only by an explicit source re-audit.
HISTORICAL_FILES = {
    "artifacts/correctness_fixes_report.md": "98d03efbcc8adc0209e2a92836ec75c3b88a0ffc520f566f6a7af12968f07c94",
    "artifacts/d013_alt_source_probe.json": "18e330ec4abae2ea7316988072c23a86d5bfb3338832b31639eb1c1ded70c653",
    "artifacts/eval_integration_report.md": "3de6a8b586c530b0c7f875520781acb7b8361d53cee7b481f2d91961bcfb08dc",
    "artifacts/gen0/baseline_report.md": "d8c56f6f2da8d6aacdd8454ed830a02e0785c373399ceba393e5e1da448ca0a3",
    "artifacts/gen0/checkpoint_reference.json": "c5c7cb8b26b5f6e7fbeaa4e4bf517fb48fa33571848c2f2300237829b11ebc87",
    "artifacts/gen0/eval_clean.json": "7f937e1de5d5fc35d99473d910c2d260dae8839594f52b77fe64e568d18e7c1a",
    "artifacts/gen0/eval_randomized.json": "180dfad6916cb81b4991e975f41d7bc83d30c6837a8271bca74c25310a793320",
    "artifacts/gen0/failure_episodes.jsonl": "daf06b00428c5f6318428d3b7a872d16c5ea188fa0e79503964afd6b01ef5357",
    "artifacts/gen0/success_episodes.jsonl": "b784acb9dbb0da01bb2acdc58ec2e0e134ac1b60452392bf98898cb366d2cd9f",
    "artifacts/gen0_myungkyu_59999_clean/baseline_report.md": "e276360432cfff9bcc07ce24e518fb2fe7202243d8095265d80dae58dd49366a",
    "artifacts/gen0_myungkyu_59999_clean/checkpoint_reference.json": "a98839da7c7a763f66e67e3b1bf06d7e619cc800823dea34fa3d2f4fef2ee852",
    "artifacts/gen0_myungkyu_59999_clean/eval_clean.json": "b30abe3b560935968c987ec9f431d5c72d7deac42763869a7e24a85db3f10c86",
    "artifacts/gen0_myungkyu_59999_clean/failure_episodes.jsonl": "c7b67b5b40c8851d4b8d389e57df4c17d325e34f03baf36376e093a8bef27bb7",
    "artifacts/gen0_myungkyu_59999_clean/official_reference.json": "f4e6d43bb3e214088aeb0a518f1dd1c3cd50358a4ef4a8455a019a647c3182a7",
    "artifacts/gen0_myungkyu_59999_clean/success_episodes.jsonl": "94d57ff2f6cb40260bde1a13851541a914fe7669ca8050766640c161803027f6",
    "artifacts/h2_diagnosis_prep_report.md": "9208e2e451194c5ab1f8cd0d44d862ea03214b7ba981a461f6460bdcc14d3cd9",
    "artifacts/measurement_readiness_report.md": "57ea44517825a8f88ca68c3a8028188cd7157f460b91b91f5586519a42158eea",
    "artifacts/oracle_complete_report.md": "56e627e3a7736bfda4d81afc10f0a2df4c59ebee0526912d624541e4e7781b71",
    "artifacts/paligemma_official_tokenizer_report.md": "e14f4a6b9237833ba4380ecc615ff453b9b1c4bb53a3359f8beea07afd4d5dbe",
    "artifacts/positive_control_v2_report.md": "9678b5b2497e470ea625eac2f5263833bc57616ca36d9ba82363943596bec683",
    "artifacts/robotwin_gate1_sapien_install.log": "1a217e841181d7a890d199ce21b73c86a579e116efb5eb08e410f87ac39de056",
    "artifacts/robotwin_phaseB_report.md": "041fa66443e82b87646c40ca74bdb02708555768e6121083c87f43cdd13e6055",
    "artifacts/stage0_progress_report.md": "2308b6ec8a838fca1ce33d5bff702b947b089ed16e5ac1fe0fbe23d3cf35f040",
    "artifacts/tokenizer_gate_output.txt": "a544c183bcb8cb735cad3dd50fbb8dcedd84e6396efddf8e62efff9e5d7e872a",
    "artifacts/tokenizer_guard_passed.md": "d23afad330af4ea7d0644f685e8c990073faba7edf3737acba49e7f4e7fa15c5",
    "artifacts/tokenizer_source_guard_report.md": "76d2f3354e179a95c5444053d7bb7f887a8cf7fafdf7893a7acf71a0091f0a3c",
    "data/gen1_pilot_two_task_20261010/split_manifest.json": "9bbdeccc8692812c1ec1e9918c14a8cce5f91ff4f01e3b0d0bb34887b977efe5",
    "data/gen1_task_batches_20261010/hanging_mug/historical_seed_inventory.json": "e2dd3172bb20acff77191fcae4e0d86797d061fc42dd3ce22cb6d47e22d81e13",
    "data/gen1_task_batches_20261010/hanging_mug/manifest.json": "f0fdbac2202c14f750623bba726fed1ec5ccb77ed717171b5274873c0c64206c",
    "data/gen1_task_batches_20261010/place_mouse_pad/historical_seed_inventory.json": "3ab6775fe51d417b6fdb14b5ccc10e7c466bc517c58c9a881b96d9131aaeb32f",
    "data/gen1_task_batches_20261010/place_mouse_pad/manifest.json": "8727c462277c09ac3d41cd68ebb8be0acb6f190a817c12b408556d42111ae308",
    "outputs/controlled_pair_full_20261011/manifest.json": "9ab5c5c47369788fad499586fa01cf1bde54b63e45d1358e9add69459edf5c5f",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100000/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100000/initial/hashes.json": "53d58ff1ef0a5f3649e7a26a3c24c5db8b17bfe6d781bc1ac2a863928e52033e",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100000/result.json": "f62d4235b2935232c075b6350dcd89dc0bf3e0c435ec3f4519222d6619f5e4b3",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100000/worker.log": "599ff46548a1917abec76e00e400aa635827019d8c3943511d22d8b1236f762e",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100001/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100001/initial/hashes.json": "ef0d5581d229fe618a2c9f22859dd53f88e6b285dc0307f892e4d735951ef9f6",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100001/result.json": "3b9d22c7d2f88c4c6a002f3c70af0c52f8a5559bf6e2a87aa36628bad64ee1f2",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100001/worker.log": "a5576e0141f0ca4f301188747f2aae3bb0a76dea2dbf585b8c96db6f011519e3",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100002/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100002/result.json": "5716dd45a5e1206866fbaa8fd2dfbcd8374b16924cd09fa33b46d549d41b3b1c",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100002/worker.log": "30a6d64fd09f86c6750965b85c45cf89dccd14d5d19d15dbb8a837fd6d5e7965",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100003/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100003/initial/hashes.json": "403c7696556a8d54a38a8a334e7d1e3b89058fc634f780e25de2c79097b0a17b",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100003/result.json": "5326be90e431ef82d410d48b6378290d5c777b4fd3ff4fe30c9376b894017e3f",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100003/worker.log": "36033cba2d52c94be145c9a7da5ff1868ca4e37a459e660de074737021022d24",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100004/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100004/initial/hashes.json": "d06361c815b01ac06616c827249ae3286753dc17bb6cf6624fc21d2d4d62a64a",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100004/result.json": "2d878d6a9d0667e102109448c5417e56585ab24ac063ada19a49c269a87fe21f",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100004/worker.log": "6de02653cbb9762d515c2688a84b47ac6146bdd203a85a15479beebd9b98ab4b",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100005/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100005/initial/hashes.json": "afdeb415951adc435fb12635e57855a8851a8c33eb205102000a53b60e9992e4",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100005/result.json": "3a4124240867cc2f19bfb802abbfa8919596f1d65dcfd56d7a8675bd9accc4cd",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100005/worker.log": "cd6fda1092acdf86625258e698dc1f1d7611fe6b4027fd056977bee3f03602c1",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100006/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100006/initial/hashes.json": "55135d25222b533461f7fc87be649f8e03b022fd4797d4f7296b332cd6e4c145",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100006/result.json": "32af6e38d71473dd19bd921512f4136de0a528a60a18e10b3ad332b9fc12519a",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100006/worker.log": "c4b9f4079f3b3c8ea4a9e731edcaf3bcbe67a3c5dcc79bbfd8a48a25f6bbde12",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100008/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100008/result.json": "0b0f677b728847053eb143a89d176896e1457767fb08ac3cc513181c6b33c4ef",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100008/worker.log": "a3e5f55d8896d63b09d529533296b63d00a24089da1c651b4b104d2f92b2b56f",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100009/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100009/initial/hashes.json": "f8bea9f5e06d129764d17c09921593786f1795c9716f37481294fe2da6a571aa",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100009/result.json": "b609f75cf7ed81002a4ce871a4c1ccdfc08359185f60060b4cf27abf6eda97ef",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100009/worker.log": "b6673326cb8eb04e88961832b1b6662a3f0fe59fdd9ef2c0858c4ba0b8bb4477",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100010/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100010/result.json": "e47cf2498afe5d87395f7582d14487e419450ecc97ba8ffae3f461120c79fee3",
    "outputs/controlled_pair_full_20261011/pilot/hanging_mug_100010/worker.log": "d92a45eadcdabb5b6e74a7038a1aa8cc63e93ffcbbcd865cda9112e136c3eada",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100000/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100000/initial/hashes.json": "9ac53ede98c27df455b2c150962a46849f658b3db9891960af07251913497a7c",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100000/result.json": "9a5b92f9e7dd9bb9914969210f3f1489ab70b88333df3f87ea03d5f8acccdc3b",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100000/worker.log": "cade8fe1888f18781d01482e48cb2275d8859d695408a924d501267c89b08930",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100001/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100001/initial/hashes.json": "ff3837b9096b1c3012bc74d3fd6e4c487a76b6d39128e67b62997fff975b2593",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100001/result.json": "ee0251c8a3c1369f085a3e7f9c2a909bafb1cdf0a863690b94ab8ef472a0b5e2",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100001/worker.log": "98c4d5b34c1de2a9d22919e60b53a2ab164edcfbe61915bd67ff8a10b8b316bc",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100002/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100002/initial/hashes.json": "e401354814295f857b79bcc25737a62cf193d31f1636e90af583b69965e1dc1b",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100002/result.json": "fa0f8bd96f7848e8ddb33f01866b239db752d91239e52bfdd07c56210932dcc3",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100002/worker.log": "bb6c35aa588eb988003dec5b9b1172601d147b53bdec04cca7d725d22bb44b9d",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100003/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100003/initial/hashes.json": "9e4793e20e7835f2ec26b4a5c1a9413d46a179d6305b75c682637676818f542b",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100003/result.json": "b87c322328b0025fa5f58674416f90a8686a71e876543a72e4562fed1db87fc2",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100003/worker.log": "3ac039455b44f1e59c27d2f70c0babdbccdbc7ad1cd712b3aa7cfeea15ad4056",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100004/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100004/initial/hashes.json": "88915d722369e024c00368cc51562c861bbd9afe2cd3dbc16c2239a1e0e45d34",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100004/result.json": "d4701e54e21555825ffc47245bae73bf33db51dee3cc0927d41ae605951f69c9",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100004/worker.log": "a5c2115a0717cf08351ed13a20436ecd73af07dc17693c0b9e2c022440e9c6c7",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100005/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100005/initial/hashes.json": "ac2bc671e180d4f0d9180cd1134239ec16d6dab7b2cb8ef697a54ea7ed843cbd",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100005/result.json": "317943d3492cd2c840e3636cce5751ec2b06a5129290835a375d37175b931724",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100005/worker.log": "fb3af56e6f1fa8b79b01e3fc21007cf588f8e8e9463c1293954a1d566670b679",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100006/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100006/initial/hashes.json": "0a7c8c315177ff67dc57006220d4c3d1f9c661a265f3f5e82adbbc754212d155",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100006/result.json": "fd800e60d76de8fce16f24786743cc6f349433bf697ec77aaa600aade7e93154",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100006/worker.log": "82388501e2829fa87fad6e20ef983e3a1372cf6be1eeb694eea41a0a8bc6ea87",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100007/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100007/initial/hashes.json": "56361c3464c96ce298565f9cf498128b9f01996e9c7948fe4db14f77fb5e296c",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100007/result.json": "063223d731f796388920eef473eec75dc4c20311e459b7f1ecf00053143d5b9f",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100007/worker.log": "92c3b43ab47cd956e8b10bc6e4b4db40b69958bc5a5f7a97723578755f338a48",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100008/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100008/initial/hashes.json": "c86b59ae7712d54a6067b9696e57edb80558105e3ed3acee674a29e2dba0b76d",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100008/result.json": "b4d6b3124d4ac9605fe307098c0c5e268158c03af5945c9e4f747dc97fcc97af",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100008/worker.log": "54ad6e7b2a68fa67b7dedd39b3f5565abb9f483fa8edb804a5827e0ad1cbd7ef",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100009/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100009/initial/hashes.json": "974ca667bd8e780f349deaa8e1782481edfc438b98ae29af6022c7ef203b46d4",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100009/result.json": "4db90152c3d3a22efa1b42209cd157e0353218ea74ec589f4da08fa4a6f277c0",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100009/worker.log": "34be39f36ba4571a486c004408462104192b7df671e82a1278bb8b76650ca9d4",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100010/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100010/initial/hashes.json": "a15c9a7daf5ff4fd2874de28f6c69b9bd9ae75c547045052d755c822e1a57a21",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100010/result.json": "32cfdf83555f8047ccd0e710ca854b16b04fc9a10c3b0db8560b14a581dd349f",
    "outputs/controlled_pair_full_20261011/pilot/place_mouse_pad_100010/worker.log": "6aa4cb9815958b881ea053460db14e26822a994a006f1c0ef07108b2f379dbca",
    "outputs/controlled_pair_full_20261011/progress_01.json": "2a289d5bf72ad5fdabbb4e3f596710b35c17b50208230cccc02fbbb95245e16f",
    "outputs/controlled_pair_full_20261011/progress_02.json": "c467321db32816e0db73be872f9e498d70deb93d59b271624c2220a2407d7c0c",
    "outputs/controlled_pair_full_20261011/progress_03.json": "87dfeb20617efcb51879effd099577ec918cd33940384ccb6fa2931f73b34d7f",
    "outputs/controlled_pair_full_20261011/progress_04.json": "a4c4f069917c09e2f7e83f2e757e5d5044d648e2ebe767f1b960fbf424a8d8d9",
    "outputs/controlled_pair_full_20261011/progress_05.json": "bd88d36d63d5239f3cbd5628e39f43a3cd8af0da571e6aa484f937e09b9fdfd7",
    "outputs/controlled_pair_full_20261011/progress_06.json": "8cacd803b741f60de9dc3e05e31e4f4739abd5b5333fe08359320b1a3c5ea8ed",
    "outputs/controlled_pair_full_20261011/progress_07.json": "be431984b0bea30fe180f7a98a13ac9f75902aaa91cccab8bfa6e722610a22b1",
    "outputs/controlled_pair_full_20261011/progress_08.json": "7157e7f48e8425c588c7154026ba13a53ff5d5e2a7494d2f93e370c2c69f4260",
    "outputs/controlled_pair_full_20261011/progress_09.json": "3c359dc37d34d5d09c7f7a57fdec5b1c14538007280419f0ca82909f928ce471",
    "outputs/controlled_pair_full_20261011/progress_10.json": "ae6ea08e3f8a5bb1e2b13b7504f301b3f55b9aa33d55c9e4a6117e9fd8e15200",
    "outputs/controlled_pair_full_20261011/progress_11.json": "a71817fb5f5b5e4af88c5ced0bbb531e37d829706ad08b5a3ece7388927c51fe",
    "outputs/controlled_pair_full_20261011/progress_12.json": "c575f65f8dce61bb749bcaf417d4729131909d2766f926fd8f6f8b936b6326ac",
    "outputs/controlled_pair_full_20261011/progress_13.json": "dfb0f0cefe7dff7f29ba203a5a1d1d524a9d36ea42ed4daee59feb0f44d2b741",
    "outputs/controlled_pair_full_20261011/progress_14.json": "4dce2428455a6841972926ff01fd4ed0e57f59cc620fe01d23fbb0c4e25b768f",
    "outputs/controlled_pair_full_20261011/progress_15.json": "ae7541e3832b237c5af2c50a61a20c44a20dc143c6b06478dd084b551b476e99",
    "outputs/controlled_pair_full_20261011/progress_16.json": "33ae8a0f62d765949e7701d241d40f292ee90163749e8a3508908be35d0d1d61",
    "outputs/controlled_pair_full_20261011/progress_17.json": "612669ee7ecf2115a375443b85f79a8e6bba8853842eb475f2287f4b6eac1549",
    "outputs/controlled_pair_full_20261011/progress_18.json": "563bce8668d6fecab884202b58f6346c5d1bbdc18397710b68c8cb36b26056b7",
    "outputs/controlled_pair_full_20261011/progress_19.json": "35ad9503b1ad93bd0fb93861d2ff093f3faa113726355cd435f09ede60a5d9dc",
    "outputs/controlled_pair_full_20261011/progress_20.json": "a6f1474a43003fe8fc66642299654f7a609bb7316289fe6871ba88816127e5eb",
    "outputs/controlled_pair_full_20261011/progress_21.json": "9821c72de962027fc96f00a9780043b57949769b2132f577cd0e6610dae97bd2",
    "outputs/controlled_pair_full_20261011/progress_22.json": "502f4b4cf1f10fc136672dce60c172e667cdb86e6ebf3161afca5fcf1236476a",
    "outputs/controlled_pair_full_20261011/progress_23.json": "64361e1fa22557f72e721c6fdaf3da58c3678b8ad8f75df153d9a701719c508b",
    "outputs/controlled_pair_full_20261011/progress_24.json": "bc3264253d0d5a037d5e6d395b2de0b0412c399cddd13d60ae473bdc00272878",
    "outputs/controlled_pair_full_20261011/progress_25.json": "5a09d3d09ee315e940c30a8ef27a2b2160b8d78b0f92e6c44da93ca72941b99b",
    "outputs/controlled_pair_full_20261011/progress_26.json": "6608dd52347913f24fe9c6c764e889222dddeff78f5be94281ca014c7688fa8e",
    "outputs/controlled_pair_full_20261011/progress_27.json": "30d6d6c8e8dd9cd8edc5979ccf4fb83671e9231f0929f8f6d61b79934dd2c2d6",
    "outputs/controlled_pair_full_20261011/progress_28.json": "a771d7ab43d7949cf0b24e5407125b976581283b2617961eba1a6282b2ab94cb",
    "outputs/controlled_pair_full_20261011/progress_29.json": "f0fca4bfc7a6e5a04fa0cc5842f3c2236001576da2f8e332d0aab8fe3b9561c1",
    "outputs/controlled_pair_full_20261011/progress_30.json": "7fb0d74ac479657d2e4e043e503b01a794556881243dfaf867a71c0a470e1b83",
    "outputs/controlled_pair_full_20261011/progress_31.json": "8a9bcdf39214266b8c12c31c033bd8bd2e555b6e3a4bb542e9c3d6fa5e89427d",
    "outputs/controlled_pair_full_20261011/progress_32.json": "c11a15166f0a936dc17fb9ae58b8a03c353d6fc100357532c92ce4d33ce143f6",
    "outputs/controlled_pair_full_20261011/progress_33.json": "ec7cfbe2b1744f27545b8c3072b1c8c09c47728ffe479c9d66efcf3bd58991cb",
    "outputs/controlled_pair_full_20261011/progress_34.json": "f9b25e2660f2c9aa76bed4003518c46cd590604790143f4bbcc9c58f28dcb180",
    "outputs/controlled_pair_full_20261011/progress_35.json": "c6dde3a2a716c6ce782cd6527244674b4c4d8a55d89c939a318881c8c0b55005",
    "outputs/controlled_pair_full_20261011/progress_36.json": "6bb5674f931d8223db822a83f40d687e9ee105847faf880091489ca525ea5253",
    "outputs/controlled_pair_full_20261011/progress_37.json": "c4c78f9a17ea15c83b89d236ff18e92f1e2bcf6c908c80c842b83e794ef30364",
    "outputs/controlled_pair_full_20261011/progress_38.json": "64f38cd5e2fd96f3b1f73b47d27653c2c4e8e214439b113a6c46a5f39822331d",
    "outputs/controlled_pair_full_20261011/progress_39.json": "3ebdccfa542a1889e996aa20e6ba31509b4677c17f94bdbffbc43a4d217da18c",
    "outputs/controlled_pair_full_20261011/progress_40.json": "4b781fc3261405fc5881d141594132ba2a961360c6c785fa91d6855a0f5d7b18",
    "outputs/controlled_pair_full_20261011/progress_41.json": "73bb5e967456070ca1a0182ed20489b6f29c99ea7585be453a752dd0526d1719",
    "outputs/controlled_pair_full_20261011/progress_42.json": "c357f3e0ad528da63ef0c7c9787a890ee04bb614b4e23fea020b09ef213778bf",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100000/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100000/initial/hashes.json": "53d58ff1ef0a5f3649e7a26a3c24c5db8b17bfe6d781bc1ac2a863928e52033e",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100000/result.json": "f62d4235b2935232c075b6350dcd89dc0bf3e0c435ec3f4519222d6619f5e4b3",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100000/worker.log": "e97f6e7bbfb8bba6e6812df0af9d695101d33107dd1accd6e15f43501bb18ca0",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100001/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100001/initial/hashes.json": "ef0d5581d229fe618a2c9f22859dd53f88e6b285dc0307f892e4d735951ef9f6",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100001/result.json": "f6344afdfdb47deaa45236f3fa2f474ef2c2939110e2f127a039e746045a063b",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100001/worker.log": "4328a0f93af3fb98f64e297c50a55dc2ec6aa82cc1b35f2e2f55f4941d8acd8a",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100002/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100002/result.json": "4c3da3a5afc39955734bd090ca10aa93702962207108dc2219a45b0c34e64994",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100002/worker.log": "cfe9650e53e25ac868ae77e6b1b57dba157cffe0a7e004bfe696b36614b1b9fb",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100003/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100003/initial/hashes.json": "403c7696556a8d54a38a8a334e7d1e3b89058fc634f780e25de2c79097b0a17b",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100003/result.json": "5326be90e431ef82d410d48b6378290d5c777b4fd3ff4fe30c9376b894017e3f",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100003/worker.log": "f14fddc3eeac9590cb23153ed0072976a4b1646abdcbb3114aded1a8e0a7dcbe",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100004/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100004/initial/hashes.json": "d06361c815b01ac06616c827249ae3286753dc17bb6cf6624fc21d2d4d62a64a",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100004/result.json": "2d878d6a9d0667e102109448c5417e56585ab24ac063ada19a49c269a87fe21f",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100004/worker.log": "fa5e2855fa8ad74cd4c8171d5362c2c9fa2a64d8be9de5ca37f00b0db26d6ecf",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100005/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100005/initial/hashes.json": "afdeb415951adc435fb12635e57855a8851a8c33eb205102000a53b60e9992e4",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100005/result.json": "3a4124240867cc2f19bfb802abbfa8919596f1d65dcfd56d7a8675bd9accc4cd",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100005/worker.log": "34d24a5469107d5becafe95f664f3faeb1e53223d972056c8348425013df9904",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100006/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100006/initial/hashes.json": "55135d25222b533461f7fc87be649f8e03b022fd4797d4f7296b332cd6e4c145",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100006/result.json": "32af6e38d71473dd19bd921512f4136de0a528a60a18e10b3ad332b9fc12519a",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100006/worker.log": "76d973eaa57d8346e17d40040a6f362e1a5ab136a6fe8aaffca572763cfeb0b5",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100008/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100008/initial/hashes.json": "d713f9f78ef1e01c5ed68f9bd597881eb6ee8d7007f4146ae791e059007ad472",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100008/result.json": "1009968ffcbb6528515714e8f2248453dbcbd8e0ee20436f93c7826dab4f49a5",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100008/worker.log": "c870b7b67cf330743baddea62a451364e90457da15b680b7d07cfc512c35ee99",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100009/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100009/result.json": "a146268b9fdcfda73b02ec9f1fb1b6a8aaa84b898fa3028f5c49bf1329d6f551",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100009/worker.log": "c1e0f15d5233575fdc5781a8fced0445cb2d2c03034823492017d3f2bd2a48d3",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100010/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100010/initial/hashes.json": "cb2aeecdb1cbdc77b65ecc4697bace2e485a90c682b44072bb789e35043310f3",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100010/result.json": "674e9aab54489ecb5408934f76793f516c348659244695c31e4f55638423cef4",
    "outputs/controlled_pair_full_20261011/reference/hanging_mug_100010/worker.log": "81a05634763cc1a252fed3d43fbaa73d80148cd9ecd4a3a196701f248fd9a6e5",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100000/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100000/initial/hashes.json": "9ac53ede98c27df455b2c150962a46849f658b3db9891960af07251913497a7c",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100000/result.json": "9a5b92f9e7dd9bb9914969210f3f1489ab70b88333df3f87ea03d5f8acccdc3b",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100000/worker.log": "d70dc59dbdb0f77bd1b4713a2f5ccf9a46b6b4b84d1fbfc06b1e87e64b97dcb0",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100001/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100001/initial/hashes.json": "ff3837b9096b1c3012bc74d3fd6e4c487a76b6d39128e67b62997fff975b2593",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100001/result.json": "ee0251c8a3c1369f085a3e7f9c2a909bafb1cdf0a863690b94ab8ef472a0b5e2",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100001/worker.log": "68d9a91e8a6b4d0713d86d8833059adadd116c70e223b08c23039608760b3058",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100002/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100002/initial/hashes.json": "e401354814295f857b79bcc25737a62cf193d31f1636e90af583b69965e1dc1b",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100002/result.json": "fa0f8bd96f7848e8ddb33f01866b239db752d91239e52bfdd07c56210932dcc3",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100002/worker.log": "151e107e19339deeb505d254ace10d916c46c2aada0762a0b16c293bd91069ba",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100003/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100003/initial/hashes.json": "9e4793e20e7835f2ec26b4a5c1a9413d46a179d6305b75c682637676818f542b",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100003/result.json": "b87c322328b0025fa5f58674416f90a8686a71e876543a72e4562fed1db87fc2",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100003/worker.log": "4422e75e8f87c26f06d24db97ace04c808e057214dea9e8fd966eb4f6ccf483c",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100004/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100004/initial/hashes.json": "88915d722369e024c00368cc51562c861bbd9afe2cd3dbc16c2239a1e0e45d34",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100004/result.json": "d4701e54e21555825ffc47245bae73bf33db51dee3cc0927d41ae605951f69c9",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100004/worker.log": "f41d865d3b7ec5daffe77c16b2209e77b3c29e4a8eb8b30ef1ce2347be3d8712",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100005/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100005/initial/hashes.json": "ac2bc671e180d4f0d9180cd1134239ec16d6dab7b2cb8ef697a54ea7ed843cbd",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100005/result.json": "317943d3492cd2c840e3636cce5751ec2b06a5129290835a375d37175b931724",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100005/worker.log": "51b5b52e8e964b4927020f4ae8c912778c8e54a96b92a98dbfd76010c8ab17aa",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100006/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100006/initial/hashes.json": "0a7c8c315177ff67dc57006220d4c3d1f9c661a265f3f5e82adbbc754212d155",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100006/result.json": "fd800e60d76de8fce16f24786743cc6f349433bf697ec77aaa600aade7e93154",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100006/worker.log": "73a074aa12b0ffa7965af196e2d9c86a4f050f8ec2f2e70e5b8e817301e76fa6",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100007/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100007/initial/hashes.json": "56361c3464c96ce298565f9cf498128b9f01996e9c7948fe4db14f77fb5e296c",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100007/result.json": "063223d731f796388920eef473eec75dc4c20311e459b7f1ecf00053143d5b9f",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100007/worker.log": "83401e212412ed098cea939c7fe65a041d2d1943d3c2f71f0b5b19de32c5332d",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100008/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100008/initial/hashes.json": "c86b59ae7712d54a6067b9696e57edb80558105e3ed3acee674a29e2dba0b76d",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100008/result.json": "b4d6b3124d4ac9605fe307098c0c5e268158c03af5945c9e4f747dc97fcc97af",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100008/worker.log": "5ff268eebdf5a8e5bc98451567f5c2c77e754ae268fd876e1a5ae1e4b6125ac2",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100009/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100009/initial/hashes.json": "974ca667bd8e780f349deaa8e1782481edfc438b98ae29af6022c7ef203b46d4",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100009/result.json": "4db90152c3d3a22efa1b42209cd157e0353218ea74ec589f4da08fa4a6f277c0",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100009/worker.log": "064d57d47c1a364d8b93659205776ec9a09d5bfffef2cfcd53b557ae780a671e",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100010/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100010/initial/hashes.json": "a15c9a7daf5ff4fd2874de28f6c69b9bd9ae75c547045052d755c822e1a57a21",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100010/result.json": "32cfdf83555f8047ccd0e710ca854b16b04fc9a10c3b0db8560b14a581dd349f",
    "outputs/controlled_pair_full_20261011/reference/place_mouse_pad_100010/worker.log": "e65b27ee3863d4e477231b696b07f2d2efc1f2af2ce671ef63400a2601f81ce3",
    "outputs/controlled_pair_full_20261011/summary.json": "c357f3e0ad528da63ef0c7c9787a890ee04bb614b4e23fea020b09ef213778bf",
    "outputs/controlled_pair_manifest_test.json": "626015a91d74f11e4d3aa5d4b101689c7432b41fd5f213f360f89ed90989f3a8",
    "outputs/controlled_pair_manifest_test2.json": "bfdaf04e1d62770a8c2d4d709927223984fdb65229ba3f5d73eb9e454a701884",
    "outputs/controlled_pair_smoke_final_20261011/manifest.json": "9ab5c5c47369788fad499586fa01cf1bde54b63e45d1358e9add69459edf5c5f",
    "outputs/controlled_pair_smoke_final_20261011/pilot/hanging_mug_100000/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_smoke_final_20261011/pilot/hanging_mug_100000/initial/hashes.json": "53d58ff1ef0a5f3649e7a26a3c24c5db8b17bfe6d781bc1ac2a863928e52033e",
    "outputs/controlled_pair_smoke_final_20261011/pilot/hanging_mug_100000/result.json": "f62d4235b2935232c075b6350dcd89dc0bf3e0c435ec3f4519222d6619f5e4b3",
    "outputs/controlled_pair_smoke_final_20261011/pilot/hanging_mug_100000/worker.log": "d78808702da9e43754a560c65c23609ba6e530238e2a82e99907b21ddc37bc3c",
    "outputs/controlled_pair_smoke_final_20261011/progress_01.json": "2a289d5bf72ad5fdabbb4e3f596710b35c17b50208230cccc02fbbb95245e16f",
    "outputs/controlled_pair_smoke_final_20261011/progress_02.json": "d5f3ee6b4a3e9884ea96bb8c9547b73cfb901a2db354ccc12223c3c99db2825d",
    "outputs/controlled_pair_smoke_final_20261011/reference/hanging_mug_100000/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/controlled_pair_smoke_final_20261011/reference/hanging_mug_100000/initial/hashes.json": "53d58ff1ef0a5f3649e7a26a3c24c5db8b17bfe6d781bc1ac2a863928e52033e",
    "outputs/controlled_pair_smoke_final_20261011/reference/hanging_mug_100000/result.json": "f62d4235b2935232c075b6350dcd89dc0bf3e0c435ec3f4519222d6619f5e4b3",
    "outputs/controlled_pair_smoke_final_20261011/reference/hanging_mug_100000/worker.log": "3eba7cedfdc24fef39016ac510e341bc7a4020d0c4e637c7b4cd1eea08fcf7fd",
    "outputs/controlled_pair_smoke_final_20261011/summary.json": "d5f3ee6b4a3e9884ea96bb8c9547b73cfb901a2db354ccc12223c3c99db2825d",
    "outputs/controlled_paired_eval_manifest_20261011.json": "edc6a70846e5a015feb5d871392de54dbc0082226e5539410b7855cbe21ebb20",
    "outputs/controlled_paired_eval_manifest_20261011_v2.json": "eeee36bc597a1033f0cad9256736e25c08e9767e27924636e587cf8b3604e27b",
    "outputs/controlled_paired_eval_manifest_20261011_v3.json": "9ab5c5c47369788fad499586fa01cf1bde54b63e45d1358e9add69459edf5c5f",
    "outputs/gen1_diagnostic_aa_manifest_20261011/manifest.json": "dd1b2343812df646f91b38c7aa6ada936b7f34d07a1b63d6d4a45d598aa82f40",
    "outputs/gen1_diagnostic_aa_manifest_20261011/manifest.sha256.json": "33308c87f6b529c7be132ffb98d708241184af2e4d2ba430a647e8c867ecffcb",
    "outputs/gen1_diagnostic_aa_manifest_20261011_v2/manifest.json": "0ea3ee098b970361fc2700469ff879b904db7e3d0fee8e85959ec2e1796b4e28",
    "outputs/gen1_diagnostic_aa_manifest_20261011_v2/manifest.sha256.json": "5a9325c3aabfd7608d1cb41a3f08c1273db33ac243077bfd81a8b41fbea2e6e5",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r1/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r1/initial/hashes.json": "53d58ff1ef0a5f3649e7a26a3c24c5db8b17bfe6d781bc1ac2a863928e52033e",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r1/result.json": "5641a9f180dfb8f2bd0900f4d441c6b9a8aeffc937ee14382d65c30431a48b48",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r1/trace.jsonl": "d543f182571d54d92204b4e3ed6b53b38520ff731e7a6cb105bb4cbd2c9876c3",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r1/worker.log": "460d651993a830a1028e387d5a6181203fb841b8dd876ddb92e322e3b08697cd",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r2/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r2/initial/hashes.json": "53d58ff1ef0a5f3649e7a26a3c24c5db8b17bfe6d781bc1ac2a863928e52033e",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r2/result.json": "5641a9f180dfb8f2bd0900f4d441c6b9a8aeffc937ee14382d65c30431a48b48",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r2/trace.jsonl": "14abcbe8e3f9a05db0ff35963b2b253a6e9b381447e15a77c91e1488cab677ca",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100000_r2/worker.log": "00d03a8039971baee225673763da7e7cb94f4e23e25e8d2c5c13f1b16fa5cebc",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r1/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r1/initial/hashes.json": "ef0d5581d229fe618a2c9f22859dd53f88e6b285dc0307f892e4d735951ef9f6",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r1/result.json": "df8770f17e677e5be958805d9e03384da2041ffba4a76e0ade3f47b2cdb75311",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r1/trace.jsonl": "ac1ca47830eba69247e255622273c95de5d5943bb2b5d4dbc471e0a572ca8a5b",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r1/worker.log": "534ecb8ad298a23594d71fe2a2bda18da88e8a51f44236a5ace54e261989f4fb",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r2/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r2/initial/hashes.json": "ef0d5581d229fe618a2c9f22859dd53f88e6b285dc0307f892e4d735951ef9f6",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r2/result.json": "df8770f17e677e5be958805d9e03384da2041ffba4a76e0ade3f47b2cdb75311",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r2/trace.jsonl": "ac1ca47830eba69247e255622273c95de5d5943bb2b5d4dbc471e0a572ca8a5b",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/hanging_mug_100001_r2/worker.log": "54f34744801bcdf77811174f56701e497bbe38f24d0bc1b58fd46e054aee6b34",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/manifest.json": "0ea3ee098b970361fc2700469ff879b904db7e3d0fee8e85959ec2e1796b4e28",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r1/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r1/initial/hashes.json": "9ac53ede98c27df455b2c150962a46849f658b3db9891960af07251913497a7c",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r1/result.json": "78639fa05f8f710f779bffe30f85ba8c343ac5d76433e442c6056db559edbb81",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r1/trace.jsonl": "ed38fd263ece89583f5c2b675cc9f7b1079a0bdf38ac52a0ee18b9ef5db9a45b",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r1/worker.log": "391551ae52efdeb24d3853e31f8cfa1535fa4e69665f0068ba457d6a4ddd8162",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r2/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r2/initial/hashes.json": "9ac53ede98c27df455b2c150962a46849f658b3db9891960af07251913497a7c",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r2/result.json": "78639fa05f8f710f779bffe30f85ba8c343ac5d76433e442c6056db559edbb81",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r2/trace.jsonl": "ed38fd263ece89583f5c2b675cc9f7b1079a0bdf38ac52a0ee18b9ef5db9a45b",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100000_r2/worker.log": "75805ee8d74975bd14d92cefc57cad0bbd8bc7db9c8196e88346f21dc7190375",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r1/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r1/initial/hashes.json": "ff3837b9096b1c3012bc74d3fd6e4c487a76b6d39128e67b62997fff975b2593",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r1/result.json": "ce06842e21a53f113d9cb736fc23f595cb6f1218034876a8149d904eeec94d9a",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r1/trace.jsonl": "61b2400a54c94a53389e703b57c957bc32f29cfa97aac71581307b4a5a20e5a9",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r1/worker.log": "7f8e435b54e071564d8f5b8d5519601f2047d23f9c4697c3fc424b37573e57a3",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r2/faulthandler_sim.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r2/initial/hashes.json": "ff3837b9096b1c3012bc74d3fd6e4c487a76b6d39128e67b62997fff975b2593",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r2/result.json": "ce06842e21a53f113d9cb736fc23f595cb6f1218034876a8149d904eeec94d9a",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r2/trace.jsonl": "61b2400a54c94a53389e703b57c957bc32f29cfa97aac71581307b4a5a20e5a9",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/place_mouse_pad_100001_r2/worker.log": "9007b466481702c1ab990f09f5ac8d08ea0c262944ff2957c5acc6350641c772",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_01.json": "5a3418495cf0766c8831438ff091b3e460a0d2721ff33c78922ec73c630a1652",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_02.json": "8ed4de72f38ca552a91fb95ccf0dc4a77b712e3c2fe5d658b9dfe95a17c1401c",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_03.json": "123065a971bbc0ad49a68191499ac43c5f4cb3f014536b644cb9146f8a6e19a6",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_04.json": "5bb05a55145493ca645be1d63ed4147a9ef2bcfe407315a30d918e66e9d00a26",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_05.json": "4acd9282c033041f3bba6d961cb569b9b430c187c9bac18ec3e9c7c40a7e059e",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_06.json": "3c349fbe0ddcd670f854cdb994ac267f247083a6f3b3aff6a5238ed68190acd5",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_07.json": "4f60e6bb52699183d96091fe3e2e8fa555a755329694757b0b036bb8d5ea95ec",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/progress_08.json": "f3cb0802d726f5716eef248017be4bf551b378f1d7de5223943047d04d6570e7",
    "outputs/gen1_diagnostic_aa_run_20261011_v2/summary.json": "f3cb0802d726f5716eef248017be4bf551b378f1d7de5223943047d04d6570e7",
    "outputs/gen1_diagnostic_gpu_full_v3_20261011/ema/completed.json": "8cd4000f43286f1019ee44f90013a0e978f2b084ca148de6e820e90793395eae",
    "outputs/gen1_diagnostic_gpu_full_v3_20261011/ema/records.json": "1588f921f065cc2b8b950dbb91678b2ecc97078264eb492fdb0fd7991dc181a0",
    "outputs/gen1_diagnostic_gpu_full_v3_20261011/original/completed.json": "66a622c418b67d35f25d1324396e2f7011c32dc13680ab0b23a725cf3ddf830b",
    "outputs/gen1_diagnostic_gpu_full_v3_20261011/original/records.json": "9a319b9e7944219a4bc54bb14df1f19619126aad72e463ed45bb0408582a0642",
    "outputs/gen1_diagnostic_gpu_full_v3_20261011/report.json": "2424def6d7ca908428ede36909c2138157cc26b5992c523358806b0a0aac1ee5",
    "outputs/gen1_diagnostic_gpu_smoke_v2_20261011/failed.json": "67e4c83f9a26981b3352a682ac1b824579817e5a441d5b3444dfd384bddf01c5",
    "outputs/gen1_diagnostic_gpu_smoke_v3_20261011/ema/completed.json": "04041c4cde42df295255a4885cac817c977d20014623251aa8aaa800acdaf6e5",
    "outputs/gen1_diagnostic_gpu_smoke_v3_20261011/ema/records.json": "36c85f0dbb23ec9a0d58fe1ec4205baa42eb1db1e0fe86f97f0f8d5055d92f20",
    "outputs/gen1_diagnostic_gpu_smoke_v3_20261011/original/completed.json": "ea92d5e0946e4770e627f2ce0a465cd8fb46edad16a13993bf17f7cb4447b4be",
    "outputs/gen1_diagnostic_gpu_smoke_v3_20261011/original/records.json": "aaad7f71266e68fbc7714a5c917bacb8035e657c1e2a15a42dfd9107f9578582",
    "outputs/gen1_diagnostic_gpu_smoke_v3_20261011/report.json": "2e89c41b6b5e26088d3759c1b82ba36bdfe414ccee01c1ffbe763adadd5a585b",
    "outputs/gen1_diagnostic_loader_cpu_v2_20261011/original/completed.json": "845271d032d1cf31dd25328931137e04c4b29c0a3a0e6f1df20f72464c6f181b",
    "outputs/gen1_diagnostic_loader_cpu_v2_20261011/original/records.json": "3878b82e0d8f5557a066a187f8ec880aba792efc1785a0c395a1ed6667f48686",
    "outputs/gen1_diagnostic_loader_cpu_v2_20261011/report.json": "bf803a7f91e8083d0505ff20053e0116fddd63ce19cd5fabab785a6b510427d3",
    "outputs/gen1_diagnostic_loader_full_cpu_v2_20261011/original/completed.json": "4219538e761e344df1b859fa5586352c23de6c8e05fe9b38adbec2df34bb802a",
    "outputs/gen1_diagnostic_loader_full_cpu_v2_20261011/original/records.json": "1f3072bded83746f22e118dd899cdcff6cbc99b7acf28dc8ee0f0a3b0b7ac786",
    "outputs/gen1_diagnostic_loader_full_cpu_v2_20261011/report.json": "dbac4dfcc31f870010c0bedba9216f3616e3c43e79b7ea35a017d66f98eb885b",
    "outputs/gen1_diagnostic_prepare_cpu_20261011/manifest.json": "9b7653cd8915de094780a8522e1db80e9aa92f47a94d19d51a54237fb035560f",
    "outputs/gen1_diagnostic_prepare_cpu_v2_20261011/manifest.json": "cc2f146b93078c7a090ff8ef61c3d5cf2d03c696729b3be60756e7d40475b0da",
    "outputs/gen1_diagnostic_prepare_cpu_v3_20261011/manifest.json": "cfac4604b412c1da93a938996634ca19c351aa54ad853e8d5434e64162473dc0",
    "outputs/gen1_first_chunk_replay_20261011_v1/comparison.json": "fa321da85bb9e7c54b37c6571da1b4b5d58739a83dfd2a20bbc3a2731f7a830a",
    "outputs/gen1_first_chunk_replay_20261011_v1/manifest.json": "795239058c13c3d25aa48cdeaa9f1e3c85c8261ba6275d0b3227899d79fe3820",
    "outputs/gen1_first_chunk_replay_20261011_v1/manifest.sha256.json": "f9f108bae550553214344c6b558db9bdc88097ac690663c47e0a3e7bd2d10071",
    "outputs/gen1_first_chunk_replay_20261011_v1/r1/leaves.json": "f8eef3e5a9fa5f6b121fa18e0ac95b6bdf234f8e2b55f4e99e595ed804a8b628",
    "outputs/gen1_first_chunk_replay_20261011_v1/r1/runtime.json": "f4a8c635c9dc9bdde4bc7ec0dbf5a6843c72aa4b3684f8c118cd70c040e8312f",
    "outputs/gen1_first_chunk_replay_20261011_v1/r1.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_first_chunk_replay_20261011_v1/r2/leaves.json": "f8eef3e5a9fa5f6b121fa18e0ac95b6bdf234f8e2b55f4e99e595ed804a8b628",
    "outputs/gen1_first_chunk_replay_20261011_v1/r2/runtime.json": "f15517629f86157c368657f6c275e19812d00de819dfb16d61d2d4d77ddd3bd1",
    "outputs/gen1_first_chunk_replay_20261011_v1/r2.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/gen1_first_chunk_replay_20261011_v1/started.json": "60948c929dc8d73a87998c6dc1733fde1c8ba5a24a4ea9d621f8fae3aa3ec4c1",
    "outputs/gen1_pilot_2000_20261010/completed.json": "1fcb07c3b7cf6350f62ecfac32af43eca70d69201c3dabe50fe7e1c5003eabb2",
    "outputs/gen1_pilot_2000_20261010/metrics.jsonl": "89b67288e6b6b0a8a8f91d7b124483453f4adb0ec26c1e341c6b2f54fc5191e5",
    "outputs/gen1_pilot_2000_20261010/run_config.json": "7f690a4ae177b6be61678d91acbfbbb84a810807eff1d45874207f1370d8a92c",
    "outputs/gen1_pilot_2000_20261010/split_manifest.json": "9bbdeccc8692812c1ec1e9918c14a8cce5f91ff4f01e3b0d0bb34887b977efe5",
    "outputs/gen1_pilot_2000_20261010/train_conversion_audit.json": "15c71fa6cdd493a966bb5cc9d245baca6f7046e0376f6a681d3e9fa70336741d",
    "outputs/gen1_pilot_2000_20261010/val_conversion_audit.json": "f93d1143c588a02fe05cfdcaa963e824080b4ed0091c8aac219e4fc21250b748",
    "outputs/gen1_pilot_target_audit_20261010_152526_6fcaa952.json": "654a939cd3da702ba232f17fcbbf3ab508f3ae66d84b69724c51110d9b23e5cc",
    "outputs/gen1_pilot_target_audit_corrected_20261010_154345_6aa1c84c.json": "dac47906637f24b7e5e6daaaa433d275f830ad61f8696d156446b746676d2f67",
    "outputs/gen1_raw_gpu_full_v1_20261011/raw/completed.json": "ae16038c611e99ece3c8ecaec9825e0ba809eb424daa3674afcbdf1132a15b5a",
    "outputs/gen1_raw_gpu_full_v1_20261011/raw/raw_restore.json": "5b30cca63c9827fafbb917e19b008a0b430d24ca4c300106f179ac9bb4b58da3",
    "outputs/gen1_raw_gpu_full_v1_20261011/raw/records.json": "9b94dd6472dd3831fddd39941ebf36db3662d85d5480de44364cfbd02bb625f8",
    "outputs/gen1_raw_gpu_full_v1_20261011/report.json": "10ac177e61f7d3991fe9f91a22d6dcaa6b9f5174bc7caee4c3c6949f8819dd59",
    "outputs/gen1_raw_gpu_full_v1_20261011/supervision.json": "f07b3946be6c8182735ea0576a9acae17b4d618bebe1b29ec60cf141d766ee6e",
    "outputs/gen1_raw_gpu_smoke_v1_20261011/raw/completed.json": "0cdcd8dedb2f92a68f048b3d0df68e15e4fcbdb405c689e6d31d6280aeebc353",
    "outputs/gen1_raw_gpu_smoke_v1_20261011/raw/raw_restore.json": "e6ee5e911865271bb4174e050f59396c228e6df62ddce6f764b5b2ed0f2a4090",
    "outputs/gen1_raw_gpu_smoke_v1_20261011/raw/records.json": "ec9370ec4dbed988eda8e22516ac73af8bb1644e29858c8e14c8079e6ae7361f",
    "outputs/gen1_raw_gpu_smoke_v1_20261011/report.json": "10a37d7a3facf3dcc68964870ee774a00f5f7dd65ec9654ddb99bee467c17cde",
    "outputs/gen1_raw_gpu_smoke_v1_20261011/supervision.json": "8ff240cedbf240152bf8f6d0f682402a4adf95662709322a880dc41f1e19187a",
    "outputs/gen1_raw_prepare_cpu_v1_20261011/manifest.json": "832c4d4ce6ca29383528e9c42c4b5d044c6aa7caf320a7ec937fbba9d0eb1ae6",
    "outputs/gen1_readiness/env_clone.log": "d57eb583b4845d2f46d7d32867ba4ce2b683a79eca7a1b26ace27186b36c0dd2",
    "outputs/gen1_readiness/env_clone2.log": "cd81ae8b8a48dbe59e3d7117713ba4ae24bf429fc64bf9b42a317948d7f50433",
    "results/FAULTHANDLER_DUMP.txt": "002cf755219e1889e16fa74aadc918c5629463f67f927fb8bd235d6777c57556",
    "results/FAULTHANDLER_DUMP_SIM.txt": "4439f5a18da101f144763081405a2eaee477e2da9529b354fe465eb6ac05cc30",
    "results/WATCHDOG_SNAPSHOT_20261002T133408Z.txt": "0e8e65963fac4ccbfdecc3be59c16bd2ae2e9c0d586dc5bbc4df3c98e097d887",
    "results/WATCHDOG_SNAPSHOT_20261002T153400Z.txt": "243478895910dbd6e41b506b14fc3bdb3c6fedb50e9ec4044fdecc072658ffc9",
    "results/_probe_placeholder.txt": "a695e31b307030517dab0730d97d3b37a25f7d42a5643db8b351c619fdcc49c5",
    "results/appearance_gap_diagnostic.md": "9b6e9f1cd29511ec7da9eddb01ccda6b15e0d36681734d498f24a4263182dbe8",
    "results/appearance_gap_stats.json": "ccad8bba1e4603b65e9a67bbb0eb2277ff8b2d270fee7780cc804f3f3da7f652",
    "results/b2_dim_probe_20261001T091754Z.log": "a5270454613b5960b95e94a6165d1ec6ca98169dd5479ab5ee92d3d6a911f190",
    "results/background_texture_download.log": "72a9c14516fdc43c74afbea20f816256a3d9731904223bff7296c49290174fe4",
    "results/cam_dirs_du.md": "f0e72d564b0fb1367f04ca439d82b14e46b3d6453df1ba1d36871547e85b089d",
    "results/camera_probe_fix.md": "fc2570988a9b1de2a23af79c59b22fad4bf69835523436d82b1c6dfd33385b0a",
    "results/convention_probe_20261001T093547Z.log": "8e59994fbf9032665d8e0e886e325cf3acedcb7ac1d749c19523805b5e6ed09d",
    "results/converter_deps_verified.txt": "cc5d86665ef32eb68193dfb25d8afeddc6c58d0c019b169dae070c5cd268a17d",
    "results/converter_tokenizer_fix.md": "4ac120339a66937a86f181fe0b30980d2e7355b64e13e2c197963c2a6bc0f6ca",
    "results/converter_tokenizer_fix_v2.md": "0bc83c312e777cd514f0046e7663b69ac5c37fadeff07e063dfe0de1d21bf46c",
    "results/converter_tokenizer_fix_v2_header.log": "36e3bb5b20f67cdfd9c8d70d1ab9dd7eec79e61950bd6f1aacbbfaae5b865aa2",
    "results/converter_tokenizer_fix_v2_run.log": "7188170781eafb2c22fd59ca0c88a8cb90db71ff0ee4d3cc2a2e60433b396b47",
    "results/d013_smoke_training_20261001.log": "6df8c7371126998c240715d2f1c894457f0d764e44995b503cd0e008b30af797",
    "results/d147_provenance_and_repin.md": "fd3e9e9b056ec57d031c69973072e5e1d0c941f02e24185cf2686a3d7a878331",
    "results/d148_official_quantity_reconcile.md": "40d4976dd3049ae6e66436b0e00bba2c68ace4600b04bc980fa901d4f076c16d",
    "results/d149_adjudicate_serving_prep.md": "f1459f2d137b39c77b72ad3a2994a5d2ff69d5cd69e7f76b69c3ad04400de422",
    "results/d150_provenance_enum.md": "2b01761b28b972bd02cf93bfb1f78313818b965fe8aef3da90065a7b7d9f36e5",
    "results/d151_loop_trace_analysis.md": "114204443909470a1c23d5d4698d072c8b5faa1f4b6288f7ae3aeb0d521e5587",
    "results/d151_loop_trace_official_seed42.json": "50f4d5b8b108417d168f9d5d9f3111ace09b986b15fc13c8a3ebdea0ed0a0466",
    "results/d151_loop_trace_official_seed42.jsonl": "733b04c2817baf82874e463fa181fcee259cf94e1477d8670a4ba9542c385ca3",
    "results/d151_loop_trace_peer_seed42.json": "f02678f851da4e5266b9485d9c7da19d814a8f01ae2c91f291c3b75913b53a58",
    "results/d151_loop_trace_peer_seed42.jsonl": "4844266a7ba14dbcb274f8afa01706ce9d4559e89f27a93f7de79191839b151b",
    "results/d151_loop_trace_prep.md": "433345e3e188cedbe1cad9235ca693efcc22d4e1c5a0e3946b1f102287b0edd3",
    "results/d151_loop_trace_synthetic_20261003T154826Z.jsonl": "7b89d9db3d5a4ec149792d83c5e317c739e629a37a8e6f20f800b97a1c3e9844",
    "results/d151_runner_reconcile.md": "6526b9ce40b2155d6c946cfcaa8395cb47d4270166f1b735eec4b96ecfd6218e",
    "results/d156_probe_pathfix.md": "a134ff8d4acf68f17158ed9d91b54526f7da3dc901e0bcfd30996b3e47230fa3",
    "results/d171_arm_defect_prep.md": "cf1597492449c3c9ff463d12417414d22003b159457b985e75d0f78642aa537c",
    "results/d171_doc_wrapups.md": "5edd472d5f61a18f1ae69f907bf37c0bbb668e23b08e207e736a49668b29fcbf",
    "results/d171_loader_fix.md": "069f030e2956d2559007f060a429444789d3907d9dfe1e5e994ff6b28fbb59f1",
    "results/d171_official_half.json": "048c48a2aeebdc1633a606992355243769929b1bc3b469e1e61ae21caafff94a",
    "results/d173b_fix.md": "4e99642a2f5f9c46cedf9f2d9575685dd603ef13c2a7959d36c28e80c15a23ff",
    "results/d176_norm_stats.json": "b535901f65638abdf661ef6c05ed1c6d5f3ebf243eb339f00cc1b40c2e20db4e",
    "results/d176_norm_stats_audit.md": "0b2b59c220a9f200468fc76c3d5f9c28863c609c3c741debfd10cba3a0893778",
    "results/d179_step1_config_read.json": "9bb65e8353a2436473fd3c47f291a73518d4d554b9dbcc7f610d8b59c3478e7b",
    "results/d179_step1_config_read.md": "d3ff35ac28721f8efaa7b6b3ccab6b47c87f9c6d8c28bdd1380ba3e183c21515",
    "results/d181_prestage_gpu_scripts.md": "47569b56ace12ac412644e87fee1cebe8e833f3b303ccb526cbf29bdd58cedc5",
    "results/d182_token_rebin_audit.json": "8eabb94897479a3571a810c415c369c7c5c66491f75c66cf7ee5d25bfa06228c",
    "results/d182_token_rebin_audit.md": "1b823675b5130e5d6e703c006ca25c57b13ba65a0dd5f8cb7586f1e4f0a7fb2b",
    "results/d184_smoke_harden.md": "81dcb7aea1ee43fa97d5654dacd0b7041ea426c4c7beb6d873d2c3abbdd73532",
    "results/d189_deconfound_fix.md": "31aeccccee79e02bb2ccaf8c321e1ee138321083c15d8328143f57a752c089df",
    "results/d192_stalecheck_fix.md": "2eae2bbec0be3dff8566cc5b10b510cdb8b08b04ad1914c692ef4d6b2524622c",
    "results/d194_freeze_protocol.md": "45352fbda798a7971614b7df9beb2c8b0243244e6b890804eb605afdfabeda47",
    "results/d198_freeze_and_gpusanity.md": "794dcc03857bd8ee1aa024480912a57b063a99fdd9f508d0a57af900c5dce1a9",
    "results/d199_legA.json": "a679e9784f9ec2c571e6487d1622b9dad85dfad20291cf268a9dba159513957f",
    "results/d199_legB.json": "6d4eb9b0b969ae416f9e06570c7e324da0bffcf231cd4768b48d9150ba12a230",
    "results/d203_housekeeping.md": "2724ffaf8e2b513ea1ef6979d094775982d44bf6f4a3b54dda0d402220c719b0",
    "results/d203_weights_firm_headline.md": "0f09a66334c747b0272e266f3446cc886804b39f2511645591320e455adc005e",
    "results/d204_s0_s1_conversion_localize.json": "42756b46d279dc76cebffa851a244451511a01342b3bce6a67b4561115f2dcee",
    "results/d204_s0_s1_conversion_localize.md": "6238c6a1bab6faf76c730f5fdd095da2852ba604ea45fe197e61d1f3986105c8",
    "results/d205_stray_grep.md": "cf6b4bae1db52fcc788638946a0db09817192360af014f13cfa01dcba2f22c69",
    "results/d209_s3a_value_arbitration.json": "bf99f16e1b56e6eb4ae7114e8efa4dfbcd5c9a37c3064263ee1d587391628ad8",
    "results/d209_s3a_value_arbitration.md": "708bac154257f377565275dd98fdcd97b26e36f64e92ffa5cc8836e173954ca2",
    "results/d211_s3b_scope_inventory.md": "04d8f84ed3d272dd7d919a53b99b02beef979593b621d5c2ee3149554789b295",
    "results/d213_s3b_target_resolve.md": "2b3d0ce2fe7612d9e85228537c55577682d3d94b5b992a1af59c58b7de085959",
    "results/d215_s3b_fix_freeze_gate.md": "ca67e8080f96dad46eb40d942bc8fe2327be9551497105b77b5775c942970187",
    "results/d218_frozen_refreeze.md": "1ae0390b40aa5025424be7ff7b8a4774866e996cfd3d5e146ff9ac1af5e3bc86",
    "results/d220_s4_stage_reconcile.md": "aaf8f67dc864b7f08388b747f88a218dda390ef3925626e9ceab25fed9adfd7b",
    "results/d221_s4a_peer_control_prep.md": "ccfbf13392a92332287dbded43c266dc8055f6693278520c5f4cf0394f71de58",
    "results/d223_peer_prefix_fix.md": "0a1382fe3054f58ae31a6529b4aa804f60595790c49e436595b4c1d7a211bd0e",
    "results/d225_peer_device_fix.md": "b05d7e9e298fd164023a7d8c00f6527327e168486311798363637cf86add8d1f",
    "results/d226_peer_file_pin.md": "efb32c1c9eb2d6063bb0035ca821a284ef8449ca5062e36bf95f056373699861",
    "results/d230_s5_s0.md": "0a0ae7bed70c3376fe63a1dac55bf1833fc66b283c41057d11cba42b59744e6f",
    "results/d234_s2_pregates.md": "3bf5aa10f514ed37500dee887f6e6070387efd388d700d214e4648d1935a85b4",
    "results/d236_cache_byte_identity.md": "265b16f3bff7d0e99228bfee401b7aa19ba098a631349f90bea4e426e2de1a67",
    "results/d236_p1bc_probe_commands.md": "d5ef02455ecc24283fb6e2401f7138d8b9d26ee459c7e622183b387ac2861b9b",
    "results/d238_f1_normserving_audit.md": "02234a20a9af0e9416a683cdf2a082736863920c096b90b085d694614f5578b2",
    "results/d243_a1_rawleaf_coverage.md": "84ad5caa198cdee2a36c0eaac33baa63f476eff1c8f284705df87fa2de4e1729",
    "results/d243_a2a3a4_protocol_parity.md": "548e17cbe4e9c848b27c877c9cc9bf4686b26bafca4afa1ff489cf33103dcaa4",
    "results/d244_mech_summary.md": "366db29dd13fa6ea045241e15673b8c06cbd7f831b009a39c59b199181cc304c",
    "results/d244_mechanism_trajectory_tap_eb71add52852a0592cbefc8675ac9f228cb8974782283c7ca057288373fab1a4.json": "ae312fc0cb75c32e474293b5858016e5156dbcbbbf7a72b45acbebae41df7086",
    "results/d244_prefire_hygiene.md": "d291f40cafbb617b177ea0c268acbe53ca91a1e7e17f39c679d45c1da6659307",
    "results/d244b_reference_number_audit.md": "a77b9c8ac1de9234b096e6c6ee581f1c5b85b8a1f85a0f7715843fe60223c941",
    "results/d245_b3_followup.md": "870326c38a19ef616f28bf73ff4e2678bbacd8e7d7790d0a0221167a2da65d0a",
    "results/d246_c1c3_stage.md": "54ef33844233f8665642b48461f774a548b1fb2ced6832f90208671610965e03",
    "results/d246_c2_s42.json": "9d85a4c3abed0e1b855afefd676d6ec502e749634ae5d26efa64d6d3d27d782b",
    "results/d246_c2_s43.json": "eb2eb060f2a04841c74f3053241863705fe92e2cffb18f9f5f37357d39238f92",
    "results/d246_c2_s46.json": "5d97a3ac684c6b9713f13f6c0fc2f8ca5c4ff1da3afbe7c01617a066a213b98b",
    "results/d246_c2_s47.json": "44c20e2631c3f5a9f7e13f68ea8b87c625098ed04c60f49610f92623f0e2664f",
    "results/d246_c2_s48.json": "14b52a9e15d3bcdd29f35b7865aa2961a8cc1603326b0912e1044ced85bc6ee9",
    "results/d246_c2_s49.json": "9c23642a0070f69a0ca406e98bc00d44a501831626f14f666a57b872047584bf",
    "results/d246_c2_s50.json": "222f840f2a671bb406548154384069fb38840061b2fb657c31898a6ef2d52649",
    "results/d246_c2_s51.json": "32f3691b4e315b8b24abc9d7d2c07818ab09175ec806d64befbd7feced716efc",
    "results/d246_c2_s53.json": "54831e35cda1190419a6dabe5bec8e58fc609148b010461142a0d7e2893d8ba9",
    "results/d246_c2_s54.json": "30a0ba5094369f6745e77661485efa5a5d3f65c369dc7a2580f293a4bbcc6cf3",
    "results/d246_c2base400_pass1_seed42.json": "f0e0675d15cb0ad073bf2aeae5bbb4568dd5e670e221779e8a550a2c29383cac",
    "results/d246_c2base400_pass1_seed42.jsonl": "70708a48703317f829777f81fef8a14c00140457f51f9286f038a970e516784e",
    "results/d246_c2base400_pass1_seed43.json": "8f5d14c52afb47cb14f7f50ccdfd460440624949760b1411c3c08222c9a72fbb",
    "results/d246_c2base400_pass1_seed43.jsonl": "f74484a7dcce3f50f641e89d257f2e19dbc72ffe919397c5ee8b4cc60836b07b",
    "results/d246_c2base400_pass1_seed46.json": "cca6be5b8ac3639e012a158a5e209d314c4788f6b08b70be242c818cd8c142a3",
    "results/d246_c2base400_pass1_seed46.jsonl": "95454932d874e933bb3d342156c112807b523adb8e4d33ea8c15ccbf75d1faae",
    "results/d246_c2base400_pass1_seed47.json": "5bae58f4bf85b391362301b340a233a278763f204fd5c0620489f1661c1edab2",
    "results/d246_c2base400_pass1_seed47.jsonl": "dae69ec30e1de9066df5c49bd22f5fd27a723f152fc6b5728e88eca5d791c262",
    "results/d246_c2base400_pass1_seed48.json": "bbc87150ee37ed9c056eb2d82b98a7a19151d6cc3f26376c5d99fe64e958eab4",
    "results/d246_c2base400_pass1_seed48.jsonl": "8aa7c97f19dd347c3d5c84d618044da43997b02f1e4a154ce8cc4e2930d21c71",
    "results/d246_c2base400_pass1_seed49.json": "d0a9e381c842537365f49bb0278a1e67bffaf238c7b999a9e300a2cc22516e8b",
    "results/d246_c2base400_pass1_seed49.jsonl": "f9654f312b3dfa08f0508d4929a232a6b38925b7387eafee98f6822334b71a1c",
    "results/d246_c2base400_pass1_seed50.json": "b0114ae67527b9c18f02b46fea8db198996a5e204410d3613fa525514ae7e62c",
    "results/d246_c2base400_pass1_seed50.jsonl": "328d7840e8f38acd88420255f34296671800056c2a52e6f9fc48787a32c8931b",
    "results/d246_c2base400_pass1_seed51.json": "43b6bb77837fedfd2ef287bd119bbaff870eebc116b1da81fe36c9f495af799a",
    "results/d246_c2base400_pass1_seed51.jsonl": "f4328a0f80f6a12335e5aa399ff510d203067dd6409e59d6b85d391822012620",
    "results/d246_c2base400_pass1_seed53.json": "30b089bdebc7dd1ad900e93adb857a3699282122aa84d7456f1a9d53ffdb8615",
    "results/d246_c2base400_pass1_seed53.jsonl": "921e3b56dcc0c4640ef8aac5ffc976483e82b4de8c12e9284058d884115cc3f2",
    "results/d246_c2base400_pass1_seed54.json": "a5702edcc9ebbc54014ac0b40961a60ad1f1da2137274f990800cee4062364d8",
    "results/d246_c2base400_pass1_seed54.jsonl": "f442b7b65e82672943824a7691632feae0ab8fe4596a598ffc47fa8f6896a801",
    "results/d246_c2base400_pass2_seed42.json": "c72286a091f4ff3c9c837efb6f8496565061c73d8e66289eb82bdc78b434cad7",
    "results/d246_c2base400_pass2_seed42.jsonl": "37be79fff4fc2b34f37853f623ff197e2d3fd8a69abbc1deff560b59fb622cfa",
    "results/d246_c2base400_pass2_seed43.json": "3d2381f7baf3b26932dea6826c8097549b0e47caf3b0961ff7f1744c960013a8",
    "results/d246_c2base400_pass2_seed43.jsonl": "8dc14b96dcbdda07b9c78ba861b6f9e947d225b00f561b7c1e57160199ff19d8",
    "results/d246_c2base400_pass2_seed46.json": "f0ff3683b0f1fa43bcbfc1de37df297e4711a9288d68f7f83e204e4aeac68df6",
    "results/d246_c2base400_pass2_seed46.jsonl": "4f7af36f3b51b0f51cfae53346022785c127d6c58f86fe94024bf23f0b52ee7c",
    "results/d246_c2base400_pass2_seed47.json": "c09a2fc957f648f6f0f274e0ecdb511afb2871390d38083f7d428e6e2fe0de64",
    "results/d246_c2base400_pass2_seed47.jsonl": "22fd2636eb66bea5f11cd5c5381ebcd532615a355c450c60a781f2dec1b3852c",
    "results/d246_c2base400_pass2_seed48.json": "f5834885b2d85742a339e254a93187731dbaedd2360e044084d56413c86c4cde",
    "results/d246_c2base400_pass2_seed48.jsonl": "fcb8601c4d12a06caad64d4ee90f3983eaf7bfc270ce485928351c5012d3766d",
    "results/d246_c2base400_pass2_seed49.json": "66dd761cde9b31a70ead6919672020ea424b679ae4ca4ca91820e4d647b301b5",
    "results/d246_c2base400_pass2_seed49.jsonl": "c1695f80807b487d856e074fe96ea6b249fdfcf3be7f6bc21267fda2360e8df9",
    "results/d246_c2base400_pass2_seed50.json": "ea0aea519ec3cc01c55e8cb6693027d89ca70cb8344fe8f23002420dc5d2c81c",
    "results/d246_c2base400_pass2_seed50.jsonl": "3258c4648b80866481743180772eee7352a6b3fe8b171d296f8c094e9b5e30ad",
    "results/d246_c2base400_pass2_seed51.json": "c7ab3627c496ffa8a4f08e0abd07de4bca1be94ede7b1a9d220c57576b51c355",
    "results/d246_c2base400_pass2_seed51.jsonl": "0b903c7a2296d8447639fda8aceeeaf37224688fe45e385887baf55dc8da6452",
    "results/d246_c2base400_pass2_seed53.json": "12fb20cc235d2fd2308ad05dca9e6dbc1be16a2a6c8cdafe0b80c26add217715",
    "results/d246_c2base400_pass2_seed53.jsonl": "5a2a99c44f49c3d024ca8a01f9f8e6ed53f231e391bf1b9f476ec0600766f842",
    "results/d246_c2base400_pass2_seed54.json": "2fef5f9e1cd907654109af7e4e75260dad02d954c7979469359e2c339a9d8028",
    "results/d246_c2base400_pass2_seed54.jsonl": "85cd4199318ea075d1f754c49cdcbe786739a36f7469692b679ffc7e1f4d630f",
    "results/d246_c2base400_pass3_seed42.json": "a7219802a29df0fc8490c84227163ac5b3fae502335e6057cdf3fbce05849f31",
    "results/d246_c2base400_pass3_seed42.jsonl": "aa45f0c0bcb3cb3e7b216c040f85989dd0cbd723a666af6700ae2370435d7d82",
    "results/d246_c2base400_pass3_seed43.json": "68c479612b59b20007bc898d5fe0c17e2a1eba8a64b4408aed2f70ae6843e51a",
    "results/d246_c2base400_pass3_seed43.jsonl": "b262c8873e63db98e616a1c8aad9a19085f3d0144544c04ebe66c7ff98f94287",
    "results/d246_c2base400_pass3_seed46.json": "7a8759f6bb50b6808ed571f28801ca3ebf9ad4220872f1526f044e6477f057c8",
    "results/d246_c2base400_pass3_seed46.jsonl": "47d7cb64bc02018522f2d9e74d1a6a575b0098ddc626a5919f4fecd57f59b692",
    "results/d246_c2base400_pass3_seed47.json": "1dd2e6a8e4e72cc4f9c6455a0e5cc6d1669f27b9ef608175643019f1c8442a54",
    "results/d246_c2base400_pass3_seed47.jsonl": "6b8ee792408b49775ffc855eb38563cc2d083f90f2589a8c0fdbbb6d9e45d337",
    "results/d246_c2base400_pass3_seed48.json": "a4ac6247eb407c940f4575b55d1ff5af109b7559c5af5a5f150c3d47b292d6d7",
    "results/d246_c2base400_pass3_seed48.jsonl": "0b0f7c1d505a7c8c7449c385c378d3935f7932b56e3de6321ffb4451236922e4",
    "results/d246_c2base400_pass3_seed49.json": "540dfe7d54694bdaf114f12667d061bd8c8f312ca8aa653f843934ccb2e9fcc6",
    "results/d246_c2base400_pass3_seed49.jsonl": "7fcf82c7ab61b46147f9ea89c96d8b1043ef2346d0df894bb5c9688a746b1e87",
    "results/d246_c2base400_pass3_seed50.json": "445daacc93d1f63a6a19c0276dd7326d84e29ecd9ebec7af2af0da43ed2d3536",
    "results/d246_c2base400_pass3_seed50.jsonl": "f135205265d4a7826e65508595c03e824c58ec0b509c454ca643b54cb3182a10",
    "results/d246_c2base400_pass3_seed51.json": "2558df1e5502fa2c08ee0c7a815a2c4c608deb25c1cf401a810a893c50e3b936",
    "results/d246_c2base400_pass3_seed51.jsonl": "e13786c4a320225b706861692c4c5c1c539648bb6a2708e0ff1c8d49b9868691",
    "results/d246_c2base400_pass3_seed53.json": "1c68c2fbb98a2e35f5645cfb61998e9b02db27e9c3c46ba18a65fb0c4b148fea",
    "results/d246_c2base400_pass3_seed53.jsonl": "93f5af5e4b26cd513737741c098ab87fac319ecd2cea69bf6bfbe2bc2b16d6cf",
    "results/d246_c2base400_pass3_seed54.json": "c76fd858f071aa92a1ecb5042e38927a14f01067efb79cf85d87bbc68430ae55",
    "results/d246_c2base400_pass3_seed54.jsonl": "08c996eb4a03b60ee07a9dbe35e59dca7ecd61ac168c2d9375cdf1395ebdf0d8",
    "results/d246_c2base_pinned_seed_42.json": "36c2f400807a90b7ce04b5ca3f7f514218d902a89b2453e88bc778553110e9f8",
    "results/d246_c2base_pinned_seed_42.jsonl": "b79e35ec854789a84de7b06a4d0dff25ff7744f3e1bab9253818f603b4fd5438",
    "results/d246_c2base_pinned_seed_43.json": "a5511e50b09ccd0acd9a546e336b362ce56663fe6be7ea44884c71a259b702af",
    "results/d246_c2base_pinned_seed_43.jsonl": "277e77e4d8d1e77ea7c4c70279d2bde33cf35c341dcac318c2e482ff428ed75d",
    "results/d246_c2base_pinned_seed_46.json": "56a909a3117aa888b886adde09ba256fd195a765e5b7a1b2eb54f6df2a7561bb",
    "results/d246_c2base_pinned_seed_46.jsonl": "8bcab606a09a2da146ef792207b9524ab9c79668c524d8ff6ca82fd0021b737b",
    "results/d246_c2base_pinned_seed_47.json": "c6107f74d481c213a0d2f08b727ebd8f73e85720d646d791bd6d959f0a39871c",
    "results/d246_c2base_pinned_seed_47.jsonl": "5a868fbe03389d89b58a6bb0bc79918c85c472a90b146a62ba51cbb6815346fb",
    "results/d246_c2base_pinned_seed_48.json": "a6fa80095bf80fdafd1eed761df36e07972a77da616dc0bf39d0945713855942",
    "results/d246_c2base_pinned_seed_48.jsonl": "dcd24e6d8c65ace24e0fa2b2b352d2c0cbc682880ff2fcf396038d79a234822b",
    "results/d246_c2base_pinned_seed_49.json": "cd5f4e14bfaab205cb486201c00045b77992594572c57be08877a7cf6dbe1df5",
    "results/d246_c2base_pinned_seed_49.jsonl": "a8e8efafd265961cd254ab738cb0f0c75b4a632901d640762a58774941da9406",
    "results/d246_c2base_pinned_seed_50.json": "8af830b96aed253682450846499d6caee596f44788b85d07ee2a35a94879ce98",
    "results/d246_c2base_pinned_seed_50.jsonl": "c8638f04d73751626e5bff34682fce9f9dc8d7911766a5c7023c75b9187e0c10",
    "results/d246_c2base_pinned_seed_51.json": "0efafa2db298fb7132d05e7a53f642e0d4d33e96ab4791b5b8fd31191d09bd56",
    "results/d246_c2base_pinned_seed_51.jsonl": "86a437492b3ea86000bdfb08ac916c7c0452572a24172930cb7045f695a9eae9",
    "results/d246_c2base_pinned_seed_53.json": "7cb2ed26bd7bc433cc32de5cad9c5c29226c74257a8cee3ba0a72ce92b102e8f",
    "results/d246_c2base_pinned_seed_53.jsonl": "256a11e7f325cac200a2385e02a707286e105c14d462c956269fa697178f09ae",
    "results/d246_c2base_pinned_seed_54.json": "f219fb3293b52061462055e9d85dfadeea09df0da6adf1cecd7187e4b7d3b796",
    "results/d246_c2base_pinned_seed_54.jsonl": "71fa562828766af9617f34dc092fd3f03ecf67898b14f7ea9f0b2c5491d4fa81",
    "results/d246_collector_override.md": "fcb1e95d2f8b1506e1efbde746c05f1da9eda963151e6f5e9e84aa7a17ff81f6",
    "results/d246_converter_build.md": "1c7d94470982e00fec50afc29dc6f4e3ed3f209ca90bd7601072279bd0b79370",
    "results/d246_converter_mismatches.md": "76a37544771d9d9ffcdd30aebc031780c855fd1697726f1ba9400ff1ac569a6e",
    "results/d246_evolution_staging.md": "8812d4e3f2a2ae4382d1a4d21852f2e1abe23a2864ea57a048a1f006e06322f1",
    "results/d246_failure_modes.md": "da480d465760190d3a743a6891a26be28307f88a235f6e571f2253df8dc541df",
    "results/d246_fps_empirical.md": "7c9ea8feb688fe8cfdb8afaafbcf40f4cc437e08246a8df834e79c53c47ba570",
    "results/d246_localpre_config_edit.md": "fce34652d3350542d8498de7aacbdeaf699d3b2e08d5027d28373bc7ee02600c",
    "results/d246_paligemma_offline.md": "98be47a8b999409712e776464a6a903b76a2705a8417328e548acbfe30859811",
    "results/d246_parse_path_fix_report.json": "843f8112e9595a0d87002dba042ff87f6e7ca1ec2d75467c1d24476d7bb895f7",
    "results/d246_parse_path_refire.log": "72c3713647600c4735ec85367d8c4e2635e191e8725c208cd6031aaf138ddabb",
    "results/d246_peer_mbase_prep.md": "4d9c187de8ad4c2c0f50900cdc268efa54b08ff6333aa5d24084b86fc89f4c93",
    "results/d246_peer_norm_mode.md": "4873d2b99df145257c061b092149df56c85c5cdbdc9bbc5448f65fcaf2f7ed87",
    "results/d246_policy_flag_form.json": "b28246faa73c1341a2aa7ddc3f9c238bca86d917478186f016e3cf483c861e20",
    "results/d246_prefire_checks.md": "1cdaf643f208a83b429b23a391fdac984bbca10625dcdfd7b76ff1370539219e",
    "results/d246_seed_probe_realcollector.md": "e7a03971a28a4fc14ccd1a3e38f294df87c2f984626dcf53a10c2118012cc1b3",
    "results/d246_seed_probe_stage.md": "53ee57d06132bddfcf7dd893eace76fe33a2fd5b36ce0ceea0d8732b716187c6",
    "results/d246_task_field_fix.md": "e445065be9041705090f43d8b97d19c12af35c9a107dd790ccaba837231d8514",
    "results/disk_inventory.md": "25c8d6cbe1b97b217e56b8069715bf76423c260e287ca0b396a8c40072ba3bc5",
    "results/gate2_full_verification.json": "beb016b009f8c8568803a655e92bf79abb03898bf7271ebffa1f94765c1bd76e",
    "results/gate2_full_verification.md": "2201507455d7e7c2413f839adeb19f70921d55d387f285493789fb6a0d26d6de",
    "results/gate2_precheck.md": "e0fe31cc3129c1126b82d37eccb18dd11f7c00bedacc6dceda21e1b728dfdf54",
    "results/gate2_survey_cache/README.md": "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17",
    "results/gate2_survey_cache/clean/meta/info.json": "5d790c32b5d0f5ae0c84453727398da00aa2313541e479772a1c021d42e4c458",
    "results/gate2_survey_cache/random/meta/info.json": "a82ce3547c05f6f53599793f4de1ba2e4c0c63821b9cff953b3bce7b8e5981a9",
    "results/gate3_action_execution_trace.md": "6266d6253b763134dc190b83ec32d49290c32c562fb04ec9c715fca07a6891f6",
    "results/gate3_action_expert_engagement_probe.log": "1270fc847452e960e64bc269bbd8920dff0a8933c7a339c753739cc7d8287a93",
    "results/gate3_camera_permutation_repro.json": "0e501de28c71389bf7cae70a21707b4bed5874ed5514a4400df0facba04cb64c",
    "results/gate3_camera_permutation_repro.md": "988a12d3d8aef5ec57ac87a08a6ebf6718fcebba5dd9a03442259fe2f659697c",
    "results/gate3_camera_permutation_repro_official.json": "2d95bc5e465fd94d96b18a9338c7a065ab3a4cb7019a8e30aa040c2c7061485a",
    "results/gate3_camera_permutation_repro_official.md": "6c04a45787a1848ada73efab937ab822240f59a6ab2976490f257748bb07b0bf",
    "results/gate3_checkpoint_parity_audit.md": "570dc7b7021727b0f4f49653de149f676230b9bc99801cfddc7eb2c539837e80",
    "results/gate3_checkpoint_plan.md": "89af3de5159202e0ccc440ecc34a748e95e84ae172795e68189f1c0217a3954b",
    "results/gate3_checkpoint_prefetch.md": "5184bcf217e093f3227bc50a6e27e7f4fd9ea547944b83760809adc72406a539",
    "results/gate3_conditioning_prefix_probe.md": "7526bd3ff69179885b8183ecd9f1ad59587b9c4366bc99a27893bb7eb78bb312",
    "results/gate3_conditioning_rootcause.md": "c218fe8d1dc1fc2e8795d5250bec92ff794ed78da184820626df26a41ee4ee65",
    "results/gate3_forensics_vs_5seed_crosscheck.md": "52f8535e3e850c5a8ccc963e26e0cb81e0d586b394294eed1a943405677c1383",
    "results/gate3_frame_visibility.md": "fff9685b6e088299a86aaba8ab130bf77f00b27633cd6c1fd32c81fc630c02c8",
    "results/gate3_full_ab_official_s43.harness.json": "32bbf48af6a63c0eff5e52c078025994ddefcbf95ef013aed6e2431f0101c9c2",
    "results/gate3_full_ab_official_s43.json": "07eb987a57eed183a293a93a1fc38126f7b232f3d74d5f570f90bef281b5efc1",
    "results/gate3_full_ab_official_s44.harness.json": "b73daac11ee46b2a1d5bc0e7fdf799d3ac36576d3c1757cfb1c809e366ea6e21",
    "results/gate3_full_ab_official_s44.json": "82b5ecf080e3b2fcf644055c5d2f79b84e34680f0471b2189ba32e6347d8b568",
    "results/gate3_full_ab_official_s49.harness.json": "d40f97c3df031d71b200aa58300f45e20cbc54a58fe70f21d82298c9e732b3e6",
    "results/gate3_full_ab_official_s49.json": "85a77d621cb8a3d3eafc64e99c292f60c57c4bf42bed1f96f73b779b77eea8d6",
    "results/gate3_full_ab_official_s50.harness.json": "ed6e35e2d6a8ad06b66d6fe8197a323a45f11fdc8f31cef0cbc42c0d6ff0e649",
    "results/gate3_full_ab_official_s50.json": "f37d4147cd6257e3149ed4980bd499f94040274e23d789dbc5c99950a74bb2b9",
    "results/gate3_full_ab_official_s51.harness.json": "25775267e495ce3905782413a7f5cc59c70599e8f71f0af00374bc1921c4beda",
    "results/gate3_full_ab_official_s51.json": "40fd2d6508d650f79924d070d491df1898b2e9898a302732e62bacedd708f69d",
    "results/gate3_full_ab_official_s52.harness.json": "ddeb994157b76a6d04b350a0204e3321568503847a5f803867a2ecdce2172391",
    "results/gate3_full_ab_official_s52.json": "d26cc295abf63bdb4a355b4d0b923173bcd9626afca2ae2eedee1a19c426ed3b",
    "results/gate3_full_ab_sidneyxie_s43.harness.json": "c5058a721d7b41a447fde0456181d9a170ac84f7dee513f84daba060dee7061b",
    "results/gate3_full_ab_sidneyxie_s43.json": "f426c4d125b1a55d8623eaa2020ef6eb3925e7a89190a5393825936632b69403",
    "results/gate3_full_ab_sidneyxie_s44.harness.json": "406b5fffad660cf9e04419565daad0d0af272fe178a29dc64260a07425a06b52",
    "results/gate3_full_ab_sidneyxie_s44.json": "af62616f54702a8b8d7878fa9293ad34cff494a3c67d6089262654b05a61141f",
    "results/gate3_full_ab_sidneyxie_s45.harness.json": "242864dfd800dac6b436ad4924ff2623cd05c723c8d619c75c5520609a68a5f6",
    "results/gate3_full_ab_sidneyxie_s45.json": "9c5cc192e313327010254913831ee2fb3641b1ef7e959b5722bebe28e078cd8b",
    "results/gate3_full_ab_sidneyxie_s46.harness.json": "0d29a97d133f08858844de654e9895f17e596d533716940c81cfc3bc06ab303d",
    "results/gate3_full_ab_sidneyxie_s46.json": "e1c5dae71aac96a399c4769442ca816c594a9225662da5b92396bcc82ee241f9",
    "results/gate3_full_ab_sidneyxie_s47.harness.json": "b111e9831a77254d9bb1d6916d21cb19d02dca5049fb253492557028899a0ced",
    "results/gate3_full_ab_sidneyxie_s47.json": "e41dc6a30380c69fc970e1cf56a5b13157176f2d73a5e6692fabf2494a3c9009",
    "results/gate3_full_ab_sidneyxie_s48.harness.json": "92777b6baf91cb503b1938cdead1cad9b5e060d9d86a7bab6b490c8266cccafc",
    "results/gate3_full_ab_sidneyxie_s48.json": "66b12cdb00bd20a17b165c1c17924771aa8fc3e8de2b66c4332fa56395ab10e8",
    "results/gate3_full_weight_parity.md": "c24ea21d8e60ebe349eee799d79bdd220f715014909ac0f9d29f81035c11107b",
    "results/gate3_instruction_paired_prep.md": "376ccf6fa0cc434a50018e9410aa7b5ced5cdc398bbe3cefc38aa918f56ec3d5",
    "results/gate3_instruction_paired_test_README.md": "b846dc0e7c2701616d11ad60bbd5a5cd08edfee4311cc96a6af66599d1c93007",
    "results/gate3_instruction_paired_test_official_s42.harness.json": "2eabc02c085c9ea9d31c9eb297769b8ceb6aa7bb0e8f4d5cd5382c0a02c67db7",
    "results/gate3_instruction_paired_test_official_s42.json": "1e5c8178d96ca380eace151ea3b75837e821e8159e6609fe901dca5f13ec5a63",
    "results/gate3_instruction_paired_test_official_s43.harness.json": "3de2d5baa73fd6d8e07b218a84231c1a06bdb266d57d95b851d6026659383f97",
    "results/gate3_instruction_paired_test_official_s43.json": "332a620ec0ad35ad68cc6b1f4613441c27e9f54d1c379204217bde218850bdc3",
    "results/gate3_instruction_paired_test_ours_s42.harness.json": "16c95c2599b841b2cc1573b09a595f12c875c377c812b4faacd5e41d4833e71d",
    "results/gate3_instruction_paired_test_ours_s42.json": "3e458cd3809d02209df6b090db506f19f61a03e71d22505d52ac2db33e0af915",
    "results/gate3_instruction_paired_test_ours_s43.harness.json": "98092288db8b2168ee1cfbd18d1e9a155b1ffbad4d177db2e03b64ecc9bf19d3",
    "results/gate3_instruction_paired_test_ours_s43.json": "4d10bcd1bc540614a2da1d668716cc8e9ce6245e52844cd705c6327a36c98a9e",
    "results/gate3_instruction_paired_test_v2_analysis.md": "69ac7bcfbabd1aff8016bb89920f4b682e5e4f10e006c3173b44ffd0cb0025fe",
    "results/gate3_ipc_crash_diagnosis.txt": "da7e17657a1bdb53da74414ad940af1d8fc46c7ea456c684e46adb4e67ef6df8",
    "results/gate3_nondet_repeat_analysis.md": "a7ac127e318d17ad7a70ff935f0f7a7b6ee95a1b915e92556990f72790950d00",
    "results/gate3_norm_fix_oracle_validation.md": "c082027ca9e6f404884bd4834e33860b3475c3c9d10c7b55f0e4dd4df93a660d",
    "results/gate3_norm_roundtrip_audit.md": "d2f0d08a6bf65f2e33452cb239dcfe9628145d227996b9f15d2ecd85a75f1663",
    "results/gate3_obs_range_contract.md": "b16c357e33b764a6bf6b87046ba32e9eb6bdc54e95543ba5303e8529eef66204",
    "results/gate3_official_ab_official_s42_v2_extract.md": "060592e9da66cd1b3a5777a16a503c58d43fe9184b93049934235b6c0411f982",
    "results/gate3_official_ab_official_s42_v2_traj.jsonl": "a85630c9c5637b8a04c0edf5f12b87baa0312bf1ff11fe29da7cf70b8e11b7cf",
    "results/gate3_official_beat_block_hammer.log": "f72cba3148b20bc7c1eae061c9d2048f4d43076ad5e30a6499926cc7c5c3ff37",
    "results/gate3_official_eval_v3.log": "f81060aaba268d9174d6be80a6294a92a9c7ae3cb64b8e07b4cf0c7326b8dcc0",
    "results/gate3_official_eval_v4.log": "55066a52c21b7694d6d78a6d2f7ba4ff22421840a790a3a6a6b696e63df5882b",
    "results/gate3_official_eval_v5_raster_analysis.md": "78bd846c2de15d96231d438868bedcae4a600f218b4c88b349da7725f7c3d1e2",
    "results/gate3_official_v2.log": "514c264c29a2d09be2a3a2e6641f1883c573d9b6cb19e1d4eddea2ce41db2856",
    "results/gate3_oracle_action_reproduction.log": "5cb01da225fb139c65f1d5264b91f2b49dfe1884a14cfc44f5243f1ad749b46f",
    "results/gate3_oracle_action_reproduction.md": "3f07ec804405987447c50b3ace04a619850816982882fde9da3847232f302bc9",
    "results/gate3_perception_eval_vs_training.md": "f1620e42e86726341918f3b7fde313905834eda8c4c573ca8d843f699244b31f",
    "results/gate3_perception_refiltered.md": "915d373321f8a5aed1c1b54c6781bd6eb8ef39d86afdcfcfcc89f3cf613266e0",
    "results/gate3_pregate_fix.md": "5ca6b6f8f996b1f8afa2ba97012ed0e00f4e3512e4e36c6e435f655c0060b477",
    "results/gate3_preprocessor_fix.md": "a63899d4534e73a9967ab80e6b1328cc9bbcf1560f7bc2be6bd023b991d6c8a2",
    "results/gate3_proxy_correlation_prep.md": "f28868a93663867350e31d3df9bb7d75a5f42f36f0adf4654a1fb1e3ebe41a19",
    "results/gate3_raster_0of5_audit.md": "628f4dec88688568aa9b2d271b73597f94f10838e02247bbf8b251e05676375b",
    "results/gate3_raster_forensics_1x400_analysis.md": "73f662bf767e895d08c751580ec47eb26a08bea315c5f925f5632ab2de240ba8",
    "results/gate3_serving_diff.md": "5e7285fb97b2d063b07b681f2373f9c30dcf5e0fd9bcdbba9d4aa7afcb87f7f8",
    "results/gate3_short_budget_1x250.log": "0300980107edc7f359e7eb4e62c3e14378e52b26143a56dbb824ad2f0d3e593f",
    "results/gate3_short_budget_1x400.log": "2f3fd61c1e356125d944bbcdb6443d22b57f82ce3845b2923e3473b34093b3d0",
    "results/gate3_short_rollout.json": "4fe5754ffe3c1284099d1f4ba5d585ac297c065126e261fc0f08e8e79d67daaf",
    "results/gate3_short_rollout.log": "1de7fdd9bea2478e15b15869170044e1c8ca1272186a4962a9b6ee9eb60e3657",
    "results/gate3_short_rollout_trajectory.jsonl": "f3c5fbfcb4e29cae263c4de4ef7db5399ea192270fd07140c6583ef711cebe36",
    "results/gate3_single_forward.log": "d831c6a5c7a5a1e6a3f367e72ea5aff73037ed35454b96d686a789d2995603bb",
    "results/gate3_skip_expert_1x400.log": "7a3cc13a58abb4a3e241b8769a42caeb322cfc3b924f39d7cce43ae89276df61",
    "results/gate3_skip_expert_5ep.log": "b5ffe23f47f09e0a76802c2991c07a53fff5711d383e0c31a93403753949ac25",
    "results/gate3_state_order_test.json": "f2cf4b1a11188a09f0a063f85dd538d479b9d7376b76b8144763ccc0df056306",
    "results/gate3_teacher_forcing.md": "a45f5929c09226dd70a52dabbb4e22fe31077d18c93f12cadde9b4f75fd3bc2e",
    "results/gate3_tokenizer_status.md": "d32e15fbb786d0e09c5b21fdb859c6d1ef203c7c992e01d26837bf3915372de5",
    "results/gate3_two_checkpoint_discriminator.md": "5defda751bb6a99b2dbb9f087cb06a7b7fc5f6d9be22ae1b5454340eae9fc140",
    "results/gate3_weight_swap_fixed_config.json": "9d598515da3d727fc5ee3f6183aa4ea5e0fcd5233cb82d9fcc2e955a2466800f",
    "results/gate3_weight_swap_fixed_config.md": "ab92fca7ccf49956570220029c9b9b8b608ff5db374c841e8f1ad692c913f7a8",
    "results/gate3_zero_diagnosis.md": "f492073a97db7bdd9eaaae39ddf6748240df69854d444fd86e9b5d714be55b9a",
    "results/gen1_pilot_2000_clean_two_task_20261010/hanging_mug/result.json": "4953645ea6e47d123e5cc74b4ae15d6be5ef023431d41a416b61127dd9a74dc9",
    "results/gen1_pilot_2000_clean_two_task_20261010/hanging_mug/run.log": "1317eaeea214a6ab313294e0110850b0df9ac137aba689c519a9d2bdc07d07a7",
    "results/gen1_pilot_2000_clean_two_task_20261010/manifest.json": "7a37a4e5234e1bb99d7cde8fee97b6cf37d357111fe3a9c3a2784f18387dd023",
    "results/gen1_pilot_2000_clean_two_task_20261010/place_mouse_pad/result.json": "24044b3e9dbcb7b122790d83cab58e593265be9e1805060789ab757094c0ac0a",
    "results/gen1_pilot_2000_clean_two_task_20261010/place_mouse_pad/run.log": "ad8b40e130c3fe4e6ea14b7e51cb5219d0c16bd54cc015d114884c3f1b202665",
    "results/harness_cwd_fix.md": "bab2afec920cc5356baa197b73aa9a8a208f52dcd50d32d03b706332e2a3ae81",
    "results/harness_fd1_state.md": "3d5912aeb83b0cd0dd285667cb6b9f0683bdfe4ca379b8e27befcba751e9f005",
    "results/harness_fd1_verify.md": "efaf67d34d40a6dc2cecbff23280be78e98f97c6a3d50b6d6f9384b096e81a26",
    "results/harness_reconcile.md": "e083c0709aa8e34a8041780902f2abc26bed975ad5a3478f8a1557b3f15d14cc",
    "results/harness_validity_reference.json": "a8cf3da4d09778ff50b8790f39b84b752f98f11737dbf3a4b3636237aaa98180",
    "results/hf_transfer_install.log": "a488b0a7a5f317745734cb39b4095fced15ed4e12f8bd25c7fb422512eb41edc",
    "results/kill_orphan.md": "4da83985ad5403c883888f113cb85505e80e976e61488b840db6ec6be0433812",
    "results/make_policy_batch_fix.md": "01d477b3583ececad543434cfefbe3fa811a64fcd8917110a3313c8f3fd05586",
    "results/make_policy_batch_fix2.md": "91f8193b4dc45957e1a4e5ff506980f6766bc7f797881cbf438285b55541e2e1",
    "results/objects_extract_fix.md": "67a2864a72aea254ca92363afb59012b902a84afdfc62b73edf35d4336c5338a",
    "results/obs_bridge_selftest.log": "e82e7d433e8ba81c4ec22c678d4a9e4bfaaac1886ca57c17c151a920f1e82a97",
    "results/official_step_lim.md": "243a6e50306a1f486aca231a94722f6f62db295634a40db2483351657569dbb6",
    "results/oidn_guard_fix_report.md": "5a3f34ad7377ee1e53753c246b5e796b04f4a038d2b072ae1c54a3aa74550107",
    "results/openpi_converter_deps.txt": "3ff793c305a2113374c9318c8548e43efcc03f825675486c3035bf913cc8642b",
    "results/openpi_public_pip_extra.log": "7839db99e258cd4a4a44e6ddc75787e31a0e00a9ce643047caeb3874e1a66ffd",
    "results/openpi_public_pip_install.log": "5d828e557f588f55079a23f38fd88b615ff5e62e0c607231555c37c756286bfe",
    "results/oracle_complete_raw.log": "5a33572392aa28dab121fa6248b6efd33e367757e25ad8ea6de866569902a729",
    "results/orbax_conversion_result.md": "ac7e2ded8da6ca3c20c6a3ec757c5567f8b70dda3b3fb95575ae082083cfe750",
    "results/orbax_cpu_restore.md": "75cf40ad02a5c752a25ffee4c605ac0b2a8b5534e92a3804f11889ac1cc2298b",
    "results/orbax_cpu_restore_probe.log": "dfca6cc25973a6fbf41887f306ea800bb8f8b4fcf747b3970dfce63bceeb9f3b",
    "results/orphan_identity_probe.txt": "e923fc5e1f1affa26b4dd0fb3430d6ae350bbd477953fd84cecb5b690b85aaa0",
    "results/pi05_key_mapping.md": "bf877396d572cc09bd125e2d63ca8cbfd948d9dc3d96a5b6731ecb3df517acb8",
    "results/pi05_official_orbax_baseline/checkpoint_protocol_audit.md": "c790d8309b5d8558d691dce64776fabbb1cb38e0718eb794397ed0ea354ef66e",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702/adjust_bottle/run.log": "407ab5fec13cb6c887ffa15ffb01ecbb18f80ea90e54efe7418473460881dbf1",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702_v2/adjust_bottle/result.json": "5a22f6d89b4b98de24ce5481a378e11007b3bcd2a9e49fd3dd8698a0227242c7",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702_v2/adjust_bottle/run.log": "1500dbf2e790498f5aaf56f3af6001a05f5364a09e70ce09e8d76347d6338c6a",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702_v2/beat_block_hammer/result.json": "6f47ec1ca500d2a85217d6d9b56bb48dfcab86945e110c564163e8f2a655a583",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702_v2/beat_block_hammer/run.log": "fdff5b82d60083b1119e18d19c9236b98ff2481e68e24261b8ef0a1d6675206f",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702_v2/click_bell/result.json": "2c168d1c9687ddfe5aba9b71fc7cef5ccf8bccf1e79a4f5b9faf7fd7c821ca0e",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702_v2/click_bell/run.log": "33ea694df5e57420015e4ca014a2dc3376d8aac7bef3e2a861db3ac803ce0eff",
    "results/pi05_official_orbax_baseline/diagnostic_3task_clean_10_e154702_v2/manifest.json": "54d1500a8f6aba9382591c02ffeec0239bf9a1839c7d37d8e627ed6f5225de5a",
    "results/pi05_official_orbax_baseline/diagnostic_adjust_bell_clean_10/adjust_bottle/result.json": "2217c2ea915a320ecb3ba2f2b85f3f7c5b9556a0fc913b72ee872fe7807b803f",
    "results/pi05_official_orbax_baseline/diagnostic_adjust_bell_clean_10/adjust_bottle/run.log": "909061cb53a3e3144a854cc32447c519996c5fa00ae64deeed8140008bbb24b0",
    "results/pi05_official_orbax_baseline/diagnostic_adjust_bell_clean_10/click_bell/result.json": "3382490e9eabe24acd01cae57431bd38a44c107e8ec50819a174cea3f0ece5f0",
    "results/pi05_official_orbax_baseline/diagnostic_adjust_bell_clean_10/click_bell/run.log": "99ca8e1bcdb6aa3af2fb5eeb8dceacd2016e2b134917e1af0e13cb0d351f4d4a",
    "results/pi05_official_orbax_baseline/diagnostic_adjust_bell_clean_10/manifest.json": "d5401b67564cd1e55db1d03d5cb43a9fbc374db867c3b734f655348ef3e49a5a",
    "results/pi05_official_orbax_baseline/diagnostic_hammer_clean_10/beat_block_hammer/result.json": "5c197f6adcb4a09da7e6cdb29d31c72ca863ccba7d0a7c843d544e068b78ae8c",
    "results/pi05_official_orbax_baseline/diagnostic_hammer_clean_10/beat_block_hammer/run.log": "740d41d18cfc46218885ae25a72a926d3f86f08a0aa1a925983afbeae1b9cff0",
    "results/pi05_official_orbax_baseline/diagnostic_hammer_clean_10/manifest.json": "aec6d329c27f27e503f4d8bc5c990cdb9255b148055c2276e3985ba7d54a9871",
    "results/pi05_official_orbax_baseline/myungkyu_59999_clean_report/clean/per_task.json": "e36842435a90685c21c954c27898d62620e67f9ae4318787bf4cc0078dceea08",
    "results/pi05_official_orbax_baseline/myungkyu_59999_clean_report/clean/summary.json": "3932ffcabb0aea8f48337f41dc4d10a873d6b0f74ca559dde6ea1d8a9553ba47",
    "results/pi05_official_orbax_baseline/myungkyu_59999_clean_report/comparison.json": "d59f32fb6448ca78c0df8c6f40e78e8e61651135e2b675886f2dc667fb8fdef0",
    "results/pi05_official_orbax_baseline/myungkyu_59999_clean_report/final_report.md": "e276360432cfff9bcc07ce24e518fb2fe7202243d8095265d80dae58dd49366a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_clean_report/official_reference.json": "f4e6d43bb3e214088aeb0a518f1dd1c3cd50358a4ef4a8455a019a647c3182a7",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/click_bell/result.json": "1ad37f49aec6097525ebd091673f07736bd9a3d4bb136e44fee8640b118740ae",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/click_bell/run.log": "9a4706a7574a0c5f72ca0b64a6c9f464fe7ea2fca3f6d579aa6b7ef43f707bbd",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/lift_pot/result.json": "bbbc6fb428dd6380027ae5eb7aa949f33542b0523c363ece5f670c8d4cfeb952",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/lift_pot/run.log": "ba93c5ebf4117d4962005c398c443fa141ae6b29e520110df19411374f57b0bf",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/manifest.json": "4445c3fb743b8933a572748567713917eec54011c497ab06020fd08c5280126a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/place_cans_plasticbox/result.json": "bfc6f04b11343f9154a6179ad595e48c37aa8117e10ec34a854bb493b404b4a2",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/place_cans_plasticbox/run.log": "2eba045717760d6207f8b5b4f13e66e2ad94dce5f177059d6d1fdc96393fbab6",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/place_dual_shoes/result.json": "829451b8a59c50db44c58a58799d945547566d01b52e206a1a1ae49cb553717d",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/place_dual_shoes/run.log": "5f9fae9269567b81efc11577277cffc0a012f0b6dab83bcae4a0592158f190b7",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/place_mouse_pad/result.json": "dd678f1b33e26f018f278efb58934ffd0dd4f0b790be542865183a99962e0551",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_clean_10/place_mouse_pad/run.log": "152048eb890ebbf15c0777fcacaaeb4abf2b17a0d65d72660cd2375e949f012c",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/click_bell/result.json": "6e8d1319f21d859e251e8bd4ea7d4f1385b61886ba41384d22373e483ea9dfd9",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/click_bell/run.log": "f09dc0486a33207de84659263e638620e7e2333f7303110c3428fee35080cac4",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/lift_pot/result.json": "6ac65169629f7a37087a0899f5715fa8f488f68449c7938a52974ecf1603fbe5",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/lift_pot/run.log": "90daf15b696db6b3c5c5b408fa09ef7bb9c0f1982fd1de281d3c128b2ca82838",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/manifest.json": "fd044bb3feeac308f48f0ca4eb515c7ef7ec607739926dcb23fd4e2d92dbed5b",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/place_cans_plasticbox/result.json": "49a5e9537a8f66a647b14a0a89f19461f82cb1f94d119e57e8447c75864b3364",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/place_cans_plasticbox/run.log": "a709b2cb8803fc169a5862c1766653bdb674fd3293911bb5e5674b86b2b15750",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/place_dual_shoes/result.json": "c6fda4789b91039beb2ba3c3b7bd2a54595172750c8e3ddfe633122256a343bc",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/place_dual_shoes/run.log": "cfbc709c4a99f2b0817c221416a59687b72582bd6f7447e4c963e3c288936ecf",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/place_mouse_pad/result.json": "1c1f47b9f31a7db2324f726097de263ecf90018d4a3fc6a67eb7093daa387b97",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_randomized_10/place_mouse_pad/run.log": "32a850d69c11ce760e59a9d22ffd6fd464920196c817788e287535cf314852fc",
    "results/pi05_official_orbax_baseline/myungkyu_59999_diagnostic_summary.json": "989572c9e6dcbfc4c058131699fc51f9f6301e5487ecfef2434ee6f8902f27b5",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/adjust_bottle/result.json": "ba6708a1e326eb479ce12758ada54f33c43a7741dc7e22123fd7acf0791e7a75",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/adjust_bottle/run.log": "9149035e3b5c5c0e1e809dffd59930f94265bd0a1ca3c2608f80a194f89b98de",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/beat_block_hammer/result.json": "aa54df6b1d75e31f351a52ff8fbf19231745cada87c992d8c21c60657c081a8e",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/beat_block_hammer/run.log": "482290941022a745311873d93a53c5889b698525c3a3faecd61c61395b7ca829",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/blocks_ranking_rgb/result.json": "e30d0072edd0a269d3e0a16024bf6bfb5c9089438e655eab12e72fad89554862",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/blocks_ranking_rgb/run.log": "1fa46ba790306a0c9ac8af48403be26b30a749a58e051696f6c76f92d6962a1a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/blocks_ranking_size/result.json": "2613e87d14869ed943e7439a279544d103e0b01e699b0b8bc558b7cb46587d24",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/blocks_ranking_size/run.log": "7dc06d75d0bf35d7c33f5458076a6b75bb61e5fb0a6c30026cd4400304cd19d1",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/click_alarmclock/result.json": "d8e6fb634bbb6089e0788fff51cd2d016cf5082a6e224610a25bf80f92496356",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/click_alarmclock/run.log": "003c772f8533d12fffaeed84199b73af5714b0691a50e95cf0f27c5bcbf2c350",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/click_bell/result.json": "671124c6fa8bbaa9e2f907a0e6c12b144c3513c73bc038a2c4f02b8df0a40045",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/click_bell/run.log": "14ef8f51ff26ddabe274fa869fc0a67061804c270bf325a8f74bd114f4e2afbd",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/dump_bin_bigbin/result.json": "9e11e0e8e9d200a4ea2a540e6c015a0dd5a6a29301d661114e5636b0490fd080",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/dump_bin_bigbin/run.log": "e6afd9024df469892f7888429b56f79d015ae5f6bf401d6720ab000aeb9454b3",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/grab_roller/result.json": "3b58f11e52abdc2f960064acc2fb2fd238c0b2cd821317cf7f97e1ec2373c8e0",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/grab_roller/run.log": "ead3395cd0852341274b9c1dd12a30bdf26f7fafe6b9cde697d7cac5a4378948",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/handover_block/result.json": "eb3376c3863d27bb501fdc1e17cf26036a53c964b301360436af522247536e87",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/handover_block/run.log": "868cf11b79e2bc44085a206db0f1d3c7fbaa9a7efa5101fc3c80c323397b5a23",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/handover_mic/result.json": "07b6d7390b7434181326f18b43fc9958a77929556e33f6ed79a463f7cc8fbb73",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/handover_mic/run.log": "77545b25db82481c7ca8821e53faa44a5bc683e6385e5e3f33f206015a4ee23f",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/hanging_mug/result.json": "21fd8d3872f7daf8df2f4521eabb2938e6f0d7aca4d2a059e816555587d885c2",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/hanging_mug/run.log": "fc1fd8c46b490a7c728a4d07d935d2747ae060392af53566e815d081d02f2565",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/lift_pot/result.json": "3aac1b2fa097291df3a1ee3ab2735a54ba0c6534085186fa6e2cc622b62b5d44",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/lift_pot/run.log": "34fedd4ae60e66e0cb7c81a8feae240323ee3e007ad8a9af34c7841b10a6d892",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/manifest.json": "68cc593bd397d67b7a40b9031b5e930c3a9331ca4fb798bc31c1a23abbf7394c",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_can_pot/result.json": "9080a67c63756ddf68b2b07b4f4a2af412ed2738e931d39654fc2a4267e7d24a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_can_pot/run.log": "b0372cb23ced01c9d4ec2fd43594c86f020427413dd9df4a179447f06b633253",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_pillbottle_pad/result.json": "d2ce07e475403348b2b7323278d71f8275397193866aa144f31269c276121532",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_pillbottle_pad/run.log": "5b2298a03463ccea41f0a5c1d8132babbd8fdfe713689f88fc3a6148634b7ece",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_playingcard_away/result.json": "579cbb6920703d12b5b35aa42ee6db1f3b72634bdab8322bfe545a1fb6ba3946",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_playingcard_away/run.log": "efc9971b5feded317740ca3334b36a6f6622ab95bd6160886ca4c16be5e6e2e1",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_stapler_pad/result.json": "40a2e9f19d51bf617aaa5b64f74c21c7eedf62f4a75a4a745931d272a97da280",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/move_stapler_pad/run.log": "ecf0b8100fd5d00e5e0adeee008c61a22a5bd17afc3650041452062f4daac587",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/open_laptop/result.json": "2e4f8dc4b8c7c8cfbd842d1a2d8344306c1f74b9243b832c728e6f2f320e4400",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/open_laptop/run.log": "d5dee7e96cd4cc144c058421360cc0bf7f5d0069ab1213f9a4f4419b7f3602d2",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/open_microwave/result.json": "61359541e9f18d1a1bb8f415b8507953d8097f3451a107cca5cb424c4e9af503",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/open_microwave/run.log": "6e7f3c224580c7dc50d436a36637da47d68825686c7a2a07fa1b0e2f5de95f67",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/pick_diverse_bottles/result.json": "550cc38a45e8924b439da2dea267b3ef363bde18e9d730b8d179b6f1c889b348",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/pick_diverse_bottles/run.log": "25d0f5876e735f959c91b61015f3d282695348d5288effe930fc96cb103a4d5c",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/pick_dual_bottles/result.json": "0233892b226bdcb64d48f9508fe0580fedebf9c637991a1ee8eb3aae91345b0e",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/pick_dual_bottles/run.log": "e7254928bac75bbc3ba67b953c570e56d76314d52b10f5917347913194792170",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_a2b_left/result.json": "5aac13cce750eaf4c099dd4aa82552107c4fa84172b7291fb983f6f7186aa648",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_a2b_left/run.log": "1586edad3d7b02e405a519d0dc08b5c9cbea1fadaeb2b057d3bf356e734cddc7",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_a2b_right/result.json": "6c578c8b7274e51e4d0bc99962b87e029116e10aadcb1565b616ad9a0a4962a7",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_a2b_right/run.log": "a783592500238af208f8a8f21b67eddda1aac4334074d85b3d4a1355bdcfce93",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_bread_basket/result.json": "6af476e2113b54da396364d6b0831c284775274dbe7be320475ea2de0b44dab4",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_bread_basket/run.log": "a7bacb052ca244aa263612ce1873145525412cf19536571c20cbb46346b6a68d",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_bread_skillet/result.json": "d9ab9a4898e79673da04a63b7253dd4a1268df99985129690aabba33657bf439",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_bread_skillet/run.log": "ad745ead2529926b13e28f51bcce5e900550116173f6a3ffb6ef64fc73dcc6fb",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_burger_fries/result.json": "4031f9580b6dee7184cb4804dd575a3db581fae94564758a984a7acdbb40e144",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_burger_fries/run.log": "1083f6f362feeb8d3e62b6a85069f5cbe984f071aead55a14c5edc436e4df057",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_can_basket/result.json": "dac607ec7012b48ba8324c5276e4486017d67e03d293deb2144d273ccd532337",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_can_basket/run.log": "17aa0babdf18a7e6eea996bbb754f515b6db65e3883abdc771f6853adfca752a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_cans_plasticbox/result.json": "ce4d8ab86642f1f7faf37236e58939f554d45f1669a8befba333fd7925b6fbea",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_cans_plasticbox/run.log": "78c4ffc9b85170e789c15229fb00b55eabfd3265c4e988b439bf3a83a4d42627",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_container_plate/result.json": "aaf6aa50d0407370a1b856658fc4406cb5e2e4bea172f203d1b3d84da6b002d0",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_container_plate/run.log": "62606ccd2fae9e473892f1168e932c3cb04f10f652c904df32d453cd6a98cec1",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_dual_shoes/result.json": "1ecf1a524311b47a498c839deebb4d89a4ccfa24d4e39619e7b739d74c556a27",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_dual_shoes/run.log": "e2dafd99f96c829cb44387dc6ce24dbc5bc28dcb5f9a17df7bab45ad8a3701ce",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_empty_cup/result.json": "3919b6d18207e4c605b0f6f36c7f8b6dc274d2dc8f9a38d8336ee0860b266821",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_empty_cup/run.log": "14fa8f234e543825f6e68c6d9198e7d47fba08f38a09f5ab145bebc473493f51",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_fan/result.json": "d6a3b144727e31f0a1caf0b34863ec1f192395f6eb6bb65a7bf2809fa5eba33e",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_fan/run.log": "6713d2090426b1ae4fde2a84232c5a9c4287fe9af7fd0f5427d85a9143f6f757",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_mouse_pad/result.json": "827b1eddc0e5df04cc518a5170b090245c71b972d60c7a70b9a72066bab34822",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_mouse_pad/run.log": "c3426003cd2198150d75eb974986bf0617831c40cecd35003cbf7c5ed51411d0",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_object_basket/result.json": "ee56f23fdaa5e84e908c78cff3a323d5261b072c59e849158c8517d38b9a4834",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_object_basket/run.log": "af1353435ff8b9361b6d71b0da3e8bf42c55f3b48522bc3060ad8eac70f896f1",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_object_scale/result.json": "939310e5b40d1270b58bd9214cc0c64d65de52c761ac6d6b69f81193b54fa642",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_object_scale/run.log": "333cf222b262698baa838c72d1f9a75885eba219cdabf6eb9a011da9c65b7946",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_object_stand/result.json": "94cf809d39c6b6ba18ea24301d12c2d8ebef214564cf23cf75ee873889be49bb",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_object_stand/run.log": "56bd372a23445815177b7b062afd8c076b549050d1601944cc16bcfba96685c8",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_phone_stand/result.json": "ac36ddd06b8e2c1b10fec7dc71b40c109490e2fbb604e9663f40960c6c81b590",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_phone_stand/run.log": "e8a0804c81bd320aef6204cf47045a0c12670fbad50f26455ade056a2e7f3a74",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_shoe/result.json": "b8e3709d30a26da3a1cedfd7f6f98119a8c40996dc3cc105d76546ec08611d36",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/place_shoe/run.log": "f6fd13319df014ae1dff95053ecec31035a15dabad4b978978b7ba59b88740cd",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/press_stapler/result.json": "242e8e14b2c11dbaed0f19499fe69e53e7b5686bf8d7d47c5513a730f87b16d3",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/press_stapler/run.log": "8de21d2323acc39db260fcba7121ff455ab49c9857448359168b5b35718b95bc",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/put_bottles_dustbin/result.json": "13370766e9e2909f959c93b993bc8da9dd3c9c82471674b7bc5e32a9d3943bc5",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/put_bottles_dustbin/run.log": "3a3a14c52028e0e0c188ff7af2e653f23734d4e0dd160980e72b2453c844bb31",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/put_object_cabinet/result.json": "961c01c870515ad85a86a01cf03f6ee64a3807ddb5db6bcb88c1e509cb639a72",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/put_object_cabinet/run.log": "624793e655ef363d285a78352f4538203043f04c28cfd470f6a8bf685644140c",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/rotate_qrcode/result.json": "f1d56520d89ab5ac23250dddbeb0d0bf8619d940a6aab3781f2c956aeb96bd5a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/rotate_qrcode/run.log": "70491c34e30c3a9b4411285698c2ec2f239b425ea6fd74cc2d4f30bc0e320fe7",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/scan_object/result.json": "ef6ceec71bb58edf0338cb70fcc198db3f56da84d0aae0fcf0b44b427157900a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/scan_object/run.log": "fb7a3d5a74bf65dced25fbdca67c08b5c45c5bd1c159d87cc5384485f70a908c",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/shake_bottle/result.json": "613991e41298671ea078e85ed4d9d85cabb1fd1e77c101e4453f80da47cc1e22",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/shake_bottle/run.log": "c717d017743472fb566c0ce63ff558afc4474d285bfa35adc8e85d5abaef0adb",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/shake_bottle_horizontally/result.json": "e48cf4290fe345d3b3e7603127f5c57dfea80228ec97f5eba0964ecbf6db5389",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/shake_bottle_horizontally/run.log": "dc63d553ee8a1805185b51eff01d95bff47acf5b5a406b79966b79297a31c3f3",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_blocks_three/result.json": "f9a6fd380fd611cd2520e73070c5ee5072c4314ee9b82f9bc6d095fc9be67e8d",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_blocks_three/run.log": "fa3d1b28f20f334b02f96a7f03738f7a5e0450aeb6238e37c9f73cffb004cdc9",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_blocks_two/result.json": "3e356e391c16d7e44df3b88b8f77407c4dea11650a410cb52892793c3c8acfbb",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_blocks_two/run.log": "2e84f0aabdb529403084af1be847d55ffd3b9eaa635b3e789b42caf20a08bc0f",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_bowls_three/result.json": "9374cbc62b9a421e674d79e5dbbe5648dc11d12dabaf75f922a0c1e707f1d5b3",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_bowls_three/run.log": "c90109988ede72a0139700a4eb823498d0527b3d1a245c6c32e21a481a60968e",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_bowls_two/result.json": "3799e06f8c68027b1f75335b0045ad0fbdb3096b4ca24e2d2d50fc3d340ab05a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stack_bowls_two/run.log": "0dca24c9b3f833922c07daaf81899373c9a993257bf475163b5076bc03014e15",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stamp_seal/result.json": "90fd1ea4989cddbc59095c0ab87eb067c7aa802471a069bf0070db1553046636",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/stamp_seal/run.log": "58db37b1c1243ad38700a4121a34ac5c0ff2d58e1632a837640af201b6fb363a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/turn_switch/result.json": "125edbf6dcd4f145f6cdeb561bbe4c72295f980e08b207b1d8a0aa67af38e33b",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/turn_switch/run.log": "d458b15a5cb90eab453d6b79e6cc8121df702a5125889fcd0b8f2c19153c5a77",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/adjust_bottle/result.json": "fc245c2288c62894f4562126a90f4f0c39dc22505a09626a0ae83fb3969a2c90",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/adjust_bottle/run.log": "db9951c9683a10c8a750fb484101c68a0bb129738d28d3d68a3e0aadcb2d16ba",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/beat_block_hammer/result.json": "f352f6aa116435a8a214a83fa464ad2cafd1050c15228e76d53db5d6a09a533d",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/beat_block_hammer/run.log": "fcb2e0b29bb63ef9cfc8289937fce42ce2a9bec564bd4b23683e4a6302e70d9e",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/blocks_ranking_rgb/result.json": "02ba399fefb814182214aec8f3cbbca774034176bac51e307f8634e2d8a10601",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/blocks_ranking_rgb/run.log": "c53dd899e84cda0ed28f3a9e10abcebe61aaaf10c7cbf15a1fe5b71dbfd6f173",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/blocks_ranking_size/result.json": "c8672f023237fbe8b0329b5719a7ab18b026d6fa28181694b437f7e53dc8557e",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/blocks_ranking_size/run.log": "eae6ea346eaee243f318a08e1a8a0cd6aadf9249f5c530ac39c3b3d09b76b657",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/click_alarmclock/result.json": "b7e079bcc1e3949a11028365bed8076ebcdcef6c1973c52b515f4ebb1eb2a462",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/click_alarmclock/run.log": "c9d14f9fc6a6428f23e6052f6e967e8648da9f62b18f740df9c5b2e535257dfb",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/click_bell/result.json": "212d4082fef18565dd4c3f5b44adfebe01cd3e825fc9c0ee1296d0784bd7a915",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/click_bell/run.log": "8005c75c13ed178d0258c9bee22453e7732a905672bbfedbfe46d33175db6e74",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/dump_bin_bigbin/result.json": "bb522f0ee6eda4fd5290207981e834ee9c96602d33ab5030f0a7fa345badb2c5",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/dump_bin_bigbin/run.log": "e432cdb34134f163ddaab504b036f55ca695d019464258045f75b6bdc36e8f5b",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/grab_roller/result.json": "e190cb2892289df7fa2d342ba283cec2d2fe45a93da37f1da14a8c77805b4918",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/grab_roller/run.log": "fe35fa7265726fbde9f37e4189e6564ef9360204fb32ffb541a8906cec13792f",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/handover_block/result.json": "2a1d308abd4fca7f59980002d55abfcfe83c9f4bb782ff12921ed37d01c37999",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/handover_block/run.log": "2552c779c1932012fd94d46f24e383b237db8544980409e35bee45c95f95462d",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/handover_mic/result.json": "45781db82b8f5388961004fc66390047b8f7460b2268ed80f38d977b6f086008",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/handover_mic/run.log": "2abb1b6c1b77c7f00ede4a79c14941540a3a0bc68ad79c0ecd4f613cae1eed11",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/hanging_mug/result.json": "4bf8aea7fe317585d7919de834c3f3f142b66bc5d81a6e352b9e1c55c4519f47",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/hanging_mug/run.log": "1961ca681e0ec02c1c0de08b3cebbd7c9f800bef8286a7568218ac5bf2ee9de9",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/lift_pot/run.log": "4018460973b8299ee335e211e38b87a1d2eb41d569127b8c5f3e1567d79f35a9",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_randomized_10/manifest.json": "2ac6ecfe69d78d0013b83ce430bf1d6bf33c50df87ad76d5f6be2ff1ec073d8a",
    "results/pi05_official_orbax_baseline/myungkyu_59999_full_run.log": "f9d0adb6d7d01737fcbb89b088b10c7b7a2f201919351e203252a88f05c0fc83",
    "results/pi05_official_orbax_baseline/myungkyu_59999_resume.log": "3339095fc89a81334c4b39d06bb461a21c3fdcc9d1177af427c2cec7a73b521a",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/adjust_bottle/result.json": "1f9ea9a6b8f708ba13f97eb3f12ce9d3d3ac7c8878ce879c9a7300e510c83198",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/adjust_bottle/run.log": "57f8d43e13aac3aac4bdf880796196e963f5a21086ec5297dd31f55597afa839",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/beat_block_hammer/result.json": "7829fc15d929e65144ccb2bbbcca67db4ff6fe6cf2fe0e80d14fdba343c52f89",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/beat_block_hammer/run.log": "73e8caafb000cf4688264b36cac74b2f61b3f53ec37cd7aff075e0cdc488bcd0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/blocks_ranking_rgb/result.json": "59096922b351744a1c3c9d11c13ca6b98ea4657155255ea8c3ee1c8f5e23c3c8",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/blocks_ranking_rgb/run.log": "13ab18dc02b7a3a160180eebdc291599ca4b73924316927d05f002861d9cd0b0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/blocks_ranking_size/result.json": "08257cb3d82768325565f816ff2b110ad264f04dd6832a1f9d83fc37f0cdce5c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/blocks_ranking_size/run.log": "d9675c97494be5544eedbc986ca3ed836233ac91181360089bb3521fbdb887f8",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/click_alarmclock/result.json": "e61b6cf1c6a47ad606ed35a6f7c630c268b066b2fe2f6e1d6eed50e5b8ece5a5",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/click_alarmclock/run.log": "17d9860532cb5a4d7c69759836517b72f0a4d863ba8c984e17ae9b56d725b342",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/click_bell/result.json": "75bd316afbb6d74dc4afda59d5dd84a42c16f08463512ebcde2209c59557fad4",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/click_bell/run.log": "544bd3543367bdb539bbe929477a55624c7795e205f4e93285b97db226a55ffb",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/dump_bin_bigbin/result.json": "8747b86a50423f94ce58b8eba7afb58d0b201eb0c12a6be07074e151229b7bdc",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/dump_bin_bigbin/run.log": "28c32e238cb141ebcee67fb06b47765c37f6707584e67ac6f857f649121d5d23",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/grab_roller/result.json": "0daf223682baab1fceb02e6c88cf7183348f30900a66bcc212662b0cefc15588",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/grab_roller/run.log": "5914bc2fdf4423c4fe3d7d813e76e56ccce7baea2228f000418ba522872f368e",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/handover_block/result.json": "989041581dba9bd84acf62f6d36222250f9d74c90d0c8be8117db688d9607808",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/handover_block/run.log": "64b3cf3458317690c5fe27f17462f827fd081820eff59dd729bb2681418fd063",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/handover_mic/result.json": "16fbf1b21f2612e1bab969d8c2107b98f3610bafafe015f03b9fe9868f918083",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/handover_mic/run.log": "44e9a623c6adaa6ad69f817ec1299880ab887a7cb023e5d004b3d80c872af3eb",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/hanging_mug/result.json": "f6c17373c5dc78eb55958599f3cbc451659a5ec650eff832dfe0cfc397a627d3",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/hanging_mug/run.log": "e17ac5c0af27319ed848c423cd3266aace555c0f5f3fc739b1eb100611328558",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/lift_pot/result.json": "302044b1e6475f5c189439c4b87fa24bee70b35b4d0dd780f9aa66b69613aa8a",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/lift_pot/run.log": "98b99778ff6be59560b9786ecb9c69f08a7a3147d29eae7feaab67cf68bb5c6a",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/manifest.json": "8097f9a4433de2f03f0c98d76eee80dd30bba8e0b40d9852ae489387220ac7a0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_can_pot/result.json": "fe62cac73f0c24435df685b8f2e9a0bb4afba755697e312360d9bf0585d46dd5",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_can_pot/run.log": "61b6f70a7015020add9b2edce50f300e5917f3a76413cb0418c2e55a011044c4",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_pillbottle_pad/result.json": "ce6df7226a8225c819b56f58179e1a969bd2102e11e0a39f750043c2c840a6d6",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_pillbottle_pad/run.log": "c5a980c2568cddee0ec4879a88f686c08b83aa07c4689a5e2a500743e5ce5402",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_playingcard_away/result.json": "779ef1f7fd3dbe04992652a229fb6b55673a66d2d90e6c881d226765d5ce62e2",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_playingcard_away/run.log": "b48173cf2771bc8a861a5113617fdaec92f54d709adf41b092eb568690014c42",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_stapler_pad/result.json": "2ebc2c7f3d96d478de1618b23d92953d3dd1f3980e0ca1b7820c90dfb94d1c3b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/move_stapler_pad/run.log": "7e912eb3a75ec05446b8335efa34db49b52ddd8292c4be1326bc02b6b1bb4b42",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/open_laptop/result.json": "79a27f0d46c5c9421c5f8167b91a66766f329f8aaf39435369e1ca170732a0b1",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/open_laptop/run.log": "58130733a181227dbd396db13131688cfd3d5a96be064efe44b519b82e9e5989",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/open_microwave/result.json": "6d5249967bbecce2d8cf7eda5884cc3adbe762d3c31a9c67f37564552e41ec53",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/open_microwave/run.log": "6d02eab17a7a2e683ebe756bbebcd67cdacba43750d58ca4cd7bb836d843151c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/pick_diverse_bottles/result.json": "723920453dabdb2ec323502eac5f60c5af1470e48c5e53978db6b0e4f28b2dd8",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/pick_diverse_bottles/run.log": "cca6d459344716bb6d0d0b1601835c9d84d7a085bb909c5107dc6061499a3514",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/pick_dual_bottles/result.json": "ce2443060247373921f4043a04e7a5a048a83f1d0dee31ff8e6803cc4e2db62f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/pick_dual_bottles/run.log": "b1ea07593356c7b790f5816cf8e287a18df176addc8897fcfe948e5780e3e4d2",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_a2b_left/result.json": "6502e2d6207102b288fe28d7caa66098dfd5d992d86674503e6894137634d662",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_a2b_left/run.log": "7d09d5531be152630fc4900b5f02043a7f0822572725f7ccd3b9fc08a6d4e467",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_a2b_right/result.json": "c423f4a0553e25502a80e0e9275b3b66a807c8435ce7ace740491d11da7884c6",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_a2b_right/run.log": "1d006eadb827b0c5d575aaefdc5be8b5c243513aace5314bb3776500d557b863",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_bread_basket/result.json": "d1529b266a49e0b044f4548ba10c5a09a21655e464e49eb539c61f4e7a918df7",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_bread_basket/run.log": "dacd2d1cd878e39031df65f5883f4c4c0c6de56ac11e3d645eae3a9bc67a0aa6",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_bread_skillet/result.json": "7eb502c333c1d246eeef5cd96ea15edbb4608978bff193fb15586c45937bac52",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_bread_skillet/resume_correction.json": "72c1d6e02178fc88bd8bca6668d0da0c3589d89d75c388f805ef5676864c3cba",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_bread_skillet/run.log": "31713d09deddaa1b7004693eaf347f03440b4ecd72d33fd53e887834b9562abf",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_burger_fries/result.json": "6c565165b2409429f64873a818b60cb208aacc6767fb144cdeeb5ce0301bdf3b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_burger_fries/run.log": "e549420118ec529979f3f505d9e8f265afb131203df51b4236a545af41c3c9f1",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_can_basket/result.json": "7c312c0dc0df78151900024d01565b6184770ea48570d5b9480d442259ca84be",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_can_basket/run.log": "b7e06d8ce3edb7ea0d6d684ce3213bd69351c7ecfed6c876a363112f14c6c5b0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_cans_plasticbox/result.json": "208195ad693cf8fad78e2dd5ed76f02932978774842125260045bb30d46c87e5",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_cans_plasticbox/run.log": "67bea2171f28963042bf12b11d7977edf4c0037a976988e54492ec8ee56254ea",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_container_plate/result.json": "fd7bb9a36797e59893fd32050ee4976dc9517bcbe3e2399fc5e6ba1899b2e557",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_container_plate/run.log": "f369534977a1cda13bcf21132ead36bc227d30f1a89b020a15e168ec123dc608",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_dual_shoes/result.json": "7bf598bee6a7fb2966fd04301ffba9bdf5ab0e9fa2bbf03a0c8f3ffc22b9e0e9",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_dual_shoes/run.log": "3a276201232607c956c06ab2ff56b3a1d69405f4456ef2d5c5c722b7d154f82f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_empty_cup/result.json": "cb880486f3fa6b054d2be8a3a0206179304ddb4ef7219e798e04f5c8f7d2356c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_empty_cup/run.log": "5cf88d9f8f5ca844f225227751d3c2566568e26d871958432309055346daf194",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_fan/result.json": "8767941f37d6741e0e9d38d88d1af2969f6d8525ef7a968a62b5df74d329f3b0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_fan/run.log": "8036442cff689d99d1026ffadcd822ee1b5eeb7145d496b9e606e75d6a8d5c0c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_mouse_pad/result.json": "ca6390f6918b95ae352a949fa7d664e563a7dd081f12486bfaadfc8435fea8b1",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_mouse_pad/run.log": "730189bc5d2f88dd7df11ffa1f232ea5f83b4f6af6c7039fd7aca9d6422040c8",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_object_basket/result.json": "76e86a40bcb5cb00768882d77b3f084ef10465fd7d0164264206d15272f3a224",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_object_basket/run.log": "0a6ef3c80c2362914516e47becddbdab3c9c22cc4c26c672f07a9db924082319",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_object_scale/result.json": "ccc1116aae3392bcd58a88ca3e44acd67914c060fd2c005840d5c4318a78d7a6",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_object_scale/run.log": "9ff6fab2d159b0af567c4eda338082109361bd909c11fcbcd4f3f7211662e579",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_object_stand/result.json": "56aa336a11621e5cd82ff8b35476c751bd038f912a77646286119b612447c951",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_object_stand/run.log": "d2667ce77379376fab1812a0a459a7aa88dd46ca1c90acd3c1fcc8a26bfb9225",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_phone_stand/result.json": "5057679f203f5ce8f477ed0740b9e87b1fc4f4a8f68687363361ca64af820dbc",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_phone_stand/run.log": "70d6c1f5ba973758c55eedd1672994d47901476102f1b834753687bceb4290cc",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_shoe/result.json": "4d0a402bd818474abe0ed56f6d351a31cbcce2af1b0028509ab5489379108ce9",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/place_shoe/run.log": "4962efe360796b0366bea0d001198b29c53778ffda3c1c007807f524e0ce0501",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/press_stapler/result.json": "47db9ba4574158cb999630fb7a1549292226ca7afac8b99c4c4b702f13274522",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/press_stapler/run.log": "e02665234cc64b3c90900eec2e5d31a6f1bbc5a01d8ed66c11b6dc669900f5a5",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/put_bottles_dustbin/result.json": "0233fce4e1808cef7764e619cb8b917ae5f16841b56009f8db62a60beca22917",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/put_bottles_dustbin/run.log": "0b1e2241b33fca4b3f2a470f25a2b09181944ec65c44b3ab3f5d84e8dde61711",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/put_object_cabinet/result.json": "5383c99f82d8f0d8db218afee0b902a1262302c8788e09a5b8917322c52d769d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/put_object_cabinet/run.log": "110fd671a73208d91f85aae0bf2a7dc21e04ebdf255a0f0ad7f7ef25b9f32379",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/rotate_qrcode/result.json": "2a5c6cefb8d2521a588cdf6ae2c6512b35ac79287c220dade5190f378dc246e3",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/rotate_qrcode/run.log": "0990302fd108d3cb7a5efb3f443fa0f2fc0ce4baf87526d70feccbd461ca9335",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/scan_object/result.json": "a1bb86bef73a6ea34582a28ae3f4040b444609188ce48a40dc4818754f3f872d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/scan_object/run.log": "1a1653483f66d678f81d41ab0c84cf83844c1f76922c2f2bee8b425de2759ede",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/shake_bottle/result.json": "cf3eabf70388e01bb223d2476ed957d3ef23fdcbb36417939e290d60b4ba8bd3",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/shake_bottle/run.log": "dd90b7ea1bf62330be7db02286dd79b377ac1a5e3fcfdca90ef337ef357f50e4",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/shake_bottle_horizontally/result.json": "868a65337d9331c66b81718d7fc5df8ff699695c1351cf4bb54a8ff03eded04d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/shake_bottle_horizontally/run.log": "bf50f9cae670acfb39c020cf58da620e34df1e0da2f4988bd7872c34cc9b48b2",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_blocks_three/result.json": "46d832c316387f1f2715af3141969519d430f2aaa48bf1d2204a60a22ca7bf32",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_blocks_three/run.log": "54cf5f5b71a868d8c7bd1a37de4bd151a39506be027c778135db3b21b08bfc42",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_blocks_two/result.json": "4f5fcc249fb87badc43b76b0fe495eeeeb48755718d22b45f981380a927f64f7",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_blocks_two/run.log": "6a4a9ae8fea102b0d7c38ef3874004d97943cab4bc51cc45d1ce49b1caf6d84f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_bowls_three/result.json": "7c2b463f19bcd2c213313a4749247e52deea0a98e7d1d75f7e20cf7aeaea6088",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_bowls_three/run.log": "b3d6af1fb2cee5f67b74c6f80dc8b3bb0b3b87cd177d196ede8f95af516586de",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_bowls_two/result.json": "4c138fb7fc8b52b9c97c518d5f447ace5c863d17b7dd3335670c744af4dc0560",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stack_bowls_two/run.log": "c0161d5575a3a2b3ac269b92ce9933a9ebd7fb16d83cc13995e9ff8d4e8d0f5c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stamp_seal/result.json": "38cb27d3db3e82287b8ec69a6751b1353d779eb843807b19d561d88fab75697f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/stamp_seal/run.log": "24b07049cc22c2f0edc006f6b75221db209039fd4489ca46ab507cfed1630d9b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/turn_switch/result.json": "63548e7936a40552297c4ec7f7eff2e39aae64c4a54a07eef64bdee929b9626b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0/turn_switch/run.log": "49b9e907fa05109eed307ba36167f46218f75992f1be22172778eee35310d976",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/adjust_bottle/result.json": "754c2a7369d8c06a1e9f8ac6cf73260d38bd40c89fdcbc6d8c1aaddfea7df21c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/adjust_bottle/run.log": "268b7b76c34d8efafd1d9783849c5f2021676d99406c89d71398e7b8ff5e4bee",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/beat_block_hammer/result.json": "7850344a873f8e8ededaacd65f9fe65d8016801c42cc7f43e72917df695bb38e",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/beat_block_hammer/run.log": "ab915520fd1c96124474b79301af04e61d144a5e86454bdeb7a92e69dd741778",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/blocks_ranking_rgb/result.json": "1dda5a7f44208109b6a18d46f4c66bf1cd90bcbe66c4706fe63f0e572d020bb1",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/blocks_ranking_rgb/run.log": "215ae59fadfee3c4de37696d10733f7eac1871aac5e9eca9871ee6c22ee0bdc4",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/blocks_ranking_size/result.json": "6febdbbdaa97e3357d24ae5a0b05392fb1a4605dfe14aaf4f3030f8fe5f74321",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/blocks_ranking_size/run.log": "a58216a2d0519fe116162ee084e7a2da58d69f44ceb7462d8983a572750cb4f0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/click_alarmclock/result.json": "df129fc25c44213995fda463386de4b56ebb2c1cbbac8e10dcad37cbc3dcbdc1",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/click_alarmclock/run.log": "e07edb97ac23add76ca3449354cd51b2902bda6f5faea7e244a8c0c72f9a5b62",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/click_bell/result.json": "1bd618c594a6b10348f54c6ae0330fbc0d7cf555a3683d3af032dc1aa80624ee",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/click_bell/run.log": "465413d68d64239869b840f1fd3ef381a51b572b92591d53367ae671fd209bbe",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/dump_bin_bigbin/result.json": "127fd98e5867e78f0ee1504411337a6b299a95ffd1a483a7684035d439bdf090",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/dump_bin_bigbin/run.log": "fa53d6cca17efc95127c5f0b46bb76fb4e69051c9e59dc98687c7d473f31ad8f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/grab_roller/result.json": "291beb43652e106039694b94c47b3983d6a90d0c96a987c70f3a055130b44f07",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/grab_roller/run.log": "1619ac78230f715c8c682b5eb7aad5ae5c8ef2b42770c4bae38de30f659bc0c3",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/handover_block/result.json": "eb14b6142f3f2761836b60c3f624a25e44fc0740a322850aea030fed8863b670",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/handover_block/run.log": "1bbac3cf710b68a8c029f72be389a708adbf3c2f832864cd6d1abf871bf6c9f8",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/handover_mic/result.json": "3cf2f16f6ece6a553fb330fe99d60c6fef90e313baa4b14c98792279f62ec4c1",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/handover_mic/run.log": "486b23eea94c52150d2f30e4e5db6706034afefcecefb98b81e8862c3147789b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/hanging_mug/result.json": "5c58c59c86a9070108fae0e529e828fce569e9b6087217a598167de01ccced20",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/hanging_mug/run.log": "2298490194b215c8b4a7fc650687a873e1ec13a2a04373ecf93cd62f0535bf98",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/lift_pot/result.json": "f80101d0d59afd54f20e1896261fd48931779140546b58c7eeef2259b03f9ec3",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/lift_pot/run.log": "173381f9751ef41b4194c88c148680fdc5eda040aff72d9d1be1eefe66e4252b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/manifest.json": "0f094b31a491b644b74b45c736762cdcd51d36b1f4aab2ce3ed7cbe6ead863b9",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_can_pot/result.json": "1a0ffb5bdc35196da5bf2736ec3b6e8bbfaaa4812aa8ff820b2de3c3655b314f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_can_pot/run.log": "7fcc6e0563cdc7d1f6ed1c1718f53ecd5abc8b05abe1ae3deaa8583db345ce9d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_pillbottle_pad/result.json": "dcb6f2291fbb20bbae2d34060cfb3acdf9da0dedc4ba8b680d678b3c52994ce1",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_pillbottle_pad/run.log": "7fd3cbf393d670c9253e6ee26a8f79c801d3524d5225074a20ed9ac5174195be",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_playingcard_away/result.json": "464fecc45316ac77e63199ce2501ce635bd8ab9b9fb823c6dc6b53c8d3849ab2",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_playingcard_away/run.log": "ffa698ffa3465a63857796c70162c8bcce60a9f5329cc860e3f99f855262212e",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_stapler_pad/result.json": "fd775449ffafe11722145426580763e9065b16b8dee61477614ccf4cdd6c81bf",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/move_stapler_pad/run.log": "afd13638f81148093d95df2ffae006d61bfeb3f1662a42c4c1690148b8738e56",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/open_laptop/result.json": "410e824c52d1f7ee9b5d31c9e3b5a8152fbd95fbd605473d1ff1c422cba5ee5a",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/open_laptop/run.log": "853a3735553443cc1a7fb4cef85cf43d37129eb9d3e7bc30c4408f04d540689f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/open_microwave/result.json": "39ae011f8b99e9b035febfbaf97dfe71d67c5d6db62e77ce9420b4c684941c22",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/open_microwave/run.log": "1be8b64424da09cf71f8d1a91380e8b22232750bc8cbb8e624dfed04a4b4690e",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/pick_diverse_bottles/result.json": "ed41303cdf98d4cf782dc7902b673ea45572734b4417d6e71a7267fe4f931310",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/pick_diverse_bottles/run.log": "bfc3e0903cdcd9fde91a15a152fe88d1981e5ae8902814653189f9b5949f14d8",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/pick_dual_bottles/result.json": "f9faf1da0960537df0cbd59c74e167e4ffae24fe258ea5f75874b956a97e992c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/pick_dual_bottles/run.log": "a949e07c25741146a7edd23d064152418d60bca8de84c7eff9fe5e6ec35103c0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_a2b_left/result.json": "d73422de9a0c230a4c10e0b46ac093b3899559128bc3a11267fdc2fe32dc6b23",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_a2b_left/run.log": "086ea28030443195ec836c5f1b939c571a0cf54a9a0ed9c8fa5656b38bafca0d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_a2b_right/result.json": "7397f091d2c9824bd2ee3ca2ef906e6de2dcff75eff1d51a8791aae77a7605e5",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_a2b_right/run.log": "ac7823baca665268c752e9bce71260c52c7e17d6af6c738107f86f651d1ca51d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_bread_basket/result.json": "7bc488bb90c47b85aa036a9ff4e61d42a3c03486637a5e5cfe5aae48fdabc737",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_bread_basket/run.log": "3ebf6d928a527f07c620ba5cf8ebc0086a0175f75bc54496d1e6f5ad3bc2549f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_bread_skillet/result.json": "f49966b8d0879ff59d03f91f2308aae730e56209e46bcbaa2f059faa8ee7fcb0",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_bread_skillet/run.log": "0145405232b3bd883461c32652d3bf4edd7f2fc38f93436cbf2b78788a442c8b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_burger_fries/result.json": "6e17a0d727d48f80d7d475fe13fca753444550add7f4fccdbb50b571d0707957",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_burger_fries/run.log": "7bef25967cdf05b944dbff70ffa5926516c5671fce1d09dcb4f65e910e85d90e",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_can_basket/result.json": "65ac6a905d30b4a7d677e62cbd2f22130af2a4c6cf4d563431262900490a9eb5",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_can_basket/run.log": "00f92dfa890ef42eb29c5a59783611d7c99ab336682e0d90f26633b2b94b3c62",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_cans_plasticbox/result.json": "c613718ff0d7b9b0d896e738c5b552b34b5a11b5c98410de4201d9e3e75a2404",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_cans_plasticbox/run.log": "2d74abc89851c731074954361fb953cab5d879eac4faedff37b37f85fd107855",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_container_plate/result.json": "5bde34905de378bb0a56f67e19d169189ddbf534040fa8cd17c3c1af202cad6a",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_container_plate/run.log": "b3b4efd5454638c518afbaa4d937b31687693c8c44bf740a87e009a45a902911",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_dual_shoes/result.json": "7cf40b69e3ed21a91733352890fe71d4a922fa5ea5fe3ed308bf44f43a5a1d0d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_dual_shoes/run.log": "7f38824fac8d4d345da027a2adccc582072d93361f0a332dcad7d352ffb42db2",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_empty_cup/result.json": "aca8195a1b3b59fdab36ad782015cf8c5b9fa755fd977ca467e773c9db25e80a",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_empty_cup/run.log": "2636be2ec5009c3b8ec439805434d7bb16bb3e053eb74c406adb2f24cc08fb7c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_fan/result.json": "c9758316a28138f9d61fd9680e4a37e93fd86fc589593558b8f426c921cee472",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_fan/run.log": "fb392e66eea872c5d9644c178b882347f1f77fb477e3290992c19e62c6034ac4",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_mouse_pad/result.json": "a1018d8616b78820fac92f2fe08f6c52140dc7bd735258edc6ad2f28d70e0630",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_mouse_pad/run.log": "a49a93d4e678d297b5a9c4e3633d40f4396cebe77fc4a943f3b706f07b2fd583",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_object_basket/result.json": "160fea5a082e923961dac94c40b82c297fdba6dd101ab4e06bd80f7263710aed",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_object_basket/run.log": "f99a0283c76d49307fee0d38b34b5f00ee7f171112a145ba7f3b119aa4509ea6",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_object_scale/result.json": "9ca0d342f6667941b5664b86f59bd6ac4de38a8577feccb3965d3e41bca98821",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_object_scale/run.log": "8ac3dd125307ddaa5dcd2560abc8677737591d110e373d3f65f05ffb5c6e1d68",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_object_stand/result.json": "af4dfcf5c4cdd337a167d2491ed76979bc5d6aabdd2f09412577609e7414fe36",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_object_stand/run.log": "46462944bd0f4fe2cc990adc861613b99a5cbe61132b12e225bcfed7ce98a074",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_phone_stand/result.json": "8acdf170e6411fefb493666f39aa1e61471a2918727871937694641e3ec24799",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_phone_stand/run.log": "b6cee45a838a2e463155f9cb47d43c1fb58fe3be0779a33e2886de89e83f4f86",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_shoe/result.json": "9ec9e290964c8bd2c4c7464da22fd3d7d6958bbc36b4aa87c2c25aeee2770dcf",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/place_shoe/run.log": "9090730ae86f802600f414ae3377e03f26f87d8af0a75992c3dd01b174406d74",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/press_stapler/result.json": "2cb4daafea9c0e1038cd764c7737da3127ba43ddc02392f0e4e92d20829a3099",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/press_stapler/run.log": "74e81e8d544a9f1498ad7a7bec3067369179aa17d6b016873c99ef1647b496ce",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/put_bottles_dustbin/result.json": "f960d526ec1a9a5d4b144d6dbff407f75dd058e19efb33d10d83c59348c68e0a",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/put_bottles_dustbin/run.log": "92641366a35c8d7c4b15090ec830271a2093e0fbc71603067f965fb13e0b3dfc",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/put_object_cabinet/result.json": "1f4c62c3974725eb267491a021533c1d211bef1f1025cd36bb29ab182bf36ad5",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/put_object_cabinet/run.log": "2294b208a6fa57f46b5a86dd489747002a6bf27e8af88535651bd2d4d9f2e567",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/rotate_qrcode/result.json": "3cef38a8022ef6b96eb80312c9db44e525ae800b3b80848236a28e08b825413f",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/rotate_qrcode/run.log": "45f76189afdaafee35c7fc032a96cdefb03f68d33240d798ca869628bd913383",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/scan_object/result.json": "dfbfc39ccbb4f33972c7c333efe2055d214e626f2ebc28ed7457422f89d1497d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/scan_object/run.log": "5c40721c356db43d0e36f538210ba669e21eee5c67a8a7b1c5968355c1c96106",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/shake_bottle/result.json": "b81f340cc3c5ae5351e80b83bb64175c9c928a99ba111a4a73938d14cecbade8",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/shake_bottle/run.log": "427f75fc8c7a26dee25f4e7b00e722d824f4b0b5f9c478b431e2b038b4c20f59",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/shake_bottle_horizontally/result.json": "89c782b95382df12bbec6f92cfdaf00ba9319250a370bd4aeada3d17a1f1946d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/shake_bottle_horizontally/run.log": "65b1e7804e70ada50c075c96ea6cc24f8dc6a7dc2f8fa4b41083ad4c0ff4be51",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_blocks_three/result.json": "d501a2324d8257ac9fcc8c5cf4bf33abf6fe4734d5488bcec5904f3495e4877c",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_blocks_three/run.log": "17196832bc88fcd6015fe7ee8e45369094025cf8e2e25e7d74ab1586e63b2aaf",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_blocks_two/result.json": "a1546462ac381e507ae73deef92e5d856d341c64d9f51afd2690dd43a53ddb69",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_blocks_two/run.log": "a997d2460374a900ebad6311eb31430d9040d215e01ec08c8825e541e6237b61",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_bowls_three/result.json": "f6c0f45a363dd7f7fa1af4e7195d418b4ef56983e3f94cad3f9cd80d4942c4ad",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_bowls_three/run.log": "3c58cf482f4ed8d85d39940bd917a4387b03790d5cecffa108a733422281a552",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_bowls_two/result.json": "c7bca68e12c674b2765d595e61e309acd5e1d4e594e7b176ee94434529b2a340",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stack_bowls_two/run.log": "1e6d45b51e640f3cb4a3ac7803a9f0e05bf3675f8f2deb7c57c6f6e2764f9c4b",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stamp_seal/result.json": "e3ad8c83f9b44c81bd50365da1a3d8acd0a9dd192cd8ec762cecbaac3adf8ef3",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/stamp_seal/run.log": "b844358880e8dfcab50b883057dc5dabeb2fc44fc9559b9754b1034c82c6a0c2",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/turn_switch/result.json": "0d4b3d4258b988ecada80df6c1fdfc046d0cd5e8b15b95b7de03548405e4fa9d",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0/turn_switch/run.log": "2eb0c8423b8669e36f2db5660f1dd48e629df39281b51a04fc3c7f05b0fff9e4",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_report/clean/per_task.json": "7910acf9fe0ac2f14b2bbb9c158919faecd8062c1961f1619668715a33f99125",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_report/clean/summary.json": "d81035f46693fd44cee0f8ac86195f7e25e648e345c1c062d9b8a8f1906894af",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_report/comparison.json": "72a9280220e6bf150286847d7230098ae2016ae7cc73863a181897e581482c50",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_report/final_report.md": "d8c56f6f2da8d6aacdd8454ed830a02e0785c373399ceba393e5e1da448ca0a3",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_report/randomized/per_task.json": "57e01d0ad07c6334599ba62d9b0940b148aba435c9b2ddf115f117c2f9a1a404",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_report/randomized/summary.json": "90b1cc0f539348c8a4a9685d2d83682f97815673e31eb1582cc710a85a8040db",
    "results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_run.log": "a57b9f658e2e196e0b495004eddf8eb35c810e9f673dde2aa7d0ddb2432270c0",
    "results/pi05_official_orbax_baseline/protocol_v11_full_clean_10_oidn_jax0/adjust_bottle/result.json": "fa2b1466dec5f380e74c3fe44492facaa958425e0248286147f1c4b960be1807",
    "results/pi05_official_orbax_baseline/protocol_v11_full_clean_10_oidn_jax0/adjust_bottle/run.log": "4bcae0a000c5d4d51a4760209e54971da18f50856aca451bac44e7a98c19430a",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/click_bell/result.json": "a88e3221d5c47f65e85ac6934c0f2d961102c825bf53f7d98e1adbac6c8d0b1e",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/click_bell/run.log": "f253d57faddd60380fec0004050a1440d65df900ba817c4f2c0f756cd00432eb",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/lift_pot/result.json": "5b7493bd92e235eecba95f2a873c0c63e335335909006c4e8177d7bc6a12cd61",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/lift_pot/run.log": "ad7eb1307a8e41b8da7858b68b8c6956dcccfd573befd5d0ba44f868e838fc83",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/manifest.json": "781edec6036e1893313cde4bd86fbf4cfebbd7366a5c8ffc582eb4509e271e57",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/place_cans_plasticbox/result.json": "faa2dce3fa617db8075f6144dce0d540f2a187f199efff70a4363d2da3df69e6",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/place_cans_plasticbox/run.log": "603a86544099665156ddab64f3f65b4a38a358bf2bdd84c21c55fd5f0719825e",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/place_dual_shoes/result.json": "5f89bd8425a7cecc0af63e0d13ece809b34f99d0ddf92d9f75af18f632b03855",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/place_dual_shoes/run.log": "b8e422dba6093099e6f8b758611d814447c360f79bc3ac7c15fc0fee2811bb70",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/place_mouse_pad/result.json": "3d438ac68dd8e49a34824722b61ae4517898f721e93c7fa38bb69174badcb4a5",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_clean_10/place_mouse_pad/run.log": "3807298df529fa0c2a518fe4dc004e62dee1212ff9dc59741caa0e686c5d6d72",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/click_bell/result.json": "d86e070a4c48867a99a0d659af6e443467f6bce3df4a071e39c87309779efcec",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/click_bell/run.log": "ade4db6ba111c3b03096eb11b335d2e01ef8984660626d2f1cb28fa2d039bab3",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/lift_pot/result.json": "6674325770c99fb296fe59cac556e7df7bc45e287b978ebad0491e6260375bd8",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/lift_pot/run.log": "a720216aac934401a3dd3a931762faa5d04222013e66aa4a13893e3cd2f8996f",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/manifest.json": "4a31b0f56c61d17094652787c40df0a14fb3a8c82110dc54acdec5ebec1f2af2",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/place_cans_plasticbox/result.json": "067b778ee38a805b57dbb0f8df326cec5f65cb5e94161af7d2303bbaa62f4a51",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/place_cans_plasticbox/run.log": "6a9775d1f511acc00acf9cd1dec9868b9dcfcd0d092109ea9e6edb5d2f961b97",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/place_dual_shoes/result.json": "424d2e34cf320e12ba1e0476626ca9069539ef0bc1dbf0ac4b530a43c71bed49",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/place_dual_shoes/run.log": "5d78faba5256f0a6b1fbf384ab165fb7a1189d1b6d8c980b53dfff425b7b7f22",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/place_mouse_pad/result.json": "421ada4212bf11f1d84dd23e8cd143f0d3d25277be14adddad8cc16cc1596860",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_randomized_10/place_mouse_pad/run.log": "3388dbeb658dfd58ce7e98c1f20927525d423009e719c613a48d3bfa6e25b4d8",
    "results/pi05_official_orbax_baseline/shijiesky_59999_diagnostic_summary.json": "ae10fb42d6c2b567b95b0ac89fa1fb81b060894ef26303a094d5103abcd3ebf7",
    "results/pi05_official_orbax_baseline/smoke_hammer_seed100000/result.json": "87d2b00b93767c3a0e7aa1f48bdb8196d621e1a6e334a8de7fc4ddeaf82c9dc7",
    "results/pi05_public_baseline/clean/failure_episodes.jsonl": "07a0801e0f252b356fe3a2195c2ba57f38a37424e41546dcbc45f98f98bbe6c4",
    "results/pi05_public_baseline/clean/per_task.json": "b7a7c84175a87509b1f4c4c66d4cd9e120499cf9edfad319cd77f12986d3fb4a",
    "results/pi05_public_baseline/clean/success_episodes.jsonl": "7927b077fccdfb7612318fc4b80d877913d49166d791ffff12530d40a8f9431f",
    "results/pi05_public_baseline/clean/summary.json": "6d496cef0fc8d277d765860ba56571419a76ed1fd56efed4939baaefbdaee76a",
    "results/pi05_public_baseline/coverage_clean/adjust_bottle/result.json": "fda633874743a5c838440fb892ecdc9a40074602afbb739476d95fddec279ded",
    "results/pi05_public_baseline/coverage_clean/adjust_bottle/run.log": "5ec51e9f6c94b83d72810e825b3bfd0b03f0a6adc9831b27c9fcf4d56f26bc4c",
    "results/pi05_public_baseline/coverage_clean/beat_block_hammer/result.json": "fd1e23c66fea2eeb3a4b68768a6f4e6fcedc3f92e0ad10b580d2a881588633a3",
    "results/pi05_public_baseline/coverage_clean/beat_block_hammer/run.log": "5e52cb8f3fbb0fb33401447b5b6247a3e06772c3951b31f4b80d763d26d65546",
    "results/pi05_public_baseline/coverage_clean/blocks_ranking_rgb/result.json": "4d71946c90e2eab70ad0968d02bbd10a6947c693d4a4cccb4d01e67148f4c31c",
    "results/pi05_public_baseline/coverage_clean/blocks_ranking_rgb/run.log": "63f660db6de4214c46564d4b6945b83b9b36b1f2c03ce183748dc09b93b21a70",
    "results/pi05_public_baseline/coverage_clean/blocks_ranking_size/result.json": "1810485c435dfa89b7ad9d0bdd9b33b9f721161c482237ce7c281a717f1b2926",
    "results/pi05_public_baseline/coverage_clean/blocks_ranking_size/run.log": "08eacfc4db816d849e03d720aee5a7848e4f59d59f37716c42724e4b759b6b6d",
    "results/pi05_public_baseline/coverage_clean/click_alarmclock/result.json": "fbe10efd0420a16ad81d5d82f229ca6c7240ddb2d0775775d4b608551fe95029",
    "results/pi05_public_baseline/coverage_clean/click_alarmclock/run.log": "cae60bceffcebabd3d4c007a6f2ea28e8ebf78fcfad2d18e2f0ba133fd66919f",
    "results/pi05_public_baseline/coverage_clean/click_bell/result.json": "627551fb73c607b532e07e75edec520a041bc7befea009a4fc36df0b7a9e7657",
    "results/pi05_public_baseline/coverage_clean/click_bell/run.log": "970db2d269be6543780caf31b853ba4aa5df839f35193cbb54b65a990b9bbc01",
    "results/pi05_public_baseline/coverage_clean/dump_bin_bigbin/result.json": "552641ba7bbcca3c1a9a879c270098452741188056d795f7e654faf6d18e15ed",
    "results/pi05_public_baseline/coverage_clean/dump_bin_bigbin/run.log": "be190087a31fd046ea412437a0a0e8e23d2961404f0919dbe8ba17c890794c9f",
    "results/pi05_public_baseline/coverage_clean/grab_roller/result.json": "2e9c9255922fd1e1920bd81968fd759d2254b00f4d57f96976d7c13563f42f9b",
    "results/pi05_public_baseline/coverage_clean/grab_roller/run.log": "f34d87978158fef4b4a3827ba4564986be9ede9c8bad652748b77e660abb569f",
    "results/pi05_public_baseline/coverage_clean/handover_block/result.json": "a24d0fe5814101fedec8bc53600078a7366bfd212793de9050ad4e107f325a20",
    "results/pi05_public_baseline/coverage_clean/handover_block/run.log": "bcb588f5d08533ab357e5d6d987a92eb18477e2b6618ab5451daaff11eab6185",
    "results/pi05_public_baseline/coverage_clean/handover_mic/result.json": "f83ebe9db8034b341b9c956c05ca5261941dba7656c5f0f5ab5912db38bc681d",
    "results/pi05_public_baseline/coverage_clean/handover_mic/run.log": "4eba3cfefa56d15ade2620d15297bd06cd9fac88e73039b0909e657c4c7ec147",
    "results/pi05_public_baseline/coverage_clean/hanging_mug/result.json": "c3149ddeb20c096be1b41bd0f8c5e0ee769a0d978f00c6d2e32dd2b1f8967594",
    "results/pi05_public_baseline/coverage_clean/hanging_mug/run.log": "7db3a2edad5bf27b30df1abd9abd31b119f338cbfbba19b50bd991475dfb7eb5",
    "results/pi05_public_baseline/coverage_clean/lift_pot/result.json": "be7451985368ccc574c598cda297f71b55d42b0e05626aca24fc919e69f8ba45",
    "results/pi05_public_baseline/coverage_clean/lift_pot/run.log": "8d19820fcc3439adebfc5405c5d5a0036289396b98ecf180765e19c181360b9c",
    "results/pi05_public_baseline/coverage_clean/manifest.json": "915b56e17a9827882687ec6e40dc043ab51f56af4d47213af2dbc38c73c827a0",
    "results/pi05_public_baseline/coverage_clean/move_can_pot/result.json": "897374e87b635235daec7e3cbb2e063013c7c3d50b067567b4a4a3caa859c641",
    "results/pi05_public_baseline/coverage_clean/move_can_pot/run.log": "6365151f0ac1273f0daa34cbee835fb308538c25e8f87d989262d82ed75a53c7",
    "results/pi05_public_baseline/coverage_clean/move_pillbottle_pad/result.json": "236fe335a91186f544584c5ab1289bc228b21ffb6cfa83503393728b9e6e6384",
    "results/pi05_public_baseline/coverage_clean/move_pillbottle_pad/run.log": "a79f096bb6fbd5b576babc036edc3473874f9d3e83cc3f22ad1b760ab2b5dbe0",
    "results/pi05_public_baseline/coverage_clean/move_playingcard_away/result.json": "3bb6c21818da76a769c58c14a982118e9b2b77baf3461bc8417c70cefbec650e",
    "results/pi05_public_baseline/coverage_clean/move_playingcard_away/run.log": "292fcfcf0a670e7f4b020660de584eb32810b32ae8f83d9ab0bb6722fa0ce410",
    "results/pi05_public_baseline/coverage_clean/move_stapler_pad/result.json": "4a4ae5e7b2c5862fe0a8ede6b66e4341dff3a4ea32ce57dbd88e032d5a574bb9",
    "results/pi05_public_baseline/coverage_clean/move_stapler_pad/run.log": "d44ba3329a6d8a5abba7731f4e0d7589f9729fde5cfa2ef1442e80dd757c0321",
    "results/pi05_public_baseline/coverage_clean/open_laptop/result.json": "843ddfe6d7439ff50e30f02366f404d534bcbab9992cca2fd9a0b8d2ea3a0dab",
    "results/pi05_public_baseline/coverage_clean/open_laptop/run.log": "d4e6ab7162b11d224f1433d22be5316ad635b3048e67b386116bdd7202bc3eba",
    "results/pi05_public_baseline/coverage_clean/open_microwave/result.json": "b9ecd31dd5978c82bfe79356071529bf4b9926fce66ea5b4d4088c2b970ab46b",
    "results/pi05_public_baseline/coverage_clean/open_microwave/run.log": "61b2ad312a49f883abb5c994e69b6f4f2a17509294856967c07a80969889b62b",
    "results/pi05_public_baseline/coverage_clean/pick_diverse_bottles/result.json": "9c5277809971b8a46696b4661d31be583039a3d85e4e39a8a4ddc319f8d49cb8",
    "results/pi05_public_baseline/coverage_clean/pick_diverse_bottles/run.log": "b6dc9585fbc0c59f5a5b9d7510987c8a2cd48936c2567eceec1cebf482933470",
    "results/pi05_public_baseline/coverage_clean/pick_dual_bottles/result.json": "73b5e09521be8416fda2f6c849fa2124683365ae81d759c4fbd167c6c71be179",
    "results/pi05_public_baseline/coverage_clean/pick_dual_bottles/run.log": "efedd677c24e3e9b683cfcb5e579de50ace60b7f8ff514dd0970cf46c40f79be",
    "results/pi05_public_baseline/coverage_clean/place_a2b_left/run.log": "ed92f8cb5d8bd9d3412b1a2b26fa5e0e659c4de54f78ebfab90e7a828082df73",
    "results/pi05_public_baseline/coverage_clean/place_a2b_right/result.json": "54f7e2719214940e8ae600ea2d617ba94c6ed311519085bbe46ab3c18401d273",
    "results/pi05_public_baseline/coverage_clean/place_a2b_right/run.log": "70120e2f74ea347eb9aec8ac3b068dee9626c4a7c7d016b2268782f96a6c19f4",
    "results/pi05_public_baseline/coverage_clean/place_bread_basket/result.json": "67029a2d29233795ca853f793e97853f217145922af529672cdd6476ecad8417",
    "results/pi05_public_baseline/coverage_clean/place_bread_basket/run.log": "efee7bde0d810dbbf3903462bcd16e7f9412b6f87e53760fa4d2f7ec98f0060d",
    "results/pi05_public_baseline/coverage_clean/place_bread_skillet/result.json": "3a8f0f18068ca7960733bc501489b4fa8f25d5d2d8e02f0afbbc3ba6da17d5e1",
    "results/pi05_public_baseline/coverage_clean/place_bread_skillet/run.log": "2285038629077146f448a0cef3f3fa40011cd7a129385c4eb6818172f6f6cd73",
    "results/pi05_public_baseline/coverage_clean/place_burger_fries/result.json": "13c96e8556effa68169869f522006efbd8d3fc72b67c3a434dbdfdb38529e871",
    "results/pi05_public_baseline/coverage_clean/place_burger_fries/run.log": "39bbfec9563e6ecf712e6891448d8a1f0951b3b0bdbe868e889472cf61fc1fc9",
    "results/pi05_public_baseline/coverage_clean.log": "6c5a6d658bf8a1b66d34206c06f8c39d602ef7290fa1c96bdceb9d4211a41aca",
    "results/pi05_public_baseline/coverage_randomized/adjust_bottle/result.json": "63954f330d81655f53f62dccb64a5e2ef9448d9779bcfa784314e3c953a41d03",
    "results/pi05_public_baseline/coverage_randomized/adjust_bottle/run.log": "a2c703277c407badd2922fecdd886f42b16d8ebfcd58b3857c7fcb59eaa37e3f",
    "results/pi05_public_baseline/coverage_randomized/beat_block_hammer/result.json": "bd598b7a07d80c6d2c81e04bdca97d623d9ec5c171761f5b2fd365f0e16b748e",
    "results/pi05_public_baseline/coverage_randomized/beat_block_hammer/run.log": "0920fcba3cfa633579f122a5c8387c96571cc4dda5e96552e75d81fb347bf950",
    "results/pi05_public_baseline/coverage_randomized/blocks_ranking_rgb/result.json": "616e599ec8a21828fd1bc7c93ff85a0f32637c559460a736107c37dff0792eb2",
    "results/pi05_public_baseline/coverage_randomized/blocks_ranking_rgb/run.log": "a050e36622176b420da50dd3bad81a9694423bb8b7557171fb5deb36fed3cd1b",
    "results/pi05_public_baseline/coverage_randomized/blocks_ranking_size/result.json": "118b478799c23b76d54aa6937159e46415d3a4c5aac8212ff065483d7a864bbf",
    "results/pi05_public_baseline/coverage_randomized/blocks_ranking_size/run.log": "8daf1cf80bf06bf0b09492443319f5954bd18c378cea6d76ea327874b88b15fb",
    "results/pi05_public_baseline/coverage_randomized/click_alarmclock/result.json": "990fd5f3211dccbcdc78b1c56532e2becfca4afcfb99069910f4041eeac8e576",
    "results/pi05_public_baseline/coverage_randomized/click_alarmclock/run.log": "f24ba8b4753c2d2eaca1a29bd7141b53e5ac8abc2c1dc5bd5fabd1152147b977",
    "results/pi05_public_baseline/coverage_randomized/click_bell/result.json": "c4b546690e1a864f48e5f0f7e393ad08542d24b18455ba6e3f3cb6ea419472bf",
    "results/pi05_public_baseline/coverage_randomized/click_bell/run.log": "d2da29e05829f8b6287fb5c3904b72e25debf7b2d17f99131e997169d415365c",
    "results/pi05_public_baseline/coverage_randomized/dump_bin_bigbin/result.json": "78d91493c3129fc8974d0a3884b00f71f7f8b2cc10075a5944b20e002969e0d2",
    "results/pi05_public_baseline/coverage_randomized/dump_bin_bigbin/run.log": "428e58a73ef74934d9c06c9e991f52bcf8f6481fe5940352eba6c81ce6b28f6f",
    "results/pi05_public_baseline/coverage_randomized/grab_roller/result.json": "65d36da6d7a39a3d8ed556ad883283a51abe06ac911169d800ef59d7ae337ca2",
    "results/pi05_public_baseline/coverage_randomized/grab_roller/run.log": "350ca24b896301434fe056448571d421f5282a5341c94e6c537cc9723321e18f",
    "results/pi05_public_baseline/coverage_randomized/handover_block/result.json": "aabbf2ed05cdf7a24536ab692602b465b04b699315e8b65898198a9f99c73cc3",
    "results/pi05_public_baseline/coverage_randomized/handover_block/run.log": "35224799f143bb2111b295477babc50c8a49da4207d3960274faec8ef1fc7a81",
    "results/pi05_public_baseline/coverage_randomized/handover_mic/result.json": "fcbbb8747f0d1fd090e4874f1994672462fdf5039258cf66e995cc53b95975c2",
    "results/pi05_public_baseline/coverage_randomized/handover_mic/run.log": "16c380414e66e4b9092008a53e8895b3bb2f2fa84d6c64581f4115bc64c51e9f",
    "results/pi05_public_baseline/coverage_randomized/hanging_mug/run.log": "be13db7775c8152f703aea9a5ce7c0b98eea3f065562ff3b2f62b81d0e82e34f",
    "results/pi05_public_baseline/coverage_randomized/lift_pot/run.log": "8c62a5cedceca3be1efe60b0ce7f3c61541c91c735599689313b6251efc4bef1",
    "results/pi05_public_baseline/coverage_randomized/manifest.json": "27c0151d08570e5763476849bd31446be0aba00723249532bfb9c5262449e69f",
    "results/pi05_public_baseline/coverage_randomized/move_can_pot/run.log": "e6763cae3aa3b91662687dca3c083639915a0376ac151bbdbd19887faa76fc66",
    "results/pi05_public_baseline/debug/adjust_bottle_clean/run.log": "bb3a03caaf75e9e71ca67b53c167c53cefb3fc246d084c99718cd4de03ae4810",
    "results/pi05_public_baseline/debug/adjust_bottle_clean_v2/run.log": "4267fcc35110ff9ef36226ec9c66c5aeeec242fe4889ad7fc3d042025ef1a114",
    "results/pi05_public_baseline/debug/adjust_bottle_clean_v3/result.json": "0cc3a1da26d5ada76645c411482a2cad4b0135c655de13abae748b784e343f30",
    "results/pi05_public_baseline/debug/adjust_bottle_clean_v3/run.log": "920b60592cc42943f02ff4aa8684a936d719804ae93e4ed79ac3e86e3ca72ea0",
    "results/pi05_public_baseline/debug/adjust_bottle_randomized/result.json": "1a27875b37ee263446a4d33ce97471f2c75e82244ce00cc1adb659f57da06590",
    "results/pi05_public_baseline/debug/adjust_bottle_randomized/run.log": "49e2c08f91728dd4af1002c44465764e2b6aae3be90e86084d43eb3aa5e58f2e",
    "results/pi05_public_baseline/debug/assets_validation.json": "85795b3b7ac8b7e68184e7bfbe82571278daabcd6c0b88351df86a5478a7c246",
    "results/pi05_public_baseline/debug/beat_block_hammer_clean/result.json": "1e51be71a9070d39c9160eea2e644b3f9d642ab3202f409a30373d8698513be9",
    "results/pi05_public_baseline/debug/beat_block_hammer_clean/run.log": "2e3a45c6dd04afab4016f8a6ae25ce3e984470be733d66431ed5491ab4a8e317",
    "results/pi05_public_baseline/debug/beat_block_hammer_randomized/result.json": "4be50dd01f231211ba53622e97e0429bc862d9ebe62fd83974e6b61ce0bc4408",
    "results/pi05_public_baseline/debug/beat_block_hammer_randomized/run.log": "a16f23ccc907f7d1da27ccb228c04007285e4723dbde302e1bfb2a8abfce7f27",
    "results/pi05_public_baseline/debug/clean_episodes.jsonl": "8f05b7345b2761c3aaf6216006745ba311e7cd540aa3355c477c8f5bdd631be1",
    "results/pi05_public_baseline/debug/click_bell_clean/result.json": "e34500f7c502a079313d925cb6b6a14dbea14338e5a53d0d7356ac962a873a35",
    "results/pi05_public_baseline/debug/click_bell_clean/run.log": "a07f3e48691cd0e84c28433572d5ad016aa452d890059db8ef480ea0e95829a5",
    "results/pi05_public_baseline/debug/click_bell_randomized/result.json": "5519afdc0d097333ba5ebb2b5759c9494a928c090ec226fd5f6ca091bb8f817b",
    "results/pi05_public_baseline/debug/click_bell_randomized/run.log": "d19ae72d05087f0bf6e821398f45e77eaa9df8ffcc7f6cf16324821b22b57dc1",
    "results/pi05_public_baseline/debug/failure_clean_episodes.jsonl": "70fe480de76ea8e5c8052912aa603857c64049f2b5d732c36825c3955120e7e1",
    "results/pi05_public_baseline/debug/failure_randomized_episodes.jsonl": "2d0679b296e21e2313c0d0b942c58afda9233b5c5d6c3c8ca815d429a43aeaad",
    "results/pi05_public_baseline/debug/randomized_episodes.jsonl": "ba83f615072fc4068624404aef8de86da72f7c8d071cb63b719a95fd8ec7879b",
    "results/pi05_public_baseline/debug/success_clean_episodes.jsonl": "0d15b583de3bd5cf2665328232379bd304fe552a69c47af4d261aacde2deadf2",
    "results/pi05_public_baseline/debug/success_randomized_episodes.jsonl": "61a0aeb0a14742e99c231f577016628687eb574503608d7d12aa2633e992fb63",
    "results/pi05_public_baseline/debug/summary.json": "51a397bb45f0169c62c725bfc539bc2aee98346a4b9de47db8c3b6a85a1ee31b",
    "results/pi05_public_baseline/final_report.md": "657820fa7153fab2ccddb972db9e8267bf832eafc562a97f1ea4d6f8e5674a1e",
    "results/pi05_public_baseline/full_clean_10/adjust_bottle/result.json": "baad4109952c7511d2ed01dcbcbc73dc160b78955d261bb7f11a2f64fce9e386",
    "results/pi05_public_baseline/full_clean_10/adjust_bottle/run.log": "f33215b4d9ff3bcc751310aa84570a215740cca6a0234e16589caa49388ca611",
    "results/pi05_public_baseline/full_clean_10/beat_block_hammer/result.json": "7ff48484352e8ba1753707af66204d18fb3cb6cb7b77b71b6dc49458f2028730",
    "results/pi05_public_baseline/full_clean_10/beat_block_hammer/run.log": "2ca9f03a77c531b012c14d28b74c49ace735187908723c0e4b2de0ad6ebf56a0",
    "results/pi05_public_baseline/full_clean_10/blocks_ranking_rgb/result.json": "58cc0d4da7edbdd017c992be6e134ff0e6091a71abfb337875a836357e857ce5",
    "results/pi05_public_baseline/full_clean_10/blocks_ranking_rgb/run.log": "86064ca5824c814d1a48e30e4e4cb215c3536fa2da87e1a225242dfbe3d1fa75",
    "results/pi05_public_baseline/full_clean_10/blocks_ranking_size/result.json": "b40b6799339ea6aef28c45b29264e66702b15bc13a4dd566e25ddd68a4534b1c",
    "results/pi05_public_baseline/full_clean_10/blocks_ranking_size/run.log": "1d4618defeedd4eba81edcf07c8e3f6b11a75792469fbf3d0d31bcd0151b5318",
    "results/pi05_public_baseline/full_clean_10/click_alarmclock/result.json": "07aa170483b812ca531d274ed4b4ab8846ca8a39ff1458816a7f44df2e543eac",
    "results/pi05_public_baseline/full_clean_10/click_alarmclock/run.log": "a97316f2ad9f3c9d2e0920a9d5bf975973e224e039e127fd421900adb85a31d3",
    "results/pi05_public_baseline/full_clean_10/click_bell/result.json": "23379eb1fb6dcd2c2d7dca3f20211d3721f93cc3260f9d15fdb4aa8a8625a716",
    "results/pi05_public_baseline/full_clean_10/click_bell/run.log": "ad24a8d8c9d897bf7e63393bcae1c140b1d668d0e15d0e156233c6d85040ac3a",
    "results/pi05_public_baseline/full_clean_10/dump_bin_bigbin/result.json": "2d5f2518bc1c9014ad293ff24d66f83e644ee1648ffb132d2a056ff8921c6afa",
    "results/pi05_public_baseline/full_clean_10/dump_bin_bigbin/run.log": "c9230a41e26269a634ff76dcabb3465700480206d1d5d5a9926b0ceca55f22f9",
    "results/pi05_public_baseline/full_clean_10/grab_roller/result.json": "2387e1862c9167dc76e7ceaa924476b6d2f6a83e7d37965ea7c7ef6c3aaa1ebe",
    "results/pi05_public_baseline/full_clean_10/grab_roller/run.log": "d99c57d557db906bc0ca7ff8defa47878b9acfcee6c9cba1174c86e38713ad38",
    "results/pi05_public_baseline/full_clean_10/handover_block/result.json": "3160ad4428da56e4d0995b43a6205deae35aa859130e953937f637cf33aaeff8",
    "results/pi05_public_baseline/full_clean_10/handover_block/run.log": "02b6941726b171ffcdeb488fe95c405015ee62f059e16c4ebf7c2e67f5019eb8",
    "results/pi05_public_baseline/full_clean_10/handover_mic/result.json": "aa5037b410a55ce1b1e275ac2a5ddd1f54c16efc0c888f2c5c7ef425a589ebce",
    "results/pi05_public_baseline/full_clean_10/handover_mic/run.log": "dcbb244a02db2a6839ada9a934cca618498fedf64f96ac0370c58895a6dc9de8",
    "results/pi05_public_baseline/full_clean_10/hanging_mug/result.json": "6e810351019f85b12d1688358f0d38d8a369c99711628b9ba5159976febc2748",
    "results/pi05_public_baseline/full_clean_10/hanging_mug/run.log": "7749334985a4310a05a9a58e18d106abcf424de5512f3b5be555a3091e177495",
    "results/pi05_public_baseline/full_clean_10/lift_pot/result.json": "d83e1dba691e26d9d80a813af2c562bad567bc22332f9ebbb760bc97f01662fb",
    "results/pi05_public_baseline/full_clean_10/lift_pot/run.log": "70803e8ebc127055ea452dd56a00428ba73cce72b7eb0afd5544e8b30e6370ab",
    "results/pi05_public_baseline/full_clean_10/manifest.json": "190d3663810f35313949d485229829e8beb7a0182f926157c9ced95d652b79e2",
    "results/pi05_public_baseline/full_clean_10/move_can_pot/result.json": "1991209f40f3fc3428453acc687bca768bdd11d500cd0e969f6f0bde8046473c",
    "results/pi05_public_baseline/full_clean_10/move_can_pot/run.log": "85a001f959e45aba2f7d287aea91ccd26209d9f3d0027171af4400a4d9b46b2b",
    "results/pi05_public_baseline/full_clean_10/move_pillbottle_pad/result.json": "5cef0c1489670258abc92938c00e17184bc8859931ff48f1a2b95f8cec9a761b",
    "results/pi05_public_baseline/full_clean_10/move_pillbottle_pad/run.log": "911a1a9253de84143efb8ca79dfafe90252ce8b2000f5ac000c9b835965ceb32",
    "results/pi05_public_baseline/full_clean_10/move_playingcard_away/result.json": "e690735a34bced1c1021dc7a1e8f84950009b24e5c56e6fc4a9823968ea59dc9",
    "results/pi05_public_baseline/full_clean_10/move_playingcard_away/run.log": "caa0022c78fc137fed2bd9f45c1b64169a63eafffeb8763704628fd7a1f1f7e7",
    "results/pi05_public_baseline/full_clean_10/move_stapler_pad/result.json": "30dcc1fa38bf591ad939bc92e6b1b241f04b2cb17b988418f8503774c168cf05",
    "results/pi05_public_baseline/full_clean_10/move_stapler_pad/run.log": "e04092f0c48cac4bd2abea8228a6ff24f3c55cfde4dcf34364b37c6fea916fe2",
    "results/pi05_public_baseline/full_clean_10/open_laptop/result.json": "05298acc0b28b9bb2c1036d454e77db3de92ee8f820316add3a280544cee193a",
    "results/pi05_public_baseline/full_clean_10/open_laptop/run.log": "77be506944d9a4f9927f8334ced30e14b3d32832f991746cc331fd4983c5eaf2",
    "results/pi05_public_baseline/full_clean_10/open_microwave/result.json": "fe2cce8ea8628060457c5129d5215cf91789429e24e816b7a8b35fdf8608c244",
    "results/pi05_public_baseline/full_clean_10/open_microwave/run.log": "60835ffeb868a31310c4d03faab6e092253c2e85579fdd30d38ed5887d4439be",
    "results/pi05_public_baseline/full_clean_10/pick_diverse_bottles/result.json": "8ef74a03fa260c5904d4f75f490df46c4db6a436bb0f448727883b4f1d2e94f0",
    "results/pi05_public_baseline/full_clean_10/pick_diverse_bottles/run.log": "f14bd67ac9a02002d312945fae90dcb78e5fe03ef0e1836efac6c03b365997ee",
    "results/pi05_public_baseline/full_clean_10/pick_dual_bottles/result.json": "967787de0ceff02e3b2305707a8d93dad05f0750a11f627786c1c94b9efb4d2d",
    "results/pi05_public_baseline/full_clean_10/pick_dual_bottles/run.log": "a3baea2962214cca2ff6e105381ec3cc1c142ff49542449138b4ccd312500797",
    "results/pi05_public_baseline/full_clean_10/place_a2b_left/result.json": "1a5c85cc6cbda7510d68701633dfab872dd79e6d936d5ea329fe84e338148092",
    "results/pi05_public_baseline/full_clean_10/place_a2b_left/run.log": "c8374fa0b124ffd7c323a788e5eb686b7f4c7ff25ef40ee57fb1ee659a279a97",
    "results/pi05_public_baseline/full_clean_10/place_a2b_right/result.json": "6901aa832ed1b49d3e97d1756bf372e663d569a96e70c9c82798340168bc7a52",
    "results/pi05_public_baseline/full_clean_10/place_a2b_right/run.log": "99b9651e57090f1c1177e0e7cfe21f7cef27a5a361c5b45f66de5f2456bf9c22",
    "results/pi05_public_baseline/full_clean_10/place_bread_basket/result.json": "914786d4cf7f99373e83864fc04d561d4ad2faa77f0d997b0075bfd8703fdae7",
    "results/pi05_public_baseline/full_clean_10/place_bread_basket/run.log": "8c1d31dd19a9d69c9351b9375621d80a35a7192431f29df1e426f0c80b2fa1a7",
    "results/pi05_public_baseline/full_clean_10/place_bread_skillet/result.json": "c9443df33c9be6ea75ee2ed3009c0e2cc5019fde5789763bc5e834fdbe7301f8",
    "results/pi05_public_baseline/full_clean_10/place_bread_skillet/run.log": "38c671e8b50f54b99cce4f9bd8b9e230f6e7a942b0cf13c0b373b6cd09995543",
    "results/pi05_public_baseline/full_clean_10/place_burger_fries/result.json": "71af51ca693259ffcd383c6c057cfdb4bb9dc56d1dde3fdbf3220825e129d969",
    "results/pi05_public_baseline/full_clean_10/place_burger_fries/run.log": "fc3e64fe004a850985a00d399e19ceb0115c1f6b666b229b3b22149f6d170df4",
    "results/pi05_public_baseline/full_clean_10/place_can_basket/result.json": "7da2dc779f1b264e23232ca32b3047d882a491183759eb8a0fd4df4d11aaedbb",
    "results/pi05_public_baseline/full_clean_10/place_can_basket/run.log": "04bcb06be312701f6720dd7c8c81e7dcf6a4e99e7364440a574a5ec9770d6191",
    "results/pi05_public_baseline/full_clean_10/place_cans_plasticbox/result.json": "bde9a778c0e54e75fa3296399edd6f804e2156648554111d052aa634fb9fbb53",
    "results/pi05_public_baseline/full_clean_10/place_cans_plasticbox/run.log": "bd96c2eb26dbcae91568dbdcc5503470b3951b75d8a7b300ae0e541551749dca",
    "results/pi05_public_baseline/full_clean_10/place_container_plate/result.json": "248cd0b69a6f92d6d3e9c944ce364be3f9b35f1289a44d00c6290c4a0deff1cc",
    "results/pi05_public_baseline/full_clean_10/place_container_plate/run.log": "f4e54e25bc723b6673f628e22dbd177db9e9354f75f241b393de8978d0b18bb1",
    "results/pi05_public_baseline/full_clean_10/place_dual_shoes/result.json": "4cf3fbf5ecbc92d5a6bea0b06a6282091bd5bbd8867baedf01a950f5f32c05db",
    "results/pi05_public_baseline/full_clean_10/place_dual_shoes/run.log": "b53b329f273ceaecd66a52e2e185e7d48ee2e7527f1601d53798097765e1b574",
    "results/pi05_public_baseline/full_clean_10/place_empty_cup/result.json": "c601bd256a6e697d822f963a2c594956d81ebf5149d55c62b552f5ac5af04f7c",
    "results/pi05_public_baseline/full_clean_10/place_empty_cup/run.log": "f3cc5ebafdc94d8cc7eee27e9b07a66b666dc454bfa0b5dd61e18ff4e5c7d889",
    "results/pi05_public_baseline/full_clean_10/place_fan/result.json": "264e71ceda2781a2e2ffe09cdf820389a4923ea078ac1298041a87a81d94d5a8",
    "results/pi05_public_baseline/full_clean_10/place_fan/run.log": "5f74ee5b274c4e24a67595438238719bdb2557514df3f37f78453e882ad9e054",
    "results/pi05_public_baseline/full_clean_10/place_mouse_pad/result.json": "d3cf62a65a11abba44af5f3de191eeeacf6654d4a89c3eafbdc950a17e52a859",
    "results/pi05_public_baseline/full_clean_10/place_mouse_pad/run.log": "34b5033d8a4555c47fe6ef9d249f236e16a2dd9d623dd2cbf8c2b6867d8aea61",
    "results/pi05_public_baseline/full_clean_10/place_object_basket/result.json": "58137e163922fa674f5b681d5cdad9b17e84a7fa26d65133666c4d30227ac08c",
    "results/pi05_public_baseline/full_clean_10/place_object_basket/run.log": "986de3457b4d4e6e70f46456e813ed7711d48d644024cf872985b6ade141362b",
    "results/pi05_public_baseline/full_clean_10/place_object_scale/result.json": "0c789e5adfb7db998581c30a95e2dea1c8d4d3e03b1dc08fb58ab0ebded7a518",
    "results/pi05_public_baseline/full_clean_10/place_object_scale/run.log": "556fe60d110afe9459fe6caf0d533256b7066aef35c189cb255e6498547f9036",
    "results/pi05_public_baseline/full_clean_10/place_object_stand/result.json": "f9401422f32248965624ac1ba264587bbe40cb4ac462ff7636721b202bb29e25",
    "results/pi05_public_baseline/full_clean_10/place_object_stand/run.log": "4b2e4bc1eb65d8db574d2e233f71e07a268f07f99899cf6c1f4fd79eaa136e21",
    "results/pi05_public_baseline/full_clean_10/place_phone_stand/result.json": "7f96b57fe262b27a8df6e25e65c725dfa5a9497f30567dd676e93443610b6a06",
    "results/pi05_public_baseline/full_clean_10/place_phone_stand/run.log": "f137eecc6bd49bd8474a61f99f9ead065aaa24f521573e40b7fc8400a01842c3",
    "results/pi05_public_baseline/full_clean_10/place_shoe/result.json": "b6ae82c1ffe75f9c9dba36c8da2ac26f44ab1f463d2e1c98ef045864a5d687fd",
    "results/pi05_public_baseline/full_clean_10/place_shoe/run.log": "e4641c4cf6f749fcc75004c295a35a19a385b2f27d5ccb40fe9a6475e3549ec3",
    "results/pi05_public_baseline/full_clean_10/press_stapler/result.json": "f10c18849a7bbdaa4fa23d5919d51cd3f42e0c9e75024fd5684d219e23344ffc",
    "results/pi05_public_baseline/full_clean_10/press_stapler/run.log": "61ff9aa2477802afc15ec3a5677b2205eb11d5b02a3f8ae9891f7df8c70f572d",
    "results/pi05_public_baseline/full_clean_10/put_bottles_dustbin/result.json": "3c983258de3db297726422e4c5035840c733727745cd9579ae81a9d5e74e8ce2",
    "results/pi05_public_baseline/full_clean_10/put_bottles_dustbin/run.log": "d16d102279cfbfb84e865f239ec37ae72df78634ebdf811fafff4fc43ce3428d",
    "results/pi05_public_baseline/full_clean_10/put_object_cabinet/result.json": "e2333a34dbc7922ba405b6a8db3f7972b24be959759c33325fcfad47b2d95bb0",
    "results/pi05_public_baseline/full_clean_10/put_object_cabinet/run.log": "6917b1bdab8b12f55a7fcb3d97701b30769a85a6b40c1ce33163a1105925529b",
    "results/pi05_public_baseline/full_clean_10/rotate_qrcode/result.json": "5d9618ff75c84bd7e803b6733f9646ff66315ffd67477c083d8ac65a1dc90e1a",
    "results/pi05_public_baseline/full_clean_10/rotate_qrcode/run.log": "00f256a212f83fad8061cf94880dc86a876786f59ba7620488a3463af40469b8",
    "results/pi05_public_baseline/full_clean_10/scan_object/result.json": "23095deaac576d7039dbb4c7ce83e69c0fafba57d6a2d748ec0aeb31debeb04c",
    "results/pi05_public_baseline/full_clean_10/scan_object/run.log": "6210044eeeac0e0f42689b1de7d4ef98d7e9fbd25fe5dd898376b0a48e54cf36",
    "results/pi05_public_baseline/full_clean_10/shake_bottle/result.json": "a3eae979e1d75e2909434818268dfb790020d0fbc15aa52a3867249931ad566c",
    "results/pi05_public_baseline/full_clean_10/shake_bottle/run.log": "2cf243393a4fa96d1cecaf4e875cf6934c309b69e3dadfe8d0992f497fd4efa5",
    "results/pi05_public_baseline/full_clean_10/shake_bottle_horizontally/result.json": "fa230d9dd046a02ece19406166fd47fdbe3fadc29cd39e959ecea4adb5276364",
    "results/pi05_public_baseline/full_clean_10/shake_bottle_horizontally/run.log": "fce24d4938bda74c9d001fbac2e571979ce06a68911a0c778025155daaa99488",
    "results/pi05_public_baseline/full_clean_10/stack_blocks_three/result.json": "cf54e0f35163573cb32aed5ec7a6c8c6dcdcc7d91cefce12275de35cf140776e",
    "results/pi05_public_baseline/full_clean_10/stack_blocks_three/run.log": "994e6d1a27005cebe3cfb5de151ecea0f60270ec36b23c519f91a4635153b270",
    "results/pi05_public_baseline/full_clean_10/stack_blocks_two/result.json": "5a38c5f77959a090c79e0122913255ca256b320aa1ea695db8aa06c261392e0a",
    "results/pi05_public_baseline/full_clean_10/stack_blocks_two/run.log": "2a0b1836500545e3d37158cbd2a46e8f9227c4e437d353473d2deee3520436c0",
    "results/pi05_public_baseline/full_clean_10/stack_bowls_three/result.json": "93912be162d07fa7193e0d14019d0d3f4a31554594fd885b6b297982e539cb85",
    "results/pi05_public_baseline/full_clean_10/stack_bowls_three/run.log": "9a87e1cc832ce5d0b871ecdae8fc2f4aece68e2811c19445d6cf29768452ea5e",
    "results/pi05_public_baseline/full_clean_10/stack_bowls_two/result.json": "61f2bda27227881921e2a90e51c81e8dc5cfc1a8208393c173aca0430a336f34",
    "results/pi05_public_baseline/full_clean_10/stack_bowls_two/run.log": "f5301823ccd4be93dd76f2abb6883890fa092651fc15215442de710b34a2391a",
    "results/pi05_public_baseline/full_clean_10/stamp_seal/result.json": "9fe24e8fbe64919ef8bff4b42953f45d2ad9bfee1c9cfe4c7995ca35d4b0a93e",
    "results/pi05_public_baseline/full_clean_10/stamp_seal/run.log": "af83fb3779af5eda0d924405fe190c47f131485a66d4bca77ba69248d2883c98",
    "results/pi05_public_baseline/full_clean_10/turn_switch/result.json": "9c25a0889336177502cd21b4c1a6f69a7746341e805d0f465792e57708d874eb",
    "results/pi05_public_baseline/full_clean_10/turn_switch/run.log": "ce9b81096868f6130d22e5ac43fa6197bde241bbae331f02b3d1afcab3a03a4d",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/adjust_bottle/result.json": "f52f29a339d3e6bd8b6d83ced1d9f8504a1cf30777cbea4cb6e59cd39819f2c2",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/adjust_bottle/run.log": "9e7ec57bb9f2a7987c0aad022b82f140847b56293f67b3bd72bbaf8c23431592",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/beat_block_hammer/result.json": "0a8cfdc64b000d550a10a2a52e0bd7f180715978be9f06fbc240ee1f13075cff",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/beat_block_hammer/run.log": "6c650cac1f5b9a07e6fb911368b8e8903e3713d00bb60b942b2bfd4dead89718",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/blocks_ranking_rgb/result.json": "eddeb99691521a5001845e0aebf4a8e7bd5ce0447561238d34bd692b145b1592",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/blocks_ranking_rgb/run.log": "8a73997ef9a5def99671f1dce8c85b03ef3370bff03a23bd4b26965d7ed13a90",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/blocks_ranking_size/result.json": "bab6b393ce6a10c886f191937eaebf255bf8f494dcad9f4f5dac42f5c081b89b",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/blocks_ranking_size/run.log": "82bbc8b50906bc18a18334afe902e4b378aebf42c9ac72a3e69b140112f7af1f",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/click_alarmclock/result.json": "fa63187c650e067cb284784337dde55203b8c534d2141fb3a913bb5df556a1e3",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/click_alarmclock/run.log": "c5a3d897ff2811f0e4eeb4583a2d83002b8e6b6f9d9bd5493e1add9db68d31be",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/click_bell/result.json": "5675cd11f8b4da308ba0d792e11b8e3cf203df29ae5832aaa0aae1b5159d67dd",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/click_bell/run.log": "0a5cba44e9cfda0ddb29c33b77d8f9b4ebacbb27915d6a160703f55a7211e52d",
    "results/pi05_public_baseline/full_clean_legacy_unprefixed_partial_20261005/manifest.json": "8654aacaf0e9c77323911f063f3e131d42e4ce82df3dec3e75edc0cace5ad25d",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/adjust_bottle/result.json": "65246afdd2c776bc3232825daa8dd96a97f5cacd273633bd4d648ae3fdba499c",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/adjust_bottle/run.log": "b7f1bee2df1b311f444d790028a27169791a68bd8b7c832d38768469a70dc7d6",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/beat_block_hammer/result.json": "633ecd7b51253a65d0594f3502212e06fb31a8ed54a9a7dfbb31d8e1d8b013d7",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/beat_block_hammer/run.log": "78d26fdbb9c76ead4bdb88177a00f1d37c84453eb3770fd69189aed7c9c8767a",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/blocks_ranking_rgb/result.json": "267f0886fd7e7c547ca1920dfe5c2be0471e2c17a3689213c9d5651191f53564",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/blocks_ranking_rgb/run.log": "4c9239eb6ecbe37a8d529c2e1493c2a59026bc15c09c14cc15dd1aaf4942f408",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/blocks_ranking_size/result.json": "008af04000568553eb44d2cd24f09d158e84cfaea44eef0e9155f3e224e09c46",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/blocks_ranking_size/run.log": "f796e9c0533cc4a5feeeafc29b2c6255931c399ebfdff343a85717bad6f4819d",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/click_alarmclock/result.json": "969b5fed774fccd3e715a48934bdc035a990575dbb21803bf1c8b51bf21defa5",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/click_alarmclock/run.log": "29b47113f86ad5fb98eb0510167bd5e31152db6fd7204b44953b8689dfdbd5b4",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/click_bell/result.json": "2540677d374111bb93e8278bf2d3ae44405ad9cbc3dff1f9eac0122cc97b06f2",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/click_bell/run.log": "e5be09c12ec2fe152ae90febde2905892528b64014108f07ea1d1308dfa6f764",
    "results/pi05_public_baseline/full_clean_official_partial_20261005/manifest.json": "ba31ab10c75f29ece8e9590466da64a43dacde8041320d73d46db022e82dae61",
    "results/pi05_public_baseline/full_randomized_10/adjust_bottle/result.json": "d64939bc29bfde8459af4b1151d7a6f2e43739f71a229e1228483674510e02b9",
    "results/pi05_public_baseline/full_randomized_10/adjust_bottle/run.log": "a831e089ca4a8ff52c4e7a78b01f578818d290b645b1dc2b5f1458c7cdef0ced",
    "results/pi05_public_baseline/full_randomized_10/beat_block_hammer/result.json": "19d6c3ab4ebd26edc211860313648fc01bc33b7d7837ad2311d191f167f022ba",
    "results/pi05_public_baseline/full_randomized_10/beat_block_hammer/run.log": "8d57921451936e463c4795846d8bc000a05914def4a77394b3cd2cab9c4ce614",
    "results/pi05_public_baseline/full_randomized_10/blocks_ranking_rgb/result.json": "c5975a2ae20483f14ebc420e1b9f0eb8546b2d52021316f12076666bcf20683a",
    "results/pi05_public_baseline/full_randomized_10/blocks_ranking_rgb/run.log": "07b1e7bf3adde8a2c4cfe53724706934ac1292c1b9be93868bb67c09df12c5a6",
    "results/pi05_public_baseline/full_randomized_10/blocks_ranking_size/result.json": "cac7a6d6aa6dca50df84021891e2641cf065d8b5bf5765754033e38b5bac68d1",
    "results/pi05_public_baseline/full_randomized_10/blocks_ranking_size/run.log": "728973efec5a5e6e8d14b6e3a44e291ccd91826b3ea3f8b09fdf72b4ac32c181",
    "results/pi05_public_baseline/full_randomized_10/click_alarmclock/result.json": "350810f469a9a179c29c487bce90d33657d1b0be3da48f2d61b0fbe1f0ceb5b1",
    "results/pi05_public_baseline/full_randomized_10/click_alarmclock/run.log": "0949ee7a7c89d7b13120d9d7eeee6c7768944464b41c46eecdb48c466a747cc7",
    "results/pi05_public_baseline/full_randomized_10/click_bell/result.json": "01c1cd5d1859dca7febbd6c82ff7c6e47b8a2b6143f2f023c6bf528be7ecce20",
    "results/pi05_public_baseline/full_randomized_10/click_bell/run.log": "47cb803dab58d25ec508974cfb6defe09f4bbdab5d3cd6edcf9ce1e9dc52f116",
    "results/pi05_public_baseline/full_randomized_10/dump_bin_bigbin/result.json": "a639eb42db2304561e1fe410a87144f313fba4fb83cd12c95e9657896b129ce1",
    "results/pi05_public_baseline/full_randomized_10/dump_bin_bigbin/run.log": "8702c3791f3e167f1d15bb0bcf24e1ade9fd10198b8cade2c30318a1beea9a4f",
    "results/pi05_public_baseline/full_randomized_10/grab_roller/result.json": "5d8a8f6059939ba0c1d182aa03c629b918734451fca5a0a5c41dc3240d174eed",
    "results/pi05_public_baseline/full_randomized_10/grab_roller/run.log": "700e899982d9b3b0bab7ec8c5f6f0798d4210b0025a237d09aca20c30e52a411",
    "results/pi05_public_baseline/full_randomized_10/handover_block/result.json": "16acd2322b1464b1e2b311c4ee67b1803440e20443ff7c4dc16ffc0afbbc1557",
    "results/pi05_public_baseline/full_randomized_10/handover_block/run.log": "11661bb955f1dba6dede08ffa4bd377f6c810c58059cd91a0024ebca94e1eefe",
    "results/pi05_public_baseline/full_randomized_10/handover_mic/result.json": "1cec7cb2289365f4e522d6686548bd4456f278dffca46c8673d779ffe2673cff",
    "results/pi05_public_baseline/full_randomized_10/handover_mic/run.log": "91fb886bdce2796631eaab223f53c9432d6bb0c9802fc29073df48c0522091d2",
    "results/pi05_public_baseline/full_randomized_10/hanging_mug/result.json": "fca0e15cfec1e06f9b9b25ff26414e158ff28e0b90520067fb6fbca89e1399f1",
    "results/pi05_public_baseline/full_randomized_10/hanging_mug/run.log": "9b357a4827dff47a33ed370532e3c4cf1578c802f9b5a5bf5e4c238eb4efd09c",
    "results/pi05_public_baseline/full_randomized_10/lift_pot/result.json": "dbc316b81841f83fc5fcdb6b3da091499d32866d31cb99a01dbaff4fcb3448bb",
    "results/pi05_public_baseline/full_randomized_10/lift_pot/run.log": "9b899f21e0af5457320cefa7ddc459df139fa3a9dc3385af3f1cec3d4c2ff0eb",
    "results/pi05_public_baseline/full_randomized_10/manifest.json": "03985e9c8aa967d6b2f843d87d459d60f07943bd8c87366edd757e3b01b4e93c",
    "results/pi05_public_baseline/full_randomized_10/move_can_pot/result.json": "27fc13351148e29e145c221cd310ede25096e7e111a9ae9195497ab245708c81",
    "results/pi05_public_baseline/full_randomized_10/move_can_pot/run.log": "d2e55be12260963abd5744135b6e7af51c30ac358d5f44fd39d2c2cd7d84c7c2",
    "results/pi05_public_baseline/full_randomized_10/move_pillbottle_pad/result.json": "ea42956a1fa9387408dd689a425c08c65f270f6c65be6cb2855687a351953c6a",
    "results/pi05_public_baseline/full_randomized_10/move_pillbottle_pad/run.log": "35395ccfb2029775775ec1af34fcbd07c3668a90ea65465f261ffd5d6cea0bb6",
    "results/pi05_public_baseline/full_randomized_10/move_playingcard_away/result.json": "c35bfb675defc2057a7d903901564b485ae087cca0bc38af16fa71559fb0b45c",
    "results/pi05_public_baseline/full_randomized_10/move_playingcard_away/run.log": "f923d884b8217f3c9cddec2728113d5ae24634b8993d58121da07ea32d4a810f",
    "results/pi05_public_baseline/full_randomized_10/move_stapler_pad/result.json": "2d0eb293e81f8ba488356d4f3f325ec60645cedacd106766752cf7a47495ace1",
    "results/pi05_public_baseline/full_randomized_10/move_stapler_pad/run.log": "037b79f0da8407eeeee2deb0b94878d1e95b0ed92b4e4c953ae5d7ec591dfc17",
    "results/pi05_public_baseline/full_randomized_10/open_laptop/result.json": "c70d350a27aa213de9e2f27bacf983364c7eb28a1ef9d52bdcc5b227332e75c5",
    "results/pi05_public_baseline/full_randomized_10/open_laptop/run.log": "a99d3050883ef7ec656c8cb2f0c09cab1f3fd52966fc1da207dd4aa88418fbaf",
    "results/pi05_public_baseline/full_randomized_10/open_microwave/result.json": "fc5cde5bed341fbd5be868dc078063269d568bc3a9079855931d608a25cdfa70",
    "results/pi05_public_baseline/full_randomized_10/open_microwave/run.log": "61b9e49d01a332fcecb350848ec73cac9713eb138b68798df05d41b60e53f7ce",
    "results/pi05_public_baseline/full_randomized_10/pick_diverse_bottles/result.json": "ec745813e737fbd847f98a31aba7bd3bc5de8b264b5d3349d1def564e8440065",
    "results/pi05_public_baseline/full_randomized_10/pick_diverse_bottles/run.log": "6412aee711a75e83a4d04017587bc2aa0319e9fc33492467294b8554d0fd486b",
    "results/pi05_public_baseline/full_randomized_10/pick_dual_bottles/result.json": "0288eb365074099df998363e04e453f0df8b6d002b24cb0c82088b6c397d6064",
    "results/pi05_public_baseline/full_randomized_10/pick_dual_bottles/run.log": "ea8cd5acaaa431e4038f36a2b9c7fcab42509c45d8c9e5c1edd57393e66a3058",
    "results/pi05_public_baseline/full_randomized_10/place_a2b_left/result.json": "371b96332632df1099cd01b6f04edd819cbd5339ae43c67a5e150b7efc433cab",
    "results/pi05_public_baseline/full_randomized_10/place_a2b_left/run.log": "f079924bdc10a43e604c2eb2b5c7642d7e5eec2e9c9dee087e69adaa2136bb34",
    "results/pi05_public_baseline/full_randomized_10/place_a2b_right/result.json": "2d8cc10fc3272937b269842eda58ddd4d3512050345f1a2b33ce9dd733063a91",
    "results/pi05_public_baseline/full_randomized_10/place_a2b_right/run.log": "05b9c762c00cec123818a191a34c988ad82d72f9658e1abacf10592756e1895b",
    "results/pi05_public_baseline/full_randomized_10/place_bread_basket/result.json": "1fcc05d3cabfae6a014d7859a5250c453859f032393f4c0aff91d985ff2a52f5",
    "results/pi05_public_baseline/full_randomized_10/place_bread_basket/run.log": "4806767097c11f5d2179c50632284bbc913c6e66ede20eed09e6a49bf11990bd",
    "results/pi05_public_baseline/full_randomized_10/place_bread_skillet/result.json": "2337a025840e71930230101dcc0cb58ee9c7f75a75cb67e476115e798f7ecf68",
    "results/pi05_public_baseline/full_randomized_10/place_bread_skillet/run.log": "9a1be294d4c515c3329eb7263d4898bc8ccac0a7b5414896d2595672566e1012",
    "results/pi05_public_baseline/full_randomized_10/place_burger_fries/result.json": "8f7e5a3e310c1981547d9e929808cda612086746d7ab71b0b39841323818928a",
    "results/pi05_public_baseline/full_randomized_10/place_burger_fries/run.log": "9fd4ce94aa28f04855dc177477fedb36bafea1b98dad3d35bb33b067d127f3f4",
    "results/pi05_public_baseline/full_randomized_10/place_can_basket/result.json": "f04b0f7b61de9e693fbb92bbabf09f1ac8f4b8f0b480b74a72e0d1447af49ddc",
    "results/pi05_public_baseline/full_randomized_10/place_can_basket/run.log": "3ca193039bbea60d0436f8e9c5351ae956063ac294a52e2ea4d6a2c4e0d6c43b",
    "results/pi05_public_baseline/full_randomized_10/place_cans_plasticbox/result.json": "27976fa376d5aa5c1ec4037ddf92f9640b89974de725c603d5a6a3ba9ec45b1d",
    "results/pi05_public_baseline/full_randomized_10/place_cans_plasticbox/run.log": "d7312cc0360e00672a43f753da7036edd01184f5fddd3baa7b4925ab5daccc81",
    "results/pi05_public_baseline/full_randomized_10/place_container_plate/result.json": "80542d7c3a4a86b903617cc0a6983ade009b3190ceea4f2d021f2d90ca5c1ec1",
    "results/pi05_public_baseline/full_randomized_10/place_container_plate/run.log": "e48ded715504bab4515df587de106a53bc783c9d042227f108050cb46ebd20dc",
    "results/pi05_public_baseline/full_randomized_10/place_dual_shoes/result.json": "d4a65ffd7b5a2aedd870b244f65d2952aa6e2620191a21b1b09f34e42518faee",
    "results/pi05_public_baseline/full_randomized_10/place_dual_shoes/run.log": "a6abd7f1220a342ca68b6be2173e287dd32ad91e89c9880b18169350605af24f",
    "results/pi05_public_baseline/full_randomized_10/place_empty_cup/result.json": "72fce581b31cbf89ad9c9662809907f06b8d9c34143319cddd5a26c59de11187",
    "results/pi05_public_baseline/full_randomized_10/place_empty_cup/run.log": "5bb7e65800daf5dfac530c9c3dc6d81ae2b5c2bfd881ee3352a49faef6efed29",
    "results/pi05_public_baseline/full_randomized_10/place_fan/result.json": "ea2e1db5709a4794cff597a50c6a11ce4da0f9d8f0f462e8b31c718b3a07ca3b",
    "results/pi05_public_baseline/full_randomized_10/place_fan/run.log": "1ccc1c7eca9a8a33937d517200685e96266e4e00f525be11acb74c60ab01c7d7",
    "results/pi05_public_baseline/full_randomized_10/place_mouse_pad/result.json": "7953367337b800706a1007f861f99676bb4a75a8d894ba604031f5f3609ce7df",
    "results/pi05_public_baseline/full_randomized_10/place_mouse_pad/run.log": "ad1b882c082ced73ea3d835e0094e5a50b23785be306f18203cc16fcd3f5407d",
    "results/pi05_public_baseline/full_randomized_10/place_object_basket/result.json": "f671da23eb45169942accf9895537ad5e6ac5ed6b3342fbfd6d235eddbb42add",
    "results/pi05_public_baseline/full_randomized_10/place_object_basket/run.log": "152c4807f6212e238a5ae93a72e15b2d6ba88e4b03e81eea1f95dd5eb667f8f7",
    "results/pi05_public_baseline/full_randomized_10/place_object_scale/result.json": "fdaf83abff72d22f8247a261d4bd71354804d33907842fe34d2cb2bb200e5757",
    "results/pi05_public_baseline/full_randomized_10/place_object_scale/run.log": "457bc04bd5047e0eaa9bf317514495127ce001d3c0065342e46e048fc3b52dd6",
    "results/pi05_public_baseline/full_randomized_10/place_object_stand/result.json": "2e9f5e1ef379ad1b3a3e1c1df9a49d516442796e4ff57dc715bdeae4fea06b27",
    "results/pi05_public_baseline/full_randomized_10/place_object_stand/run.log": "52432a95b35efe81a793de89622bc37d45fa235b1d9782fb354b64560c52ab0c",
    "results/pi05_public_baseline/full_randomized_10/place_phone_stand/result.json": "c698ff2dfc03c08f6ac93d4a2fc043e5ddffd06b81010dc686f92dbb06e7a723",
    "results/pi05_public_baseline/full_randomized_10/place_phone_stand/run.log": "437b3fa31c9587d4deac20529925c4e562c129b43833dd21969f272bdced48f2",
    "results/pi05_public_baseline/full_randomized_10/place_shoe/result.json": "aa11f2aed59fc2a383d56ca3b8b6a059ae0cbe332b56d20194d92e054728aa97",
    "results/pi05_public_baseline/full_randomized_10/place_shoe/run.log": "5f928d045767789b798cf004e664f59de51021399dea89cea40d08ff0601955c",
    "results/pi05_public_baseline/full_randomized_10/press_stapler/result.json": "359ad01cf386b3c9730580bf25fdde354ac136d78bb53c1fcb83222d938c195b",
    "results/pi05_public_baseline/full_randomized_10/press_stapler/run.log": "456820654eb1a0b662eda05db014fb693364daa256aa6e1fc4ccf16ae17de49f",
    "results/pi05_public_baseline/full_randomized_10/put_bottles_dustbin/result.json": "2593d7c880d259bda78cce922350b195d7d7812a18251b1389505fb177d86aad",
    "results/pi05_public_baseline/full_randomized_10/put_bottles_dustbin/run.log": "0270eb48c68c96f0f60979f78f882e37bcad3aed97021172209667ebbded0270",
    "results/pi05_public_baseline/full_randomized_10/put_object_cabinet/result.json": "5291fc61509bf1d08a18c537caa13d89b9b24dc29509f06214031d594ba5c611",
    "results/pi05_public_baseline/full_randomized_10/put_object_cabinet/run.log": "fe03ae79799466bd4a17f3e7f93f230d5aec7edeb22a98cf88402306bb2171f2",
    "results/pi05_public_baseline/full_randomized_10/rotate_qrcode/result.json": "a5f8839c0820751a10850ab6afffaa01f3c0bd61b2608d235bcb31f9c472ffc1",
    "results/pi05_public_baseline/full_randomized_10/rotate_qrcode/run.log": "1dad9ee8d0607ef76a9f16a12e2f6e220379da4c23e5962c48260fce3a0e52a4",
    "results/pi05_public_baseline/full_randomized_10/scan_object/result.json": "cd52c4669269769f98d95477f7494fb4d91cc6cbad9f33f6ee55606e07fb2114",
    "results/pi05_public_baseline/full_randomized_10/scan_object/run.log": "0eb276e23fcbbcf3d6021f3880558ed51f9c698b6e635ba5b1f45241539a4cc4",
    "results/pi05_public_baseline/full_randomized_10/shake_bottle/result.json": "1b56af066702fd1eddc0fee45e8fd0872163e23f75e1239556bf55add2848ff8",
    "results/pi05_public_baseline/full_randomized_10/shake_bottle/run.log": "71181942b50c2a4fc9e072bf78415ca04969b4ac38e3c05a2a808869400ad013",
    "results/pi05_public_baseline/full_randomized_10/shake_bottle_horizontally/result.json": "dec37d44316cf90ba7c24816a7b551997d423a22ada654c24af6ce0190cac370",
    "results/pi05_public_baseline/full_randomized_10/shake_bottle_horizontally/run.log": "c0af5442ac40f93f3dab580b6d473fb80da0a32d6b97d549526155f45afb1512",
    "results/pi05_public_baseline/full_randomized_10/stack_blocks_three/result.json": "f84248bb8b495a889d266d21067e28941d47180ad7ba718259647c7eb5ade198",
    "results/pi05_public_baseline/full_randomized_10/stack_blocks_three/run.log": "96326d22137d243a9661c57bc8991789fb8353c694f641b0220641e8583099c4",
    "results/pi05_public_baseline/full_randomized_10/stack_blocks_two/result.json": "8cb9c05992fd32c0c7d178702c5cfad2d38ff19425a729212e5cea49d2fe58d1",
    "results/pi05_public_baseline/full_randomized_10/stack_blocks_two/run.log": "7d191d03b9367f9394bd215b51390c8d7a5f738f8acacd2ecceedff045cf6502",
    "results/pi05_public_baseline/full_randomized_10/stack_bowls_three/result.json": "fd137f295cde745fc0fbc549084e080ad06ac32b3ae33d1e4fe292b348c02e6c",
    "results/pi05_public_baseline/full_randomized_10/stack_bowls_three/run.log": "8185f4a2f338ede255c59afed1cc1069b1b1236b8d9735c9d0ac6feb5053fab8",
    "results/pi05_public_baseline/full_randomized_10/stack_bowls_two/result.json": "5bfdeadc9adcc568a5b1536c1be6c59abad7fb1a7c8d43933aa87dd1e71d4755",
    "results/pi05_public_baseline/full_randomized_10/stack_bowls_two/run.log": "7ca240aab2a16eae8367fd995eb50ead123d5dfe9f65c0597641993eab022b62",
    "results/pi05_public_baseline/full_randomized_10/stamp_seal/result.json": "e21410fa9e9f412da45f9649ee53eab4af16c7bd2143aa17b8164a7dd72076ef",
    "results/pi05_public_baseline/full_randomized_10/stamp_seal/run.log": "54bd77183e6fd9181b4f755f2323958a9f4baa66985e2775ad632e14a7a5e5c5",
    "results/pi05_public_baseline/full_randomized_10/turn_switch/result.json": "be4b2914c6f3f3f6f39ca41e6512db49bb419357b4933ec5ebec879ee9d57a9c",
    "results/pi05_public_baseline/full_randomized_10/turn_switch/run.log": "2fa86bbfae8e19bd8c52114ea40fc6128acd4ddb59352104428262b0e3ebab68",
    "results/pi05_public_baseline/norm_ab_seed100000_quantile/result.json": "a3a97c81618320202a13d9d25146afd205dc4f0bef6fd46f5216f88d5a52662a",
    "results/pi05_public_baseline/norm_ab_seed100000_quantile/run.log": "a3d591a959253a6b58e689fa503890f2624088e6bfe52ff3fcd08f6030ac965d",
    "results/pi05_public_baseline/norm_ab_seed100000_zscore/result.json": "5873abc2f0235ea6f691689b0a60283108ca165bca19f696991cb50c69ae2c42",
    "results/pi05_public_baseline/norm_ab_seed100000_zscore/run.log": "eb578a93c0e8cdc3878284aad28162e5f3cc1a8b024baf6eee5d019827585bb7",
    "results/pi05_public_baseline/prompt_ab_hammer_scene_seed100000/result.json": "6206be18b9e2eb06d2360d92c54939ea9d0dcf89ea747ed2456fcbae0000bfb7",
    "results/pi05_public_baseline/protocol_v10_full_clean_10_oidn_torch0/adjust_bottle/result.json": "15e5f9fff840ed7ebf01627851a1f545852487dc1fa1e5dd150e2a8f8e1100e0",
    "results/pi05_public_baseline/protocol_v10_full_clean_10_oidn_torch0/adjust_bottle/run.log": "5ee40a1b1b678d6f1e85480ab6e6618f1c215c855a061a5548a070072ff92efe",
    "results/pi05_public_baseline/protocol_v10_pipeline.log": "be9b1705aec73c6214d13a5b85168746d166422ea8592826cb45246ba938f58c",
    "results/pi05_public_baseline/protocol_v1_clean_3x10/adjust_bottle/run.log": "8ea91fc5ce4624940ca44bba3318ed3705d494fc51bdd11c6f440b98838b5969",
    "results/pi05_public_baseline/protocol_v1_clean_3x10/beat_block_hammer/run.log": "0d246439f07d7c0b92d7e67283993555573c87516b557347ab09901b5d0d4b46",
    "results/pi05_public_baseline/protocol_v1_clean_3x10/click_bell/run.log": "2fc6cec6a273dcfc02e59996c11c7fdc2cbe256532f3eb310c008f2cb937182f",
    "results/pi05_public_baseline/protocol_v2_clean_3x10/adjust_bottle/run.log": "78b825938e4b2f24e43ce6bf2fd96e44fc8e9385d3236024d2b28c19d6f100c7",
    "results/pi05_public_baseline/protocol_v2_clean_3x10/beat_block_hammer/run.log": "b42726aa97d3ccd1582958a9931b8238f73efe5ad4bb886f703a37ea815fc9bd",
    "results/pi05_public_baseline/protocol_v2_clean_3x10/click_bell/run.log": "e1d034d72e656f057a9c487d8b790fbd6546b92f8c5e592522964e5cb5af8071",
    "results/pi05_public_baseline/protocol_v3_hammer_seed100000/result.json": "f89b49cf9d45709b14474c4d1033241ce6214883ac4cc34a663cd2a35c8a95bc",
    "results/pi05_public_baseline/protocol_v4_hammer_seed100000_optix/result.json": "8b6e98d59cb2b7ae7252a55ee69239d03966398f7197273a52f224f0ff272e04",
    "results/pi05_public_baseline/protocol_v4_hammer_seed100000_optix/run.log": "41c47e49d9abbc07a9711b3e1982f9d4f16f7badf825b2f4556a1f3ce42bec71",
    "results/pi05_public_baseline/protocol_v5_hammer_seed100000_zscore_optix/result.json": "bef15dad0d8e6482729d88fbd4cd0a477edaf4569ddd9fd27412f9c923047cb8",
    "results/pi05_public_baseline/protocol_v5_hammer_seed100000_zscore_optix/run.log": "3dc91d5d34d1a4d90ce97ea134d9d0f3a7cdec835c13bee60332801b59bca3a6",
    "results/pi05_public_baseline/protocol_v5_hammer_seed100000_zscore_optix/trajectory.jsonl": "7df33a07a36c9ee62121222e8458b4e09f1604570d2ed2322c01c72e2fa0a012",
    "results/pi05_public_baseline/protocol_v6_hammer_seed100000_quantile_oidn/result.json": "c5bdf802fbdc54d9b544d902bdcf13db7345269a609e4031d03c6070cc9f065f",
    "results/pi05_public_baseline/protocol_v6_hammer_seed100000_quantile_oidn/run.log": "27f8680ea6195f802669d406e0f9800b5cd152dc265b77d19b16ca2b6379f6b6",
    "results/pi05_public_baseline/protocol_v6_hammer_seed100000_quantile_oidn/trajectory.jsonl": "69db7b05e66ae982724deb8eefe4703325a9686771c394ac575afe9ef6b987b5",
    "results/pi05_public_baseline/protocol_v7_clean_3x10_oidn/adjust_bottle/result.json": "c16a216d473822c819d172bb09f7c61ca8b4401fc96df05a8906c1a3cb764878",
    "results/pi05_public_baseline/protocol_v7_clean_3x10_oidn/adjust_bottle/run.log": "c40e90766293895be04d4e41eb5b0d0c6354b77900453b051f0f0f457bf7372d",
    "results/pi05_public_baseline/protocol_v7_clean_3x10_oidn/beat_block_hammer/result.json": "215908f25e0092a3f5a2d5734cfa2f8b07a9401775df787596278c5fbd14f400",
    "results/pi05_public_baseline/protocol_v7_clean_3x10_oidn/beat_block_hammer/run.log": "c28c1db2f0549142a5e22a7a6961fb8fa3021f8c26bb058ced494d7c0fa359ed",
    "results/pi05_public_baseline/protocol_v7_clean_3x10_oidn/click_bell/result.json": "eb2a5b70512c1e25babafbbf667f60ce53674cf61d9270dc29303d1737f1e311",
    "results/pi05_public_baseline/protocol_v7_clean_3x10_oidn/click_bell/run.log": "13c8a2e48cee7677e6af8299506c79d17f91d2ca2e0785aa6944f7eea67159f7",
    "results/pi05_public_baseline/protocol_v7_clean_3x10_oidn/manifest.json": "1d774e0dc87a4bfc05a2139fa8d6d39252fb915fd42b6fbf92be9752e363fee5",
    "results/pi05_public_baseline/protocol_v8_hammer_seed100000_policy0_oidn/result.json": "f0640c7010dee7216104c6ad790c6b8ddf69dd68e1c8a966ca30830d323921cb",
    "results/pi05_public_baseline/protocol_v8_hammer_seed100000_policy0_oidn/run.log": "8bd1f08493f5b9e57f67f98404b89007cca7394408c7c2d8cce0227d507b9ce7",
    "results/pi05_public_baseline/protocol_v8_hammer_seed100000_policy0_oidn/trajectory.jsonl": "7291cf3e52b7996749a801fcddf281c3c206a4c7d708605599c32b81dc203e3a",
    "results/pi05_public_baseline/protocol_v9_hammer_seed100000_jax0_oidn/result.json": "f7bfa8b8d83db8b4e34d723324fd28e89b7e71b067a27d67c231f3c47902e17c",
    "results/pi05_public_baseline/protocol_v9_hammer_seed100000_jax0_oidn/run.log": "30dbc6fd4137bae4ff889e435beee9f205bcc02fd6879a1a403d29a3bb747fae",
    "results/pi05_public_baseline/protocol_v9_hammer_seed100000_jax0_oidn/trajectory.jsonl": "a3363a30ad396f94f803bafbe86d92c74ebc651205d505959c0a2c497b2d0703",
    "results/pi05_public_baseline/randomized/failure_episodes.jsonl": "2cacfa13e86a66c7460ddf2cdffefe1970b8be634be16e10cbaadacd5351d7f9",
    "results/pi05_public_baseline/randomized/per_task.json": "ddf37902ad4093fde5e8e7f48ccbf7b49f3a4827622c12e9c5d80406f456dc2a",
    "results/pi05_public_baseline/randomized/success_episodes.jsonl": "9a29cfba4b22ff6cb47d1882321ac9b28380ffe3dfc186649a22c1ec26fd8960",
    "results/pi05_public_baseline/randomized/summary.json": "01bb65db056eb21f3dbdc10548d825235705da7376d7ccb53a165cf5125249e8",
    "results/pi05_public_baseline/stage0_pipeline.log": "46cbc17917b50303a8d1911579c3efa6e3154866cbc63f9241bb468ddcec45aa",
    "results/pi05_public_baseline/stage0_supervisor.log": "b4a6c06672677cf0e25ec72bec83beae33305cee4b8087f168962014373b02bb",
    "results/pi05_public_download.log": "56837c77440d8d76f6a4d385d00aad82f61513126e21aef0545234700a58dbe0",
    "results/pi05_public_download_v4.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/pi05_public_hfxet.log": "9d967e36a61a1db7e2e7e729686cb2bda859620a1e175e0cd02e41303f06ec41",
    "results/pi05_robocasa_competence.json": "1f6c1929fc769dfeb390dd2ad8dca258f212067f7481a1115857b730390c317d",
    "results/pi05_robotwin2_public_download_status.json": "fca13df1f1e2db6b9331d720195986e07a8cb5b24e8171686a65a3d39efa0c64",
    "results/positive_control_v2_probe.log": "73614f4602a7bf2e2fe0c6cbf1dddf318b8b08d6e1bcd4b905922a05f1cb226a",
    "results/probe_check_cleanup.md": "55e1fe031884f95895aba26ec8ebe5a8709990b14c4b323cfa45165b44afc6eb",
    "results/raster_fix_diff_provenance.md": "edff40e77c458b5e70b07b6fb318e122893f748d223b448fbe95f8db45d18665",
    "results/render_load_l0_dryrun.log": "57f6bbc46d05f1cf33b9a40ecb83c3a8c701962b9b05578ca7e56e1fe966a581",
    "results/render_load_l1_dryrun.log": "1e3d0dc40c8c1b8add65e38943bc4aab75a8f272a7252b390950ad533e695174",
    "results/robotwin_ckpt_paths_A_C.md": "9818c0469798eb23436a23d27bc6fa95f563087a403d658c6a940aab223e72e0",
    "results/robotwin_convert_fix_20261001.log": "6a6547798a2cd0aafb8e27d66f80342fa4d9dd9828c369945ced3dd607afac09",
    "results/robotwin_convert_fix_final_20261001.log": "41d97662f2295d6cbfb295d3211ef97c16d09f707b8d857f7ce29afe91cf9cfc",
    "results/robotwin_data_checkpoint_survey.md": "09cfbedb95088cb836484a8d95485170f8ce3bd766c6db3f43737c507c93efa4",
    "results/robotwin_env_analysis.md": "59fbde28c1f3be895178bb62a2a3ae4636a37e684709877f466458b486379d53",
    "results/robotwin_gate1_render.log": "af7dc896cb9d8c2e0fd2f73156cd73a5568b4bfa03762165018595f695a35bf6",
    "results/robotwin_gate3_adapter_plan.md": "d4cc0f4b65a4fa6d27ddbd8b73b775ecfdfbe40b826b72578accea1386c54826",
    "results/robotwin_official_loading_path.md": "1ea33d59d22e34d301355046ad4f9383342a3b81d547ae781184a291cceefafe",
    "results/robotwin_phaseB.log": "3f910e952f52b32afa95e5a8a0fffab5eaa1b67ec7c46ea066aa8aa2e307a04b",
    "results/robotwin_phaseB_smoke_rerun.log": "d0ce43baafb7dfd6f29ac2deed550edc7a79fb929996bf79abc58cf2f3f3d353",
    "results/robotwin_phaseB_smoke_rerun_v2.log": "03e41c9ce0a6a62a1c2c878caa58896dce90804e27f41bfc4a01673fcd50327f",
    "results/robotwin_phaseB_v2.log": "996609adf78ee050f473b76861c4c8997bc604bf60c499ae38d5a98241d11c0c",
    "results/robotwin_phaseB_v3.log": "aac657d8b1e4f42fe3ef1d43f2f5e57140142e9292d4028476dddadac480878d",
    "results/robotwin_positive_control.log": "16458217a3ab94c3f0b0ce936529e2945d7f5633259bee8077ce877782cdbe47",
    "results/robotwin_stage1_dump.log": "78f2fd7ffdce8667c4d591d5333fc1f9c4ae0680c784447e4150a6a40ede6f90",
    "results/seed_dedup_tally.md": "d48e5ad0f622000d6ef056bc75508bf9e0fa624bc7c6aee1aa7aac8b4e1aa801",
    "results/sidneyxie_load_test.log": "42a9f7add358b73286753f3292c49da8ec462d1748c4fd2744e08fb5dc7a27cf",
    "results/sidneyxie_load_test.md": "31d08292380a47da2ddb813b21fce71e1a8a42e8037e773bed5e2a5baca8eea7",
    "results/sim_child_error_decoded.md": "3a22d4d89ccb809f38b0019c6c57973e724254db463b7441f7176d5eaefc339d",
    "results/sim_child_stderr.log": "f0823f009c601e2bd4c1533743b042e80124856873fb8d27eddd0f872dad8d47",
    "results/sim_child_stdout.log": "4f270c16aa3c9223f6ff26ed4115100d573752104c9f73e7645ac18f4d89414b",
    "results/sim_render_config_investigation.md": "80e602011c537e7d763f13bb9431e0ad3c32712a291d94b6c7772372aaf3fed2",
    "results/single_repin.md": "12367cf3578a7b0bdffd1dea5b640c7cae24adf53cf2f1172bc12d2f061c52a4",
    "results/smoke_camera_keys.json": "21b893ebf9da8518d7fda2b6bd091b368970758f4384bae9550e5bd0996f18b1",
    "results/smoke_public_openpi/result.json": "5ccd64a92ab2113bd3424ae8732c2321af899123ea8a63e56c2cb2caa5c3c053",
    "results/smoke_public_openpi/run.log": "54d09f95f866136f771e6d41f0ac990078d943fd360c7cf523aafd4c3942bd04",
    "results/smoke_public_openpi_full/report.json": "ef473a97fe74c95bc59a53335250a32bdcdbca3600fe9ed0ad1e493e32eba2c1",
    "results/smoke_public_openpi_full/result.json": "9507e34d487e63172cdaffb29e29b4192f159c3b007969e4ed601cb175687179",
    "results/smoke_public_openpi_full/run.log": "19a8dd922b70de8b9d4a9df6bab1ca9573e6a8bc0c2fc8378c6150f8d056475d",
    "results/smoke_public_openpi_prefixed/result.json": "0497e41ec6bc2f1198e157d871b26c69da73264864e95dc5dc5956cddb5d7fc8",
    "results/smoke_public_openpi_randomized/run.log": "d8aeab50bf9ed9222cc0ae29db1247d73a5f3f62af6e991eb3be416680f61f5f",
    "results/smoke_public_openpi_short/result.json": "cdbbbfe91c55a24667783a8350a372bf864adf826bf531406c8e5dd98761db41",
    "results/smoke_public_openpi_short.log": "4d7713165c72446abc6a0a0154e0f1a03da6e6fa578dd3af8b6654287a85d5f7",
    "results/smoke_public_probe/beat_run.log": "8b3a0ee3ca647409b98d011b6f12941a0c27ef8af3bcc763d126bb48b68740e8",
    "results/smoke_public_probe/heartbeat2.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/smoke_public_probe/local_model_beat_1x20.json": "5384b27d062ed601f07f178f895db250b16495b8961d4d14d95289be0cdedfcd",
    "results/smoke_public_probe/run.log": "ce78e6ecd473c7f27ad29e4788b2507b528796f871ee3d60f3bfaf39c12d5eb0",
    "results/smoke_video_probe/heartbeat.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/smoke_video_probe/run.log": "1f45fe0b9be49f91ab8b1cc6aea1897ff7ea8f4c23ad3a3e6d4dde018e372158",
    "results/smoke_video_probe2/heartbeat.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/smoke_video_probe2/run.log": "2861ce22ec10c10984fb0c304a4e16e14ec811cea20861881b3e9ede28398844",
    "results/smoke_video_probe3/heartbeat.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/smoke_video_probe3/result.json": "a27077d863e32be74490d959e0a5757b67c8e8ca786d98bf982b11f224bd2bc3",
    "results/smoke_video_probe3/run.log": "c5bcba98c5f6d7a0c3ff2c88f8d4eae19a1505e6bf2605202741dac6b7393fb0",
    "results/stage0_environment_inspection.md": "8281f6b2aebd237b7881f680a82c8fbdb2a2c812b399fab3df77857684cf24f6",
    "results/stage1_leaf_manifest.txt": "cbf278742a4cda596be1cc7869e0bf66e83d24524e2cba88dde79e7b1b8d4a60",
    "results/state_ablation_asis.json": "affd2b2ab691c83484e61aad2bcf3a55026801d5bc6318d6412a4dd95960eda7",
    "results/state_ablation_convention.json": "b2532de5a8dbb5244cb3196262e218d8e96f83fcaf8bba985f65471fab7dd48a",
    "results/state_ablation_sanitized.json": "2189e52ded001dc950e32b4fcccdeaabf93ef575f5da78868e80a5ecbd3f0c57",
    "results/state_ablation_static_20261001T093547Z.log": "f5c233d565419a50077851d9b4a24ac6ec6c1941a74969a9e1a331c3e5065679",
    "results/success_signal_contract.md": "f13f066690db99110fda879a44e869748484b5e65464cbf5d40006f698df7724",
    "results/test_parallel_one/adjust_bottle/result.json": "bc4fa5735cc378ec7961de5236bb5a8be846a0363fe5e51aed0310520fb13a86",
    "results/test_parallel_one/adjust_bottle/run.log": "2517144816d37e60d72798a273ce154fd50308cd4aa49cbe9206654fd8d0a142",
    "results/test_parallel_one/manifest.json": "16843baa8f9d82e7cfb70398a65b2d4c73779377a1589513e153cf5e2b73aa69"
}
HISTORY_SUFFIXES = {'.json', '.jsonl', '.log', '.txt', '.md'}


def historical_paths():
    paths = set()
    for name in ('outputs', 'results', 'artifacts'):
        root = ROOT/name
        def visit(directory):
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    # Tokenizer cache is model vocabulary, not episode metadata.
                    if path == ROOT/'artifacts/quarantine_tokenizer_provenance':
                        continue
                    if entry.is_symlink():
                        if path.suffix in HISTORY_SUFFIXES:
                            raise ValueError(f'Linked historical metadata: {path}')
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        visit(path)
                    elif entry.is_file(follow_symlinks=False) and path.suffix in HISTORY_SUFFIXES:
                        paths.add(str(path.relative_to(ROOT)))
        visit(root)
    return paths


def audit_inventory_membership(manifest_path=None):
    actual = historical_paths()
    if manifest_path is not None:
        path = Path(manifest_path).resolve()
        if path.is_relative_to(ROOT):
            name = str(path.relative_to(ROOT))
            if name not in HISTORICAL_FILES:
                actual.discard(name)
    pinned = {name for name in HISTORICAL_FILES if name.split('/')[0] in ('outputs', 'results', 'artifacts')}
    # The frozen external inventory is complete only relative to this audit.
    # A later unrelated file causes fail-closed review rather than silent reuse.
    if actual != pinned:
        raise ValueError(f'Historical inventory changed; audit required: added={sorted(actual-pinned)}, '
                         f'removed={sorted(pinned-actual)}')


def extract_seeds(node, excluded):
    """Strict known seed schemas; reserve 1000 for configured start ranges."""
    single = {'seed', 'episode_seed', 'explicit_seed'}
    lists = {'seeds', 'known_seeds', 'seed_list', 'accepted_seeds'}
    starts = {'start_seed', 'seed_start', 'evaluation_seed_start'}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in single | starts:
                if key == 'explicit_seed' and type(value) is bool:
                    continue  # Provenance flag in legacy summaries, not an episode number.
                if key == 'seed' and value is None and {'global_step', 'episode', 'step'} <= set(node):
                    continue  # Audited per-step null marker, not an episode declaration.
                if type(value) is not int or value < 0:
                    raise ValueError(f'Unparseable historical seed: {key}={value!r}')
                excluded.update(range(value, value+1000) if key in starts else (value,))
            elif key in lists:
                if not isinstance(value, list) or any(type(x) is not int or x < 0 for x in value):
                    raise ValueError(f'Unparseable historical seed list: {key}')
                excluded.update(value)
            else:
                extract_seeds(value, excluded)
    elif isinstance(node, list):
        for value in node:
            extract_seeds(value, excluded)


def seed_inventories():
    excluded, sources = set(), {}
    for name, expected in sorted(HISTORICAL_FILES.items()):
        path = ROOT/name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or linked historical source: {path}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Historical source changed: {path}')
        sources[str(path)] = actual
        if name == 'results/sim_child_stdout.log':
            # Audited binary IPC: exact hash above pins its episode_start seed.
            excluded.add(100000)
            continue
        if path.suffix == '.json':
            if path.stat().st_size > 512*2**20:
                raise ValueError(f'Oversized historical metadata: {path}')
            extract_seeds(json.loads(path.read_bytes()), excluded)
        elif path.suffix == '.jsonl':
            with path.open() as stream:
                for line in stream:
                    extract_seeds(json.loads(line), excluded)
        else:
            try:
                with path.open(errors='strict') as stream:
                    for line in stream:
                        # Covers seed=123, seed 123, seed_00123 and --seed 123.
                        for match in re.finditer(r'(?i)\bseed(?:_start|_list|s)?[\s_:=\"\'\[,-]+(\d+)', line):
                            value = int(match[1])
                            excluded.update(range(value, value+1000))
            except UnicodeDecodeError as exc:
                raise ValueError(f'Unparseable historical source requires audit: {path}') from exc
    if not HISTORICAL_FILES:
        raise ValueError('No audited historical inventory')
    return excluded, sources


def selected_cases(inventories, description_root):
    excluded, sources = inventories
    cases = []
    for task in ('hanging_mug', 'place_mouse_pad'):
        seed = 200000
        while len([x for x in cases if x['task'] == task]) < 10:
            if seed not in excluded:
                spec = template_spec(task, seed, description_root)
                sources[spec['source']] = spec['source_sha256']
                cases.append(dict(id=f'{task}_{seed}', task=task, seed=seed, template=spec))
            seed += 1
            if seed >= 300000:
                raise RuntimeError('No eligible fixed panel seeds')
    return cases


def simulator_config_paths(repo):
    """Read YAML only, without importing simulator code; follow selected robot refs.

    load_task_args uses env_cfg/task_config and repo-relative embodiment paths.
    Robot.set_planner selects curobo_left/right for the fixed dual-arm aloha.
    Their collision_spheres paths currently point outside the pinned repo; hash
    those actual referenced files, not similarly named local copies.
    """
    import yaml
    repo = Path(repo)
    paths = set()
    def load(path):
        path = Path(path)
        if not path.is_file() or not path.read_bytes().strip():
            raise ValueError(f'Missing or empty simulator configuration: {path}')
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict) or not value:
            raise ValueError(f'Invalid simulator configuration mapping: {path}')
        paths.add(path)
        return value
    cfg = repo/'env_cfg/task_config'
    task = load(cfg/'demo_clean.yml')
    embodiments = load(cfg/'_embodiment_config.yml')
    load(cfg/'_camera_config.yml')
    load(cfg/'_eval_step_limit.yml')
    if task.get('embodiment') != ['aloha-agilex']:
        raise ValueError('Fixed evaluation requires aloha-agilex embodiment')
    robot_path = embodiments['aloha-agilex']['file_path']
    if not isinstance(robot_path, str) or not robot_path.strip():
        raise ValueError('Missing selected embodiment path')
    robot = repo/robot_path
    config = load(robot/'config.yml')
    if config.get('planner') != 'curobo' or config.get('dual_arm') is not True:
        raise ValueError('Unexpected selected embodiment planner')
    def references(value, parent):
        if isinstance(value, dict):
            for item in value.values():
                references(item, parent)
        elif isinstance(value, list):
            for item in value:
                references(item, parent)
        elif isinstance(value, str) and Path(value).suffix in ('.yml', '.yaml'):
            target = parent/value
            if target not in paths:
                references(load(target), target.parent)
    for name in ('curobo_left.yml', 'curobo_right.yml'):
        path = robot/name
        references(load(path), path.parent)
    # Also pin the full real task-config directory, including optional variants.
    for path in sorted(cfg.rglob('*')):
        if path.suffix in ('.yml', '.yaml'):
            load(path)
    return sorted(paths)


def manifest_code_paths():
    scripts = ('gen1_raw_paired_eval.py', 'gen1_raw_diagnostic.py',
               'gen1_diagnose_pilot.py', 'controlled_paired_eval.py',
               'pi05_openpi_adapter.py', 'robotwin_eval_harness.py',
               'robotwin_public_openpi_harness.py')
    paths = [ROOT / 'scripts' / name for name in scripts]
    for directory in (OPENPI/'src/openpi', OPENPI/'packages/openpi-client/src/openpi_client'):
        paths.extend(sorted(directory.rglob('*.py')))
    repo = ROOT/'robotwin_repo_ba0fd21'
    paths.extend(sorted((repo/'envs').rglob('*.py')))
    evaluator = repo/'scripts/eval_policy_xpolicylab.py'
    if not evaluator.is_file() or not evaluator.read_bytes().strip():
        raise ValueError('Missing or empty executed evaluator source')
    paths.append(evaluator)
    paths.extend(simulator_config_paths(repo))
    return sorted(set(paths))


def file_identity(directory, subdirs):
    directory = Path(directory)
    result = {}
    for name in subdirs:
        base = directory / name
        if not base.is_dir() or base.is_symlink():
            raise ValueError('Missing or linked model subtree')
        for path in sorted(base.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlink in model subtree')
            if path.is_file():
                with path.open('rb') as stream:
                    result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not result:
        raise ValueError('Empty model identity')
    return result


def expected_manifest(include_weights=True):
    """Deterministic reconstruction from pinned inventory, never rollout outcomes."""
    excluded, sources = seed_inventories()
    description = ROOT / 'robotwin_repo_ba0fd21/description'
    cases = selected_cases((excluded, sources), description)
    # Freeze the complete bounded task object pools, so dynamic expert bindings
    # cannot silently select descriptions edited after panel preparation.
    for name in ('039_mug', '040_rack', '047_mouse'):
        files = sorted((description/'objects_description'/name).glob('*.json'))
        if not files:
            raise ValueError('Missing official object description pool')
        for path in files:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    repo = ROOT/'robotwin_repo_ba0fd21'
    for path in [repo/'scripts/eval_policy_xpolicylab.py',
                 repo/'envs/hanging_mug.py', repo/'envs/place_mouse_pad.py',
                 repo/'description/utils/generate_episode_instructions.py']:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    import controlled_paired_eval as c
    m = dict(runner='gen1_raw_paired_eval_v1', version=1, cases=cases, sources=sources,
             models=dict(original=c.MODELS['reference'], raw=str(PILOT)), config=c.CONFIG,
             limits=dict(output_budget_bytes=LIMIT, min_free_bytes=MIN_FREE,
                         worker_timeout_seconds=1800, min_prelaunch_cgroup_bytes=24*GIB,
                         min_running_cgroup_bytes=12*GIB),
             code={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in manifest_code_paths()},
             selection='First ten eligible seeds per task from 200000; excluded historical inventory; no outcomes',
             protocol='Frozen controlled expert gate; fixed official seen template; deterministic expert-bound objects; '
                      'not historical noncontrolled random prompt generation; full 50-action chunks and explicit JAX noise',
             limitation='Initial observed inputs only; agent_pos is drive-target state, not contacts or full physics')
    if include_weights:
        m['identities'] = {
            'original': file_identity(c.MODELS['reference'], ('params', 'assets')),
            'raw': file_identity(PILOT, ('train_state', 'assets'))}
    return m


def prepare(path):
    path = Path(path).resolve()
    if path.parent != OUTPUTS.resolve() or path.exists() or path.is_symlink():
        raise ValueError('Manifest must be fresh direct child of outputs')
    # Byte and path guards are evaluated before hashing large identities.
    guard(path, 64*2**20)
    audit_inventory_membership()
    m = expected_manifest()
    validate(m, check_files=False)
    encoded = (json.dumps(m, indent=2, allow_nan=False) + '\n').encode()
    guard(path, len(encoded))
    with path.open('xb') as stream:
        stream.write(encoded)
    return m


def validate(m, check_files=True):
    if type(m) is not dict or set(m) != set(expected_manifest(include_weights=False)) | {'identities'}:
        raise ValueError('Manifest field set changed')
    expected = expected_manifest(include_weights=check_files)
    actual = m if check_files else {k: v for k, v in m.items() if k != 'identities'}
    if actual != expected:
        raise ValueError('Panel configuration or source/weight identities changed')
    # The reconstructed identity includes all exact file paths and hashes. No
    # unknown or missing shard/path entry is tolerated.
    return True


def require_same_binding(original, raw):
    if not original or not raw or original != raw:
        raise ValueError('Original/raw expert prompt bindings differ or are absent')


@contextlib.contextmanager
def bound_task(task, spec, description_root, record):
    """Observe one existing expert return and bind the existing instruction call.

    This primitive needs an audited class_decorator hook before panel use. It
    performs no extra expert, observation, simulation, success or RNG calls.
    """
    original_play = task.play_once
    original_set = task.set_instruction
    calls = {'expert': 0, 'instruction': 0}
    def play(*args, **kwargs):
        result = original_play(*args, **kwargs)
        calls['expert'] += 1
        record['_expert_payload'] = result
        return result
    def set_instruction(instruction):
        calls['instruction'] += 1
        if calls['instruction'] != 1 or calls['expert'] != 1 or instruction != spec['template']:
            raise ValueError('Unexpected instruction binding protocol')
        record.update(bind_prompt(spec, record.pop('_expert_payload'), description_root))
        return original_set(instruction=record['prompt'])
    with replaced(task, 'play_once', play), replaced(task, 'set_instruction', set_instruction):
        yield


def array_record(value, shape):
    import numpy as np
    a = np.ascontiguousarray(value, dtype='<f4')
    if list(a.shape) != shape or not np.isfinite(a).all():
        raise ValueError('Invalid diagnostic array')
    return dict(shape=shape, sha256=hashlib.sha256(a.tobytes()).hexdigest(), values=a.tolist())


class NumericTrace:
    def __init__(self, stream, reserve):
        self.stream, self.reserve = stream, reserve
        self.chunks, self.steps, self.noises = 0, 0, []

    def emit(self, kind, **fields):
        encoded = json.dumps(dict(kind=kind, **fields), allow_nan=False) + '\n'
        self.reserve(len(encoded.encode()))
        self.stream.write(encoded)
        self.stream.flush()

    @contextlib.contextmanager
    def policy(self, adapter, reference_noise=None):
        if adapter.checkpoint_format != 'openpi_orbax' or adapter.noise_source != 'jax':
            raise ValueError('Expected Orbax explicit JAX noise policy')
        original_infer, original_chunk = adapter.policy.infer, adapter.infer_chunk
        pending = []
        def infer(*args, **kwargs):
            if 'noise' not in kwargs:
                raise ValueError('Missing explicit noise')
            noise = array_record(kwargs['noise'], [50, 32])
            if reference_noise is not None and self.chunks < len(reference_noise):
                if noise['sha256'] != reference_noise[self.chunks]:
                    raise ValueError('Common explicit noise differs before inference')
            pending.append(noise)
            return original_infer(*args, **kwargs)
        def chunk(**kwargs):
            before = len(pending)
            value = original_chunk(**kwargs)
            if len(pending) != before + 1:
                raise ValueError('Expected exactly one inference/noise per chunk')
            actions = array_record(value, [50, 14])
            noise = pending.pop()
            self.noises.append(noise['sha256'])
            self.emit('chunk', index=self.chunks, before_step=self.steps,
                      noise=noise, actions=actions)
            self.chunks += 1
            return value
        with replaced(adapter.policy, 'infer', infer), replaced(adapter, 'infer_chunk', chunk):
            yield adapter

    def response(self, payload):
        if payload.get('type') == 'step':
            self.steps += 1
            state = payload.get('diagnostic_state')
            if state is None and not (payload.get('terminated') or payload.get('truncated')):
                raise ValueError('Missing existing nonterminal observed state')
            if state is not None and state != array_record(state['values'], [14]):
                raise ValueError('Malformed diagnostic state')
            self.emit('step', index=self.steps, state=state,
                      terminated=bool(payload.get('terminated')), truncated=bool(payload.get('truncated')),
                      success=bool(payload.get('info', {}).get('is_success')))
        return payload


@contextlib.contextmanager
def simulator_shim(eval_module, harness, spec, description_root):
    """Wrap existing task construction/calls and enrich IPC envelopes only.

    episode_start.instruction stays the frozen template in the original envelope;
    diagnostic_binding carries the full bound prompt. The inference receiver
    checks both then normalizes a local copy for the frozen episode loop.
    """
    record = {}
    original_construct, original_send = eval_module.class_decorator, harness.send_message
    with contextlib.ExitStack() as stack:
        def construct(*args, **kwargs):
            task = original_construct(*args, **kwargs)
            stack.enter_context(bound_task(task, spec, description_root, record))
            return task
        def send(stream, payload):
            extra = {}
            if payload.get('type') == 'episode_start':
                if payload.get('instruction') != spec['template'] or 'prompt' not in record:
                    raise ValueError('Missing bound episode instruction')
                extra['diagnostic_binding'] = dict(record)
            if payload.get('type') == 'step':
                obs = payload.get('observation')
                extra['diagnostic_state'] = array_record(obs['agent_pos'], [14]) if obs is not None else None
            return original_send(stream, dict(payload, **extra) if extra else payload)
        stack.enter_context(replaced(eval_module, 'class_decorator', construct))
        stack.enter_context(replaced(harness, 'send_message', send))
        yield


def kill_group(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    proc.wait()


def drain_log(proc, log, output, state, selector):
    """Bound each read/write; stop the process group at 8 MiB, never discard silently."""
    for key, _ in selector.select(timeout=0):
        data = os.read(key.fd, min(65536, 8*2**20-state['bytes']+1))
        if not data:
            selector.unregister(key.fileobj)
            continue
        if state['bytes'] + len(data) > 8*2**20:
            state['truncated'] = True
            raise RuntimeError('Worker log cap reached; remaining pipe bytes not saved')
        guard(output, len(data)+64*2**20)
        log.buffer.write(data)
        log.flush()
        state['bytes'] += len(data)


def supervise(command, env, log, output, raw, timeout=1800):
    """Supervise combined inference/simulator group; any resource error aborts."""
    paths = raw.discover_cgroups()
    baseline = raw.memory_snapshot(paths)
    raw.check_memory(baseline, needed=24*GIB)
    guard(output, 64*2**20)
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    log_state = dict(bytes=0, truncated=False)
    evidence = dict(status='running', baseline=baseline, samples=0, log=log_state)
    selector = None
    try:
        selector = selectors.DefaultSelector()
        deadline = time.monotonic() + timeout
        if getattr(proc, 'stdout', None) is not None:
            os.set_blocking(proc.stdout.fileno(), False)
            selector.register(proc.stdout, selectors.EVENT_READ)
        while True:
            drain_log(proc, log, output, log_state, selector)
            snapshot = raw.memory_snapshot(raw.discover_cgroups())
            raw.check_memory(snapshot, needed=12*GIB, baseline=baseline)
            evidence.update(last=snapshot, samples=evidence['samples']+1)
            guard(output, 64*2**20)
            if time.monotonic() >= deadline:
                raise RuntimeError('Worker wall-clock timeout')
            try:
                rc = proc.wait(timeout=0.25)
                if rc:
                    raise RuntimeError(f'Worker exited {rc}')
                # Terminate any descendant still holding the log pipe open.
                if selector.get_map():
                    kill_group(proc)
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Log pipe drain timeout')
                    selector.select(timeout=0.05)
                    drain_log(proc, log, output, log_state, selector)
                evidence['status'] = 'passed'
                return evidence
            except subprocess.TimeoutExpired:
                continue
    except Exception as exc:
        evidence.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        return evidence
    finally:
        kill_group(proc)
        if selector is not None:
            selector.close()
        if getattr(proc, 'stdout', None) is not None:
            proc.stdout.close()


def restrict_worker_files():
    """Inherited per-file ceiling covers native/faulthandler files; disable core dumps.

    The parent additionally reserves 64 MiB and polls the entire output tree.
    This is not a filesystem quota or a bound on unrelated shared-disk writers.
    Frozen sim mode does not start the parent-only heartbeat snapshot writer.
    """
    import resource
    for kind, cap in ((resource.RLIMIT_FSIZE, 16*2**20), (resource.RLIMIT_CORE, 0)):
        soft, hard = resource.getrlimit(kind)
        effective = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
        resource.setrlimit(kind, (effective, effective))


def runtime_environment():
    if any(os.environ.get(k) for k in ('ROBOTWIN_SKIP_EXPERT', 'ROBOTWIN_TRAJECTORY_JSONL')):
        raise ValueError('Expert bypass or legacy trajectory capture forbidden')
    env = {k: v for k, v in os.environ.items() if not k.startswith('ROBOTWIN_')}
    for key in ('PYTHONPATH', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_ALLOCATOR',
                'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_CLIENT_MEM_FRACTION'):
        env.pop(key, None)
    env.update(OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.7', CUDA_VISIBLE_DEVICES='0',
               ROBOTWIN_RENDER_BACKEND='rt', ROBOTWIN_RT_DENOISER='oidn',
               ROBOTWIN_OIDN_RUNTIME=str(ROOT/'third_party/oidn-2.5.1/lib'),
               ROBOTWIN_OPENPI_SRC=str(OPENPI/'src'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def checked_write(path, value, output):
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    guard(output, len(encoded.encode()))
    with Path(path).open('x') as stream:
        stream.write(encoded)


def paired_episode(first, receive, send, directory, case, label, output, reference=None):
    import controlled_paired_eval as c
    if first.get('type') == 'case_status':
        if first.get('seed') != case['seed']:
            raise ValueError('Excluded case seed mismatch')
        return {key: value for key, value in first.items() if key != 'type'}
    binding = first.get('diagnostic_binding')
    if not isinstance(binding, dict) or binding.get('template') != case['template']:
        raise ValueError('Initial binding identity missing or mismatched')
    expected = bind_prompt(case['template'], {'info': binding['fields']},
                           ROOT/'robotwin_repo_ba0fd21/description')
    require_same_binding(expected, binding)
    if first.get('instruction') != case['template']['template']:
        raise ValueError('Simulator envelope template changed')
    if label == 'raw':
        if reference is None or reference.get('status') != 'completed':
            raise ValueError('Missing valid original result')
        require_same_binding(reference['binding'], binding)
    normalized = dict(first, instruction=binding['prompt'])
    episode_case = dict(case, prompt=binding['prompt'])
    guard(output, 64*2**20)
    with (directory/'trace.jsonl').open('x') as stream, contextlib.ExitStack() as stack:
        trace = NumericTrace(stream, lambda pending: guard(output, pending))
        def make_policy():
            return stack.enter_context(trace.policy(build_adapter(label, directory),
                                       reference_noise=reference['noise_hashes'] if reference else None))
        original_capture = c.capture
        def capture(path, images, state, prompt):
            import numpy as np
            # NPY headers, state and hashes fit comfortably inside this overhead.
            pending = sum(np.asarray(images[k]).size for k in
                          ('cam_high', 'cam_left_wrist', 'cam_right_wrist')) + 65536
            if pending > 16*2**20:
                raise ValueError('Initial capture exceeds conservative 16 MiB cap')
            guard(output, pending+64*2**20)
            value = original_capture(path, images, state, prompt)
            guard(output, 64*2**20)
            return value
        stack.enter_context(replaced(c, 'capture', capture))
        result = c.episode(normalized, lambda: trace.response(receive()), send,
                           make_policy, directory, episode_case,
                           reference=reference.get('hashes') if reference else None,
                           pilot=label == 'raw')
    result.update(binding=binding, noise_hashes=trace.noises, chunks=trace.chunks)
    if label == 'raw':
        common = min(len(trace.noises), len(reference['noise_hashes']))
        if common == 0 or trace.noises[:common] != reference['noise_hashes'][:common]:
            raise ValueError('Common chunk explicit noise mismatch')
    return result


def verified_manifest(path, expected_sha):
    data = Path(path).read_bytes()
    if not expected_sha or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('Worker manifest SHA256 mismatch')
    manifest = json.loads(data)
    if manifest.get('runner') != 'gen1_raw_paired_eval_v1' or manifest.get('version') != 1:
        raise ValueError('Worker manifest version mismatch')
    validate(manifest, check_files=True)
    return manifest


def check_dependency_versions():
    import importlib.metadata as metadata
    required = {'jax': '0.5.3', 'jaxlib': '0.5.3', 'orbax-checkpoint': '0.11.13',
                'flax': '0.10.2', 'numpy': '1.26.4', 'tensorstore': '0.1.74',
                'torch': '2.7.1', 'lerobot': '0.3.3', 'transformers': '4.53.2'}
    if any(metadata.version(name) != version for name, version in required.items()):
        raise ValueError('Runtime dependency version mismatch')


def worker(m, label, case, output):
    import controlled_paired_eval as c
    from robotwin_public_openpi_harness import _configure_sim_renderer_env
    from robotwin_eval_harness import receive_message, send_message
    directory = output/f"{case['id']}_{label}"
    reference_path = output/f"{case['id']}_original"/'result.json'
    reference = json.loads(reference_path.read_text()) if label == 'raw' and reference_path.exists() else None
    env = runtime_environment()
    _configure_sim_renderer_env(env)
    restrict_worker_files()
    env['ROBOTWIN_CONTROLLED_OUTPUT'] = str(directory)
    env['GEN1_RAW_TEMPLATE_SPEC'] = json.dumps(case['template'])
    command = [c.CONFIG['sim_python'], '-I', '-B', str(Path(__file__).resolve()), 'sim',
               '--role', 'sim', '--controlled-case', '--episodes', '1', '--task', case['task'],
               '--seed', str(case['seed']), '--episode-steps', str(c.CONFIG['episode_steps']),
               '--robotwin-root', c.CONFIG['robotwin_root'], '--task-config', c.CONFIG['task_config'],
               '--first-instruction', case['template']['template'], '--first-instruction-seed', str(case['seed'])]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             env=env, cwd=c.CONFIG['robotwin_root'])
    try:
        result = paired_episode(receive_message(child.stdout), lambda: receive_message(child.stdout),
                                lambda value: send_message(child.stdin, value), directory,
                                case, label, output, reference)
    except Exception as exc:
        result = dict(status='worker_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            send_message(child.stdin, {'cmd': 'stop'})
            child.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    if child.returncode:
        result = dict(status='simulator_error', returncode=child.returncode, episode=result)
    if label == 'raw':
        if result['status'] == 'completed':
            result['raw_restore'] = check_raw_evidence(directory/'raw_restore.json')
        elif (directory/'raw_restore.json').exists():
            raise ValueError('Raw restore evidence present without completed rollout')
    checked_write(directory/'result.json', result, output)
    return 0 if result['status'] in ('completed', 'expert_rejected') else 1


def check_raw_evidence(path):
    evidence = json.loads(Path(path).read_text())
    if (evidence.get('kind') != 'raw' or evidence.get('internal_step') != 2000 or
            evidence.get('checkpoint') != str(PILOT/'train_state')):
        raise ValueError('Raw restore evidence identity mismatch')
    reads, leaves = evidence.get('logical_reads'), evidence.get('leaves')
    if not isinstance(reads, list) or not isinstance(leaves, list) or not leaves:
        raise ValueError('Missing logical raw restore evidence')
    expected = [['step']] + [leaf['key'] for leaf in leaves]
    if reads != expected or len({tuple(key) for key in reads}) != len(reads):
        raise ValueError('Unexpected or duplicated logical raw reads')
    for leaf in leaves:
        key = leaf['key']
        if len(key) < 3 or key[0] != 'params' or key[-1] != 'value':
            raise ValueError('Non-parameter raw tensor selected')
        if leaf['native_dtype'] != 'float32' or leaf['inference_dtype'] != 'bfloat16':
            raise ValueError('Raw inference parameter dtype mismatch')
    return dict(internal_step=2000, logical_parameter_leaves=len(leaves),
                logical_reads=len(reads), native_dtype='float32', inference_dtype='bfloat16')


def pair_gate(original, raw):
    if original.get('status') == raw.get('status') == 'expert_rejected':
        return 'excluded'
    if original.get('status') != 'completed' or raw.get('status') != 'completed':
        raise ValueError('Incomplete or asymmetric pair')
    if original.get('hashes') != raw.get('hashes') or not original.get('hashes'):
        raise ValueError('Initial observed inputs differ')
    require_same_binding(original.get('binding'), raw.get('binding'))
    common = min(len(original['noise_hashes']), len(raw['noise_hashes']))
    if common == 0 or original['noise_hashes'][:common] != raw['noise_hashes'][:common]:
        raise ValueError('Common explicit noise differs')
    # Task success is intentionally irrelevant to protocol integration validity.
    return 'valid'


def cpu_preflight():
    """CPU imports only: no parameter restoration, cache writes, or compilation."""
    import importlib.metadata as metadata
    import json
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise RuntimeError('Preflight requires CUDA_VISIBLE_DEVICES empty and JAX_PLATFORMS=cpu')
    if not sys.dont_write_bytecode or os.environ.get('JAX_ENABLE_COMPILATION_CACHE') != 'false':
        raise RuntimeError('Preflight requires -B and disabled compilation cache')
    sys.path.insert(0, str(OPENPI / 'src'))
    sys.path.insert(0, str(OPENPI / 'packages/openpi-client/src'))
    import openpi
    import openpi_client
    from openpi.policies import policy_config
    import jax
    expected = {'openpi': OPENPI / 'src/openpi/__init__.py',
                'openpi_client': OPENPI / 'packages/openpi-client/src/openpi_client/__init__.py'}
    modules = {'openpi': str(Path(openpi.__file__).resolve()),
               'openpi_client': str(Path(openpi_client.__file__).resolve()),
               'policy_config': str(Path(policy_config.__file__).resolve())}
    for name, path in expected.items():
        if Path(modules[name]) != path.resolve():
            raise RuntimeError('Unexpected import source: ' + name)
    if Path(modules['policy_config']) != (OPENPI / 'src/openpi/policies/policy_config.py').resolve():
        raise RuntimeError('Unexpected policy factory source')
    versions = {name: metadata.version(name) for name in
                ('jax', 'jaxlib', 'orbax-checkpoint', 'flax', 'numpy', 'tensorstore',
                 'torch', 'lerobot', 'transformers')}
    required = {'jax': '0.5.3', 'jaxlib': '0.5.3', 'orbax-checkpoint': '0.11.13',
                'flax': '0.10.2', 'numpy': '1.26.4'}
    if any(versions[name] != value for name, value in required.items()):
        raise RuntimeError('Preflight dependency versions differ')
    platforms = [device.platform for device in jax.devices()]
    if not platforms or set(platforms) != {'cpu'}:
        raise RuntimeError('Preflight backend is not exclusively CPU')
    print(json.dumps(dict(status='passed', modules=modules, versions=versions,
                          platforms=platforms, panel_execution='disabled'), indent=2))


def run(m, manifest_path, output):
    import controlled_paired_eval as c
    import gen1_raw_diagnostic as raw
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh output required')
    data = Path(manifest_path).read_bytes()
    if json.loads(data) != m:
        raise ValueError('Parent manifest changed after validation')
    audit_inventory_membership(manifest_path)
    guard(output, 64*2**20)
    output.mkdir()
    checked_write(output/'manifest.json', m, output)
    pinned_manifest = hashlib.sha256(data).hexdigest()
    records, gate_status = [], 'not_started'
    try:
        for index, case in enumerate(m['cases']):
            pair = {}
            for label in ('original', 'raw'):
                directory = output/f"{case['id']}_{label}"
                directory.mkdir()
                command = [c.CONFIG['inference_python'], '-I', '-B', str(Path(__file__).resolve()),
                           'worker', '--manifest', str(manifest_path), '--output', str(output),
                           '--manifest-sha256', pinned_manifest,
                           '--label', label, '--case', str(index)]
                with (directory/'worker.log').open('x') as log:
                    supervision = supervise(command, runtime_environment(), log, output, raw)
                checked_write(directory/'supervision.json', supervision, output)
                if supervision['status'] != 'passed':
                    raise RuntimeError('Worker supervision failed')
                result = json.loads((directory/'result.json').read_text())
                pair[label] = result
                records.append(dict(case=case['id'], label=label, result=result))
                checked_write(output/f'progress_{len(records):02d}.json', records, output)
            gate = pair_gate(pair['original'], pair['raw'])
            if index == 0:
                gate_status = gate
                if gate != 'valid':
                    raise RuntimeError('Fixed first pair excluded; no integration evidence; no replacement')
        status = 'completed'
    except Exception as exc:
        status = 'aborted'
        error = f'{type(exc).__name__}: {exc}'
    checked_write(output/'summary.json', dict(status=status, integration_gate=gate_status,
                  records=records, error=error if status == 'aborted' else None,
                  limitation=m['limitation'], protocol=m['protocol']), output)
    return 0 if status == 'completed' else 1


def simulator():
    import importlib
    import robotwin_eval_harness as harness
    spec = json.loads(os.environ['GEN1_RAW_TEMPLATE_SPEC'])
    repo = ROOT/'robotwin_repo_ba0fd21'
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo/'scripts'))
    # Preserve frozen harness native stdout redirection: wrap its import hook
    # instead of importing simulator/evaluator libraries before that redirect.
    original_import = importlib.import_module
    with contextlib.ExitStack() as stack:
        attached = False
        def hooked(name, package=None):
            nonlocal attached
            module = original_import(name, package)
            if name == 'eval_policy_xpolicylab' and not attached:
                if Path(module.__file__).resolve() != repo/'scripts/eval_policy_xpolicylab.py':
                    raise ValueError('Unexpected evaluator source')
                attached = True
                stack.enter_context(simulator_shim(module, harness, spec, repo/'description'))
            return module
        stack.enter_context(replaced(importlib, 'import_module', hooked))
        return harness.main()


# CPU safety review and 27 regressions passed, including actual configuration
# identities. Runtime integration remains gated by the first fixed pair.
RUNNER_REVIEWED = True


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if sys.argv[1:] == ['cpu-preflight']:
        return cpu_preflight()
    if not RUNNER_REVIEWED:
        raise SystemExit('Panel prepare/run disabled: complete CPU safety review pending')
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        return simulator()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--label', choices=('original', 'raw'))
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.manifest)
        return 0
    m = json.loads(args.manifest.read_text())
    if args.mode == 'run':
        validate(m)
        return run(m, args.manifest.resolve(), args.output)
    if args.output is None or args.label is None or args.case is None or not 0 <= args.case < 20:
        parser.error('worker requires output, label and case 0..19')
    if args.manifest_sha256 is None:
        parser.error('worker requires parent-pinned manifest SHA256')
    m = verified_manifest(args.manifest, args.manifest_sha256)
    check_dependency_versions()
    env = runtime_environment()
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(OPENPI/'src'))
    sys.path.insert(0, str(OPENPI/'packages/openpi-client/src'))
    return worker(m, args.label, m['cases'][args.case], args.output)


if __name__ == '__main__':
    raise SystemExit(main())
