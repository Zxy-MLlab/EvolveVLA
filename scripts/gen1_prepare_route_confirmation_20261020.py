#!/usr/bin/env python3
"""Freeze specified novel local seeds and original-controller routing protocol, CPU only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OUT = ROOT / 'outputs/gen1_control_repair_20261020/route_confirmation'
sys.path.insert(0, str(ROOT / 'scripts'))
import gen1_raw_paired_eval as binding


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def seeds_in(value, path=''):
    out = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = f'{path}/{key}'
            if key in ('seed', 'requested_seed', 'start_seed', 'seed_start', 'evaluation_seed_start') and type(item) is int:
                out.append((here, item))
            elif key == 'seeds' and isinstance(item, list):
                out.extend((f'{here}/{i}', s) for i, s in enumerate(item) if type(s) is int)
            elif isinstance(item, (dict, list)):
                out.extend(seeds_in(item, here))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                out.extend(seeds_in(item, f'{path}/{i}'))
    return out


def main():
    cases = [dict(task=task, seed=seed, episode_index=0)
             for task, base in [('place_mouse_pad', 2102000), ('hanging_mug', 2103000)]
             for seed in range(base, base + 10)]
    target_seeds = {c['seed'] for c in cases}
    local_paths = sorted((ROOT / 'data').glob('*/split_manifest.json'))
    local_paths += [ROOT / 'data/gen1_task_batches_20261010' / task / 'manifest.json'
                    for task in ['place_mouse_pad', 'hanging_mug']]
    local_paths += [ROOT / 'data/gen1_demo_clean/manifest.json']
    local_checks = []
    for p in local_paths:
        records = seeds_in(read(p))
        collisions = [(key, seed) for key, seed in records if seed in target_seeds]
        local_checks.append(dict(path=str(p), sha256=sha(p), explicit_seed_records=len(records),
                                 unique_seed_count=len({s for _, s in records}), collisions=collisions))
    assert not any(c['collisions'] for c in local_checks)
    command = ['rg', '--files', '--no-ignore', 'outputs', 'results', 'artifacts',
               '-g', 'result.json', '-g', 'results.jsonl', '-g', '*manifest*.json',
               '-g', '*cases*.json', '-g', 'summary.json', '-g', 'progress.json']
    scan = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    paths = [ROOT / line for line in scan.stdout.splitlines() if line.strip()
             and not line.startswith('outputs/gen1_control_repair_20261020/route_confirmation/')]
    pattern = re.compile(r'(?<!\d)(?:210200\d|210300\d)(?!\d)')
    history_hits, inventory = [], {}
    for p in paths:
        data = p.read_bytes()
        inventory[str(p.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        text = data.decode()
        if pattern.search(str(p)) or pattern.search(text):
            parsed = ([json.loads(s) for s in text.splitlines(keepends=True) if s.endswith('\n') and s.strip()]
                      if p.suffix == '.jsonl' else [json.loads(text)])
            exact = [(key, seed) for obj in parsed for key, seed in seeds_in(obj) if seed in target_seeds]
            if exact or pattern.search(str(p)):
                history_hits.append(dict(path=str(p), seed_records=exact))
    assert not history_hits, history_hits
    OUT.mkdir(parents=True, exist_ok=False)
    write('cases.json', cases)
    descriptions = ROOT / 'robotwin_repo_ba0fd21/description'
    templates = [dict(case=case, template_spec=binding.template_spec(case['task'], case['seed'], descriptions)) for case in cases]
    write('instruction_templates.json', templates)
    checkpoint_refs = dict(
        baseline=ROOT / 'outputs/gen1_timing_validation_20261020/baseline_policy_ab/manifest.json',
        candidate=ROOT / 'outputs/gen1_control_repair_20261020/candidate_prepare/manifest.json')
    identities = {label: read(p)['checkpoint'] for label, p in checkpoint_refs.items()}
    write('checkpoint_reference.json', dict(checkpoints=identities,
        reference_manifests={label: dict(path=str(p), sha256=sha(p)) for label, p in checkpoint_refs.items()},
        verification='Copied prior verified full-file identities; resource wrapper must independently verify full checkpoint identity at preparation/run and inventory before each case.'))
    source_files = set(binding.manifest_code_paths())
    source_files.update(ROOT / 'scripts' / name for name in [
        'gen1_route_confirmation_20261020.py', 'gen1_recovery_confirm.py', 'gen1_recovery_eval.py',
        'gen1_recovery_compare.py', 'gen1_execution_horizon_ablation_20261019.py',
        'gen1_policy_duration_repair_20261020.py', 'gen1_prepare_route_confirmation_20261020.py'])
    write('seed_overlap_audit.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        requested_seeds=sorted(target_seeds), local_manifest_checks=local_checks,
        local_manifest_collisions=[], history_scan_roots=['outputs', 'results', 'artifacts'],
        history_filename_globs=command[command.index('-g'):], history_metadata_files_checked=len(paths),
        history_explicit_seed_collisions=history_hits, history_metadata_sha256=inventory,
        limitations=['Checks declared local collection/split seeds and recorded output metadata. This does not prove distinct physical scenes or absence from official public data/pretraining.',
                     'Current running development metadata was snapshotted; future outputs are outside this preparation audit.',
                     'No confirmation outcomes were read; the new output did not exist during the historical scan.']))
    commands = {label: [str(ROOT / 'conda_envs/scratch_orbax/bin/python')] for label in []}
    python = '/root/autodl-tmp/conda_envs/scratch_orbax/bin/python'
    wrapper = ROOT / 'scripts/gen1_route_confirmation_20261020.py'
    commands = {label: {mode: [python, '-I', '-u', '-B', str(wrapper), mode,
        '--cases', str(OUT / 'cases.json'), '--protocol', str(OUT / 'protocol.json'),
        '--checkpoint', identity['path'], '--output', str(OUT / (f'{label}_prepare' if mode == 'prepare' else label))]
        for mode in ['prepare', 'run']} for label, identity in identities.items()}
    protocol = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        status='prepared_no_gpu', fixed_cases=cases, cases_sha256=sha(OUT / 'cases.json'),
        checkpoint_identities={label: identity['identity_sha256'] for label, identity in identities.items()},
        checkpoint_paths={label: identity['path'] for label, identity in identities.items()},
        instruction_templates_sha256=sha(OUT / 'instruction_templates.json'),
        instruction_rule='Freeze SHA-selected seen template by task/seed; bind object/arm from expert episode_info identically for both checkpoints; require final binding and prompt exact.',
        scene_config='demo_clean', controller='unchanged original take_action', reset_episode_index=0,
        policy_seed=0, reset_policy_rng_per_episode=True, execute_prefix=50,
        models=['baseline', 'candidate'], attempts_per_model=20, attempts_per_task_per_model=10,
        same_case_retries=0, replenish_rejected_seeds=False,
        schedule_order='mouse2102000..2102009 then mug2103000..2103009; same list for both checkpoints',
        resources=dict(preallocate=False, memory_fraction=.12, omp_threads=2, openblas_threads=2,
                       per_case_timeout_seconds=1800, wrapper_scope='process-local environment override only'),
        pair_admission=['Both scheduled records completed with boolean success and simulator_returncode0',
                        'Requested task/seed/episode_index match frozen list',
                        'Checkpoint identity and unchanged code/inventory proven by wrapper',
                        'Final prompt binding exact across checkpoints',
                        'Saved initial three camera arrays and state exact across checkpoints',
                        'First physical.jsonl physical record exact across checkpoints'],
        exclusions=['expert_rejected', 'pending_or_missing', 'setup_or_simulator_error',
                    'nonzero_returncode', 'input_binding_physical_or_identity_mismatch'],
        route_rule_per_task=dict(minimum_valid_pairs=8, candidate_successes_strictly_greater=True,
            maximum_baseline_success_candidate_failure_pairs=0,
            qualifying_decision='provisional_candidate_recommendation', other_decision='keep_baseline_original',
            significance_claim=False),
        intervals=dict(success_rates='two-sided95% exact Clopper-Pearson per model on the valid paired denominator',
            paired_effect='report full discordant2x2 counts and exact two-sided paired sign/McNemar p; small fixed panel does not establish significance or population effect',
            discordant_direction='when discordants>0, exact Clopper-Pearson95% interval for candidate-only share among discordants'),
        source_sha256={str(p): sha(p) for p in sorted(source_files)},
        limitations=['Seeds are absent from checked local training/validation collection manifests and prior recorded metadata; this does not establish absence from official data or distinct physical scenes.',
            'Expert rejection is reported, never replaced. Valid pair count can be below10.',
            'Different checkpoints need not produce equal first predictions; exact initial condition is required.',
            'No changes to training, controller, or routing are made by this preparation. A provisional recommendation is not an activated routing configuration.',
            'Other48-task retention and broad generalization remain unassessed by this two-task panel.'],
        launch_argv=commands)
    write('protocol.json', protocol)
    write('freeze.json', dict(created_utc=protocol['created_utc'], files_sha256={name: sha(OUT / name) for name in
        ['cases.json', 'instruction_templates.json', 'checkpoint_reference.json', 'seed_overlap_audit.json', 'protocol.json']},
        wrapper_sha256=sha(wrapper)))
    (OUT / 'README.md').write_text('\n'.join([
        '# 固定新seed路由确认面板', '',
        '鼠标2102000–2102009、挂杯2103000–2103009，每例episode_index=0；原模型和24k模型均使用原控制器。', '',
        '本地训练/验证声明seed及已有outputs/results/artifacts元数据未发现重叠；不能据此证明与公开数据场景不重合。所有计划尝试保留，拒绝不补位。', '',
        '每任务预定条件：有效配对≥8，候选成功数严格更多，且原模型成功/候选失败为0；达到时仅暂列candidate推荐，其他情况保留baseline。不得称统计显著。', '',
        'protocol.json包含确切prepare/run argv。通过独立resource wrapper执行，禁止直接使用默认0.7显存设置的confirm.panel并跑。', '',
        '准备只进行了CPU文件/seed核验；未启动GPU。后续独立审计须核对输入NPY、binding、初始physical、checkpoint身份及返回码，主统计使用严格有效配对分母并报告精确区间。', '']))
    print(json.dumps(dict(output=str(OUT), local_manifests=len(local_checks), historical_metadata_files=len(paths),
                          collisions=0, cases=20, cases_sha256=protocol['cases_sha256'],
                          wrapper_sha256=sha(wrapper)), indent=2))


if __name__ == '__main__':
    main()
