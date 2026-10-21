#!/usr/bin/env python3
"""Independent CPU audit of frozen novel-seed baseline/candidate original-control pairs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OUT = ROOT / 'outputs/gen1_control_repair_20261020/route_confirmation'
sys.path.insert(0, str(ROOT / 'scripts'))
import gen1_raw_paired_eval as binding


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.strip() and s.endswith('\n')]


def atomic(name, text):
    temp = OUT / (name + '.tmp')
    temp.write_text(text)
    temp.replace(OUT / name)


def interval(k, n):
    if not n:
        return None
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))]


def classification(row, finalized):
    if row is None or not finalized:
        return 'pending'
    if row.get('status') == 'expert_rejected':
        return 'expert_rejected'
    if row.get('status') != 'completed':
        return 'infrastructure_or_protocol_error'
    if row.get('simulator_returncode') != 0 or type(row.get('success')) is not bool:
        return 'invalid_terminal_record'
    return 'success' if row['success'] else 'failure'


def manifest_checks(manifest, protocol, label, cases):
    if not manifest:
        return {'present': False}
    return dict(present=True,
        checkpoint=manifest.get('checkpoint', {}).get('identity_sha256') == protocol['checkpoint_identities'][label]
            and manifest.get('checkpoint', {}).get('path') == protocol['checkpoint_paths'][label],
        fixed_cases=manifest.get('cases') == cases,
        original_control=manifest.get('controller') == 'original',
        fixed_runtime=manifest.get('reset_episode_index') == 0
            and manifest.get('execute_prefix') == 50 and manifest.get('policy_seed') == 0,
        no_retries=manifest.get('same_case_retries') == 0,
        no_replenishment=manifest.get('expert_rejection_replenishment') is False,
        original_sources=bool(manifest.get('source_sha256'))
            and all(sha(p) == value for p, value in manifest['source_sha256'].items()))


def inspect_case(case, label, protocol, manifest, global_errors, spec):
    directory = OUT / label / f'{case["task"]}_{case["seed"]}'
    result_path = directory / 'result.json'
    row = read(result_path) if result_path.exists() else None
    # Wrapper audit is the successful post-worker finalization marker. Errors may abort before it exists.
    audited = (directory / 'wrapper_audit.json').exists()
    exceptional = bool(row and row.get('status') not in ('completed', 'expert_rejected'))
    outcome = classification(row, audited or exceptional)
    report = dict(task=case['task'], seed=case['seed'], episode_index=case['episode_index'], model=label,
        outcome=outcome, status=(row or {}).get('status', 'pending'),
        success=(row or {}).get('success') if outcome in ('success', 'failure') else None,
        eligible=False, exclusions=[], output=str(directory))
    if outcome == 'pending':
        report['exclusions'] = ['pending']
        return report
    errors = list(global_errors)
    try:
        requested = read(directory / 'requested_case.json')
        expected_identity = protocol['checkpoint_identities'][label]
        if requested != dict(case, checkpoint_identity_sha256=expected_identity):
            errors.append('requested_identity_not_frozen_case')
        if row.get('task') != case['task'] or row.get('seed') != case['seed'] or row.get('checkpoint') != protocol['checkpoint_paths'][label]:
            errors.append('result_identity_not_requested')
        wrapper = read(directory / 'wrapper_audit.json') if audited else {}
        if not (wrapper.get('checkpoint_identity_sha256') == expected_identity
                and all(wrapper.get(k) is True for k in ['source_unchanged', 'checkpoint_inventory_unchanged', 'requested_identity_matches'])):
            errors.append('wrapper_integrity_missing_or_failed')
        if outcome not in ('success', 'failure'):
            errors.append(outcome)
        else:
            b = row.get('binding') or {}
            expected_binding = binding.bind_prompt(spec, {'info': b['fields']}, ROOT / 'robotwin_repo_ba0fd21/description')
            if b != expected_binding or b.get('template') != spec:
                errors.append('binding_not_frozen_template_and_scene_fields')
            images = {camera: np.load(directory / 'initial' / f'{camera}.npy', allow_pickle=False)
                      for camera in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}
            state = np.load(directory / 'initial/state.npy', allow_pickle=False)
            hashes = {key: dict(shape=list(a.shape), sha256=hashlib.sha256(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).hexdigest())
                      for key, a in images.items()}
            hashes.update(state=hashlib.sha256(np.asarray(state, dtype='<f4').reshape(14).tobytes()).hexdigest(),
                          prompt=hashlib.sha256(b['prompt'].encode()).hexdigest())
            if hashes != row.get('hashes') or not np.isfinite(state).all():
                errors.append('saved_input_hashes_or_state_invalid')
            trace = rows(directory / 'physical.jsonl')
            if [r['step'] for r in trace] != list(range(row['steps'] + 1)):
                errors.append('physical_trace_incomplete_or_noncontiguous')
            if not trace or trace[-1].get('success') != row['success']:
                errors.append('final_physical_success_disagrees')
            report.update(binding=b, input_hashes=hashes, initial_physical=trace[0]['physical'] if trace else None,
                          steps=row['steps'], checkpoint_identity=expected_identity,
                          artifacts_sha256={p.name: sha(p) for p in [result_path, directory / 'requested_case.json',
                              directory / 'wrapper_audit.json', directory / 'physical.jsonl'] if p.exists()})
    except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
        errors.append(f'artifact_error:{type(error).__name__}:{error}')
    report.update(eligible=not errors, exclusions=sorted(set(errors)))
    return report


def pair(a, b):
    errors = []
    if a['outcome'] == 'pending' or b['outcome'] == 'pending':
        errors.append('pending')
    elif a['outcome'] == 'expert_rejected' or b['outcome'] == 'expert_rejected':
        errors.append('expert_rejected')
    elif not a['eligible'] or not b['eligible']:
        errors.append('condition_excluded')
    else:
        if not a.get('input_hashes') or a['input_hashes'] != b.get('input_hashes'):
            errors.append('initial_input_mismatch_or_missing')
        if not a.get('binding') or a['binding'] != b.get('binding'):
            errors.append('instruction_binding_mismatch_or_missing')
        if not a.get('initial_physical') or a['initial_physical'] != b.get('initial_physical'):
            errors.append('initial_physical_mismatch_or_missing')
    return dict(task=a['task'], seed=a['seed'], episode_index=a['episode_index'],
        baseline_outcome=a['outcome'], candidate_outcome=b['outcome'],
        baseline_success=a['success'], candidate_success=b['success'], valid_pair=not errors, exclusions=errors)


def summarize(pairs):
    valid = [p for p in pairs if p['valid_pair']]
    n = len(valid)
    baseline = sum(p['baseline_success'] for p in valid)
    candidate = sum(p['candidate_success'] for p in valid)
    improved = [p['seed'] for p in valid if not p['baseline_success'] and p['candidate_success']]
    regressed = [p['seed'] for p in valid if p['baseline_success'] and not p['candidate_success']]
    discordant = len(improved) + len(regressed)
    complete = not any('pending' in p['exclusions'] for p in pairs)
    meets = n >= 8 and candidate > baseline and not regressed
    return dict(scheduled_pairs=len(pairs), scheduled_attempts_finalized=complete, valid_pairs=n,
        baseline_successes=baseline, candidate_successes=candidate,
        baseline_rate=baseline / n if n else None, candidate_rate=candidate / n if n else None,
        baseline_exact95=interval(baseline, n), candidate_exact95=interval(candidate, n),
        paired_delta=(candidate - baseline) / n if n else None,
        candidate_only_seeds=improved, baseline_only_seeds=regressed,
        both_successes=sum(p['baseline_success'] and p['candidate_success'] for p in valid),
        both_failures=sum(not p['baseline_success'] and not p['candidate_success'] for p in valid),
        candidate_share_among_discordants_exact95=interval(len(improved), discordant),
        exact_two_sided_paired_p=min(1., 2 * sum(comb(discordant, k) for k in range(min(len(improved), len(regressed)) + 1)) / 2 ** discordant) if discordant else 1.,
        exclusions=dict(Counter(e for p in pairs for e in p['exclusions'])),
        provisional_rule_met=meets if complete else None,
        route_recommendation=('pending_complete_fixed_panel' if not complete else
                              'provisional_candidate_recommendation' if meets else 'keep_baseline_original'),
        significance_claim=False)


def main():
    protocol, frozen = read(OUT / 'protocol.json'), read(OUT / 'freeze.json')
    cases = read(OUT / 'cases.json')
    templates = {(r['case']['task'], r['case']['seed']): r['template_spec'] for r in read(OUT / 'instruction_templates.json')}
    global_checks = dict(frozen_files=all(sha(OUT / p) == value for p, value in frozen['files_sha256'].items()),
        frozen_protocol_sources=all(sha(p) == value for p, value in protocol['source_sha256'].items()),
        cases_hash=sha(OUT / 'cases.json') == protocol['cases_sha256'])
    model_runs, conditions = {}, {}
    for label in ['baseline', 'candidate']:
        run = OUT / label
        manifest = read(run / 'manifest.json') if (run / 'manifest.json').exists() else {}
        checks = manifest_checks(manifest, protocol, label, cases)
        errors = ['global_' + k for k, v in global_checks.items() if not v] + ['manifest_' + k for k, v in checks.items() if not v]
        conditions[label] = [inspect_case(case, label, protocol, manifest, errors, templates[case['task'], case['seed']]) for case in cases]
        model_runs[label] = dict(manifest_present=bool(manifest), manifest_checks=checks,
            summary_present=(run / 'summary.json').exists(), outcomes=dict(Counter(c['outcome'] for c in conditions[label])),
            finalized=sum(c['outcome'] != 'pending' for c in conditions[label]), planned=len(cases), conditions=conditions[label])
    pairs = [pair(a, b) for a, b in zip(conditions['baseline'], conditions['candidate'], strict=True)]
    totals = {task: summarize([p for p in pairs if p['task'] == task]) for task in ['place_mouse_pad', 'hanging_mug']}
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), gpu_used=False, original_control=True,
        audit_source_sha256=sha(__file__), global_checks=global_checks, runs=model_runs, pairs=pairs, per_task=totals,
        interval_method='Two-sided95% exact Clopper-Pearson on paired denominator; paired discordance direction interval conditional on discordants.',
        limitations=['Only scheduled new local seeds are evaluated; experts may reject and seeds are never replenished.',
            'No claim of unseen official data or independent physical scenes follows from seed inequality.',
            'Routing rule is prespecified and descriptive; recommendation is provisional and does not assert statistical significance.',
            'The 20 training-scene control confirmation cases are not read by this auditor.',
            'Checkpoint integrity uses prior full-file verification and per-case wrapper inventory checks; auditor does not reread all model shards.',
            'Equal observed inputs and initial physical records do not establish equality of all hidden physics states.',
            'Different checkpoint first predictions are allowed; they are the treatment difference, not an input mismatch.',
            'Task recommendations remain pending until all ten scheduled pairs for that task have final statuses.'])
    atomic('independent_summary.json', json.dumps(report, indent=2) + '\n')
    md = ['# 固定新seed：原模型与24k模型独立配对审计', '', f'更新时间：{report["created_utc"]}', '',
        '| 任务 | 有效配对 | 原模型成功 | 候选成功 | 原模型精确95%区间 | 候选精确95%区间 | 候选新增成功seed | 原模型独有成功seed | 预定路由建议 |',
        '|---|---:|---:|---:|---|---|---|---|---|']
    fmt = lambda x: '—' if x is None else f'{100*x[0]:.1f}%–{100*x[1]:.1f}%'
    for task, row in totals.items():
        md.append(f'| {task} | {row["valid_pairs"]}/10 | {row["baseline_successes"]} | {row["candidate_successes"]} | '
            f'{fmt(row["baseline_exact95"])} | {fmt(row["candidate_exact95"])} | {row["candidate_only_seeds"]} | '
            f'{row["baseline_only_seeds"]} | {row["route_recommendation"]} |')
    md += ['', '| task/seed | 原模型 | 候选 | 有效配对 | 排除项 |', '|---|---|---|---|---|']
    for p in pairs:
        md.append(f'| {p["task"]}/{p["seed"]} | {p["baseline_outcome"]} | {p["candidate_outcome"]} | '
                  f'{p["valid_pair"]} | {", ".join(p["exclusions"]) or "—"} |')
    md += ['', '预定规则：每任务有效配对≥8、候选成功数严格更多、原模型成功转候选失败为0；全部计划case结束后才给临时建议。拒绝与异常不当作失败、不补seed。', '',
           '两模型均使用原控制器；新seed只证明不在已核对本地记录中，不能证明不在公开数据或物理场景完全独立。样本小，区间宽，暂时建议不代表统计显著。', '']
    atomic('independent_summary.md', '\n'.join(md))
    print(json.dumps(dict(finalized={k: v['finalized'] for k, v in model_runs.items()}, checks=global_checks, per_task=totals), indent=2))


if __name__ == '__main__':
    main()
