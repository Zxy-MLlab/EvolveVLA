#!/usr/bin/env python3
"""CPU-only fixed-panel oracle and policy timing A/B audit; never launches GPU work."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_causal_diagnosis_summary_20261019 as prior

ROOT = Path('/root/autodl-tmp/EvolveVLA')
DEFAULT = ROOT/'outputs/gen1_timing_validation_20261020'
PANEL = ROOT/'outputs/gen1_causal_diagnosis_20261019/panel.json'
TASKS = ('place_mouse_pad', 'hanging_mug')
EXPERT_TERMINAL = {'completed', 'source_exhausted', 'eval_limit_reached'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, content):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.strip() and line.endswith('\n')]


def outcome(row, kind):
    """Classify outcome independently of False defaults present in error records."""
    if row is None:
        return 'pending'
    status = row.get('status')
    if status == 'expert_rejected':
        return 'expert_rejected'
    if status not in (EXPERT_TERMINAL if kind == 'expert' else {'completed'}):
        return 'error_or_excluded'
    if type(row.get('success')) is not bool or (kind != 'dense' and row.get('simulator_returncode') != 0):
        return 'invalid_terminal_record'
    if kind == 'expert' and status in ('source_exhausted', 'eval_limit_reached') and row['success']:
        return 'invalid_terminal_record'
    return 'success' if row['success'] else 'failure'


def source_exact(row):
    return prior.source_match(row or {})['tier'] == 'exact_reencoded_jpeg_and_state'


def physical(row):
    return ((row or {}).get('initial_parity') or {}).get('physical') or (row or {}).get('initial_physical')


def gate_passed(row):
    gate = (row or {}).get('expert_gate') or {}
    return gate.get('status') == 'passed' and gate.get('bypassed') is False


def row_report(row, kind):
    state = outcome(row, kind)
    exact, gate = source_exact(row), gate_passed(row)
    admission = 'observed_expert_gate'
    if kind == 'dense':
        gate = (row or {}).get('source_capture_count_equal') is True
        admission = 'stored_planner_positive_control_no_separate_expert_prefilter'
    return dict(status=(row or {}).get('status', 'pending'), outcome=state,
                success=(row or {}).get('success') if state in ('success', 'failure') else None,
                source_exact=exact, gate_passed_without_bypass=gate if kind != 'dense' else None,
                admission_valid=gate,
                admission_evidence=admission,
                protocol_eligible=state in ('success', 'failure') and exact and gate,
                timing_summary=(row or {}).get('timing_summary'),
                steps=(row or {}).get('steps'), result_path=(row or {}).get('_result_path'),
                error=(row or {}).get('error') or (row or {}).get('message'))


def counts(rows, planned):
    eligible = [r for r in rows if r['protocol_eligible']]
    successes = sum(r['success'] is True for r in eligible)
    return dict(planned=planned, recorded=sum(r['outcome'] != 'pending' for r in rows),
                outcomes=dict(Counter(r['outcome'] for r in rows)),
                raw_statuses=dict(Counter(r['status'] for r in rows)),
                eligible=len(eligible), successes=successes, failures=len(eligible)-successes,
                success_rate=successes/len(eligible) if eligible else None,
                excluded_source_or_gate=sum(r['outcome'] in ('success','failure') and not r['protocol_eligible'] for r in rows),
                fallback_actions=sum((r.get('timing_summary') or {}).get('fallback_actions',0) for r in rows),
                modified_fallback_actions=sum((r.get('timing_summary') or {}).get('modified_fallback_actions',0) for r in rows),
                cases_with_modified_fallback=sum((r.get('timing_summary') or {}).get('modified_fallback_actions',0)>0 for r in rows))


def basic_pair(a, b, kind, same_checkpoint=False):
    aa, bb = row_report(a, kind), row_report(b, kind)
    inputs = bool(a and b and a.get('hashes') and b.get('hashes') and a['hashes'] == b['hashes'])
    exact = aa['source_exact'] and bb['source_exact']
    physical_equal = bool(physical(a) and physical(b)) and physical(a) == physical(b)
    checkpoint_equal = bool(a and b and a.get('checkpoint_identity_sha256')
                            and a.get('checkpoint_identity_sha256') == b.get('checkpoint_identity_sha256')
                            and a.get('checkpoint') == b.get('checkpoint'))
    status = ('pending' if a is None or b is None else
              'expert_rejected' if 'expert_rejected' in (aa['outcome'], bb['outcome']) else
              'error_or_excluded' if not (aa['protocol_eligible'] and bb['protocol_eligible']) else
              'initial_physical_mismatch_or_missing' if not physical_equal else
              'input_mismatch_or_missing' if kind == 'policy' and not inputs else
              'checkpoint_mismatch_or_missing' if same_checkpoint and not checkpoint_equal else 'paired')
    return dict(status=status, original_status=aa['status'], aligned_status=bb['status'],
                original_success=aa['success'], aligned_success=bb['success'],
                source_inputs_exact=exact, observed_input_hashes_equal=inputs,
                initial_physical_equal=physical_equal, checkpoint_identity_equal=checkpoint_equal)


def pair_counts(rows):
    pp = [r for r in rows if r['status'] == 'paired']
    a = sum(r['original_success'] for r in pp)
    b = sum(r['aligned_success'] for r in pp)
    return dict(scheduled=len(rows), eligible_pairs=len(pp), statuses=dict(Counter(r['status'] for r in rows)),
                original_successes=a, aligned_successes=b,
                original_rate=a/len(pp) if pp else None, aligned_rate=b/len(pp) if pp else None,
                paired_delta=(b-a)/len(pp) if pp else None,
                original_only=sum(r['original_success'] and not r['aligned_success'] for r in pp),
                aligned_only=sum(not r['original_success'] and r['aligned_success'] for r in pp),
                both_success=sum(r['original_success'] and r['aligned_success'] for r in pp),
                both_failure=sum(not r['original_success'] and not r['aligned_success'] for r in pp), cases=rows)


def load_run(directory, expected_cases, modes, kind):
    manifest_path = directory/'manifest.json'
    manifest = read(manifest_path) if manifest_path.exists() else {}
    expected = {c['id']: c for c in expected_cases}
    manifest_cases = manifest.get('cases', [])
    if manifest and [c['id'] for c in manifest_cases] != list(expected):
        raise ValueError(f'Run fixed-panel identity/order differs: {directory}')
    for case in manifest_cases:
        fields=('task','seed','episode_index','prompt','source_hdf5_sha256')
        if any(case.get(k)!=expected[case['id']].get(k) for k in fields):
            raise ValueError(f'Run case contents differ from fixed panel: {case["id"]}')
    sources = {str(manifest_path): digest(manifest_path)} if manifest else {}
    indexed, unfinalized = {}, []
    for path in sorted(directory.rglob('result.json')) if directory.exists() else []:
        row = read(path)
        identity = row.get('id') or row.get('case_id')
        mode = row.get('condition') or row.get('timing_mode') or row.get('mode')
        if mode is None and len(modes) == 1:
            mode = modes[0]
        if mode is None and path.parent.name in modes:
            # The runners first write their core worker result, then atomically
            # add condition/timing metadata; do not consume that live gap.
            unfinalized.append(str(path))
            continue
        key = (identity, mode)
        if identity not in expected or mode not in modes or key in indexed:
            raise ValueError(f'Foreign/duplicate result {key}: {path}')
        row['_result_path'] = str(path)
        identity_errors=[]
        for field in ('task','seed','episode_index'):
            if row.get(field)!=expected[identity].get(field):
                identity_errors.append(field)
        if kind == 'expert' and row.get('source_hash') != expected[identity].get('source_hdf5_sha256'):
            identity_errors.append('source_hash')
        if identity_errors:
            row.update(status='case_identity_record_mismatch',error='Mismatched result identity fields: '+','.join(identity_errors))
        if kind == 'policy':
            parity = row.get('source_parity')
            if parity:
                row['initial_parity'] = dict(parity)
            physical_rows = jsonl(path.with_name('physical.jsonl'))
            if physical_rows:
                row['initial_physical'] = physical_rows[0].get('physical')
            if (manifest.get('checkpoint') or {}).get('identity_sha256') != row.get('checkpoint_identity_sha256'):
                row['status'] = 'checkpoint_identity_record_mismatch'
            if (manifest.get('checkpoint') or {}).get('path') != row.get('checkpoint'):
                row['status'] = 'checkpoint_path_record_mismatch'
        indexed[key] = row
        sources[str(path)] = digest(path)
        for name in ('trace.jsonl','physical.jsonl','timing.jsonl','retiming.jsonl','expert_gate.json','controller_patch.json'):
            source=path.with_name(name)
            if source.exists():
                sources[str(source)]=digest(source)
    planned = len(expected)*len(modes)
    state = ('not_started' if not manifest else 'complete' if len(indexed) == planned and (directory/'summary.json').exists() else 'partial')
    reports = {mode: {task: counts([row_report(indexed.get((c['id'], mode)), kind)
                   for c in expected_cases if c['task'] == task], sum(c['task']==task for c in expected_cases))
              for task in TASKS} for mode in modes}
    return dict(path=str(directory), state=state, planned=planned, recorded=len(indexed),
                cells=reports, unfinalized_result_paths=unfinalized,
                result_rows=[dict(id=identity,condition=mode,**row_report(row,kind)) for (identity,mode),row in indexed.items()],
                checkpoint_identity=manifest.get('checkpoint'), intervention=manifest.get('intervention'),
                code_identity=manifest.get('source_sha256') or manifest.get('code_sha256'),
                sources_sha256=sources, manifest=manifest), indexed


def indexed_rows(rows, field):
    indices = [r[field] for r in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f'Duplicate {field} in trace')
    return dict(zip(indices, rows))


def policy_comparison(a, b):
    result = basic_pair(a, b, 'policy', same_checkpoint=True)
    result.update(original_timing_summary=(a or {}).get('timing_summary'),
                  aligned_timing_summary=(b or {}).get('timing_summary'))
    if not a or not b:
        return result
    ap,bp=(Path(r['_result_path']).parent for r in (a,b))
    at,bt=(indexed_rows(jsonl(p/'timing.jsonl'),'action_index') for p in (ap,bp))
    aph,bph=(indexed_rows(jsonl(p/'physical.jsonl'),'step') for p in (ap,bp))
    activated=[i for i,r in bt.items() if r.get('modified_fallback_triggered') is True]
    first=min(activated) if activated else None
    expected=list(range(first)) if first is not None else sorted(at.keys() | bt.keys())
    missing=[i for i in expected if i not in at or i not in bt or i+1 not in aph or i+1 not in bph]
    present=[i for i in expected if i not in missing]
    targets=[i for i in present if at[i].get('target') is None or at[i].get('target')!=bt[i].get('target')]
    fields=('physics_steps','total_physics_steps','fallback_branches','controller_locals')
    timing=[i for i in present if any(at[i].get(k)!=bt[i].get(k) for k in fields)]
    phys=[i for i in present if not aph[i+1].get('physical') or aph[i+1].get('physical')!=bph[i+1].get('physical')]
    prefix_ok=not (missing or targets or timing or phys)
    initial_actions=[jsonl(p/'actions.jsonl') for p in (ap,bp)]
    first_predictions_equal=bool(all(initial_actions)) and initial_actions[0][0].get('actions')==initial_actions[1][0].get('actions')
    common=sorted(at.keys() & bt.keys())
    actual_changes=[i for i in common if at[i].get('physics_steps')!=bt[i].get('physics_steps')]
    changed_while_same_target=[i for i in actual_changes if at[i].get('target')==bt[i].get('target')]
    trace_complete=bool(at and bt) and all(sorted(t)==list(range(r.get('steps',-1))) for t,r in ((at,a),(bt,b)))
    result.update(modified_fallback_action_indices=activated, first_modified_fallback_action=first,
        before_first_activation=dict(verified=prefix_ok, compared_actions=len(present), missing_indices=missing,
            target_mismatch_indices=targets,timing_mismatch_indices=timing,physical_mismatch_indices=phys,
            scope='all preceding actions; if never activated, entire recorded action sequences'),
        first_full_predictions_equal=first_predictions_equal, timing_trace_complete=trace_complete,
        actual_duration_changed_indices=actual_changes, duration_changed_with_identical_target_indices=changed_while_same_target,
        full_trajectory_equal_when_no_activation=prefix_ok and trace_complete if first is None else None,
        mechanism_control_valid=result['status']=='paired' and prefix_ok and first_predictions_equal and trace_complete,
        mechanism_note='A modified fallback branch can be subordinate to the other arm duration; branch activation does not itself prove total action duration changed.')
    return result


def common_four_cases(cases,indexed):
    all_rows=[]
    for case in cases:
        rows={f'{label}/{mode}':indexed[label+'_policy_ab'].get((case['id'],mode))
              for label in ('baseline','candidate') for mode in ('original','fallback15')}
        reference=rows['baseline/original']
        comparisons={name:basic_pair(reference,row,'policy',same_checkpoint=name.startswith('baseline/'))
                     for name,row in rows.items()}
        statuses=[r['status'] for r in comparisons.values()]
        status='paired' if all(s=='paired' for s in statuses) else 'pending' if 'pending' in statuses else 'excluded'
        all_rows.append(dict(id=case['id'],task=case['task'],status=status,
                            comparisons=comparisons,outcomes={name:(row or {}).get('success') for name,row in rows.items()}))
    report={}
    for task in TASKS:
        group=[r for r in all_rows if r['task']==task]; valid=[r for r in group if r['status']=='paired']
        report[task]=dict(planned=len(group),common_eligible=len(valid),statuses=dict(Counter(r['status'] for r in group)),
            cells={name:dict(successes=sum(r['outcomes'][name] for r in valid),trials=len(valid),
                    rate=sum(r['outcomes'][name] for r in valid)/len(valid) if valid else None)
                   for name in ('baseline/original','baseline/fallback15','candidate/original','candidate/fallback15')},cases=group)
    return report


def expert_comparison(a, b, manifest):
    result = basic_pair(a, b, 'expert')
    if not a or not b:
        return result
    ap, bp = (Path(r['_result_path']).parent for r in (a,b))
    at, bt = prior.indexed_trace(ap/'trace.jsonl'), prior.indexed_trace(bp/'trace.jsonl')
    events = jsonl(bp/'retiming.jsonl')
    indices = [r['source_action_index'] for r in events]
    release = b.get('release') or {}
    start = release.get('start')
    prefix = prior.pretreatment_equality(at, bt, start)
    allowed = b.get('release_indices', [])
    valid_activations = bool(indices) and len(indices) == len(set(indices)) and set(indices).issubset(set(allowed))
    actual = []
    for index in indices:
        av, bv = at.get(index, {}), bt.get(index, {})
        actual.append(dict(index=index, original_ticks=av.get('timing', {}).get('physics_steps'),
                           aligned_ticks=bv.get('timing', {}).get('physics_steps'),
                           source_target_equal=av.get('source_target') is not None and av.get('source_target')==bv.get('source_target')))
    changed = [r['index'] for r in actual if type(r['original_ticks']) is int and type(r['aligned_ticks']) is int and r['original_ticks'] != r['aligned_ticks']]
    source_hash_equal = bool(a.get('source_hash')) and a.get('source_hash') == b.get('source_hash')
    patch_ok = manifest.get('method_patch', {}).get('only_four_qpos_fallback_literals_changed') is True
    causal = result['status']=='paired' and prefix['verified'] and source_hash_equal and patch_ok and valid_activations and bool(changed) and all(r['source_target_equal'] for r in actual)
    result.update(before_release=prefix, source_hash_equal=source_hash_equal,
                  only_four_fallback_literals_changed=patch_ok,
                  release=release, activation_indices=indices, activations_within_declared_release=valid_activations,
                  actual_treatment_actions=actual, observed_duration_changed_indices=changed,
                  controlled_duration_intervention=bool(causal),
                  mechanism_note='Only paired, matching-prefix, activated comparisons support duration attribution; unchanged success counts alone do not.')
    return result


def dense_comparison(dense, sparse):
    dd,ss=row_report(dense,'dense'),row_report(sparse,'expert')
    same_physics=bool(physical(dense) and physical(sparse)) and physical(dense)==physical(sparse)
    status=('pending' if dense is None or sparse is None else
            'expert_rejected' if ss['outcome']=='expert_rejected' else
            'error_or_excluded' if not (dd['protocol_eligible'] and ss['protocol_eligible']) else
            'initial_physical_mismatch_or_missing' if not same_physics else 'paired')
    return dict(status=status,original_status=dd['status'],aligned_status=ss['status'],
                original_success=dd['success'],aligned_success=ss['success'],initial_physical_equal=same_physics,
                source_inputs_exact=dd['source_exact'] and ss['source_exact'],
                original_condition='stored_dense_expert', candidate_condition='sparse_controller',
                limitation='Dense is stored planner positive control, without a separate expert prefilter; interpolation and timing both differ from sparse.')


def background():
    train = ROOT/'outputs/gen1_trainset_rollout_20261018'
    multi = ROOT/'outputs/gen1_all_tasks_clean10_20261018'
    report = {}
    if (train/'progress.json').exists():
        r=read(train/'progress.json'); report['trainset'] = dict(path=str(train/'progress.json'), sha256=digest(train/'progress.json'), complete=(train/'summary.json').exists(), **r)
    if (multi/'manifest.json').exists():
        manifest=read(multi/'manifest.json'); rows=[]
        for path in sorted(multi.glob('*_comparison.json')):
            if '_round_' in path.name:
                continue
            for task,data in read(path)['tasks'].items():
                rows.append(dict(task=task, **{k:data[k] for k in ('paired_completed','original_successes','candidate_successes','original_only','candidate_only')}, source=str(path), sha256=digest(path)))
        report['other_tasks'] = dict(path=str(multi), requested_tasks=manifest['tasks'], requested_task_count=len(manifest['tasks']),
              completed_task_count=len(rows), summary_present=(multi/'summary.json').exists(),
              failed=read(multi/'failed.json') if (multi/'failed.json').exists() else None,
              paired_completed=sum(r['paired_completed'] for r in rows),
              original_successes=sum(r['original_successes'] for r in rows), candidate_successes=sum(r['candidate_successes'] for r in rows), per_task=rows,
              non_target_tasks=dict(completed=sum(r['task'] not in TASKS for r in rows),
                  paired_completed=sum(r['paired_completed'] for r in rows if r['task'] not in TASKS),
                  original_successes=sum(r['original_successes'] for r in rows if r['task'] not in TASKS),
                  candidate_successes=sum(r['candidate_successes'] for r in rows if r['task'] not in TASKS)),
              limitation='Historical original controller, separate seeds and prompt protocol, incomplete 18-task panel; not a 50-task retention result or new timing experiment.')
    return report


def render(report):
    lines=['# 固定20案例：控制时序验证', '', f"更新UTC：{report['generated_utc']}", '',
           '**每任务10例按历史候选成功5例、失败5例分层选择；这里报告固定诊断面板成功率，不代表任务总体成功率。**', '',
           'source_exhausted / eval_limit_reached 是执行完成后的有效失败；专家拒绝、初始化异常、缺失或格式错误不计为策略失败。正式配对要求源JPEG/state精确匹配、初始记录物理状态一致；policy还要求初始输入哈希相同。', '']
    for name in ('expert_ab','expert_dense','baseline_policy_ab','candidate_policy_ab'):
        run=report[name]
        lines += [f"## {name}", '', f"状态 {run['state']}；结果 {run['recorded']}/{run['planned']}。", '',
                  '| 条件 | 任务 | 成功/有效 | 成功率 | 专家拒绝 | 错误/排除 | 待运行 |', '|---|---|---:|---:|---:|---:|---:|']
        for mode,tasks in run['cells'].items():
            for task,c in tasks.items():
                rate='—' if c['success_rate'] is None else f"{100*c['success_rate']:.1f}%"
                bad=sum(c['outcomes'].get(k,0) for k in ('error_or_excluded','invalid_terminal_record'))+c['excluded_source_or_gate']
                lines.append(f"| {mode} | {task} | {c['successes']}/{c['eligible']} | {rate} | {c['outcomes'].get('expert_rejected',0)} | {bad} | {c['outcomes'].get('pending',0)} |")
        lines += ['']
    for name, tasks in report['comparisons'].items():
        lines += [f"## 配对：{name}", '', '| 任务 | 可配对/固定面板 | 原条件成功 | 候选条件成功 | 原独有/候选独有 |', '|---|---:|---:|---:|---:|']
        for task,c in tasks.items():
            lines.append(f"| {task} | {c['eligible_pairs']}/{c['scheduled']} | {c['original_successes']} | {c['aligned_successes']} | {c['original_only']}/{c['aligned_only']} |")
        lines += ['']
    lines += ['## Policy四格共同可配对案例', '', '| 任务 | 共同案例 | 基线原控制 | 基线fallback15 | LoRA原控制 | LoRAfallback15 |', '|---|---:|---:|---:|---:|---:|']
    for task,row in report['policy_common_four'].items():
        n=row['common_eligible']
        cells=row['cells']
        lines.append(f"| {task} | {n}/{row['planned']} | "+' | '.join(f"{cells[name]['successes']}/{n}" for name in ('baseline/original','baseline/fallback15','candidate/original','candidate/fallback15'))+' |')
    lines += ['', '四格表统一使用四个条件都完成且初始输入/物理状态一致的案例；上面的各对比较表保留各自有效配对，分母可能不同。', '', '## 干预激活与前缀核对', '']
    for task,data in report['comparisons']['expert_original_vs_terminal15'].items():
        rows=data['cases']
        n=sum(r.get('controlled_duration_intervention') is True for r in rows)
        lines.append(f"- Oracle {task}：{n}/{data['scheduled']}例已确认初始状态、干预前完整前缀、源目标相同且实际时长改变。")
    for label in ('baseline','candidate'):
        for task,data in report['comparisons'][label+'_original_vs_fallback15'].items():
            activated=sum(bool(r.get('modified_fallback_action_indices')) for r in data['cases'])
            verified=sum(r.get('mechanism_control_valid') is True for r in data['cases'])
            lines.append(f"- Policy {label} {task}：{activated}例出现修改后fallback分支；{verified}例完成首个修改分支之前的目标、计时、物理轨迹和首次预测一致性检查。")
    lines += ['']
    lines += ['## 判读限制', '',
              '- Oracle只调整源轨迹末段静止关节释放区间；policy全局调整四处fallback常数，两者干预范围不同。',
              '- policy预测关节通常不完全静止，实际fallback触发情况必须与成功率一起报告；不能把常数修改直接当作动作时长已改变。',
              '- 原即时成功判据不变，正常成功会提前结束；本轮没有逐例评估成功后的长时间稳定性。',
              '- 不按结果补种子，不将专家拒绝重试到通过；缺失条件保留pending。', '',
              '完整逐例状态、配对排除原因、干预前缀与数据哈希见同名JSON。', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--panel', type=Path, default=PANEL)
    args=parser.parse_args(); root=args.root.resolve()
    panel=read(args.panel); cases=[c for c in panel['cases'] if c['selection_phase']=='initial']
    if Counter(c['task'] for c in cases) != Counter({'place_mouse_pad':10,'hanging_mug':10}):
        raise ValueError('Exactly 10+10 fixed initial cases required')
    report=dict(generated_utc=datetime.now(timezone.utc).isoformat(), panel=str(args.panel.resolve()),
                panel_sha256=digest(args.panel), cases=cases, comparisons={}, background=background())
    configs={'expert_ab':(('raw50','retime15'),'expert'), 'expert_dense':(('dense',),'dense'),
             'baseline_policy_ab':(('original','fallback15'),'policy'), 'candidate_policy_ab':(('original','fallback15'),'policy')}
    indexed={}
    for name,(modes,kind) in configs.items():
        report[name],indexed[name]=load_run(root/name,cases,modes,kind)
    er=report['expert_ab']
    report['comparisons']['expert_original_vs_terminal15']={task:pair_counts([
        dict(id=c['id'], **expert_comparison(indexed['expert_ab'].get((c['id'],'raw50')),indexed['expert_ab'].get((c['id'],'retime15')),er['manifest']))
        for c in cases if c['task']==task]) for task in TASKS}
    for mode in ('raw50','retime15'):
        report['comparisons']['dense_vs_'+mode]={task:pair_counts([
            dict(id=c['id'],**dense_comparison(indexed['expert_dense'].get((c['id'],'dense')),indexed['expert_ab'].get((c['id'],mode))))
            for c in cases if c['task']==task]) for task in TASKS}
    for label in ('baseline','candidate'):
        idx=indexed[label+'_policy_ab']
        report['comparisons'][label+'_original_vs_fallback15']={task:pair_counts([
            dict(id=c['id'], **policy_comparison(idx.get((c['id'],'original')),idx.get((c['id'],'fallback15'))))
            for c in cases if c['task']==task]) for task in TASKS}
    report['policy_common_four']=common_four_cases(cases,indexed)
    for mode in ('original','fallback15'):
        report['comparisons']['baseline_vs_candidate_'+mode]={task:pair_counts([
            dict(id=c['id'], **basic_pair(indexed['baseline_policy_ab'].get((c['id'],mode)),indexed['candidate_policy_ab'].get((c['id'],mode)),'policy'))
            for c in cases if c['task']==task]) for task in TASKS}
    for name in configs:
        report[name].pop('manifest',None)
    root.mkdir(parents=True,exist_ok=True)
    write(root/'summary.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write(root/'summary.md',render(report))
    print(json.dumps({name:{k:report[name][k] for k in ('state','recorded','planned')} for name in configs},ensure_ascii=False))


if __name__=='__main__':
    main()
