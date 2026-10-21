#!/usr/bin/env python3
"""CPU-only summary of fixed-case expert controls and execution-prefix probes.

Each run remains separate. Pair eligibility requires matching visible inputs
and the recorded initial physical trace; missing results remain pending.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/EvolveVLA')
DEFAULT = ROOT / 'outputs/gen1_causal_diagnosis_20261019'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_state(directory, planned, recorded):
    summary_present = (directory / 'summary.json').exists()
    return dict(run_complete=recorded == planned and summary_present,
                summary_artifact_present=summary_present,
                run_state=('complete' if recorded == planned and summary_present else
                           'awaiting_summary' if recorded == planned else 'partial'),
                remaining_results=planned - recorded)


def physical_hash(result_path):
    trace = Path(result_path).with_name('physical.jsonl')
    if not trace.exists():
        return None
    with trace.open() as stream:
        first = stream.readline().strip()
    return hashlib.sha256(first.encode()).hexdigest() if first else None


def physical_equal(a, b):
    aa, bb = a.get('initial_physical_sha256'), b.get('initial_physical_sha256')
    return bool(aa and bb) and aa == bb


def compatible_checkpoint(a, b):
    if not a or not b:
        return False
    path_equal = bool(a.get('checkpoint')) and a.get('checkpoint') == b.get('checkpoint')
    identities = [r.get('checkpoint_identity_sha256') for r in (a, b)]
    return path_equal and bool(identities[0] and identities[1]) and identities[0] == identities[1]


def input_equal(a, b):
    return bool(a.get('hashes') and b.get('hashes')) and a['hashes'] == b['hashes']


def completed(row):
    return row is not None and row.get('status') == 'completed' and type(row.get('success')) is bool


def paired_case(baseline, candidate):
    status = ('pending' if baseline is None or candidate is None else
              'incomplete_or_excluded' if not (completed(baseline) and completed(candidate)) else
              'initial_mismatch_or_missing' if not (input_equal(baseline, candidate)
                                                  and physical_equal(baseline, candidate)) else 'paired')
    return dict(status=status, baseline_status=baseline.get('status') if baseline else None,
                candidate_status=candidate.get('status') if candidate else None,
                baseline_success=baseline.get('success') if baseline else None,
                candidate_success=candidate.get('success') if candidate else None,
                observed_inputs_equal=input_equal(baseline or {}, candidate or {}),
                initial_physical_trace_equal=physical_equal(baseline or {}, candidate or {}))


def pair_counts(rows):
    valid = [r for r in rows if r['status'] == 'paired']
    return dict(scheduled=len(rows), valid_pairs=len(valid),
                statuses=dict(Counter(r['status'] for r in rows)),
                baseline_successes=sum(r['baseline_success'] for r in valid),
                candidate_successes=sum(r['candidate_success'] for r in valid),
                both_success=sum(r['baseline_success'] and r['candidate_success'] for r in valid),
                candidate_only_success=sum(not r['baseline_success'] and r['candidate_success'] for r in valid),
                baseline_only_success=sum(r['baseline_success'] and not r['candidate_success'] for r in valid),
                both_failure=sum(not r['baseline_success'] and not r['candidate_success'] for r in valid))


def source_match(row):
    initial = row.get('initial_parity') or {}
    images = initial.get('images') or {}
    expected = {'cam_high', 'cam_left_wrist', 'cam_right_wrist'}
    exact_jpeg = set(images) == expected and all(v.get('jpeg_bytes_equal') is True for v in images.values())
    state_error = initial.get('state_max_abs')
    exact_state = state_error is not None and state_error == 0
    approximate = initial.get('observed_source_match') is True
    tier = ('exact_reencoded_jpeg_and_state' if exact_jpeg and exact_state else
            'approximate_only' if approximate else
            'mismatch' if initial else 'unavailable')
    return dict(tier=tier, exact_reencoded_jpeg_all_cameras=exact_jpeg,
                exact_drive_target_state=exact_state, state_max_abs=state_error,
                approximate_threshold_passed=approximate,
                image_mae={k: v.get('mae') for k, v in images.items()})


def physics_timing(row):
    timing = row.get('final_timing') or {}
    ticks = timing.get('total_physics_steps', row.get('physics_steps'))
    method = timing.get('method', row.get('timing_method'))
    valid = isinstance(ticks, int) and ticks > 0 and bool(method)
    return dict(valid_counter=valid, physics_steps=ticks,
                timestep=timing.get('timestep', row.get('timestep')),
                simulation_seconds=timing.get('simulation_seconds', row.get('simulation_seconds')),
                method=method,
                limitation=None if valid else 'Missing/zero legacy counter is not a valid duration measurement.')


def load_run(directory, panel_lookup):
    manifest = read(directory / 'manifest.json')
    cases = manifest.get('cases', [])
    if not isinstance(cases, list) or not all(isinstance(c, dict) and c.get('id') in panel_lookup for c in cases):
        raise ValueError(f'Invalid or foreign cases: {directory}')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError(f'Duplicate scheduled cases: {directory}')
    rows, sources = [], {str(directory / 'manifest.json'): sha(directory / 'manifest.json')}
    for path in sorted(directory.rglob('result.json')):
        row = read(path)
        if row.get('id') not in {c['id'] for c in cases}:
            raise ValueError(f'Foreign result: {path}')
        row = dict(row, result_path=str(path))
        if row.get('initial_physical_sha256') is None:
            observed = physical_hash(path)
            if observed is not None:
                row['initial_physical_sha256'] = observed
        rows.append(row)
        sources[str(path)] = sha(path)
    return manifest, cases, rows, sources


def horizon_run(directory, panel_lookup):
    manifest, cases, rows, sources = load_run(directory, panel_lookup)
    prefixes = manifest['prefixes']
    by_key = {}
    for row in rows:
        key = (row['id'], row['execute_prefix'])
        if key in by_key or row['execute_prefix'] not in prefixes:
            raise ValueError(f'Duplicate/foreign prefix result: {key}')
        by_key[key] = row
    comparisons = {}
    for prefix in prefixes:
        if prefix == 50:
            continue
        tasks = {}
        for task in dict.fromkeys(c['task'] for c in cases):
            pair_rows = [dict(id=c['id'], task=task,
                             **paired_case(by_key.get((c['id'], 50)), by_key.get((c['id'], prefix))))
                         for c in cases if c['task'] == task]
            for item in pair_rows:
                same_model = compatible_checkpoint(by_key.get((item['id'], 50)),
                                                   by_key.get((item['id'], prefix)))
                item['checkpoint_identity_equal'] = same_model
                if item['status'] == 'paired' and not same_model:
                    item['status'] = 'checkpoint_mismatch_or_missing'
            tasks[task] = dict(**pair_counts(pair_rows), cases=pair_rows)
        comparisons[str(prefix)] = tasks
    repeat_rows = []
    for case in cases:
        historical_path = panel_lookup[case['id']].get('known_candidate_result')
        historical = read(historical_path) if historical_path and Path(historical_path).exists() else None
        if historical is not None:
            historical['initial_physical_sha256'] = physical_hash(historical_path)
            sources[historical_path] = sha(historical_path)
        current = by_key.get((case['id'], 50))
        item = paired_case(historical, current)
        same_checkpoint = bool(historical and current and historical.get('checkpoint') == current.get('checkpoint'))
        if item['status'] == 'paired' and not same_checkpoint:
            item['status'] = 'checkpoint_mismatch_or_missing'
        repeat_rows.append(dict(id=case['id'], task=case['task'], **item,
                                same_checkpoint_path=same_checkpoint,
                                outcome_equal=(historical['success'] == current['success'])
                                if completed(historical) and completed(current) else None))
    repeats = {task: dict(**pair_counts([r for r in repeat_rows if r['task'] == task]),
                          cases=[r for r in repeat_rows if r['task'] == task])
               for task in dict.fromkeys(c['task'] for c in cases)}
    return dict(path=str(directory), planned=len(cases) * len(prefixes),
                recorded=len(rows), prefixes=prefixes,
                **completion_state(directory, len(cases) * len(prefixes), len(rows)),
                statuses=dict(Counter(r['status'] for r in rows)),
                paired_vs_50=comparisons, repeat50_vs_historical=repeats,
                result_rows=[{k: r.get(k) for k in ('id', 'task', 'execute_prefix', 'status', 'success',
                             'steps', 'inference_count', 'inference_seconds', 'result_path')} for r in rows],
                sources_sha256=sources,
                noise_note=manifest.get('noise_note'),
                limitations=['Pairs require both completed plus equal initial camera/state/prompt hashes and physical trace hash.',
                             'The physical trace contains recorded poses/joints, not all hidden physics state.',
                             'Inference totals include first-call compilation where applicable; they are not comparable steady-state latency estimates.',
                             'Historical checkpoint identity is checked by path; immutable weight hashes were not recorded in that historical rollout.'])


def oracle_run(directory, panel_lookup, dense=False):
    manifest, cases, rows, sources = load_run(directory, panel_lookup)
    indexed = {}
    checks = {r['id']: r for r in manifest.get('preflight', [])}
    for row in rows:
        if row['id'] in indexed:
            raise ValueError(f'Duplicate oracle result: {row["id"]}')
        indexed[row['id']] = row
    reports = []
    for case in cases:
        row = indexed.get(case['id'])
        reports.append(dict(id=case['id'], task=case['task'],
                            status=row.get('status') if row else 'pending',
                            success=row.get('success') if row else None,
                            steps=row.get('steps') if row else None,
                            source_observation_match=source_match(row or {}),
                            expert_gate=row.get('expert_gate') if row else None,
                            cpu_label_check=checks.get(case['id']),
                            source_capture_count_equal=row.get('source_capture_count_equal') if row else None,
                            physics_timing=physics_timing(row or {}),
                            initial_physical=(row or {}).get('initial_parity', {}).get('physical'),
                            result_path=row.get('result_path') if row else None))
    tasks = {}
    for task in dict.fromkeys(c['task'] for c in cases):
        rr = [r for r in reports if r['task'] == task]
        tasks[task] = dict(planned=len(rr), recorded=sum(r['status'] != 'pending' for r in rr),
                           successes=sum(r['success'] is True for r in rr),
                           statuses=dict(Counter(r['status'] for r in rr)),
                           exact_source_observation_cases=sum(r['source_observation_match']['tier'] == 'exact_reencoded_jpeg_and_state' for r in rr),
                           approximate_only_cases=sum(r['source_observation_match']['tier'] == 'approximate_only' for r in rr))
    return dict(path=str(directory), kind='dense_expert_control' if dense else 'sparse_expert_labels',
                planned=len(cases), recorded=len(rows), tasks=tasks, cases=reports,
                **completion_state(directory, len(cases), len(rows)),
                sources_sha256=sources,
                limitations=['Success counts describe deliberately selected diagnostic cases, not population success rates.',
                             'Exact source image parity means identical re-encoded stored JPEG bytes, not equality of all raw pixels or hidden physics.',
                             'CPU normalization roundtrip checks use the reference formula, not a production OpenPI transform invocation.',
                             'Sparse v1 timing counters can be invalid zero; no timing conclusions are drawn here.'])


def dense_sparse_pairs(dense_runs, sparse_runs):
    reports = []
    for dense in dense_runs:
        dd = {r['id']: r for r in dense['cases']}
        for sparse in sparse_runs:
            ss = {r['id']: r for r in sparse['cases']}
            rows = []
            for name in sorted(dd.keys() & ss.keys()):
                a, b = dd[name], ss[name]
                exact_source = all(r['source_observation_match']['tier'] == 'exact_reencoded_jpeg_and_state' for r in (a, b))
                phys_equal = bool(a['initial_physical'] and b['initial_physical']) and a['initial_physical'] == b['initial_physical']
                eligible = (a['status'] == 'completed' and a['source_capture_count_equal'] is True
                            and b['status'] in ('completed', 'source_exhausted', 'eval_limit_reached')
                            and exact_source and phys_equal)
                rows.append(dict(id=name, task=a['task'], dense_status=a['status'], sparse_status=b['status'],
                                 dense_success=a['success'], sparse_success=b['success'],
                                 both_exact_source_observation=exact_source,
                                 initial_recorded_physical_equal=phys_equal,
                                 dense_capture_count_equal=a['source_capture_count_equal'],
                                 dense_physics_timing=a['physics_timing'], sparse_physics_timing=b['physics_timing'],
                                 sparse_to_dense_physics_step_ratio=(b['physics_timing']['physics_steps'] / a['physics_timing']['physics_steps'])
                                 if a['physics_timing']['valid_counter'] and b['physics_timing']['valid_counter'] else None,
                                 controlled_comparison_available=eligible,
                                 dense_success_sparse_failure=eligible and a['success'] is True and b['success'] is False))
            reports.append(dict(dense_run=dense['path'], sparse_run=sparse['path'], cases=rows,
                                limitation='Compare source capture count and reset parity before interpreting a dense/sparse discrepancy; controllers differ in timing and interpolation.'))
    return reports


def sparse_repeat_pairs(sparse_runs):
    reports = []
    for index, left in enumerate(sparse_runs):
        aa = {row['id']: row for row in left['cases']}
        for right in sparse_runs[index + 1:]:
            bb = {row['id']: row for row in right['cases']}
            for name in sorted(aa.keys() & bb.keys()):
                a, b = aa[name], bb[name]
                if not a['result_path'] or not b['result_path']:
                    continue
                ap, bp = Path(a['result_path']).with_name('trace.jsonl'), Path(b['result_path']).with_name('trace.jsonl')
                at, bt = indexed_trace(ap), indexed_trace(bp)
                union = sorted(at.keys() | bt.keys())
                missing = [i for i in union if i not in at or i not in bt]
                present = [i for i in union if i not in missing]
                phys_diff = [i for i in present if not at[i].get('physical') or at[i].get('physical') != bt[i].get('physical')]
                target_diff = [i for i in present if at[i].get('source_target') is None or at[i].get('source_target') != bt[i].get('source_target')]
                initial_equal = bool(a['initial_physical'] and b['initial_physical']) and a['initial_physical'] == b['initial_physical']
                exact_source = all(r['source_observation_match']['tier'] == 'exact_reencoded_jpeg_and_state' for r in (a, b))
                full = bool(union) and not (missing or phys_diff or target_diff) and union == list(range(max(union) + 1))
                reports.append(dict(id=name, task=a['task'], left_run=left['path'], right_run=right['path'],
                                    left_status=a['status'], right_status=b['status'],
                                    left_success=a['success'], right_success=b['success'],
                                    compared_actions=len(present), missing_indices=missing,
                                    physical_mismatch_indices=phys_diff, source_target_mismatch_indices=target_diff,
                                    both_exact_source_observation=exact_source, initial_recorded_physical_equal=initial_equal,
                                    complete_recorded_trajectory_equal=full and exact_source and initial_equal,
                                    trace_sha256={str(p): sha(p) for p in (ap, bp) if p.exists()},
                                    limitation='Physics and source-target records compared; timing fields intentionally excluded because legacy v1 counters are invalid. No hidden contacts or velocities were recorded.'))
    return reports


def indexed_trace(path):
    if not path.exists():
        return {}
    lines = path.read_text().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.strip() and line.endswith('\n')]
    indices = [row['source_action_index'] for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError(f'Duplicate action indices in {path}')
    return {row['source_action_index']: row for row in rows}


def pretreatment_equality(baseline_trace, treated_trace, first_treated_index):
    if first_treated_index is None:
        return dict(verified=False, reason='No recorded treatment activation')
    expected = list(range(first_treated_index))
    missing = [i for i in expected if i not in baseline_trace or i not in treated_trace]
    present = [i for i in expected if i not in missing]
    physical_differences = [i for i in present if not baseline_trace[i].get('physical')
                            or baseline_trace[i].get('physical') != treated_trace[i].get('physical')]
    target_differences = [i for i in present if baseline_trace[i].get('source_target') is None
                         or baseline_trace[i].get('source_target') != treated_trace[i].get('source_target')]
    timing_differences = [i for i in present if not baseline_trace[i].get('timing')
                         or baseline_trace[i].get('timing') != treated_trace[i].get('timing')]
    payload = lambda trace: [{k: trace[i].get(k) for k in ('source_action_index', 'source_target', 'physical', 'timing')}
                             for i in present]
    digest = lambda trace: hashlib.sha256(json.dumps(payload(trace), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(verified=not (missing or physical_differences or target_differences or timing_differences),
                first_treated_index=first_treated_index, expected_action_indices=expected,
                compared_actions=len(present), missing_indices=missing,
                physical_mismatch_indices=physical_differences, source_target_mismatch_indices=target_differences,
                timing_mismatch_indices=timing_differences,
                baseline_prefix_sha256=digest(baseline_trace), treated_prefix_sha256=digest(treated_trace))


def retime_run(directory, panel_lookup, baseline_directory):
    manifest, cases, rows, sources = load_run(directory, panel_lookup)
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f'Duplicate retime results in {directory}')
    baseline_manifest_path = baseline_directory / 'manifest.json'
    baseline_manifest = read(baseline_manifest_path) if baseline_manifest_path.exists() else {}
    baseline_cases = {case['id']: case for case in baseline_manifest.get('cases', [])}
    eligibility = {item['id']: item.get('eligible_source_indices', []) for item in manifest.get('preflight', [])}
    if baseline_manifest_path.exists():
        sources[str(baseline_manifest_path)] = sha(baseline_manifest_path)
    reports = []
    for case in cases:
        row = by_id.get(case['id'])
        baseline_path = baseline_directory / case['id'] / 'result.json'
        baseline = read(baseline_path) if baseline_path.exists() else {}
        if baseline:
            sources[str(baseline_path)] = sha(baseline_path)
        treated_path = directory / case['id']
        retiming = (row or {}).get('retiming') or {}
        event_path = treated_path / 'retiming.jsonl'
        events = retiming.get('activations')
        if events is None:
            events = [json.loads(line) for line in event_path.read_text().splitlines(keepends=True) if line.strip() and line.endswith('\n')] if event_path.exists() else []
        activated = [event['source_action_index'] for event in events]
        if len(set(activated)) != len(activated):
            raise ValueError(f'Duplicate treatment activation: {case["id"]}')
        bt = indexed_trace(baseline_path.with_name('trace.jsonl'))
        tt = indexed_trace(treated_path / 'trace.jsonl')
        for path in (baseline_path.with_name('trace.jsonl'), treated_path / 'trace.jsonl', event_path):
            if path.exists():
                sources[str(path)] = sha(path)
        prefix = pretreatment_equality(bt, tt, min(activated) if activated else None)
        initial_a = (baseline.get('initial_parity') or {}).get('physical')
        initial_b = ((row or {}).get('initial_parity') or {}).get('physical')
        initial_equal = bool(initial_a and initial_b) and initial_a == initial_b
        both_exact_source = all(source_match(r)['tier'] == 'exact_reencoded_jpeg_and_state' for r in (baseline, row or {}))
        method_patch = retiming.get('method_patch') or manifest.get('method_patch') or {}
        isolated_duration_change = method_patch.get('only_four_qpos_fallback_literals_changed') is True
        source_hash = case.get('source_hdf5_sha256', case.get('source_sha256'))
        baseline_case = baseline_cases.get(case['id'], {})
        baseline_source_hash = baseline_case.get('source_hdf5_sha256', baseline_case.get('source_sha256'))
        same_source = bool(source_hash and baseline_source_hash) and source_hash == baseline_source_hash
        applied = []
        for event in events:
            index = event['source_action_index']
            baseline_timing, treated_timing = bt.get(index, {}).get('timing', {}), tt.get(index, {}).get('timing', {})
            applied.append(dict(**event,
                                baseline_actual_physics_steps=baseline_timing.get('physics_steps'),
                                treated_actual_physics_steps=treated_timing.get('physics_steps'),
                                source_target_equal=bt.get(index, {}).get('source_target') == tt.get(index, {}).get('source_target')
                                if index in bt and index in tt else None))
        changed_duration = [event['source_action_index'] for event in applied
                            if type(event['baseline_actual_physics_steps']) is int
                            and type(event['treated_actual_physics_steps']) is int
                            and event['baseline_actual_physics_steps'] != event['treated_actual_physics_steps']]
        allowed_indices = eligibility.get(case['id'], [])
        planned_treatment = (bool(activated) and set(activated).issubset(set(allowed_indices))
                             and min(activated) >= manifest.get('intervention', {}).get('retime_from_index', 0)
                             and retiming.get('config') == manifest.get('intervention'))
        terminal_statuses = ('completed', 'source_exhausted', 'eval_limit_reached')
        controlled = bool(row and row.get('status') in terminal_statuses and baseline.get('status') in terminal_statuses
                          and isolated_duration_change and planned_treatment and changed_duration and prefix['verified'] and initial_equal and both_exact_source and same_source
                          and all(event['source_target_equal'] is True for event in applied)
                          and physics_timing(baseline)['valid_counter'] and physics_timing(row)['valid_counter'])
        reports.append(dict(id=case['id'], task=case['task'], status=(row or {}).get('status', 'pending'),
                            success=(row or {}).get('success'), baseline_status=baseline.get('status'),
                            baseline_success=baseline.get('success'), baseline_result=str(baseline_path),
                            treatment_result=(row or {}).get('result_path'), intervention=manifest.get('intervention'),
                            activated_indices=activated, actual_treatment_actions=applied,
                            observed_duration_changed_indices=changed_duration,
                            planned_eligible_source_indices=allowed_indices, treatment_matches_plan=planned_treatment,
                            source_hdf5_sha256=source_hash, baseline_source_hdf5_sha256=baseline_source_hash,
                            source_hdf5_hash_equal=same_source,
                            method_patch=method_patch, before_treatment=prefix,
                            initial_recorded_physical_equal=initial_equal, both_exact_source_observation=both_exact_source,
                            baseline_physics_timing=physics_timing(baseline), treated_physics_timing=physics_timing(row or {}),
                            controlled_within_case_comparison=controlled,
                            outcome_changed_after_matched_prefix=(baseline.get('success') != row.get('success')) if controlled else None,
                            interpretation='Within-case evidence only; do not attribute a change to duration if the pre-treatment physical/target/timing prefix differs or is missing. This official run stops at the unchanged instantaneous benchmark predicate; any post-success evidence is recorded separately in continuation_runs.'))
    return dict(path=str(directory), baseline_run=str(baseline_directory), planned=len(cases), recorded=len(rows),
                **completion_state(directory, len(cases), len(rows)),
                cases=reports, intervention=manifest.get('intervention'),
                code_sha256=manifest.get('code_sha256'), source_manifest_sha256=manifest.get('source_manifest_sha256'),
                baseline_code_sha256=baseline_manifest.get('code_sha256'),
                sources_sha256=sources)


def retime_repeat_pairs(runs):
    comparisons = []
    for index, left in enumerate(runs):
        for right in runs[index + 1:]:
            if left['intervention'] != right['intervention']:
                continue
            aa, bb = ({r['id']: r for r in run['cases']} for run in (left, right))
            for name in sorted(aa.keys() & bb.keys()):
                a, b = aa[name], bb[name]
                if not a['treatment_result'] or not b['treatment_result']:
                    continue
                ap = Path(a['treatment_result']).with_name('trace.jsonl')
                bp = Path(b['treatment_result']).with_name('trace.jsonl')
                at, bt = indexed_trace(ap), indexed_trace(bp)
                indices_equal = bool(at) and at.keys() == bt.keys()
                common = sorted(at.keys() & bt.keys())
                exact = indices_equal and all(at[i] == bt[i] for i in common)
                comparisons.append(dict(id=name, left_run=left['path'], right_run=right['path'],
                                        left_success=a['success'], right_success=b['success'],
                                        left_recorded_actions=len(at), right_recorded_actions=len(bt),
                                        complete_trace_dicts_exact=exact,
                                        both_preintervention_prefixes_match_baseline=a['before_treatment']['verified'] and b['before_treatment']['verified'],
                                        trace_sha256={str(p): sha(p) for p in (ap, bp) if p.exists()},
                                        limitation='Repeat executions of one selected case are not independent task samples or post-success durability tests.'))
    return comparisons


def continuation_run(directory, panel_lookup, reference_directory):
    """Summarize continued execution separately from official early-stop runs."""
    manifest, cases, rows, sources = load_run(directory, panel_lookup)
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate continuation results: {directory}')
    reference_manifest_path = reference_directory / 'manifest.json'
    reference_manifest = read(reference_manifest_path) if reference_manifest_path.exists() else {}
    reference_cases = {case['id']: case for case in reference_manifest.get('cases', [])}
    if reference_manifest_path.exists():
        sources[str(reference_manifest_path)] = sha(reference_manifest_path)
    reports = []
    for case in cases:
        row = indexed.get(case['id'], {})
        reference_path = reference_directory / case['id'] / 'result.json'
        reference = read(reference_path) if reference_path.exists() else {}
        trace_path = directory / case['id'] / 'trace.jsonl'
        events_path = trace_path.with_name('true_success.jsonl')
        reference_trace_path = reference_path.with_name('trace.jsonl')
        trace, reference_trace = indexed_trace(trace_path), indexed_trace(reference_trace_path)
        events = indexed_trace(events_path)
        for path in (reference_path, trace_path, reference_trace_path, events_path):
            if path.exists():
                sources[str(path)] = sha(path)
        first = row.get('first_success') or {}
        first_index = first.get('source_action_index')
        prefix = pretreatment_equality(reference_trace, trace, first_index)
        expected_prefix = list(range(first_index)) if first_index is not None else []
        full_prefix_equal = bool(expected_prefix) and all(
            index in trace and index in reference_trace and trace[index] == reference_trace[index]
            for index in expected_prefix)
        expected_all = list(range(row.get('source_frames', 0)))
        trace_complete = bool(expected_all) and sorted(trace) == expected_all and sorted(events) == expected_all
        events_physical_equal = bool(trace) and trace.keys() == events.keys() and all(
            trace[i].get('physical') and trace[i].get('physical') == events[i].get('physical') for i in trace)
        initial_a = (reference.get('initial_parity') or {}).get('physical')
        initial_b = (row.get('initial_parity') or {}).get('physical')
        initial_equal = bool(initial_a and initial_b) and initial_a == initial_b
        both_exact_source = all(source_match(r)['tier'] == 'exact_reencoded_jpeg_and_state' for r in (reference, row))
        source_hash = case.get('source_hdf5_sha256', case.get('source_sha256'))
        reference_case = reference_cases.get(case['id'], {})
        reference_hash = reference_case.get('source_hdf5_sha256', reference_case.get('source_sha256'))
        same_source = bool(source_hash and reference_hash) and source_hash == reference_hash
        same_intervention = bool(manifest.get('intervention')) and manifest.get('intervention') == reference_manifest.get('intervention')
        before_ticks = trace.get(first_index - 1, {}).get('timing', {}).get('total_physics_steps', 0) if first_index is not None else None
        first_tick = before_ticks + first.get('check_in_action', 0) if before_ticks is not None else None
        final_timing = physics_timing(row)
        final_tick = final_timing['physics_steps'] if final_timing['valid_counter'] else None
        inclusive_ticks = final_tick - first_tick + 1 if first_tick is not None and final_tick is not None else None
        first_matches_reference_stop = bool(reference_trace and first_index is not None) and (
            max(reference_trace) == first_index
            and reference_trace[first_index].get('timing', {}).get('physics_steps') == first.get('check_in_action')
            and reference.get('success') is True)
        post_events = [events[i] for i in sorted(events) if first_index is not None and i >= first_index]
        endpoint_false = [r['source_action_index'] for r in post_events if r.get('final_after_action_predicate') is not True]
        all_endpoints_true = bool(post_events) and not endpoint_false
        final_event = events[max(events)] if events else {}
        result_events_consistent = bool(events) and (
            row.get('first_success') == final_event.get('first_true')
            and row.get('last_success') == final_event.get('last_true')
            and row.get('true_tick_checks') == final_event.get('true_tick_checks')
            and row.get('final_success') == final_event.get('final_after_action_predicate'))
        true_checks_cover_span = bool(inclusive_ticks) and row.get('true_tick_checks') == inclusive_ticks
        controlled = (row.get('all_source_labels_executed') is True and trace_complete and full_prefix_equal
                      and prefix['verified'] and initial_equal and both_exact_source and same_source
                      and same_intervention and events_physical_equal and result_events_consistent
                      and first_matches_reference_stop and 'success' not in row)
        reports.append(dict(id=case['id'], task=case['task'], status=row.get('status', 'pending'),
            protocol=row.get('protocol'), result_path=row.get('result_path'),
            reference_result=str(reference_path), reference_official_success=reference.get('success'),
            controller_loop_success_masked=row.get('controller_loop_success_masked'),
            ever_success=row.get('ever_success'), final_success=row.get('final_success'),
            first_success=row.get('first_success'), last_success=row.get('last_success'),
            true_tick_checks=row.get('true_tick_checks'), source_labels_executed=row.get('steps'),
            source_frames=row.get('source_frames'), all_source_labels_executed=row.get('all_source_labels_executed'),
            complete_source_trace_and_predicate_records=trace_complete,
            initial_recorded_physical_equal=initial_equal, both_exact_source_observation=both_exact_source,
            source_hdf5_hash_equal=same_source, same_retime_intervention=same_intervention,
            before_first_success=prefix, complete_pre_success_trace_dicts_equal=full_prefix_equal,
            first_true_matches_official_termination=first_matches_reference_stop,
            physical_records_equal_between_trace_and_predicate=events_physical_equal,
            result_fields_match_predicate_stream=result_events_consistent,
            first_success_total_physics_tick=first_tick, final_physics_timing=final_timing,
            physics_ticks_after_first_success=(final_tick - first_tick) if first_tick is not None and final_tick is not None else None,
            simulation_seconds_after_first_success=(final_tick - first_tick) * final_timing['timestep']
                if first_tick is not None and final_tick is not None and final_timing['timestep'] else None,
            inclusive_physics_ticks_first_success_to_end=inclusive_ticks,
            true_check_count_covers_inclusive_tick_span=true_checks_cover_span,
            post_success_endpoint_count=len(post_events), false_post_success_endpoint_indices=endpoint_false,
            all_post_success_action_endpoints_true=all_endpoints_true,
            controlled_continuation_comparison=bool(controlled),
            persistent_through_remaining_source_actions=bool(controlled and all_endpoints_true and true_checks_cover_span and row.get('final_success')),
            limitation='The original predicate is recorded while early termination is masked; this is a separate continuation protocol, not an official success-rate sample. Remaining source commands are executed; no independent idle/settling period follows the final source action.'))
    return dict(path=str(directory), reference_run=str(reference_directory), planned=len(cases), recorded=len(rows),
                **completion_state(directory, len(cases), len(rows)), cases=reports,
                code_sha256=manifest.get('code_sha256'), sources_sha256=sources)


def cpu_context(root):
    result = {}
    tracking = root / 'cpu_tracking_residuals.json'
    stage = root / 'cpu_mug_tracking_stage.json'
    if tracking.exists():
        d = read(tracking)
        result['tracking'] = dict(path=str(tracking), count=d['count'], groups=d['groups'],
                                   mouse_release=d['mouse_release'], limitations=d['limitations'])
    if stage.exists():
        d = read(stage)
        result['mug_stage'] = dict(path=str(stage), groups=d['groups'], limitations=d['limitations'])
    result['interpretation'] = 'Mouse release-position error is already present before release while observed joint tracking is usually close. Mug late tracking error often appears after release; it is not established as the cause. Successful traces are censored at task completion.'
    return result


def render(report):
    local_time = datetime.fromisoformat(report['generated_utc']).astimezone(timezone(timedelta(hours=8)))
    lines = ['# 固定案例验证实验：CPU 汇总', '', f"生成时间（北京时间）：{local_time:%Y-%m-%d %H:%M:%S}", '',
             '**初始案例按旧成功/失败结果分层选取；以下数字用于定位问题，不能估计总体成功率。**', '',
             '这里只比较同一运行内已完成的对应案例；缺失结果记为 pending。相机/state/prompt 与初始物理 trace 必须同时一致才纳入 horizon 配对。', '']
    for kind in ('sparse_expert_runs', 'dense_expert_runs'):
        for run in report[kind]:
            lines += [f"## {'离散专家标签' if kind == 'sparse_expert_runs' else 'Dense 专家控制'}：{Path(run['path']).name}", '',
                      f"状态：{run['run_state']}；已记录 {run['recorded']}/{run['planned']} 条结果。", '',
                      '| 案例 | 状态 | 成功 | 初始源输入核对 |', '|---|---|---:|---|']
            for row in run['cases']:
                success = '—' if row['success'] is None else str(row['success'])
                lines.append(f"| {row['id']} | {row['status']} | {success} | {row['source_observation_match']['tier']} |")
            lines += ['', 'exact 指三相机重编码 JPEG 字节相同且 drive-target state 完全相同；approximate_only 仅通过原评估阈值。CPU 归一化往返属于参考公式检查。', '']
    for control in report['dense_vs_sparse']:
        lines += [f"## Dense / 离散控制对比：{Path(control['dense_run']).name} / {Path(control['sparse_run']).name}", '',
                  '| 案例 | Dense成功 | 离散成功 | 两侧源JPEG/state精确一致 | 初始物理记录一致 | Dense采集帧数一致 | 可对照 |',
                  '|---|---|---|---|---|---|---|']
        timing_notes = []
        for row in control['cases']:
            lines.append(f"| {row['id']} | {row['dense_success']} | {row['sparse_success']} | {row['both_exact_source_observation']} | {row['initial_recorded_physical_equal']} | {row['dense_capture_count_equal']} | {row['controlled_comparison_available']} |")
            ratio = row.get('sparse_to_dense_physics_step_ratio')
            if ratio is not None:
                dense_time, sparse_time = row['dense_physics_timing'], row['sparse_physics_timing']
                timing_notes.append(f"{row['id']} 的实际仿真步数：Dense {dense_time['physics_steps']}，离散 {sparse_time['physics_steps']}，比值 {ratio:.3f}；模拟时长分别为 {dense_time['simulation_seconds']:.3f}/{sparse_time['simulation_seconds']:.3f} 秒。")
        for note in timing_notes:
            lines += ['', note, '']
        lines += ['', '两种执行路径的时序及插值不同；即使 Dense成功而离散失败，也需要进一步确定具体控制差异。', '']
    if report['sparse_repeats']:
        lines += ['## 离散重放重复性', '']
        for repeat in report['sparse_repeats']:
            lines += [f"- {repeat['id']}：{Path(repeat['left_run']).name} / {Path(repeat['right_run']).name}，比较 {repeat['compared_actions']} 条动作；初始输入及记录的物理状态一致，且全部源目标/物理轨迹精确复现：{repeat['complete_recorded_trajectory_equal']}。结果 {repeat['left_success']} / {repeat['right_success']}。"]
        lines += ['', '此重复检查不比较无效的v1计时字段，也不证明未记录的物理内部状态相同。', '']
    for run in report['horizon_runs']:
        lines += [f"## 执行动作数：{Path(run['path']).name}", '', f"状态：{run['run_state']}；已记录 {run['recorded']}/{run['planned']} 次运行。", '',
                  '| 对比 | 任务 | 有效配对 | 50步成功 | 候选成功 | 候选独有 / 50步独有 |', '|---|---|---:|---:|---:|---:|']
        for prefix, tasks in run['paired_vs_50'].items():
            for task, data in tasks.items():
                lines.append(f"| 50 → {prefix} | {task} | {data['valid_pairs']}/{data['scheduled']} | {data['baseline_successes']} | {data['candidate_successes']} | {data['candidate_only_success']} / {data['baseline_only_success']} |")
        lines += ['', '| 案例对比 | 状态 | 输入一致 | 初始物理 trace 一致 | 原 → 候选 |', '|---|---|---|---|---|']
        for prefix, tasks in run['paired_vs_50'].items():
            for data in tasks.values():
                for row in data['cases']:
                    lines.append(f"| {row['id']}：50→{prefix} | {row['status']} | {row['observed_inputs_equal']} | {row['initial_physical_trace_equal']} | {row['baseline_success']} → {row['candidate_success']} |")
        lines += ['', '与历史同一 checkpoint 的50步运行重复性：', '']
        for task, data in run['repeat50_vs_historical'].items():
            changes = [r['id'] for r in data['cases'] if r['status'] == 'paired' and not r['outcome_equal']]
            lines.append(f"- {task}：{data['valid_pairs']}/{data['scheduled']} 条初始输入及物理 trace 可配对；结果改变案例：{', '.join(changes) or '无已确认改变'}。")
        lines += ['', '不同执行步数会改变推理次数，因此噪声按推理调用序号对应，不能声称各物理时刻使用相同噪声。', '']
    for run in report['retime_runs']:
        lines += [f"## 释放时长干预：{Path(run['path']).name}", '',
                  f"状态：{run['run_state']}；已记录 {run['recorded']}/{run['planned']} 条结果。", '',
                  '只与明确指定的 sparse v2 基准比较。先核对干预之前的全部物理状态、源目标与实际仿真步数，再解释结果。', '']
        for row in run['cases']:
            prefix = row['before_treatment']
            lines += [f"- {row['id']}：状态 {row['status']}，基准成功={row['baseline_success']}，干预成功={row['success']}；实际干预索引 {row['activated_indices']}。",
                      f"- 干预前严格一致：{prefix['verified']}；初始物理记录一致：{row['initial_recorded_physical_equal']}；两侧源JPEG/state精确一致：{row['both_exact_source_observation']}。",
                      f"- 可形成该案例内的受控对照：{row['controlled_within_case_comparison']}。基准/干预总仿真步数：{row['baseline_physics_timing']['physics_steps']}/{row['treated_physics_timing']['physics_steps']}。", '']
            if row['actual_treatment_actions']:
                lines += ['| 源动作索引 | 请求步数 | 基准实际仿真步数 | 干预实际仿真步数 | 源动作相同 |', '|---:|---:|---:|---:|---|']
                for action in row['actual_treatment_actions']:
                    lines.append(f"| {action['source_action_index']} | {action.get('requested_physics_ticks')} | {action['baseline_actual_physics_steps']} | {action['treated_actual_physics_steps']} | {action['source_target_equal']} |")
                lines += ['']
        lines += ['这是特定案例的机制实验，不能据此估计总体收益；干预前若存在差异，不得归因于释放时长。此运行在原基准即时判据通过后终止，后续持续性证据单独列于continuation协议。成功会立即终止执行，因此总模拟时长差额还包括终止时刻变化，不能全部解释为干预直接节省的时长。', '']
    for repeat in report['retime_repeats']:
        lines += [f"释放时长干预重复性：{repeat['id']} 两次结果 {repeat['left_success']}/{repeat['right_success']}；完整 {repeat['left_recorded_actions']}/{repeat['right_recorded_actions']} 条trace字典精确一致：{repeat['complete_trace_dicts_exact']}。这是同一案例的官方即时成功重复；成功后持续性另见continuation协议。", '']
    for run in report['continuation_runs']:
        lines += [f"## 成功后继续执行剩余源动作：{Path(run['path']).name}", '',
                  f"状态：{run['run_state']}；已记录 {run['recorded']}/{run['planned']} 条结果。", '',
                  '此协议屏蔽控制器及harness的成功提前终止信号，逐次记录原始成功判据；与前述官方即时成功结果分开，不增加官方成功率样本。', '']
        for row in run['cases']:
            lines += [f"- {row['id']}：执行 {row['source_labels_executed']}/{row['source_frames']} 条源动作；曾成功={row['ever_success']}，最后仍成功={row['final_success']}。",
                      f"- 首次成功之前的完整trace字典与官方retime运行一致：{row['complete_pre_success_trace_dicts_equal']}；首次成功事件对应原官方终止时刻：{row['first_true_matches_official_termination']}；受控延续对照有效：{row['controlled_continuation_comparison']}。"]
            if row['simulation_seconds_after_first_success'] is not None:
                lines += [f"- 从首次成功到执行完剩余源动作，继续 {row['physics_ticks_after_first_success']} 个物理步（{row['simulation_seconds_after_first_success']:.3f}秒）；{row['true_tick_checks']} 次true检查覆盖包含首次成功的{row['inclusive_physics_ticks_first_success_to_end']}步区间：{row['true_check_count_covers_inclusive_tick_span']}；随后{row['post_success_endpoint_count']}个动作端点全部通过：{row['all_post_success_action_endpoints_true']}。"]
            lines += ['', '该结果检查了完成剩余源动作期间的判据持续性；源动作结束之后没有另加静止等待，也不能推广为所有场景的可靠悬挂。', '']
    if not report['horizon_runs']:
        lines += ['尚无已启动的 horizon 实验结果。', '']
    context = report['cpu_context']
    if context.get('tracking'):
        release = context['tracking']['mouse_release']
        good, bad = release['True'], release['False']
        lines += ['## 已有轨迹的 CPU 证据', '',
                  f"鼠标首次释放时，成功/失败的 XY 误差中位数为{1000 * good['first_release_xy_norm_m']['median']:.2f}/{1000 * bad['first_release_xy_norm_m']['median']:.2f} mm；失败在释放前5个动作已偏离{1000 * bad['first_release_xy_norm_5_actions_before_m']['median']:.2f} mm。对应关节跟踪残差中位数为{good['first_release_tracking_error_rad']['median']:.5f}/{bad['first_release_tracking_error_rad']['median']:.5f} rad，不能将普遍放置偏差直接归因于严重关节跟踪失败。", '']
    if context.get('mug_stage'):
        group = context['mug_stage']['groups']['False']
        lines += [f"挂杯{group['episodes']}条失败中{group['has_sustained_error']}条出现持续右臂跟踪异常，{group['sustained_error_at_or_after_release']}条首次发生在释放时或之后。成功轨迹通常在释放时终止，后续观察受到截断；晚期残差不能直接作为失败起因。", '']
    lines += ['所有完整逐案例配对、状态、源文件哈希和未完成项见同名 JSON。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--retime-baseline', type=Path,
                        help='Explicit sparse v2 baseline; default ROOT/expert_sparse_mug88038_v2')
    args = parser.parse_args()
    root = args.root.resolve()
    panel_path = args.panel or root / 'panel.json'
    panel = read(panel_path)
    lookup = {c['id']: c for c in panel['cases']}
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(), panel=str(panel_path),
                  panel_sha256=sha(panel_path), protocol=panel['protocol'],
                  sparse_expert_runs=[], dense_expert_runs=[], horizon_runs=[], retime_runs=[], continuation_runs=[],
                  unparsed_runs=[], cpu_context=cpu_context(root))
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and (p / 'manifest.json').exists()):
        if (directory / 'preflight.json').exists() or 'preflight' in directory.name:
            continue
        manifest = read(directory / 'manifest.json')
        if directory.name.startswith('expert_continuation'):
            report['continuation_runs'].append(continuation_run(directory, lookup, root / 'gripper_retime15_mug88038'))
        elif directory.name.startswith('gripper_retime'):
            baseline = args.retime_baseline or root / 'expert_sparse_mug88038_v2'
            report['retime_runs'].append(retime_run(directory, lookup, baseline))
        elif 'prefixes' in manifest:
            report['horizon_runs'].append(horizon_run(directory, lookup))
        elif 'dense' in directory.name:
            report['dense_expert_runs'].append(oracle_run(directory, lookup, dense=True))
        elif directory.name.startswith(('expert_label', 'expert_sparse')):
            report['sparse_expert_runs'].append(oracle_run(directory, lookup))
        else:
            report['unparsed_runs'].append(str(directory))
    report['dense_vs_sparse'] = dense_sparse_pairs(report['dense_expert_runs'], report['sparse_expert_runs'])
    report['sparse_repeats'] = sparse_repeat_pairs(report['sparse_expert_runs'])
    report['retime_repeats'] = retime_repeat_pairs(report['retime_runs'])
    output = args.output or root / 'summary_cpu.json'
    markdown = args.markdown or root / 'summary_cpu.md'
    for path, content in ((output, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                          (markdown, render(report))):
        path = path.resolve()
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(content)
        temporary.replace(path)
    print(json.dumps(dict(output=str(output), markdown=str(markdown),
                          sparse_runs=len(report['sparse_expert_runs']), dense_runs=len(report['dense_expert_runs']),
                          horizon_runs=len(report['horizon_runs']), retime_runs=len(report['retime_runs']),
                          continuation_runs=len(report['continuation_runs'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
