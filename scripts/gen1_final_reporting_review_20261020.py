#!/usr/bin/env python3
"""Read-only CPU evidence audit; writes only reporting review artifacts."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/EvolveVLA')
DEST = ROOT/'outputs/gen1_timing_validation_20261020'
PANEL = ROOT/'outputs/gen1_causal_diagnosis_20261019/panel.json'
BROAD = ROOT/'outputs/gen1_all_tasks_clean10_20261018'
TRAIN = ROOT/'outputs/gen1_trainset_rollout_20261018'
TASKS = ('place_mouse_pad', 'hanging_mug')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines(True)
            if line.strip() and line.endswith('\n')]


def identity(path):
    return dict(path=str(path), sha256=sha(path))


def first_physical(path):
    with Path(path).open() as stream:
        return json.loads(stream.readline())


def panel_review():
    panel = read(PANEL)
    selected = [c for c in panel['cases'] if c['selection_phase'] == 'initial']
    split = read(panel['source_manifest'])
    local = {f"{r['task']}_{r['seed']}": r for r in split['episodes']
             if r['split'] == 'train' and r['task'] in TASKS}
    frozen = panel['frozen_rollout_snapshot']
    lines = Path(frozen['source']).read_text().splitlines(True)[:frozen['first_rows']]
    snapshot = [json.loads(x) for x in lines]
    historical = {r['id']: r for r in snapshot}
    checks = dict(split_manifest_hash=sha(panel['source_manifest']) == panel['source_manifest_sha256'],
                  frozen_snapshot_hash=hashlib.sha256(''.join(lines).encode()).hexdigest() == frozen['canonical_lines_sha256'],
                  frozen_last_id=snapshot[-1]['id'] == frozen['last_id'],
                  initial_order=[c['id'] for c in selected] == panel['panels']['initial'],
                  initial_unique=len({c['id'] for c in selected}) == 20,
                  local_train_counts=Counter(r['task'] for r in local.values()) == {t: 995 for t in TASKS})
    case_checks = []
    for case in selected:
        source = local.get(case['id'], {})
        prior = historical.get(case['id'], {})
        result = read(case['known_candidate_result'])
        instruction = read(case['source_instruction_file'])
        arm = result.get('binding', {}).get('fields', {}).get('{a}', 'both')
        row = dict(id=case['id'], task=case['task'], seed=case['seed'],
                   episode_index=case['episode_index'], prompt=case['prompt'],
                   stratum=case['selection_stratum'],
                   checks=dict(
                       train_membership=bool(source),
                       source_path=source.get('path') == case['source_hdf5'],
                       source_bytes_sha256=sha(case['source_hdf5']) == case['source_hdf5_sha256'] == source.get('sha256'),
                       source_revision=source.get('source_revision') == case['source_revision'],
                       source_frames=source.get('frames') == case['source_frames'],
                       directory=source.get('episode_directory') == case['source_episode_directory'],
                       episode_index=int(Path(source['episode_directory']).name.split('_')[-1]) == case['episode_index'],
                       instruction_hash=sha(case['source_instruction_file']) == case['source_instruction_sha256'],
                       first_seen_prompt=instruction['seen'][0] == case['prompt'],
                       historical_snapshot_status=prior.get('status') == result.get('status') == case['known_candidate_status'] == 'completed',
                       historical_snapshot_success=prior.get('success') == result.get('success') == case['known_candidate_success'],
                       historical_seed=prior.get('seed') == result.get('seed') == case['seed'],
                       historical_episode_index=prior.get('episode_index') == case['episode_index'],
                       historical_prompt=result.get('binding', {}).get('prompt') == case['prompt'],
                       active_arm=arm == case['active_arm'],
                       stratum=case['selection_stratum'] == arm + (':success' if prior.get('success') else ':failure')),
                   source_hdf5=identity(case['source_hdf5']),
                   instruction=identity(case['source_instruction_file']),
                   historical_result=identity(case['known_candidate_result']))
        case_checks.append(row)
    salt = panel['selection_salt']
    rank = lambda phase, identifier: hashlib.sha256(f'{salt}:{phase}:{identifier}'.encode()).hexdigest()
    supplemental = {t: sorted((k for k, r in local.items() if r['task'] == t),
                              key=lambda k: rank('supplemental', k))[:10] for t in TASKS}
    reconstructed_supplemental = sum((supplemental[t] for t in TASKS), [])
    checks['supplemental_preselection_reconstructed'] = reconstructed_supplemental == panel['panels']['supplemental']
    excluded = set(reconstructed_supplemental)
    strata = defaultdict(list)
    for row in snapshot:
        if row['status'] != 'completed' or row['id'] in excluded:
            continue
        prior = read(TRAIN/row['id']/'result.json')
        arm = prior.get('binding', {}).get('fields', {}).get('{a}', 'both')
        strata[(row['task'], arm + (':success' if row['success'] else ':failure'))].append(row['id'])
    for values in strata.values():
        values.sort(key=lambda k: rank('stratified', k))
    ms = strata[('place_mouse_pad', 'left:success')][:3] + strata[('place_mouse_pad', 'right:success')][:2]
    mf = strata[('place_mouse_pad', 'right:failure')][:3] + strata[('place_mouse_pad', 'left:failure')][:2]
    hs = strata[('hanging_mug', 'both:success')][:5]
    hf = strata[('hanging_mug', 'both:failure')][:5]
    reconstructed_initial = [v for pair in zip(ms, mf) for v in pair] + [v for pair in zip(hs, hf) for v in pair]
    checks['initial_strata_and_order_reconstructed'] = reconstructed_initial == panel['panels']['initial']
    return dict(panel=identity(PANEL), source_manifest=identity(panel['source_manifest']), checks=checks,
                all_checks_pass=all(checks.values()) and all(all(c['checks'].values()) for c in case_checks),
                initial=case_checks, reconstructed_initial=reconstructed_initial,
                reconstructed_supplemental=reconstructed_supplemental,
                initial_stratum_counts=dict(Counter(c['task'] + '/' + c['selection_stratum'] for c in selected)),
                frozen_snapshot=dict(rows=len(snapshot), last_id=snapshot[-1]['id'],
                    counts=dict(Counter(r['task'] for r in snapshot)), **identity(frozen['source'])),
                limitations=['All 20 are training scenes selected using historical candidate outcomes.',
                    'Each task has exactly 5 historical successes and 5 failures by construction.',
                    'Mouse primary sampling pool was the first 396 recorded attempts, not all 995 episodes.',
                    'This is a diagnostic panel; its success rate is not an unbiased task or held-out estimate.',
                    'Reconstructing a declared outcome-blind supplemental rule does not prove the author was unaware of outcomes.'])


def broad_review():
    manifest = read(BROAD/'manifest.json')
    tasks, all_pairs = [], []
    for comparison in sorted(BROAD.glob('*_comparison.json')):
        if '_round_' in comparison.name:
            continue
        for task, data in read(comparison)['tasks'].items():
            pairs = []
            for pair in data['pairs']:
                seed = pair['seed']
                paths = {label: list(BROAD.glob(f'{task}_round_*_{label}/{task}_{seed}/result.json'))
                         for label in ('original', 'candidate')}
                assert all(len(paths[label]) == 1 for label in paths), paths
                paths = {label: rows[0] for label, rows in paths.items()}
                results = {label: read(path) for label, path in paths.items()}
                physical = {label: first_physical(path.with_name('physical.jsonl')) for label, path in paths.items()}
                a, b = results['original'], results['candidate']
                tests = dict(identity=all(r['task'] == task and r['seed'] == seed for r in results.values()),
                    completed=all(r['status'] == 'completed' for r in results.values()),
                    boolean_outcomes=all(type(r['success']) is bool for r in results.values()),
                    normal_simulator_exit=all(r.get('simulator_returncode') == 0 for r in results.values()),
                    comparison_matches_results=all(results[label]['success'] == pair[label] for label in results),
                    input_hashes=bool(a.get('hashes')) and a.get('hashes') == b.get('hashes'),
                    input_fields=all(bool(a.get('hashes', {}).get(k)) for k in ('cam_high','cam_left_wrist','cam_right_wrist','state','prompt')),
                    bound_prompt_and_scene_fields=bool(a.get('binding')) and a.get('binding') == b.get('binding'),
                    checkpoint_paths=all(results[label]['checkpoint'] == manifest[label] for label in results),
                    initial_physical_step_zero=all(r['step'] == 0 for r in physical.values()),
                    recorded_initial_physical=bool(physical['original'].get('physical')) and physical['original']['physical'] == physical['candidate']['physical'])
                row = dict(task=task, seed=seed, original=a['success'], candidate=b['success'], checks=tests,
                           recorded_physical_fields=sorted(physical['original']['physical']),
                           sources={label: dict(result=identity(path), physical=identity(path.with_name('physical.jsonl')))
                                    for label,path in paths.items()})
                pairs.append(row)
            counts = dict(pairs=len(pairs), original_successes=sum(p['original'] for p in pairs),
                          candidate_successes=sum(p['candidate'] for p in pairs),
                          original_only=sum(p['original'] and not p['candidate'] for p in pairs),
                          candidate_only=sum(not p['original'] and p['candidate'] for p in pairs),
                          both_success=sum(p['original'] and p['candidate'] for p in pairs),
                          both_failure=sum(not p['original'] and not p['candidate'] for p in pairs))
            tasks.append(dict(task=task, **counts, seeds_unique=len({p['seed'] for p in pairs}) == 10,
                              all_pair_checks_pass=all(all(p['checks'].values()) for p in pairs),
                              comparison=identity(comparison)))
            all_pairs.extend(pairs)
    non_target = [r for r in tasks if r['task'] not in TASKS]
    return dict(manifest=identity(BROAD/'manifest.json'), task_count_requested=len(manifest['tasks']),
                source_protocol=manifest['protocol'], per_task=tasks, pairs=all_pairs,
                non_target={key: sum(r[key] for r in non_target) for key in
                    ('pairs','original_successes','candidate_successes','original_only','candidate_only','both_success','both_failure')},
                all_pair_checks_pass=all(r['all_pair_checks_pass'] and r['seeds_unique'] for r in tasks),
                limitations=['Only the first six tasks completed; five are non-target tasks.',
                    'The 5-task subtotal is not a complete or random sample of 50 tasks.',
                    'Saved non-target physical records contain arms only; object physics, contacts, velocities and solver state were not compared.',
                    'The historical run checked checkpoint paths and production export verification; it did not store a complete immutable weight identity in every result.',
                    'Fresh seeds, scene-bound seen prompts and original controller differ from the new training-scene timing panel.'])


def expert_censor_review():
    rows = []
    cases = [c for c in read(PANEL)['cases'] if c['selection_phase'] == 'initial']
    for case in cases:
        paths = {mode: DEST/'expert_ab'/case['id']/mode/'result.json' for mode in ('raw50','retime15')}
        if not all(p.exists() for p in paths.values()):
            continue
        results = {mode: read(path) for mode,path in paths.items()}
        if not all(results[mode].get('condition') == mode for mode in results):
            continue
        traces = {mode: {r['source_action_index']:r for r in jsonl(path.with_name('trace.jsonl'))}
                  for mode,path in paths.items()}
        start = results['retime15'].get('release', {}).get('start')
        events = jsonl(paths['retime15'].with_name('retiming.jsonl'))
        activated = [e['source_action_index'] for e in events]
        a, b = results['raw50'], results['retime15']
        rows.append(dict(id=case['id'], task=case['task'], release_start=start,
            original_status=a['status'], original_success=a['success'], original_steps=a.get('steps'),
            treated_status=b['status'], treated_success=b['success'], treated_steps=b.get('steps'),
            success_before_release={mode: r.get('success') is True and r.get('steps', 10**9) <= start for mode,r in results.items()},
            activated_indices=activated,
            treated_indices_without_original=[i for i in activated if i not in traces['raw50']],
            sources={mode:identity(path) for mode,path in paths.items()}))
    return dict(pairs_observed=len(rows), cases=rows,
                early_success_before_release_cases=sum(any(r['success_before_release'].values()) for r in rows),
                treated_actions_censored_by_original_success_cases=sum(bool(r['treated_indices_without_original']) and r['original_success'] for r in rows),
                interpretation=['The primary paired outcome includes valid early successes even when the intervention has not occurred.',
                    'controlled_duration_intervention is an additional mechanism-evidence flag, not an outcome eligibility filter.',
                    'A success before release cannot establish a duration effect.',
                    'If raw50 succeeds earlier, later treated actions lack matched raw observations; current causal flag conservatively becomes false.',
                    'Stopping at first instantaneous success does not establish post-success stability; the old 88038 continuation alone supports stability for that case.'])


def background_update(review):
    path = DEST/'background_status.json'
    existing = read(path)
    progress = read(TRAIN/'progress.json')
    rows = jsonl(TRAIN/'results.jsonl')
    done = {r['id'] for r in rows}
    unfinalized = [p for p in TRAIN.iterdir() if p.is_dir() and p.name not in done]
    evidence = dict(review_utc=review['generated_utc'], recorded_cases=len(rows),
                    progress_record_count_agrees=progress['completed_cases'] == len(rows),
                    last_record=rows[-1],
                    unfinalized_case_directories=[dict(id=p.name, files=[dict(name=f.name, bytes=f.stat().st_size) for f in sorted(p.iterdir()) if f.is_file()]) for p in sorted(unfinalized)],
                    broad_failure=identity(BROAD/'failed.json'),
                    broad_specific_failure=dict(**identity(BROAD/'pick_diverse_bottles_round_00_original/pick_diverse_bottles_1800000/result.json'),
                        result=read(BROAD/'pick_diverse_bottles_round_00_original/pick_diverse_bottles_1800000/result.json')),
                    broad_parent_log=identity(ROOT/'outputs/gen1_all_tasks_clean10_20261018.queue.log'),
                    trainset_progress=identity(TRAIN/'progress.json'),
                    trainset_results=identity(TRAIN/'results.jsonl'),
                    trainset_termination_cause='unknown; no saved parent exit/traceback established',
                    trainset_parent_log_search='No filename match in /tmp or /root/autodl-tmp for *trainset*.log or *trainset*.out; per-case stderr exists.',
                    non_target_pair_independent_audit=review['background_pair_audit']['non_target'],
                    non_target_pair_all_checks_pass=review['background_pair_audit']['all_pair_checks_pass'])
    existing['independent_review'] = evidence
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2)+'\n')
    return evidence


def main():
    review = dict(generated_utc=datetime.now(timezone.utc).isoformat(), script=identity(__file__),
                  panel_provenance=panel_review(), background_pair_audit=broad_review(),
                  expert_censoring=expert_censor_review())
    review['background_status_audit'] = background_update(review)
    path = DEST/'final_reporting_review_evidence.json'
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(path=str(path), panel_checks=review['panel_provenance']['all_checks_pass'],
        broad_pair_checks=review['background_pair_audit']['all_pair_checks_pass'],
        non_target=review['background_pair_audit']['non_target'],
        censor_pairs=review['expert_censoring']['pairs_observed']), ensure_ascii=False))


if __name__ == '__main__':
    main()
