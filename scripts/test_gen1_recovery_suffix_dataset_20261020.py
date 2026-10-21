"""CPU-only tests with actual synthetic RGB/14D capture files and frozen hashes."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import gen1_recovery_suffix_dataset_20261020 as dataset


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')
    return dataset.digest(path)


def make_case(root, transitions=52, case_id='place_mouse_pad_1'):
    directory = root/case_id
    dense = directory/'dense'
    raw = dense/'raw'
    raw.mkdir(parents=True)
    source = directory/'source.hdf5'
    source.write_bytes(b'Original expert source identity, not used as suffix labels')
    case = dict(id=case_id, task='place_mouse_pad', source_revision='test_revision',
        seed=1, episode_index=2, source_hdf5=str(source), source_hdf5_sha256=dataset.digest(source),
        prompt='Place the mouse on the pad.', takeover_boundary_step=100,
        source_checkpoint_identity_sha256='ancestor_checkpoint', terminal_hold_ticks=250)
    case_sha = write_json(directory/'case.json',case)
    states, captures = [], []
    for index in range(transitions+1):
        state = np.arange(14,dtype=np.float64)*.01 + index*.001
        states.append(state.astype(np.float32))
        observation = dict(joint_action=dict(vector=state),
            observation={camera: dict(rgb=np.full((3,4,3),index%256,dtype=np.uint8))
                         for camera in dataset.CAMERAS.values()})
        target = raw/f'{index:06d}.pkl'
        with target.open('wb') as stream:
            pickle.dump(observation,stream)
        captures.append(dict(index=index,physics_tick=index*15,simulation_seconds=index*.06,
            path=str(target),sha256=dataset.digest(target)))
    capture_path = dense/'uniform_captures.jsonl'
    capture_path.write_text(''.join(json.dumps(row)+'\n' for row in captures))
    verified = {key: case[key] for key in ('id', 'task', 'seed', 'episode_index', 'takeover_boundary_step')}
    verified.update(success=True, final_success=True, exact_prefix=True, stable_success=True,
        simulator_returncode=0, physics_ticks=transitions*15,
        expert_gate=dict(status='passed', bypassed=False),
        terminal_hold=dict(requested_ticks=250, physics_ticks=250, start_tick=transitions*15-250,
            success=True, initial_success=True, all_ticks_success=True, final_success=True,
            positions_preserved=True, measured_state_reset=False))
    dense_result = dense/'result.json'
    dense_sha = write_json(dense_result,dict(verified, status='dense_replay_completed', timestep=.004,
                                           dense_source_match=True, source_mismatches=[]))
    sparse = directory/'original'
    sparse.mkdir()
    sparse_result = sparse/'result.json'
    sparse_sha = write_json(sparse_result,dict(verified, status='sparse_replay_completed', timing_mode='original',
        native_eval_success=True, official_rollout_success=True))
    ready = dict(schema_version=1,id=case_id,task=case['task'],case_path=str(directory/'case.json'),
        case_sha256=case_sha,exact_prefix=True,dense_success=True,dense_source_match=True,
        timestep=.004,dense_result_path=str(dense_result),dense_result_sha256=dense_sha,
        dense_captures_path=str(capture_path),dense_captures_sha256=dataset.digest(capture_path),
        sparse_modes=dict(original=dict(result_path=str(sparse_result),sha256=sparse_sha,
                                        exact_prefix=True,success=True)))
    write_json(directory/'readiness.json',ready)
    return directory/'readiness.json',np.stack(states)


class RecoveryDatasetTests(unittest.TestCase):
    def test_build_preserves_rgb_next_target_norm_and_real_window_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=root/'input'
            ready,states=make_case(inputs)
            norm=root/'norm.json'
            norm.write_text('{"original_quantile_stats":"unchanged"}\n')
            with patch.object(dataset,'NORM',norm):
                result=dataset.process(inputs,root/'built','original','build')
            self.assertTrue(result['dataset_ready'])
            self.assertEqual(result['use_scope'],'smallfit_diagnostic_only')
            self.assertEqual(result['independent_source_episodes'],1)
            row=result['episodes'][0]
            self.assertEqual(row['valid_start_count'],3)
            self.assertEqual(row['valid_start_range'],[0,3])
            self.assertEqual(row['split'],'diagnostic_train')
            with np.load(row['dataset_path'],allow_pickle=False) as data:
                np.testing.assert_array_equal(data['observation_state'],states[:-1])
                np.testing.assert_array_equal(data['action'],states[1:])
                np.testing.assert_array_equal(data['action'][-1],states[-1])
                self.assertEqual(data['cam_high'].dtype,np.uint8)
                self.assertTrue(np.all(data['cam_high'][3]==3))
                np.testing.assert_array_equal(data['action_physics_tick']-data['observation_physics_tick'],15)
            self.assertEqual((root/'built/norm_stats.json').read_bytes(),norm.read_bytes())

    def test_short_success_is_not_mislabeled_as_failed_recovery_or_ready_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready,_=make_case(Path(tmp),49)
            arrays,row=dataset.validate_case(ready,'original')
            self.assertIsNone(arrays)
            self.assertTrue(row['recovery_verified'])
            self.assertFalse(row['dataset_ready'])
            self.assertEqual(row['status'],'excluded_short_suffix')

    def test_dense_success_does_not_authorize_unpassed_sparse_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready,_=make_case(Path(tmp))
            with self.assertRaisesRegex(ValueError,'Selected deployment mode'):
                dataset.validate_case(ready,'fixed15')
            record=dataset.read(ready)
            record['exact_prefix']=False
            write_json(ready,record)
            with self.assertRaisesRegex(ValueError,'Dense/exact-prefix'):
                dataset.validate_case(ready,'original')

    def test_capture_tamper_and_nonuniform_times_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready,_=make_case(Path(tmp))
            record=dataset.read(ready)
            capture=Path(record['dense_captures_path'])
            rows=[json.loads(line) for line in capture.read_text().splitlines()]
            raw=Path(rows[2]['path'])
            before=raw.read_bytes()
            raw.write_bytes(before+b'tamper')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                dataset.validate_case(ready,'original')
            raw.write_bytes(before)
            rows[2]['physics_tick']=31
            capture.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            record['dense_captures_sha256']=dataset.digest(capture)
            write_json(ready,record)
            with self.assertRaisesRegex(ValueError,'exact 0,15,30'):
                dataset.validate_case(ready,'original')

    def test_partial_final_capture_is_excluded_without_shifting_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready,states=make_case(Path(tmp),51)
            record=dataset.read(ready)
            path=Path(record['dense_captures_path'])
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[-2]['phase']='explicit_terminal_hold'
            rows[-1].update(physics_tick=50*15+7,simulation_seconds=(50*15+7)*.004,
                            partial_final_interval=True)
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            record['dense_captures_sha256']=dataset.digest(path)
            write_json(ready,record)
            arrays,row=dataset.validate_case(ready,'original')
            self.assertEqual(row['frames'],50)
            self.assertEqual(row['valid_start_count'],1)
            np.testing.assert_array_equal(arrays['action'][-1],states[50])
            self.assertEqual(len(row['capture_audit']['dropped_capture_records']),1)
            self.assertEqual(row['capture_audit']['terminal_hold_transition_count'],1)
            self.assertTrue(arrays['action_is_terminal_hold'][-1])

    def test_split_uses_source_group_and_rejects_conflicting_original_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready,_=make_case(Path(tmp))
            record=dataset.read(ready)
            path=Path(record['case_path'])
            case=dataset.read(path)
            group,_=dataset.parent_identity(case)
            arrays,row=dataset.validate_case(ready,'original',{group:'val'})
            self.assertEqual(row['split'],'val')
            case['source_split']='train'
            record['case_sha256']=write_json(path,case)
            write_json(ready,record)
            with self.assertRaisesRegex(ValueError,'conflicts'):
                dataset.validate_case(ready,'original',{group:'val'})

    def test_empty_future_replay_root_stays_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'input').mkdir()
            norm=root/'norm.json'
            norm.write_text('{}')
            with patch.object(dataset,'NORM',norm):
                result=dataset.process(root/'input',root/'preflight','original')
            self.assertFalse(result['dataset_ready'])
            self.assertEqual(result['accepted_suffixes'],0)
            self.assertEqual(result['status'],'no_eligible_suffixes')
            self.assertEqual(list((root/'preflight').glob('*.npz')),[])

    def test_ready_summary_cannot_override_invalid_raw_replay_outcomes(self):
        mutations = [('dense', 'status', 'expert_rejected'), ('sparse', 'simulator_returncode', 2),
            ('dense', 'exact_prefix', False), ('dense', 'dense_source_match', False),
            ('sparse', 'stable_success', False), ('sparse', 'official_rollout_success', False)]
        for label, key, value in mutations:
            with self.subTest(label=label, key=key), tempfile.TemporaryDirectory() as tmp:
                ready, _ = make_case(Path(tmp))
                summary = dataset.read(ready)
                path = Path(summary['dense_result_path'] if label == 'dense' else summary['sparse_modes']['original']['result_path'])
                result = dataset.read(path)
                result[key] = value
                digest = write_json(path, result)
                if label == 'dense': summary['dense_result_sha256'] = digest
                else: summary['sparse_modes']['original']['sha256'] = digest
                write_json(ready, summary)
                with self.assertRaisesRegex(ValueError, 'Raw'):
                    dataset.validate_case(ready, 'original')

    def test_short_or_unstable_hold_is_rejected_despite_success_flags(self):
        for key, value in [('physics_ticks', 249), ('all_ticks_success', False), ('measured_state_reset', True)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                ready, _ = make_case(Path(tmp))
                summary = dataset.read(ready)
                path = Path(summary['sparse_modes']['original']['result_path'])
                result = dataset.read(path)
                result['terminal_hold'][key] = value
                summary['sparse_modes']['original']['sha256'] = write_json(path, result)
                write_json(ready, summary)
                with self.assertRaisesRegex(ValueError, 'stable250tick'):
                    dataset.validate_case(ready, 'original')

    def test_repeated_source_episode_cannot_be_counted_as_extra_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_case(root/'input', case_id='first')
            make_case(root/'input', case_id='repeat')
            norm = root/'norm.json'
            norm.write_text('{}')
            with patch.object(dataset, 'NORM', norm):
                report = dataset.process(root/'input', root/'preflight', 'original')
            self.assertEqual(report['accepted_suffixes'], 1)
            self.assertEqual(report['independent_source_episodes'], 1)
            self.assertEqual(report['valid_training_starts'], 3)
            self.assertEqual(len(report['excluded']), 1)
            self.assertIn('Duplicate parent source episode', report['excluded'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
