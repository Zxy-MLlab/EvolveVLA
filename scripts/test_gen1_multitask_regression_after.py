import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import gen1_multitask_regression_after as gate
from gen1_recovery_confirm import assess


class RegressionGateTests(unittest.TestCase):
    def fixture(self, root, gains):
        manager=root/'outputs/manager'; manager.mkdir(parents=True)
        training=root/'outputs/training'; training.mkdir()
        selected=dict(step=4000, score=.8, checkpoint=str(root/'checkpoints/candidate'))
        (training/'completed.json').write_text(json.dumps(dict(status='completed',steps=24000)))
        (training/'selection.json').write_text(json.dumps(dict(selected=selected)))
        entries=[]
        for number,gain in enumerate(gains,1):
            directory=root/f'outputs/gen1_multitask_confirmation{number}_20261012'
            directory.mkdir()
            seed=700000 if number==1 else 800000
            tasks={}
            for task in ('place_mouse_pad','hanging_mug'):
                pairs=[dict(seed=seed+i,original=i==0,candidate=i<(2 if gain else 1)) for i in range(10)]
                tasks[task]=dict(pairs=pairs,unpaired=[],paired_completed=10,
                    original_successes=1,candidate_successes=2 if gain else 1,
                    original_only=0,candidate_only=1 if gain else 0)
            result=dict(tasks=tasks,checkpoint=selected['checkpoint'])
            result['outcome_assessment']=assess(result)
            (directory/'comparison.json').write_text(json.dumps(result))
            (directory/'manifest.json').write_text(json.dumps(dict(checkpoint=selected['checkpoint'],seed_start=seed)))
            entries.append(dict(path=str(directory/'comparison.json'),assessment=result['outcome_assessment']))
        (manager/'completed.json').write_text(json.dumps(dict(status='completed',selection=selected,confirmations=entries)))
        return manager,training

    def check(self, gains, mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manager,training=self.fixture(root,gains)
            if mutation:mutation(root,manager,training)
            with patch.object(gate,'ROOT',root),patch.object(gate,'MANAGER',manager),patch.object(gate,'TRAINING',training):
                return gate.decision()

    def test_only_two_gains_allow_broad_regression(self):
        self.assertEqual(self.check([True,True])['action'],'broad_clean_regression')
        for gains in ([False],[True,False]):
            self.assertEqual(self.check(gains)['action'],'needs_failure_analysis')
        with self.assertRaisesRegex(ValueError,'Second independent'):
            self.check([True])

    def test_failed_upstream_never_launches(self):
        with self.assertRaisesRegex(RuntimeError,'Upstream'):
            self.check([True,True],lambda r,m,t:(m/'failed.json').write_text('{}'))

    def test_checkpoint_seed_and_counts_mismatch_fail(self):
        for kind in ('checkpoint','seed','counts','selection','steps'):
            def mutate(root,manager,training):
                directory=root/'outputs/gen1_multitask_confirmation2_20261012'
                if kind in ('checkpoint','seed'):
                    path=directory/'manifest.json'; data=json.loads(path.read_text())
                    data['checkpoint' if kind=='checkpoint' else 'seed_start']='wrong'
                elif kind=='counts':
                    path=directory/'comparison.json'; data=json.loads(path.read_text())
                    data['tasks']['hanging_mug']['candidate_successes']=10
                elif kind=='selection':
                    path=training/'selection.json'; data=json.loads(path.read_text()); data['selected']['step']=12000
                else:
                    path=training/'completed.json'; data=json.loads(path.read_text()); data['steps']=4000
                path.write_text(json.dumps(data))
            with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):
                self.check([True,True],mutate)

    def phaseaware_check(self, gains, mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager, training = self.fixture(root, gains)
            manager = manager.rename(root / 'outputs/gen1_phaseaware_after_20261013')
            training.rename(root / 'outputs/gen1_multitask_phaseaware_v3_20261013')
            completed_path = manager / 'completed.json'
            completed = json.loads(completed_path.read_text())
            del completed['status']
            for number, entry in enumerate(completed['confirmations'], 1):
                old = root / f'outputs/gen1_multitask_confirmation{number}_20261012'
                directory = old.rename(root / f'outputs/gen1_phaseaware_confirmation{number}_20261013')
                entry['path'] = str(directory)
                manifest_path = directory / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['seed_start'] += 400000
                manifest_path.write_text(json.dumps(manifest))
                result_path = directory / 'comparison.json'
                result = json.loads(result_path.read_text())
                for task in result['tasks'].values():
                    for pair in task['pairs']:
                        pair['seed'] += 400000
                result_path.write_text(json.dumps(result))
            completed_path.write_text(json.dumps(completed))
            (manager / 'checked_selection.json').write_text(json.dumps(completed['selection']))
            if mutation:
                mutation(root, manager)
            with patch.object(gate, 'ROOT', root):
                return gate.decision('phaseaware')

    def test_phaseaware_actual_completion_format(self):
        self.assertEqual(self.phaseaware_check([True, True])['action'], 'broad_clean_regression')
        for gains in ([False], [True, False]):
            self.assertEqual(self.phaseaware_check(gains)['action'], 'needs_failure_analysis')
        with self.assertRaisesRegex(ValueError, 'Second independent'):
            self.phaseaware_check([True])

    def test_phaseaware_rejects_wrong_checked_selection(self):
        with self.assertRaisesRegex(ValueError, 'Checked selection'):
            self.phaseaware_check([True, True], lambda r, m:
                (m / 'checked_selection.json').write_text('{}'))

    def test_phaseaware_rejects_reused_pair_seeds(self):
        def mutate(root, manager):
            path = root / 'outputs/gen1_phaseaware_confirmation2_20261013/comparison.json'
            result = json.loads(path.read_text())
            for task in result['tasks'].values():
                for pair in task['pairs']:
                    pair['seed'] -= 100000
            path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'seed outside'):
            self.phaseaware_check([True, True], mutate)


if __name__=='__main__':unittest.main()
