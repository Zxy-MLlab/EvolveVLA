import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gen1_phaseaware_after import checked_selection, METRICS


class SelectionTests(unittest.TestCase):
    def fixture(self, root):
        training, checkpoints = root / 'training', root / 'checkpoints'
        training.mkdir()
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        steps = [1000, 4000, 12000, 24000]
        tasks = [f'task_{i}' for i in range(18)]
        write(training / 'run_config.json', dict(tasks=tasks, save_steps=steps))
        write(training / 'completed.json', dict(status='completed', steps=24000, save_steps=steps))
        candidates = []
        for step in [0] + steps:
            value = 1 if step == 0 else .8
            records = [dict(task=t, episode=e, frame=f, **{k:value for k in METRICS})
                       for t in tasks for e in range(5) for f in range(8)]
            write(training / f'validation_{step:04d}.json', dict(step=step, records=records,
                metrics={t:{k:value for k in METRICS} for t in tasks}))
            if step:
                ckpt = checkpoints / str(step)
                write(ckpt / 'params/_METADATA', {})
                write(ckpt / 'provenance.json', dict(step=step))
                candidates.append(dict(step=step, score=.8, checkpoint=str(ckpt)))
        write(training / 'selection.json', dict(selected=candidates[0], candidates=candidates))
        return training, checkpoints

    def test_accepts_tie_with_earlier_step(self):
        with TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            self.assertEqual(checked_selection(*paths)['step'], 1000)

    def test_rejects_incomplete_corrupt_summary_and_wrong_choice(self):
        for kind in ('incomplete', 'summary', 'choice', 'foreign', 'panel'):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                training, checkpoints = self.fixture(Path(directory))
                name = ('completed.json' if kind == 'incomplete' else
                        'validation_1000.json' if kind in ('summary', 'panel') else 'selection.json')
                path = training / name
                value = json.loads(path.read_text())
                if kind == 'incomplete': value['steps'] = 12000
                elif kind == 'summary': value['metrics']['task_0']['joint_mae'] = .1
                elif kind == 'panel': value['records'][0]['frame'] = 999
                elif kind == 'choice': value['selected'] = value['candidates'][-1]
                else: value['candidates'][0]['checkpoint'] = '/foreign'
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError): checked_selection(training, checkpoints)


if __name__ == '__main__':
    unittest.main()
