import json
from pathlib import Path
import tempfile
import unittest
import gen1_recovery_compare as compare


class ComparisonTests(unittest.TestCase):
    def row(self,seed,success,status='completed',identity='same'):
        return dict(task='place_mouse_pad',seed=seed,status=status,success=success,
                    hashes={'camera':identity},binding={'prompt':'fixed'})

    def test_exclusions_and_initial_mismatches_are_not_policy_losses(self):
        a={'records':[self.row(1,True),self.row(2,True),self.row(3,True)]}
        b={'records':[self.row(1,False),self.row(2,False,'expert_rejected'),self.row(3,False,identity='different')]}
        result=compare.summarize(a,b)['tasks']['place_mouse_pad']
        self.assertEqual(result['paired_completed'],1)
        self.assertEqual(result['original_only'],1)
        self.assertEqual(len(result['unpaired']),2)

    def test_selection_uses_all_four_metrics_and_requires_saved_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);training=root/'train';training.mkdir();checkpoints=root/'checkpoints';checkpoints.mkdir()
            for step,values in [(0,[1,1,1,1]),(500,[.1,.1,2,2]),(1000,[.7,.7,.7,.7]),(1500,[.1,.1,.1,.1])]:
                (training/f'validation_{step:04d}.json').write_text(json.dumps(dict(step=step,metrics={'all':dict(zip(compare.KEYS,values))})))
                if step in (500,1000):
                    path=checkpoints/str(step)/'params';path.mkdir(parents=True);(path/'_METADATA').write_text('{}')
            self.assertEqual(compare.select(training,checkpoints)['selected']['step'],1000)


if __name__=='__main__':unittest.main()
