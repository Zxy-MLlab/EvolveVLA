import copy
import unittest
from gen1_multitask_validation_report import KEYS, compare


def panel(step, a=1., b=1.):
    records = [dict(task=t, episode=i, frame=0, **{k:v for k in KEYS})
               for i,(t,v) in enumerate((('a',a),('b',b)))]
    return dict(step=step, anchors=2, records=records,
                metrics={t:{k:v for k in KEYS} for t,v in (('a',a),('b',b),('all',(a+b)/2))})


class ReportTests(unittest.TestCase):
    def test_mean_improvement_does_not_hide_task_regression(self):
        result=compare(panel(0),panel(1000,.2,1.2))
        self.assertAlmostEqual(result['prespecified_composite_score'],.7)
        self.assertEqual(result['increased_error_tasks']['gripper_mae'],['b'])

    def test_changed_anchor_and_corrupt_summary_fail(self):
        for mode in ('anchor','summary','duplicate','nan'):
            candidate=panel(1000)
            if mode=='anchor':candidate['records'][0]['frame']=1
            if mode=='summary':candidate['metrics']['a']['joint_mae']=2
            if mode=='duplicate':candidate['records'][1]=copy.deepcopy(candidate['records'][0])
            if mode=='nan':candidate['records'][0]['joint_mae']=float('nan')
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                compare(panel(0),candidate)

    def test_zero_baseline_has_no_percentage(self):
        result=compare(panel(0,0,1),panel(1000,0,1))
        self.assertIsNone(result['tasks']['a']['metrics']['joint_mae']['percent_change'])


if __name__=='__main__':unittest.main()
