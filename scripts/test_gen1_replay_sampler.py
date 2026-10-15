"""No GPU, model imports, or source data mutations."""
import copy
import unittest

from gen1_replay_sampler import ReplaySampler


def fixture():
    rows, start = [], 0
    for source in ('local', 'official'):
        for episode, (task, frames) in enumerate((('a',3), ('a',7), ('b',20))):
            rows.append(dict(source=source, episode=episode, task=task, split='train',
                             frames=frames, start=start, stop=start+frames))
            start += frames
    return rows, start


class SamplerTests(unittest.TestCase):
    def test_exact_allocation_and_bounds(self):
        rows, length = fixture()
        sampler = ReplaySampler(rows, length, 4000)
        values = list(sampler)
        self.assertEqual(len(values),4000)
        self.assertTrue(all(0 <= i < length for i in values))
        self.assertEqual(set(sampler.counts.values()),{1000})
        self.assertEqual(set(values),set(range(length)))
        for start in range(0,len(values),4):
            strata = {(r['task'],r['source']) for i in values[start:start+4]
                      for r in rows if r['start'] <= i < r['stop']}
            self.assertEqual(len(strata),4)

    def test_determinism_and_epoch(self):
        rows,length = fixture()
        a,b = ReplaySampler(rows,length,41),ReplaySampler(rows,length,41)
        first = list(a)
        self.assertEqual(first,list(b))
        second = list(a)
        self.assertEqual(second,list(b))
        self.assertNotEqual(first,second)
        self.assertLessEqual(max(a.counts.values())-min(a.counts.values()),2)

    def test_reject_invalid_rows(self):
        rows,length = fixture()
        for field,value in [('split','val'),('source','unknown'),('start',1),
                            ('frames',4),('stop',0)]:
            changed=copy.deepcopy(rows)
            changed[0][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ReplaySampler(changed,length,10)
        with self.assertRaises(ValueError):
            ReplaySampler(rows,length+1,10)
        changed=copy.deepcopy(rows)
        changed[1]['episode']=changed[0]['episode']
        with self.assertRaises(ValueError):
            ReplaySampler(changed,length,10)
        changed=copy.deepcopy(rows)
        changed[-1]['task']='c'
        with self.assertRaises(ValueError):
            ReplaySampler(changed,length,10)


if __name__ == '__main__':
    unittest.main()
