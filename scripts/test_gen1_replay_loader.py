import unittest

from gen1_replay_loader import JoinedDataset, OfficialPrompts, EpisodeSubset
from gen1_replay_sampler import ReplaySampler


class LoaderTests(unittest.TestCase):
    def test_subset_excludes_other_tasks_and_preserves_boundaries(self):
        rows = [dict(task=t, episode=i, split='train', start=i*3, stop=i*3+3, frames=3)
                for i,t in enumerate(['excluded', 'mouse', 'excluded', 'mug', 'excluded'])]
        subset = EpisodeSubset(list(range(15)), rows, ['mouse','mug'])
        self.assertEqual([subset[i] for i in range(len(subset))], [3,4,5,9,10,11])
        self.assertEqual([r['episode'] for r in subset.rows], [1,3])
        for i in [-1,6]:
            with self.assertRaises(IndexError): subset[i]
        selected = [dict(r, source='local') for r in subset.rows]
        for i in range(18):
            selected.append(dict(task=['mouse','mug'][i] if i<2 else str(i),
                source='official', episode=i, split='train', start=6+i, stop=7+i, frames=1))
        sampler = ReplaySampler(selected,24,288,allow_local_subset=True)
        joined = JoinedDataset([subset, list(range(100,118))])
        values = [joined[i] for i in sampler]
        self.assertEqual(sum(x<100 for x in values),144)
        self.assertTrue(set(values) <= {3,4,5,9,10,11,*range(100,118)})
        self.assertEqual(sampler.counts['mouse','local'],72)
        self.assertEqual(sampler.counts['mug','local'],72)
        self.assertTrue(all(n==8 for (t,s),n in sampler.counts.items() if s=='official'))

    def test_join_boundaries(self):
        joined = JoinedDataset([[10,11],[20,21,22]])
        self.assertEqual([joined[i] for i in range(len(joined))],[10,11,20,21,22])
        for index in (-1,5):
            with self.assertRaises(IndexError):
                joined[index]

    def test_prompt_binding(self):
        rows = [dict(source='official',split='train',start=0,stop=2,prompts=['source'])]
        base = [dict(episode_index=0,frame_index=i,prompt='source') for i in range(2)]
        wrapped = OfficialPrompts(base,rows)
        self.assertEqual(wrapped[1]['prompt'],'source')
        base[1]['frame_index']=0
        with self.assertRaises(ValueError):
            wrapped[1]
        base[1]['frame_index']=1
        base[1]['prompt']='foreign'
        with self.assertRaises(ValueError):
            wrapped[1]

    def test_validation_rejected(self):
        rows=[dict(source='official',split='val',start=0,stop=1,prompts=['x'])]
        with self.assertRaises(ValueError):
            OfficialPrompts([{}],rows)


if __name__=='__main__':
    unittest.main()
