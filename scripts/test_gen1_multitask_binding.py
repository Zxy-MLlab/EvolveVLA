import json
from pathlib import Path
import random
import tempfile
import unittest
from gen1_multitask_binding import bind, eligible


class BindingTests(unittest.TestCase):
    def test_official_field_matching(self):
        self.assertTrue(eligible('{A} {B}', {'A','B','a','b'}))
        self.assertFalse(eligible('{A} {B} {a}', {'A','B','a','b'}))
        self.assertFalse(eligible('{A} {B} {C}', {'A','B','a'}))
        self.assertTrue(eligible('pass the block', set()))

    def test_scene_specific_choice_and_rng_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'task_instruction').mkdir(); (root/'objects_description').mkdir()
            (root/'task_instruction/task.json').write_text(json.dumps({'seen':['{A} {B} {a}', '{A} {B} {C}']}))
            before=random.getstate()
            info={'info':{'{A}':'bread','{B}':'basket','{a}':'left'}}
            result=bind('task',900000,info,root)
            self.assertEqual(result['prompt'],'bread basket the left arm')
            self.assertEqual(result,bind('task',900000,info,root))
            self.assertEqual(random.getstate(),before)
            info['info']['{a}']='dual'
            self.assertEqual(bind('task',900000,info,root)['prompt'],'bread basket the dual arm')
            info['info']['{C}']='plate'
            self.assertEqual(bind('task',900000,info,root)['prompt'],'bread basket plate')

    def test_missing_object_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'task_instruction').mkdir()
            (root/'task_instruction/task.json').write_text(json.dumps({'seen':['{A}']}))
            for value in ('missing/base0','../escape'):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    bind('task',1,{'info':{'{A}':value}},root)


if __name__=='__main__':unittest.main()
