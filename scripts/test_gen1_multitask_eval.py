import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import gen1_multitask_eval as broad
from gen1_multitask_eval import summarize, next_cases, installed, evaluation, SCRIPT, scene_spec, scene_bind


def row(seed, status='completed', success=False):
    return dict(task='lift_pot', seed=seed, status=status, success=success,
                hashes={'state':'same'}, binding={'prompt':'same'})


class BroadPanelTests(unittest.TestCase):
    def test_serial_runner_admission_supplement_and_saved_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'outputs').mkdir()
            scripts=root/'scripts'; scripts.mkdir()
            for name in ('gen1_multitask_binding.py','robotwin_eval_harness.py',
                         'robotwin_public_openpi_harness.py','pi05_openpi_adapter.py'):
                (scripts/name).write_text('# immutable fixture\n')
            repo=root/'repo'
            (repo/'envs').mkdir(parents=True)
            (repo/'description/task_instruction').mkdir(parents=True)
            (repo/'description/objects_description').mkdir()
            tasks=['lift_pot','handover_block']
            for task in tasks:
                (repo/'envs'/f'{task}.py').write_text('# fixture\n')
                (repo/'description/task_instruction'/f'{task}.json').write_text('{}')
            checkpoint=root/'checkpoints/candidate'
            (checkpoint/'params').mkdir(parents=True)
            (checkpoint/'params/_METADATA').write_text('{}')
            (checkpoint/'verification.json').write_text(json.dumps(dict(status='passed',
                checkpoint=str(checkpoint),normalization_exact=True,all_finite=True)))
            output=root/'outputs/broad'
            args=SimpleNamespace(tasks=tasks,seed=900000,output=output,checkpoint=checkpoint)
            calls=[]
            def panel(directory,model,cases_path):
                self.assertEqual(broad.confirmation.SCRIPT,SCRIPT)
                cases=json.loads(cases_path.read_text())
                self.assertEqual(len({r['task'] for r in cases}),1)
                self.assertLessEqual(len(cases),10)
                calls.append((directory.name,str(model)))
                records=[]
                for case in cases:
                    record=row(case['seed'],success=model==checkpoint)
                    record.update(task=case['task'],checkpoint=str(model))
                    if case['task']=='lift_pot' and case['seed']==900000 and model==checkpoint:
                        record['status']='expert_rejected'
                    records.append(record)
                return records
            original_script=broad.confirmation.SCRIPT
            with patch.object(broad,'ROOT',root),patch.object(broad,'task_names',return_value=tasks), \
                 patch.object(broad,'gpu_free') as gpu, \
                 patch.dict(broad.evaluation.c.CONFIG,robotwin_root=str(repo)), \
                 patch.object(broad.confirmation,'run_panel',side_effect=panel):
                broad.run(args)
            self.assertIs(broad.confirmation.SCRIPT,original_script)
            result=json.loads((output/'comparison.json').read_text())
            self.assertEqual(len(calls),6)
            self.assertEqual(gpu.call_count,7)
            self.assertEqual(result['outcome_assessment']['total_pairs'],20)
            self.assertEqual(result['outcome_assessment']['candidate_successes'],20)
            self.assertEqual(result['tasks']['lift_pot']['pairs'][0]['seed'],900001)
            self.assertEqual(len(result['tasks']['lift_pot']['unpaired']),1)
            self.assertFalse(result['goal_achieved'])

    def test_outcome_independent_supplement(self):
        a={'records':[row(900000)]}
        b={'records':[row(900000,'expert_rejected')]}
        first=next_cases(a,b,['lift_pot'],900000)
        a['records'][0]['success']=True
        self.assertEqual(first,next_cases(a,b,['lift_pot'],900000))
        self.assertEqual([r['seed'] for r in first],list(range(900001,900011)))

    def test_mismatch_errors_and_schedule_gaps_fail(self):
        a={'records':[row(900000)]}
        for mode in ('input','binding','error','missing','duplicate','gap'):
            b=copy.deepcopy(a)
            if mode=='input':b['records'][0]['hashes']={'state':'different'}
            if mode=='binding':b['records'][0]['binding']={'prompt':'different'}
            if mode=='error':b['records'][0]['status']='worker_error'
            if mode=='missing':b['records']=[]
            if mode=='duplicate':b['records']*=2
            if mode=='gap':
                with self.assertRaises(ValueError):next_cases(b,b,['lift_pot'],899999)
                continue
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                summarize(a,b,['lift_pot'])

    def test_ten_pairs_stop_and_patches_restore(self):
        a={'records':[row(900000+i) for i in range(10)]}
        self.assertEqual(next_cases(a,a,['lift_pot'],900000),[])
        original=evaluation.SCRIPT
        binder=evaluation.binding.bind_prompt
        with installed():self.assertEqual(evaluation.SCRIPT,SCRIPT)
        self.assertEqual(evaluation.SCRIPT,original)
        self.assertIs(evaluation.binding.bind_prompt,binder)

    def test_expert_binding_reaches_task_and_receiver_without_marker(self):
        class Task:
            def __init__(self):
                self.calls = 0
                self.instruction = None
            def play_once(self):
                self.calls += 1
                return {'info': {'{A}':'pot'}}
            def set_instruction(self, instruction):
                self.instruction = instruction
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'task_instruction').mkdir()
            (root/'task_instruction/lift_pot.json').write_text(json.dumps({'seen':['Lift {A}.']}))
            messages=[]
            module=SimpleNamespace(class_decorator=lambda *a,**kw:Task())
            harness=SimpleNamespace(send_message=lambda channel,payload:messages.append(payload))
            spec=scene_spec('lift_pot',900000,root)
            original_factory=module.class_decorator
            original_send=harness.send_message
            with installed(), evaluation.binding.simulator_shim(module,harness,spec,root):
                task=module.class_decorator('lift_pot')
                task.play_once()
                task.set_instruction(spec['template'])
                harness.send_message(None,dict(type='episode_start',instruction=spec['template']))
                bound=messages[0]['diagnostic_binding']
                self.assertEqual(task.calls,1)
                self.assertEqual(task.instruction,'Lift pot.')
                self.assertEqual(bound['prompt'],task.instruction)
                self.assertEqual(scene_bind(spec,{'info':bound['fields']},root),bound)
            self.assertIs(module.class_decorator,original_factory)
            self.assertIs(harness.send_message,original_send)


if __name__=='__main__':unittest.main()
