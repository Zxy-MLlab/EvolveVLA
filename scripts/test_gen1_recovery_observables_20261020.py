import types
import unittest
import numpy as np
import gen1_recovery_observables_20261020 as obs


class ObservablesTests(unittest.TestCase):
    def test_relative_pose_rotates_translation_into_object_coordinates(self):
        q=np.array([np.sqrt(.5),0,0,np.sqrt(.5)])
        p=types.SimpleNamespace(p=np.array([1.,2.,3.]),q=q)
        value=obs.relative_pose(p,[1.,3.,3.,*q])
        np.testing.assert_allclose(value['position'],[1,0,0],atol=1e-12)
        np.testing.assert_allclose(value['quaternion_wxyz'],[1,0,0,0],atol=1e-12)

    def test_missing_interfaces_are_explicit_not_fabricated_zeroes(self):
        value=obs.observe(types.SimpleNamespace(robot=object(),scene=object()))
        self.assertIsNone(value['arms']['right']['articulation_qvel'])
        self.assertIn('right/qvel',value['unavailable'])
        self.assertIsNone(value['object_contacts'])
        self.assertIn('contacts',value['unavailable'])

    def test_contacts_keep_physical_body_names_without_grasp_inference(self):
        entity=types.SimpleNamespace(name='mug',get_components=lambda:[])
        actor=types.SimpleNamespace(actor=entity,get_pose=lambda:types.SimpleNamespace(p=[0,0,1],q=[1,0,0,0]))
        contact=types.SimpleNamespace(bodies=[types.SimpleNamespace(entity=entity),types.SimpleNamespace(entity=types.SimpleNamespace(name='right_finger'))],points=[types.SimpleNamespace(position=[0,0,1],impulse=[1,0,0])])
        task=types.SimpleNamespace(robot=types.SimpleNamespace(gripper_name=['right_finger']),mug=actor,scene=types.SimpleNamespace(get_contacts=lambda:[contact]))
        value=obs.observe(task)
        self.assertTrue(value['object_contacts'][0]['gripper_object'])
        self.assertEqual(value['object_contacts'][0]['bodies'],['mug','right_finger'])
        self.assertNotIn('grasped',value)


if __name__=='__main__':unittest.main()
