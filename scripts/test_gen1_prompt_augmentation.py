import unittest
from gen1_prompt_augmentation import compatible


class ArmConsistencyTests(unittest.TestCase):
    def test_rejects_actual_contradictory_initial_grasp(self):
        self.assertFalse(compatible('hanging_mug','Use the right arm to lift the mug, rotate it, and hang it.'))

    def test_keeps_left_to_right_transfer_and_generic_descriptions(self):
        self.assertTrue(compatible('hanging_mug','Use the left arm to lift it, then the right arm to hang it.'))
        self.assertTrue(compatible('hanging_mug','Lift the mug and hang it on the rack.'))

    def test_official_final_right_arm_is_not_mistaken_for_initial_grasp(self):
        self.assertTrue(compatible('hanging_mug',
            'Lift the mug, turn it, place it back, and move it onto the rack with the right arm.'))
        self.assertFalse(compatible('hanging_mug',
            'Use the right arm to lift the mug, rotate it, and attach it to the rack after setting it down.'))

    def test_mouse_is_allowed_to_use_either_arm(self):
        self.assertTrue(compatible('place_mouse_pad','Use the right arm to lift the mouse.'))
        self.assertTrue(compatible('place_mouse_pad','Use the left arm to lift the mouse.'))


if __name__=='__main__':unittest.main()
