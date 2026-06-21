import unittest

from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.property_verifier import TransitionPropertyVerifier
from shrdlu_blocks.state_pred import predict_world_state_after_actions


class StatePredictionTests(unittest.TestCase):
    def setUp(self):
        self.verifier = TransitionPropertyVerifier.from_file()

    def _derived_aps(self, state):
        return self.verifier.verify_transition(
            state,
            {'name': 'probe', 'args': {}},
            state,
        )['derived_aps']

    def test_geometry_prediction_matches_green_big_block_obstruction(self):
        env = ShrdluBlocksEnv()
        init = env.snapshot()
        actions = [
            {'name': 'move_grasper', 'args': {'x': -0.3, 'y': 0.05}},
            {'name': 'lower_grasper', 'args': {}},
            {'name': 'close_grasper', 'args': {}},
            {'name': 'raise_grasper', 'args': {}},
            {'name': 'move_grasper', 'args': {'x': 0.1, 'y': -0.15}},
            {'name': 'lower_grasper', 'args': {}},
        ]

        predicted, notes = predict_world_state_after_actions(init, actions)
        for action in actions:
            env.execute_action(action)
        actual = env.snapshot()

        predicted_obj5 = next(obj for obj in predicted['objects'] if obj['obj_id'] == 5)
        actual_obj5 = next(obj for obj in actual['objects'] if obj['obj_id'] == 5)
        self.assertEqual(8, predicted_obj5['resting_on'])
        self.assertEqual(actual_obj5['resting_on'], predicted_obj5['resting_on'])
        self.assertEqual(self._derived_aps(actual), self._derived_aps(predicted))
        self.assertIn('lowered onto obj 8', notes[-1])

    def test_invalid_release_is_predicted_as_precondition_failure(self):
        env = ShrdluBlocksEnv()
        init = env.snapshot()
        actions = [
            {'name': 'move_grasper', 'args': {'x': -0.3, 'y': 0.05}},
            {'name': 'lower_grasper', 'args': {}},
            {'name': 'close_grasper', 'args': {}},
            {'name': 'raise_grasper', 'args': {}},
            {'name': 'move_grasper', 'args': {'x': 0.1, 'y': -0.15}},
            {'name': 'lower_grasper', 'args': {}},
            {'name': 'open_grasper', 'args': {}},
        ]

        predicted, notes = predict_world_state_after_actions(init, actions)
        for action in actions[:-1]:
            env.execute_action(action)
        with self.assertRaisesRegex(Exception, 'support'):
            env.execute_action(actions[-1])
        actual = env.snapshot()

        self.assertIn('precondition failed', notes[-1])
        self.assertEqual(self._derived_aps(actual), self._derived_aps(predicted))

    def test_raise_while_holding_matches_grasper_resting_on_ap(self):
        env = ShrdluBlocksEnv()
        init = env.snapshot()
        actions = [
            {'name': 'move_grasper', 'args': {'x': -0.25, 'y': -0.2}},
            {'name': 'lower_grasper', 'args': {}},
            {'name': 'close_grasper', 'args': {}},
            {'name': 'raise_grasper', 'args': {}},
        ]

        predicted, _notes = predict_world_state_after_actions(init, actions)
        for action in actions:
            env.execute_action(action)
        actual = env.snapshot()

        predicted_grasper = next(obj for obj in predicted['objects'] if obj['obj_id'] == 0)
        actual_grasper = next(obj for obj in actual['objects'] if obj['obj_id'] == 0)
        self.assertEqual(4, predicted_grasper['resting_on'])
        self.assertEqual(actual_grasper['resting_on'], predicted_grasper['resting_on'])
        self.assertTrue(self._derived_aps(predicted)['some_object_resting_on_4'])
        self.assertEqual(self._derived_aps(actual), self._derived_aps(predicted))


if __name__ == '__main__':
    unittest.main()
