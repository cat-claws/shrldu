import copy
import unittest

from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.property_verifier import ACTIVE_PROPERTY_IDS, TransitionPropertyVerifier
from shrdlu_blocks.tla_verifier import verify_ap_trace


class PropertyVerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = TransitionPropertyVerifier.from_file()

    def test_close_transition_satisfies_all_properties(self):
        env = ShrdluBlocksEnv()
        env.execute_action({'name': 'move_grasper', 'args': {'x': -0.1, 'y': 0.4}})
        env.execute_action({'name': 'lower_grasper', 'args': {}})

        pre_state = env.snapshot()
        pre_scene = copy.deepcopy(env.scene)

        env.execute_action({'name': 'close_grasper', 'args': {}})

        result = self.verifier.verify_transition(
            pre_state,
            {'name': 'close_grasper', 'args': {}},
            env.snapshot(),
            pre_scene=pre_scene,
            post_scene=env.scene,
        )

        self.assertTrue(result['all_satisfied'])
        self.assertEqual([], result['violations'])
        self.assertTrue(result['derived_aps']['last_action_close_grasper'])
        self.assertTrue(result['derived_aps']['pre_grasper_resting_on_graspable'])
        self.assertTrue(result['derived_aps']['post_grasper_holding'])

    def test_raise_transition_satisfies_all_properties(self):
        env = ShrdluBlocksEnv()
        env.execute_action({'name': 'lower_grasper', 'args': {}})

        pre_state = env.snapshot()
        pre_scene = copy.deepcopy(env.scene)

        env.execute_action({'name': 'raise_grasper', 'args': {}})

        result = self.verifier.verify_transition(
            pre_state,
            {'name': 'raise_grasper', 'args': {}},
            env.snapshot(),
            pre_scene=pre_scene,
            post_scene=env.scene,
        )

        self.assertTrue(result['all_satisfied'])
        self.assertTrue(result['derived_aps']['last_action_raise_grasper'])
        self.assertTrue(result['derived_aps']['pre_grasper_lowered'])
        self.assertFalse(result['derived_aps']['post_grasper_lowered'])

    def test_open_while_holding_satisfies_all_properties(self):
        env = ShrdluBlocksEnv()
        env.execute_action({'name': 'move_grasper', 'args': {'x': -0.1, 'y': 0.4}})
        env.execute_action({'name': 'lower_grasper', 'args': {}})
        env.execute_action({'name': 'close_grasper', 'args': {}})
        env.execute_action({'name': 'raise_grasper', 'args': {}})
        env.execute_action({'name': 'move_grasper', 'args': {'x': 0.4, 'y': -0.4}})
        env.execute_action({'name': 'lower_grasper', 'args': {}})

        pre_state = env.snapshot()
        pre_scene = copy.deepcopy(env.scene)

        env.execute_action({'name': 'open_grasper', 'args': {}})

        result = self.verifier.verify_transition(
            pre_state,
            {'name': 'open_grasper', 'args': {}},
            env.snapshot(),
            pre_scene=pre_scene,
            post_scene=env.scene,
        )

        self.assertTrue(result['all_satisfied'])
        self.assertTrue(result['derived_aps']['pre_grasper_holding'])
        self.assertTrue(result['derived_aps']['pre_held_object_resting_on_object'])
        self.assertTrue(result['derived_aps']['pre_support_can_support_held'])
        self.assertFalse(result['derived_aps']['post_grasper_holding'])

    def test_grounded_property_can_be_violated(self):
        env = ShrdluBlocksEnv()

        pre_state = env.snapshot()
        pre_scene = copy.deepcopy(env.scene)

        env.execute_action({'name': 'highlight_object', 'args': {'obj_id': 10}})

        result = self.verifier.verify_transition(
            pre_state,
            {'name': 'highlight_object', 'args': {'obj_id': 10}},
            env.snapshot(),
            pre_scene=pre_scene,
            post_scene=env.scene,
        )

        self.assertFalse(result['all_satisfied'])
        violated_ids = {item['id'] for item in result['violations']}
        self.assertIn('prop.object_10_never_highlighted', violated_ids)
        self.assertTrue(result['derived_aps']['object_10_highlighted'])

    def test_active_property_set_excludes_eventuality_checks(self):
        property_ids = [item['id'] for item in self.verifier.properties]

        self.assertEqual(list(ACTIVE_PROPERTY_IDS), property_ids)
        self.assertEqual(6, len(property_ids))
        self.assertNotIn('prop.lowered_eventually_raised', property_ids)
        self.assertNotIn('prop.closed_eventually_open', property_ids)

    def test_tla_verifier_uses_active_property_set(self):
        env = ShrdluBlocksEnv()
        state = env.snapshot()
        ap_trace = [self.verifier._evaluate_state_aps(state)]
        ap_names = [item['name'] for item in self.verifier.aps]

        result = verify_ap_trace(ap_trace, ap_names)

        self.assertEqual(list(ACTIVE_PROPERTY_IDS), result['properties_checked'])
        self.assertEqual(6, len(result['properties_checked']))


if __name__ == '__main__':
    unittest.main()
