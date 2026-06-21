import json
from pathlib import Path
import tempfile
import unittest

from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.suffix_predictive_preplanned_agent import (
    SuffixPredictivePreplannedOllamaShrdluAgent,
)


class SuffixPredictivePreplannedAgentTests(unittest.TestCase):
    def test_suffix_cleanup_skips_unsupported_held_object(self):
        class FakeEnv:
            def __init__(self):
                self.actions = []

            def snapshot(self):
                return {
                    'grasper_lowered': True,
                    'grasper_closed': True,
                    'grasped_object': 4,
                    'objects': [
                        {'obj_id': 4, 'resting_on': None, 'can_support': False},
                        {'obj_id': 9, 'resting_on': 1, 'can_support': True},
                    ],
                }

            def execute_action(self, action):
                self.actions.append(action)
                return 'OK'

        agent = object.__new__(SuffixPredictivePreplannedOllamaShrdluAgent)
        agent._env = FakeEnv()
        trace = {}

        agent._execute_grasper_cleanup(trace)

        self.assertEqual([], agent._env.actions)
        self.assertNotIn('cleanup_steps', trace)
        self.assertIn('cleanup_notes', trace)
        self.assertIn('Skipped grasper cleanup', trace['cleanup_notes'][0])

    def test_symbolic_prediction_foresees_object_lowered_onto_obj4(self):
        agent = SuffixPredictivePreplannedOllamaShrdluAgent(
            ShrdluBlocksEnv(),
            trace_dir=None,
            max_steps=8,
            max_branch_retries=1,
        )
        initial_world_state = {
            'default_grasper': None,
            'grasper_closed': False,
            'grasper_lowered': False,
            'grasped_object': None,
            'objects': [
                {'obj_id': 0, 'kind': 'grasper', 'graspable': False, 'can_support': False, 'resting_on': None, 'position': {'x': -0.1, 'y': 0.4, 'z': 0.45}, 'tags': {}},
                {'obj_id': 1, 'kind': 'table', 'graspable': False, 'can_support': True, 'resting_on': None, 'position': {'x': 0, 'y': 0, 'z': 0}, 'tags': {}},
                {'obj_id': 2, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': 1, 'position': {'x': -0.3, 'y': 0.1, 'z': 0}, 'tags': {}},
                {'obj_id': 4, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 6, 'position': {'x': -0.1, 'y': 0.4, 'z': 0.2}, 'tags': {}},
                {'obj_id': 5, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': 2, 'position': {'x': -0.3, 'y': 0.05, 'z': 0.15}, 'tags': {}},
                {'obj_id': 6, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': 1, 'position': {'x': -0.1, 'y': 0.4, 'z': 0}, 'tags': {}},
                {'obj_id': 7, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': 1, 'position': {'x': 0.1, 'y': -0.15, 'z': 0}, 'tags': {}},
                {'obj_id': 8, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 7, 'position': {'x': 0.15, 'y': -0.1, 'z': 0.15}, 'tags': {}},
                {'obj_id': 9, 'kind': 'box', 'graspable': False, 'can_support': True, 'resting_on': 1, 'position': {'x': 0.25, 'y': 0.25, 'z': 0}, 'tags': {}},
                {'obj_id': 10, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 9, 'position': {'x': 0.25, 'y': 0.25, 'z': 0}, 'tags': {}},
            ],
        }
        actions = [
            {'name': 'move_grasper', 'args': {'x': -0.3, 'y': 0.05}},
            {'name': 'lower_grasper', 'args': {}},
            {'name': 'close_grasper', 'args': {}},
            {'name': 'raise_grasper', 'args': {}},
            {'name': 'move_grasper', 'args': {'x': -0.1, 'y': 0.4}},
            {'name': 'lower_grasper', 'args': {}},
        ]

        predicted_world_state, _notes = agent._predict_world_state_after_actions(
            initial_world_state,
            actions,
        )
        predicted_ap_state = agent._build_initial_ap_state(predicted_world_state)

        obj5 = next(obj for obj in predicted_world_state['objects'] if obj['obj_id'] == 5)
        self.assertEqual(4, obj5['resting_on'])
        self.assertTrue(predicted_ap_state['some_object_resting_on_4'])

    def test_symbolic_prediction_uses_top_object_when_snapshot_z_ties(self):
        agent = SuffixPredictivePreplannedOllamaShrdluAgent(
            ShrdluBlocksEnv(),
            trace_dir=None,
            max_steps=8,
            max_branch_retries=1,
        )
        initial_world_state = {
            'default_grasper': 0,
            'grasper_closed': True,
            'grasper_lowered': False,
            'grasped_object': 2,
            'objects': [
                {'obj_id': 0, 'kind': 'grasper', 'graspable': False, 'can_support': False, 'resting_on': None, 'position': {'x': 0.25, 'y': 0.25, 'z': 0.45}, 'tags': {}},
                {'obj_id': 1, 'kind': 'table', 'graspable': False, 'can_support': True, 'resting_on': None, 'position': {'x': 0, 'y': 0, 'z': 0}, 'tags': {}},
                {'obj_id': 2, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': None, 'position': {'x': 0.25, 'y': 0.25, 'z': 0.4}, 'tags': {}},
                {'obj_id': 4, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 3, 'position': {'x': -0.25, 'y': -0.2, 'z': 0.08}, 'tags': {}},
                {'obj_id': 7, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': 1, 'position': {'x': 0.1, 'y': -0.15, 'z': 0}, 'tags': {}},
                {'obj_id': 8, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 7, 'position': {'x': 0.15, 'y': -0.1, 'z': 0.15}, 'tags': {}},
                {'obj_id': 9, 'kind': 'box', 'graspable': False, 'can_support': True, 'resting_on': 1, 'position': {'x': 0.25, 'y': 0.25, 'z': 0}, 'tags': {}},
                {'obj_id': 10, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 9, 'position': {'x': 0.25, 'y': 0.25, 'z': 0}, 'tags': {}},
            ],
        }
        actions = [{'name': 'lower_grasper', 'args': {}}]

        predicted_world_state, _notes = agent._predict_world_state_after_actions(
            initial_world_state,
            actions,
        )
        predicted_ap_state = agent._build_initial_ap_state(predicted_world_state)

        obj2 = next(obj for obj in predicted_world_state['objects'] if obj['obj_id'] == 2)
        self.assertEqual(10, obj2['resting_on'])
        self.assertTrue(predicted_ap_state['some_object_resting_on_10'])

    def test_symbolic_prediction_foresees_object_lowered_onto_obj8(self):
        agent = SuffixPredictivePreplannedOllamaShrdluAgent(
            ShrdluBlocksEnv(),
            trace_dir=None,
            max_steps=8,
            max_branch_retries=1,
        )
        initial_world_state = {
            'default_grasper': 0,
            'grasper_closed': True,
            'grasper_lowered': False,
            'grasped_object': 2,
            'objects': [
                {'obj_id': 0, 'kind': 'grasper', 'graspable': False, 'can_support': False, 'resting_on': None, 'position': {'x': 0.15, 'y': -0.1, 'z': 0.45}, 'tags': {}},
                {'obj_id': 1, 'kind': 'table', 'graspable': False, 'can_support': True, 'resting_on': None, 'position': {'x': 0, 'y': 0, 'z': 0}, 'tags': {}},
                {'obj_id': 2, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': None, 'position': {'x': 0.15, 'y': -0.1, 'z': 0.4}, 'tags': {}},
                {'obj_id': 4, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 3, 'position': {'x': -0.25, 'y': -0.2, 'z': 0.08}, 'tags': {}},
                {'obj_id': 7, 'kind': 'block', 'graspable': True, 'can_support': True, 'resting_on': 1, 'position': {'x': 0.1, 'y': -0.15, 'z': 0}, 'tags': {}},
                {'obj_id': 8, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 7, 'position': {'x': 0.15, 'y': -0.1, 'z': 0.15}, 'tags': {}},
                {'obj_id': 10, 'kind': 'pyramid', 'graspable': True, 'can_support': False, 'resting_on': 9, 'position': {'x': 0.25, 'y': 0.25, 'z': 0}, 'tags': {}},
            ],
        }
        actions = [{'name': 'lower_grasper', 'args': {}}]

        predicted_world_state, _notes = agent._predict_world_state_after_actions(
            initial_world_state,
            actions,
        )
        predicted_ap_state = agent._build_initial_ap_state(predicted_world_state)

        obj2 = next(obj for obj in predicted_world_state['objects'] if obj['obj_id'] == 2)
        self.assertEqual(8, obj2['resting_on'])
        self.assertTrue(predicted_ap_state['some_object_resting_on_8'])

    def test_suffix_predictive_parser_recovers_nested_world_state_from_response_string(self):
        agent = SuffixPredictivePreplannedOllamaShrdluAgent(
            ShrdluBlocksEnv(),
            trace_dir=None,
            max_steps=4,
            max_branch_retries=1,
        )
        current_world_state = agent.env.snapshot()
        current_state = agent._build_initial_property_state(
            current_world_state,
            scene=agent.env.scene,
        )
        content = json.dumps({
            'response': json.dumps({
                'response': 'The grasper is now holding object 10.',
                'predicted_state': {
                    'property_results': [
                        {'id': item['id'], 'satisfied': item['satisfied']}
                        for item in current_state['property_results']
                    ],
                },
                'predicted_world_state': {
                    'grasper_closed': True,
                    'grasper_lowered': True,
                    'grasped_object': 10,
                    'objects': [
                        {'obj_id': 10, 'grasped_by': 0, 'resting_on': None},
                    ],
                },
            }),
        })

        parsed = agent._force_parse_property_state_prediction(
            content,
            current_state,
            current_world_state,
        )

        self.assertEqual('The grasper is now holding object 10.', parsed['response'])
        self.assertTrue(parsed['predicted_world_state']['grasper_closed'])
        self.assertTrue(parsed['predicted_world_state']['grasper_lowered'])
        self.assertEqual(10, parsed['predicted_world_state']['grasped_object'])
        obj10 = next(
            item for item in parsed['predicted_world_state']['objects']
            if item['obj_id'] == 10
        )
        self.assertEqual(0, obj10['grasped_by'])
        self.assertIsNone(obj10['resting_on'])

    def test_suffix_predictive_agent_rejects_trace_style_bad_release_when_prediction_is_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SuffixPredictivePreplannedOllamaShrdluAgent(
                ShrdluBlocksEnv(),
                trace_dir=tmpdir,
                max_steps=8,
                max_branch_retries=1,
            )
            initial_world_state = agent.env.snapshot()
            initial_property_state = agent._build_initial_property_state(
                initial_world_state,
                scene=agent.env.scene,
            )
            property_results = [
                {'id': item['id'], 'satisfied': item['satisfied']}
                for item in initial_property_state['property_results']
            ]

            def nested_prediction(response_text, world_updates, overrides=None):
                results = []
                for item in property_results:
                    satisfied = item['satisfied']
                    if overrides and item['id'] in overrides:
                        satisfied = overrides[item['id']]
                    results.append({'id': item['id'], 'satisfied': satisfied})
                return json.dumps({
                    'response': json.dumps({
                        'response': response_text,
                        'predicted_state': {'property_results': results},
                        'predicted_world_state': world_updates,
                    }),
                })

            replies = iter([
                json.dumps({
                    'response': 'Pick up the blue pyramid and place it on the green small block.',
                    'plan': [
                        {'name': 'move_grasper', 'args': {'x': 0.25, 'y': 0.25}},
                        {'name': 'lower_grasper', 'args': {}},
                        {'name': 'close_grasper', 'args': {}},
                        {'name': 'raise_grasper', 'args': {}},
                        {'name': 'move_grasper', 'args': {'x': -0.25, 'y': -0.2}},
                        {'name': 'lower_grasper', 'args': {}},
                        {'name': 'open_grasper', 'args': {}},
                    ],
                    'finish_response': 'Done.',
                }),
                nested_prediction(
                    'Moved to the blue pyramid.',
                    {'objects': [{'obj_id': 0, 'position': {'x': 0.25, 'y': 0.25}}]},
                ),
                nested_prediction(
                    'Lowered over the blue pyramid.',
                    {'grasper_lowered': True},
                ),
                nested_prediction(
                    'Closed around and grasped the blue pyramid.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': True,
                        'grasped_object': 10,
                        'objects': [{'obj_id': 10, 'grasped_by': 0, 'resting_on': None}],
                    },
                ),
                nested_prediction(
                    'Raised while holding the blue pyramid.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': False,
                        'grasped_object': 10,
                        'objects': [{'obj_id': 10, 'grasped_by': 0, 'resting_on': None}],
                    },
                ),
                nested_prediction(
                    'Moved above the green small block.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': False,
                        'grasped_object': 10,
                        'objects': [
                            {'obj_id': 0, 'position': {'x': -0.25, 'y': -0.2}},
                            {'obj_id': 10, 'grasped_by': 0, 'resting_on': None},
                        ],
                    },
                ),
                nested_prediction(
                    'Lowered while still holding the blue pyramid.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': True,
                        'grasped_object': 10,
                        'objects': [{'obj_id': 10, 'grasped_by': 0, 'resting_on': None}],
                    },
                ),
                nested_prediction(
                    'Open would fail because the held pyramid is not resting on a support.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': True,
                        'grasped_object': 10,
                        'objects': [{'obj_id': 10, 'grasped_by': 0, 'resting_on': None}],
                    },
                    overrides={
                        'prop.open_while_holding_starts_from_supported_state': False,
                        'prop.open_while_holding_starts_from_supporting_state': False,
                        'prop.open_ends_in_open_state': False,
                        'prop.open_while_holding_ends_in_not_holding_state': False,
                    },
                ),
            ])

            def fake_chat(messages, schema=None):
                return next(replies)

            agent._chat = fake_chat
            result = agent.handle_user_input('put blue pyramid on the green small block')

            self.assertIn('No feasible property-satisfying plan found.', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            self.assertEqual('infeasible', trace['status'])
            self.assertEqual([], trace['steps'])
            rollout = trace['planning_tree']['nodes'][0]['attempts'][0]['predicted_rollout']
            self.assertEqual(7, len(rollout))
            self.assertEqual(10, rollout[2]['predicted_world_state']['grasped_object'])
            self.assertFalse(rollout[6]['property_verification']['all_satisfied'])

    def test_suffix_predictive_agent_plans_full_suffix_then_executes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SuffixPredictivePreplannedOllamaShrdluAgent(
                ShrdluBlocksEnv(),
                trace_dir=tmpdir,
                max_steps=4,
                max_branch_retries=2,
            )
            replies = iter([
                json.dumps({
                    'response': 'I will lower the grasper.',
                    'plan': [{'name': 'lower_grasper', 'args': {}}],
                    'finish_response': 'Done.',
                }),
                json.dumps({
                    'response': 'Yes, the goal is satisfied.',
                    'goal_satisfied': True,
                }),
            ])

            def fake_chat(messages, schema=None):
                return next(replies)

            agent._chat = fake_chat
            result = agent.handle_user_input('lower the grasper')

            self.assertEqual('I will lower the grasper.\n\nDone.', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            self.assertEqual('suffix_predictive_preplanned', trace['planning_mode'])
            self.assertTrue(trace['planning_tree']['feasible'])
            self.assertEqual(1, len(trace['planning_tree']['accepted_plan']))
            self.assertEqual('lower_grasper', trace['planning_tree']['accepted_plan'][0]['name'])
            self.assertEqual(1, len(trace['steps']))

    def test_suffix_predictive_agent_replans_from_first_violation_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SuffixPredictivePreplannedOllamaShrdluAgent(
                ShrdluBlocksEnv(),
                trace_dir=tmpdir,
                max_steps=5,
                max_branch_retries=2,
            )
            property_ids = [item['id'] for item in agent._property_verifier.properties]
            plan_replies = iter([
                json.dumps({
                    'response': 'I will lower the grasper and then move it.',
                    'plan': [
                        {'name': 'lower_grasper', 'args': {}},
                        {'name': 'move_grasper', 'args': {'x': 0.5, 'y': 0.5}},
                    ],
                    'finish_response': 'Done.',
                }),
                json.dumps({
                    'response': 'I will raise the grasper.',
                    'plan': [{'name': 'raise_grasper', 'args': {}}],
                    'finish_response': 'Done.',
                }),
                json.dumps({
                    'response': 'Done.',
                    'plan': [],
                    'finish_response': 'Done.',
                }),
            ])
            prediction_replies = iter([
                json.dumps({
                    'response': 'Lowering is valid here.',
                    'predicted_state': {
                        'property_results': [
                            {'id': pid, 'satisfied': True}
                            for pid in property_ids
                        ],
                    },
                    'predicted_world_state': {'grasper_lowered': True},
                }),
                json.dumps({
                    'response': 'Moving while lowered violates the property.',
                    'predicted_state': {
                        'property_results': [
                            {
                                'id': pid,
                                'satisfied': pid != 'prop.no_move_from_lowered_state',
                            }
                            for pid in property_ids
                        ],
                    },
                    'predicted_world_state': {'grasper_lowered': True},
                }),
                json.dumps({
                    'response': 'Raising is valid.',
                    'predicted_state': {
                        'property_results': [
                            {'id': pid, 'satisfied': True}
                            for pid in property_ids
                        ],
                    },
                    'predicted_world_state': {'grasper_lowered': False},
                }),
            ])

            def fake_chat(messages, schema=None):
                if schema and 'plan' in schema.get('properties', {}):
                    return next(plan_replies)
                return next(prediction_replies)

            agent._chat = fake_chat
            result = agent.handle_user_input('raise the grasper after lowering it')

            self.assertEqual('I will raise the grasper.\n\nDone.', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            self.assertEqual(
                ['lower_grasper', 'raise_grasper'],
                [item['name'] for item in trace['planning_tree']['accepted_plan']],
            )
            root_attempt = trace['planning_tree']['nodes'][0]['attempts'][0]
            self.assertEqual(2, len(root_attempt['predicted_rollout']))
            self.assertFalse(
                root_attempt['predicted_rollout'][1]['property_verification']['all_satisfied']
            )
            self.assertEqual(1, len(trace['planning_tree']['nodes'][0]['children']))

    def test_suffix_predictive_agent_replans_from_verified_prefix_after_late_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SuffixPredictivePreplannedOllamaShrdluAgent(
                ShrdluBlocksEnv(),
                trace_dir=tmpdir,
                max_steps=12,
                max_branch_retries=2,
            )
            property_ids = [item['id'] for item in agent._property_verifier.properties]
            plan_replies = iter([
                json.dumps({
                    'response': 'I will pick up the green pyramid, move above the white box, and release.',
                    'plan': [
                        {'name': 'move_grasper', 'args': {'x': -0.25, 'y': -0.2}},
                        {'name': 'lower_grasper', 'args': {}},
                        {'name': 'close_grasper', 'args': {}},
                        {'name': 'raise_grasper', 'args': {}},
                        {'name': 'move_grasper', 'args': {'x': 0.25, 'y': 0.25}},
                        {'name': 'open_grasper', 'args': {}},
                    ],
                    'finish_response': 'Done.',
                }),
                json.dumps({
                    'response': 'I will lower and release onto the white box.',
                    'plan': [
                        {'name': 'lower_grasper', 'args': {}},
                        {'name': 'open_grasper', 'args': {}},
                    ],
                    'finish_response': 'Done.',
                }),
            ])

            def prediction_reply(response_text, world_state, false_props=None):
                false_props = set(false_props or [])
                return json.dumps({
                    'response': response_text,
                    'predicted_state': {
                        'property_results': [
                            {'id': pid, 'satisfied': pid not in false_props}
                            for pid in property_ids
                        ],
                    },
                    'predicted_world_state': world_state,
                })

            prediction_replies = iter([
                prediction_reply('Moved over the green pyramid.', {'objects': [{'obj_id': 0, 'position': {'x': -0.25, 'y': -0.2}}]}),
                prediction_reply('Lowered onto the green pyramid.', {'grasper_lowered': True}),
                prediction_reply(
                    'Closed and grasped the green pyramid.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': True,
                        'grasped_object': 4,
                        'objects': [{'obj_id': 4, 'grasped_by': 0, 'resting_on': None}],
                    },
                ),
                prediction_reply(
                    'Raised while holding the green pyramid.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': False,
                        'grasped_object': 4,
                        'objects': [{'obj_id': 4, 'grasped_by': 0, 'resting_on': None}],
                    },
                ),
                prediction_reply(
                    'Moved above the white box.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': False,
                        'grasped_object': 4,
                        'objects': [
                            {'obj_id': 0, 'position': {'x': 0.25, 'y': 0.25}},
                            {'obj_id': 4, 'grasped_by': 0, 'resting_on': None},
                        ],
                    },
                ),
                prediction_reply(
                    'Opening now would violate the support properties because the grasper is still raised.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': False,
                        'grasped_object': 4,
                        'objects': [{'obj_id': 4, 'grasped_by': 0, 'resting_on': None}],
                    },
                    false_props={
                        'prop.open_while_holding_starts_from_supported_state',
                        'prop.open_while_holding_starts_from_supporting_state',
                        'prop.open_ends_in_open_state',
                        'prop.open_while_holding_ends_in_not_holding_state',
                    },
                ),
                prediction_reply(
                    'Lowered until the pyramid is resting on the white box.',
                    {
                        'grasper_closed': True,
                        'grasper_lowered': True,
                        'grasped_object': 4,
                        'objects': [{'obj_id': 4, 'grasped_by': 0, 'resting_on': 9}],
                    },
                ),
                prediction_reply(
                    'Opened and released the pyramid onto the white box.',
                    {
                        'grasper_closed': False,
                        'grasper_lowered': True,
                        'grasped_object': None,
                        'objects': [{'obj_id': 4, 'grasped_by': None, 'resting_on': 9}],
                    },
                ),
            ])

            def fake_chat(messages, schema=None):
                if schema and 'plan' in schema.get('properties', {}):
                    return next(plan_replies)
                return next(prediction_replies)

            agent._chat = fake_chat
            result = agent.handle_user_input('put green pyramid onto white box')

            self.assertIn('I will lower and release onto the white box.', result)
            self.assertIn('Plan execution failed.', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            self.assertEqual('error', trace['status'])
            self.assertEqual(7, len(trace['planning_tree']['accepted_plan']))
            self.assertEqual(2, len(trace['planning_tree']['nodes']))
            first_attempt = trace['planning_tree']['nodes'][0]['attempts'][0]
            self.assertEqual('property_violation', first_attempt['failure_feedback']['type'])
            self.assertEqual(1, first_attempt['child_node_id'])


if __name__ == '__main__':
    unittest.main()
