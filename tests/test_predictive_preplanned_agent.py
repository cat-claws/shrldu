import json
from pathlib import Path
import tempfile
import unittest

from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.predictive_preplanned_agent import PredictivePreplannedOllamaShrdluAgent


class PredictivePreplannedAgentTests(unittest.TestCase):
    def test_planning_state_summary_surfaces_described_objects(self):
        agent = PredictivePreplannedOllamaShrdluAgent(
            ShrdluBlocksEnv(),
            trace_dir=None,
            max_steps=4,
            max_branch_retries=1,
        )

        summary = json.loads(agent._planning_state_summary(
            agent.env.snapshot(),
            [],
            request='put blue block on small green block',
        ))

        self.assertEqual('put blue block on small green block', summary['request_focus']['raw_request'])
        self.assertEqual(['blue'], summary['request_focus']['source']['colors'])
        self.assertEqual(['block'], summary['request_focus']['source']['kinds'])
        self.assertEqual(['green'], summary['request_focus']['destination']['colors'])
        self.assertEqual(['small'], summary['request_focus']['destination']['sizes'])
        self.assertEqual(['block'], summary['request_focus']['destination']['kinds'])
        self.assertIn('described_source_objects', summary)
        self.assertIn('described_destination_objects', summary)
        self.assertEqual([6], [item['obj_id'] for item in summary['described_source_objects']])
        self.assertEqual([], summary['described_destination_objects'])

    def test_predictive_agent_plans_then_executes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = PredictivePreplannedOllamaShrdluAgent(
                ShrdluBlocksEnv(),
                trace_dir=tmpdir,
                max_steps=4,
                max_branch_retries=2,
            )
            replies = iter([
                '{"response": "I will lower the grasper.", "action": {"name": "lower_grasper", "args": {}}}',
                '{"response": "Done.", "action": {"name": "finish", "args": {}}}',
            ])

            def fake_chat(messages, schema=None):
                return next(replies)

            agent._chat = fake_chat
            result = agent.handle_user_input('lower the grasper')

            self.assertEqual('I will lower the grasper.\n\nDone.', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            self.assertEqual('predictive_preplanned', trace['planning_mode'])
            self.assertTrue(trace['planning_tree']['feasible'])
            self.assertEqual(1, len(trace['planning_tree']['accepted_plan']))
            self.assertEqual('lower_grasper', trace['planning_tree']['accepted_plan'][0]['name'])
            self.assertEqual(1, len(trace['steps']))
            self.assertTrue(trace['steps'][0]['property_verification']['all_satisfied'])

    def test_predictive_agent_retries_after_property_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ShrdluBlocksEnv()
            env.execute_action({'name': 'lower_grasper', 'args': {}})
            agent = PredictivePreplannedOllamaShrdluAgent(
                env,
                trace_dir=tmpdir,
                max_steps=4,
                max_branch_retries=3,
            )
            replies = iter([
                '{"response": "I will move the grasper.", "action": {"name": "move_grasper", "args": {"x": 0.5, "y": 0.5}}}',
                '{"response": "I will raise the grasper first.", "action": {"name": "raise_grasper", "args": {}}}',
                '{"response": "Done.", "action": {"name": "finish", "args": {}}}',
            ])

            def fake_chat(messages, schema=None):
                return next(replies)

            agent._chat = fake_chat
            result = agent.handle_user_input('raise grasper')

            self.assertEqual('I will raise the grasper first.\n\nDone.', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            property_attempts = [
                item
                for node in trace['planning_tree']['nodes']
                for item in node['attempts']
                if 'property_verification' in item
            ]
            self.assertGreaterEqual(len(trace['planning_tree']['nodes']), 1)
            self.assertEqual('raise_grasper', trace['planning_tree']['accepted_plan'][0]['name'])


if __name__ == '__main__':
    unittest.main()
