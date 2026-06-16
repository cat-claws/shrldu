import json
from pathlib import Path
import tempfile
import unittest

from shrdlu_blocks.agent import (
    DECISION_SCHEMA,
    OllamaShrdluAgent,
    OpenAICompatibleShrdluAgent,
    PLAN_SCHEMA,
    PreplannedOllamaShrdluAgent,
)
from shrdlu_blocks.env import ShrdluBlocksEnv


class ParsingTests(unittest.TestCase):
    def test_parse_accepts_string_action(self):
        decision = OllamaShrdluAgent._parse_decision(
            '{"response": "Done.", "action": "finish", "args": {}}'
        )
        self.assertEqual('finish', decision['action']['name'])
        self.assertEqual({}, decision['action']['args'])

    def test_parse_accepts_top_level_name(self):
        decision = OllamaShrdluAgent._parse_decision(
            '{"response": "Done.", "name": "finish", "arguments": {}}'
        )
        self.assertEqual('finish', decision['action']['name'])
        self.assertEqual({}, decision['action']['args'])

    def test_parse_rejects_missing_action(self):
        with self.assertRaisesRegex(ValueError, 'action object'):
            OllamaShrdluAgent._parse_decision('{"response": "Hello"}')

    def test_parse_plan_accepts_full_plan(self):
        plan = PreplannedOllamaShrdluAgent._parse_plan(
            '{"response": "Plan ready.", "plan": [{"name": "raise_grasper", "args": {}}], "finish_response": "Done."}'
        )
        self.assertEqual('Plan ready.', plan['response'])
        self.assertEqual('Done.', plan['finish_response'])
        self.assertEqual('raise_grasper', plan['plan'][0]['name'])

    def test_parse_plan_rejects_missing_plan_array(self):
        with self.assertRaisesRegex(ValueError, 'plan array'):
            PreplannedOllamaShrdluAgent._parse_plan(
                '{"response": "Hello", "finish_response": "Done."}'
            )


class RetryTests(unittest.TestCase):
    def test_request_decision_retries_after_invalid_reply(self):
        agent = OllamaShrdluAgent(ShrdluBlocksEnv())
        replies = iter([
            '{"response": "Hello"}',
            '{"response": "Done.", "action": {"name": "finish", "args": {}}}',
        ])
        agent._chat = lambda messages: next(replies)

        content, decision, attempts = agent._request_decision([
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'user'},
        ])

        self.assertIn('"name": "finish"', content)
        self.assertEqual('finish', decision['action']['name'])
        self.assertEqual(2, len(attempts))

    def test_chat_uses_json_schema(self):
        agent = OllamaShrdluAgent(ShrdluBlocksEnv())
        captured = {}

        def fake_urlopen(req):
            import json

            captured.update(json.loads(req.data.decode('utf-8')))

            class Response:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

                def read(self_inner):
                    return b'{"message": {"content": "{\\"response\\": \\"Done.\\", \\"action\\": {\\"name\\": \\"finish\\", \\"args\\": {}}}"}}'

            return Response()

        from unittest.mock import patch

        with patch('shrdlu_blocks.agent.request.urlopen', fake_urlopen):
            agent._chat([{'role': 'system', 'content': 'system'}])

        self.assertEqual(DECISION_SCHEMA, captured['format'])

    def test_chat_accepts_custom_schema(self):
        agent = OllamaShrdluAgent(ShrdluBlocksEnv())
        captured = {}

        def fake_urlopen(req):
            import json

            captured.update(json.loads(req.data.decode('utf-8')))

            class Response:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

                def read(self_inner):
                    return b'{"message": {"content": "{\\"response\\": \\"Done.\\", \\"plan\\": [], \\"finish_response\\": \\"Done.\\"}"}}'

            return Response()

        from unittest.mock import patch

        with patch('shrdlu_blocks.agent.request.urlopen', fake_urlopen):
            agent._chat([{'role': 'system', 'content': 'system'}], schema=PLAN_SCHEMA)

        self.assertEqual(PLAN_SCHEMA, captured['format'])

    def test_error_trace_is_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = OllamaShrdluAgent(ShrdluBlocksEnv(), trace_dir=tmpdir)
            agent._chat = lambda messages: '{"response": "Hello"}'

            result = agent.handle_user_input('hello')

            self.assertIn('Trace saved to', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            self.assertEqual('error', trace['status'])
            self.assertEqual('hello', trace['request'])

    def test_finish_trace_is_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = OllamaShrdluAgent(ShrdluBlocksEnv(), trace_dir=tmpdir)
            agent._chat = lambda messages: '{"response": "Done.", "action": {"name": "finish", "args": {}}}'

            result = agent.handle_user_input('hello')

            self.assertEqual('Done.', result)
            trace_files = list(Path(tmpdir).glob('trace_*.json'))
            self.assertEqual(1, len(trace_files))
            trace = json.loads(trace_files[0].read_text(encoding='utf-8'))
            self.assertEqual('finished', trace['status'])
            self.assertEqual('Done.', trace['final_message'])

    def test_openai_compatible_chat_returns_message_content(self):
        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                self.assertEqual('model-name', kwargs['model'])
                self.assertEqual(0.2, kwargs['temperature'])
                self.assertEqual(512, kwargs['max_tokens'])
                return type(
                    'Response',
                    (),
                    {
                        'choices': [
                            type(
                                'Choice',
                                (),
                                {
                                    'message': type(
                                        'Message',
                                        (),
                                        {
                                            'content': '{"response": "Done.", "action": {"name": "finish", "args": {}}}',
                                        },
                                    )(),
                                },
                            )(),
                        ],
                    },
                )()

        fake_client = type(
            'Client',
            (),
            {
                'chat': type(
                    'Chat',
                    (),
                    {'completions': FakeCompletions()},
                )(),
            },
        )()

        agent = OpenAICompatibleShrdluAgent(
            ShrdluBlocksEnv(),
            model='model-name',
            client=fake_client,
        )

        content = agent._chat([{'role': 'system', 'content': 'system'}])

        self.assertIn('"finish"', content)

    def test_preplanned_agent_executes_without_replanning(self):
        agent = PreplannedOllamaShrdluAgent(ShrdluBlocksEnv(), trace_dir=None)
        calls = []

        def fake_chat(messages, schema=DECISION_SCHEMA):
            calls.append({'messages': messages, 'schema': schema})
            return (
                '{"response": "Plan ready.", "plan": ['
                '{"name": "move_grasper", "args": {"x": -0.1, "y": 0.4}}, '
                '{"name": "lower_grasper", "args": {}}, '
                '{"name": "close_grasper", "args": {}}'
                '], "finish_response": "Done."}'
            )

        agent._chat = fake_chat

        result = agent.handle_user_input('pick up the blue block')

        self.assertEqual('Plan ready.\n\nDone.', result)
        self.assertEqual(1, len(calls))
        self.assertEqual(PLAN_SCHEMA, calls[0]['schema'])


if __name__ == '__main__':
    unittest.main()
