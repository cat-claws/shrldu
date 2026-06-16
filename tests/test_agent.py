import json
from pathlib import Path
import tempfile
import unittest

from shrdlu_blocks.agent import DECISION_SCHEMA, OllamaShrdluAgent
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


if __name__ == '__main__':
    unittest.main()
