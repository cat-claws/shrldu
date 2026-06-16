"""Ollama-backed natural-language agent for the SHRDLU blocks world."""

import json
from urllib import error, request
from typing import Dict, List, Optional

from shrdlu_blocks.env import ShrdluBlocksEnv

__all__ = ['OllamaShrdluAgent']


SYSTEM_PROMPT = """You control a blocks-world simulator through a small validated action API.

Rules:
- Return exactly one action at a time.
- Use only the allowed action names and argument types you are given.
- Base decisions only on the current world state and latest action result.
- If the task is complete, return the finish action instead of another simulator action.
- If the user asks a conversational question, asks for a status summary, or explicitly says not to act, use the finish action.
- Keep the response short and factual.

Return strict JSON only.

Examples:
{"response": "I will move the grasper over the blue block.", "action": {"name": "move_grasper", "args": {"x": -0.1, "y": 0.4}}}
{"response": "Done.", "action": {"name": "finish", "args": {}}}
"""


class OllamaShrdluAgent:
    """A small tool-using agent loop for the SHRDLU blocks environment."""

    def __init__(self, env: ShrdluBlocksEnv, model: str = 'qwen3:14b',
                 host: str = 'http://127.0.0.1:11434', max_steps: int = 8):
        self._env = env
        self._model = model
        self._host = host.rstrip('/')
        self._max_steps = max_steps

    @property
    def env(self) -> ShrdluBlocksEnv:
        return self._env

    def handle_user_input(self, text: str) -> str:
        """Handle a natural-language request against the live environment."""
        request = (text or '').strip()
        if not request:
            return 'Please enter a command or instruction.'
        if request.lower() in {'help', '/help'}:
            return self._env.command_help()
        if request.lower() in {'reset', '/reset'}:
            self._env.reset()
            return 'Environment reset.\n\n' + self._env.snapshot_text()
        if request.lower().startswith('/command '):
            result = self._env.execute_command(request[len('/command '):])
            return self._format_reply('Executed direct simulator command.', result)
        return self._run_agent_loop(request)

    def _run_agent_loop(self, request: str) -> str:
        history: List[Dict[str, str]] = [{
            'role': 'system',
            'content': SYSTEM_PROMPT,
        }]
        action_help = self._env.action_help()
        observation = self._env.snapshot_text()
        last_result = 'No simulator command has been executed yet.'

        for step_index in range(self._max_steps):
            history.append({
                'role': 'user',
                'content': self._build_user_prompt(request, action_help, observation, last_result),
            })
            try:
                content = self._chat(history).strip()
                decision = self._parse_decision(content)
            except Exception as exc:
                return "Agent error: %s" % exc
            history.append({'role': 'assistant', 'content': content})
            action = decision.get('action', {})
            response_text = self._normalize_response_text(
                decision.get('response', ''),
                action.get('name') == 'finish',
            )
            if action.get('name') == 'finish':
                return self._format_reply(response_text, None)

            try:
                result = self._env.execute_action(action)
            except Exception as exc:
                result = "ERROR: %s" % exc
            executed_action = self._format_action(action)
            observation = self._env.snapshot_text()
            last_result = "Executed %s.\nResult: %s" % (executed_action, result)
            if step_index == self._max_steps - 1:
                return self._format_reply(
                    response_text + "\n\nReached max agent steps.",
                    last_result,
                )
        return 'Agent stopped without producing a result.'

    @staticmethod
    def _build_user_prompt(request: str, action_help: str, observation: str,
                           last_result: str) -> str:
        return "\n\n".join([
            "User request:\n%s" % request,
            action_help,
            observation,
            "Latest simulator result:\n%s" % last_result,
            'JSON schema: {"response": "...", "action": {"name": "...", "args": {...}}}',
            'Use {"response": "...", "action": {"name": "finish", "args": {}}} when done.',
            "Return strict JSON only.",
        ])

    @staticmethod
    def _parse_decision(content: str) -> Dict[str, str]:
        content = OllamaShrdluAgent._extract_json_object(content)
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Model did not return valid JSON: %s" % content) from exc
        if not isinstance(decision, dict):
            raise ValueError("Model reply must be a JSON object.")
        action = decision.get('action', {})
        if not isinstance(action, dict):
            raise ValueError("Model reply must include an action object.")
        return {
            'response': str(decision.get('response', '')),
            'action': {
                'name': str(action.get('name', '')),
                'args': action.get('args', {}) if isinstance(action.get('args', {}), dict) else {},
            },
        }

    @staticmethod
    def _format_reply(response_text: str, command_result: Optional[str]) -> str:
        if not command_result:
            return response_text
        return response_text + "\n\n" + command_result

    @staticmethod
    def _format_action(action: Dict[str, object]) -> str:
        return json.dumps(action, sort_keys=True)

    @staticmethod
    def _extract_json_object(content: str) -> str:
        content = content.strip()
        if content.startswith('{') and content.endswith('}'):
            return content
        start = content.find('{')
        end = content.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return content
        return content[start:end + 1]

    @staticmethod
    def _normalize_response_text(text: str, is_finish: bool) -> str:
        text = (text or '').strip()
        if text:
            return text
        if is_finish:
            return 'Done.'
        return 'No response provided.'

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        payload = json.dumps({
            'model': self._model,
            'messages': messages,
            'stream': False,
        }).encode('utf-8')
        req = request.Request(
            self._host + '/api/chat',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with request.urlopen(req) as response:
                body = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:
            details = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError("Ollama HTTP error %s: %s" % (exc.code, details)) from exc
        except error.URLError as exc:
            raise RuntimeError("Could not reach Ollama at %s" % self._host) from exc
        try:
            return body['message']['content']
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Unexpected Ollama response: %r" % body) from exc
