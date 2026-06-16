"""Ollama-backed natural-language agent for the SHRDLU blocks world."""

from datetime import datetime, timezone
import json
from pathlib import Path
from urllib import error, request
from typing import Dict, List, Optional

from shrdlu_blocks.env import ShrdluBlocksEnv

__all__ = ['OllamaShrdluAgent']

DEFAULT_MODEL = 'qwen3.5:27b'
DEFAULT_MAX_STEPS = 50
DEFAULT_TRACE_DIR = 'agent_traces'


SYSTEM_PROMPT = """You control a blocks-world simulator through a small validated action API.

Rules:
- Return exactly one action at a time.
- Use only the allowed action names and argument types you are given.
- Base decisions only on the current world state and latest action result.
- If the task is complete, return the finish action instead of another simulator action.
- If the user asks a conversational question, asks for a status summary, or explicitly says not to act, use the finish action.
- Do not repeat an action that already succeeded unless the latest simulator result clearly shows it failed or the world state changed.
- After a successful move, highlight, open, close, lower, or raise action that satisfies the request, return finish on the next step.
- Keep the response short and factual.

Return strict JSON only.

Examples:
{"response": "I will move the grasper over the blue block.", "action": {"name": "move_grasper", "args": {"x": -0.1, "y": 0.4}}}
{"response": "Done.", "action": {"name": "finish", "args": {}}}
"""

DECISION_SCHEMA = {
    'type': 'object',
    'properties': {
        'response': {
            'type': 'string',
        },
        'action': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                },
                'args': {
                    'type': 'object',
                },
            },
            'required': ['name', 'args'],
        },
    },
    'required': ['response', 'action'],
}


class OllamaShrdluAgent:
    """A small tool-using agent loop for the SHRDLU blocks environment."""

    def __init__(self, env: ShrdluBlocksEnv, model: str = DEFAULT_MODEL,
                 host: str = 'http://127.0.0.1:11434', max_steps: int = DEFAULT_MAX_STEPS,
                 trace_dir: Optional[str] = DEFAULT_TRACE_DIR):
        self._env = env
        self._model = model
        self._host = host.rstrip('/')
        self._max_steps = max_steps
        self._trace_dir = Path(trace_dir) if trace_dir else None

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
        trace = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'model': self._model,
            'host': self._host,
            'max_steps': self._max_steps,
            'request': request,
            'steps': [],
        }
        history: List[Dict[str, str]] = [{
            'role': 'system',
            'content': SYSTEM_PROMPT,
        }]
        action_help = self._env.action_help()
        observation = self._env.snapshot_text()
        last_result = 'No simulator command has been executed yet.'

        for step_index in range(self._max_steps):
            prompt = self._build_user_prompt(request, action_help, observation, last_result)
            history.append({
                'role': 'user',
                'content': prompt,
            })
            try:
                content, decision, attempts = self._request_decision(history)
            except Exception as exc:
                trace['steps'].append({
                    'step_index': step_index,
                    'prompt': prompt,
                    'error': str(exc),
                })
                trace['status'] = 'error'
                trace['final_message'] = "Agent error: %s" % exc
                trace_path = self._write_trace(trace)
                return self._append_trace_notice(
                    "Agent error: %s" % exc,
                    trace_path,
                )
            history.append({'role': 'assistant', 'content': content})
            action = decision.get('action', {})
            response_text = self._normalize_response_text(
                decision.get('response', ''),
                action.get('name') == 'finish',
            )
            step_trace = {
                'step_index': step_index,
                'prompt': prompt,
                'attempts': attempts,
                'decision': decision,
            }
            if action.get('name') == 'finish':
                trace['steps'].append(step_trace)
                trace['status'] = 'finished'
                trace['final_message'] = response_text
                self._write_trace(trace)
                return self._format_reply(response_text, None)

            try:
                result = self._env.execute_action(action)
            except Exception as exc:
                result = "ERROR: %s" % exc
            executed_action = self._format_action(action)
            observation = self._env.snapshot_text()
            last_result = "Executed %s.\nResult: %s" % (executed_action, result)
            step_trace.update({
                'executed_action': action,
                'action_result': result,
                'observation_after': observation,
            })
            trace['steps'].append(step_trace)
            if step_index == self._max_steps - 1:
                final_message = self._format_reply(
                    response_text + "\n\nReached max agent steps.",
                    last_result,
                )
                trace['status'] = 'max_steps'
                trace['final_message'] = final_message
                trace_path = self._write_trace(trace)
                return self._append_trace_notice(final_message, trace_path)
        trace['status'] = 'stopped'
        trace['final_message'] = 'Agent stopped without producing a result.'
        trace_path = self._write_trace(trace)
        return self._append_trace_notice('Agent stopped without producing a result.', trace_path)

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
        action = OllamaShrdluAgent._normalize_action(decision)
        return {
            'response': str(decision.get('response', '')),
            'action': {
                'name': str(action.get('name', '')).strip(),
                'args': action.get('args', {}) if isinstance(action.get('args', {}), dict) else {},
            },
        }

    def _request_decision(self, history: List[Dict[str, str]]):
        attempts = list(history)
        errors = []
        attempt_log = []
        for attempt_index in range(2):
            content = self._chat(attempts).strip()
            try:
                decision = self._parse_decision(content)
            except ValueError as exc:
                errors.append(str(exc))
                attempt_log.append({
                    'attempt_index': attempt_index,
                    'raw_content': content,
                    'error': str(exc),
                })
                if attempt_index == 1:
                    break
                attempts.extend([
                    {'role': 'assistant', 'content': content},
                    {
                        'role': 'user',
                        'content': (
                            "Your previous reply was invalid: %s\n"
                            "Rewrite it as strict JSON only using this schema:\n"
                            '{"response": "...", "action": {"name": "...", "args": {...}}}'
                        ) % exc,
                    },
                ])
                continue
            action_name = decision['action']['name']
            if not action_name:
                errors.append('Model reply must include a non-empty action name.')
                attempt_log.append({
                    'attempt_index': attempt_index,
                    'raw_content': content,
                    'error': 'Model reply must include a non-empty action name.',
                    'parsed_decision': decision,
                })
                if attempt_index == 1:
                    break
                attempts.extend([
                    {'role': 'assistant', 'content': content},
                    {
                        'role': 'user',
                        'content': (
                            "Your previous reply used an empty action name.\n"
                            "Return strict JSON only and choose a valid action name or finish."
                        ),
                    },
                ])
                continue
            attempt_log.append({
                'attempt_index': attempt_index,
                'raw_content': content,
                'parsed_decision': decision,
            })
            return content, decision, attempt_log
        raise ValueError("Invalid model reply after retry: %s" % errors[-1])

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

    @staticmethod
    def _normalize_action(decision: Dict[str, object]) -> Dict[str, object]:
        raw_action = decision.get('action')
        if isinstance(raw_action, dict):
            return raw_action
        if isinstance(raw_action, str):
            return {
                'name': raw_action,
                'args': OllamaShrdluAgent._extract_action_args(decision),
            }
        action_name = decision.get('name') or decision.get('action_name')
        if isinstance(action_name, str):
            return {
                'name': action_name,
                'args': OllamaShrdluAgent._extract_action_args(decision),
            }
        raise ValueError("Model reply must include an action object.")

    @staticmethod
    def _extract_action_args(decision: Dict[str, object]) -> Dict[str, object]:
        for key in ('args', 'arguments', 'parameters'):
            value = decision.get(key)
            if isinstance(value, dict):
                return value
        raw_action = decision.get('action')
        if isinstance(raw_action, dict):
            for key in ('args', 'arguments', 'parameters'):
                value = raw_action.get(key)
                if isinstance(value, dict):
                    return value
        return {}

    def _write_trace(self, trace: Dict[str, object]) -> Optional[str]:
        if self._trace_dir is None:
            return None
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        trace_path = self._trace_dir / ('trace_%s.json' % timestamp)
        trace_path.write_text(json.dumps(trace, indent=2), encoding='utf-8')
        return str(trace_path)

    @staticmethod
    def _append_trace_notice(message: str, trace_path: Optional[str]) -> str:
        if not trace_path:
            return message
        return message + "\n\nTrace saved to %s" % trace_path

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        payload = json.dumps({
            'model': self._model,
            'messages': messages,
            'stream': False,
            'format': DECISION_SCHEMA,
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
