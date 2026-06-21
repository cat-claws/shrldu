"""Suffix-replanning predictive agents for the SHRDLU blocks environment."""

import copy
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from shrdlu_blocks.agent import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TRACE_DIR,
    OllamaShrdluAgent,
    OpenAICompatibleShrdluAgent,
    PLAN_SCHEMA,
)
from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.predictive_preplanned_agent import (
    _PredictivePreplannedShrdluAgentMixin,
)

__all__ = [
    'SuffixPredictivePreplannedOllamaShrdluAgent',
    'SuffixPredictivePreplannedOpenAICompatibleShrdluAgent',
]


SUFFIX_PREDICTIVE_PLAN_SYSTEM_PROMPT = """You are planning a SHRDLU blocks-world task before execution.

Rules:
- Read the current predicted scene carefully before planning.
- Read the object descriptions and candidate object ids before scanning the larger object lists.
- Think through the full remaining task first, then return the complete remaining action suffix up front.
- Treat each suffix attempt as self-contained and do not assume you will get to repair it later.
- Treat the plan as a dry-run sequence of primitive simulator calls that will be verified before execution.
- Use only the allowed primitive action names and the matching JSON args listed in the allowed actions schema.
- Never invent argument names that are not listed in the allowed action schema.
- Never use null, placeholders, or descriptive strings where a concrete numeric or integer argument is required.
- Ground every action argument from the structured planning state summary or current predicted state JSON.
- Base the suffix plan only on the goal, the current predicted state, the accepted action trace so far, and any failed suffix feedback shown to you.
- Prefer the shortest valid remaining plan.
- Do not include undo steps unless the current predicted state makes them necessary for the goal.
- Avoid repeated alternating patterns or repeated identical actions unless they are strictly required by a new state change.
- Resolve object references by the full description, not by one attribute alone. For example, "green box" must match both color=green and kind=box; do not substitute a green block or a white box.
- Never combine attributes across stacked or co-located objects. A phrase like "green small block" must match one object id with color=green, kind=block, and size=small.
- Match every user-mentioned attribute that exists in the scene representation, including kind, color, size, height, width, and supportability when relevant.
- Bind all user-mentioned attributes to one single object. For example, "green small block" means one object with color=green, kind=block, and size=small; it does not mean a green object on top of a small block.
- If the request is ambiguous or the described objects do not support a confident binding to one destination object id, do not substitute a merged or invented object description. Return an empty plan and explain briefly.
- For pick/place or stacking goals, identify one concrete source object from source_candidates and one concrete destination object from destination_candidates before planning the suffix.
- For pick/place or stacking goals, use only a destination object with can_support=true. Never plan to place an object onto a pyramid or any object that cannot support it.
- Ground move_grasper(x, y) by copying the coordinates of the chosen source or destination object from the object catalog.
- Before planning open_grasper while holding an object, ensure the immediately previous lowered state would leave the held object resting_on the chosen destination object in the predicted world state.
- If the held object would not yet be supported after lowering at the chosen destination coordinates, do not include open_grasper in the suffix.
- If the grasper is currently holding an object (grasper_closed=true, grasped_object is set) and the goal requires picking a different object, first complete the full drop sequence for the held object: lower_grasper -> open_grasper -> raise_grasper, then begin the new pick.
- Do not explain alternatives or reasoning.
- Keep the response short and factual.

Return strict JSON only.
"""

SUFFIX_PLAN_USER_PROMPT_TEMPLATE = """\
Goal:
{request}

{grounding_verdict}

Current predicted property truth values (property_id: true/false):
{current_property_bools_json}

Current predicted world state JSON:
{current_world_state_json}

Structured planning state summary:
{planning_state_summary}

Accepted action trace so far:
{accepted_trace_json}

Properties to satisfy:
{property_text}

Allowed primitive actions:
{action_help}

Failed suffix attempts and backtrack feedback:
{failed_attempts_json}

Use the manipulation skeleton only when it is valid: move_grasper(source) -> lower_grasper -> close_grasper -> raise_grasper -> move_grasper(destination) -> lower_grasper -> open_grasper.
Do not use open_grasper as a default final step. Use it only when the chosen destination object can support the held object in the lowered predicted state.
If grasper_closed=true and grasped_object is already set to a different object than your next pick target, insert a drop-and-raise sequence first: lower_grasper -> open_grasper -> raise_grasper.
JSON schema: {{"response": "...", "plan": [{{"name": "...", "args": {{...}}}}], "finish_response": "..."}}
Return strict JSON only."""

SUFFIX_PLAN_REPAIR_PROMPT_TEMPLATE = """\
Your previous reply was invalid: {error}
Rewrite it as strict JSON only using this schema:
{{"response": "...", "plan": [{{"name": "...", "args": {{...}}}}], "finish_response": "..."}}"""

PROPERTY_STATE_PREDICTION_SYSTEM_PROMPT = """You are predicting property-truth states in a SHRDLU blocks-world simulator.

Rules:
- Predict only the next state's property truth values after one action; do not explain the properties.
- Predict conservatively from the simulator preconditions, not from the planner's intent.
- When an action would fail a simulator precondition, keep the world state unchanged and mark any violated transition properties as unsatisfied in predicted_state.
- Return ONLY property_results with property id and boolean satisfied. The satisfied field must be exactly true or false — no strings, no nulls, no other values.
- Do NOT include natural_language, all_satisfied, violations, or any extra keys inside predicted_state.
- Include every property id from the input exactly once and keep ids unchanged.
- Also include predicted_world_state on every reply.
- In predicted_world_state, include enough concrete state to check the next step: grasper_closed, grasper_lowered, grasped_object, and any changed objects with updated resting_on, grasped_by, can_support, tags, and position fields.
- For open_grasper while holding an object, if the held object is not resting on a supporting object, predict the action as failing with no world-state change and mark the relevant open_* support properties unsatisfied.
- Keep the response as short as possible.

Required JSON shape:
{"response":"...", "predicted_state":{"property_results":[{"id":"prop.example","satisfied":true}]}, "predicted_world_state":{...}}
Return strict JSON only."""

PROPERTY_STATE_PREDICTION_PROMPT_TEMPLATE = """\
Current property truth values (property_id: true/false):
{current_property_bools_json}

Current world state JSON:
{current_world_state_json}

Planned next action JSON:
{action_json}

Property catalog (id and condition for each property):
{property_text}

Predict the next property-truth state after this one action.
Each property must be predicted as exactly true or false — no other values.
If the action would fail, preserve the current world state in predicted_world_state and reflect the failure in property truth values instead of imagining a successful effect.
Return strict JSON only."""

PROPERTY_STATE_SCHEMA = {
    'type': 'object',
    'properties': {
        'response': {
            'type': 'string',
        },
        'predicted_state': {
            'type': 'object',
            'properties': {
                'property_results': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {
                                'type': 'string',
                            },
                            'satisfied': {
                                'type': 'boolean',
                            },
                        },
                        'required': ['id', 'satisfied'],
                    },
                },
            },
            'required': ['property_results'],
        },
        'predicted_world_state': {
            'type': 'object',
        },
    },
    'required': ['response', 'predicted_state'],
}


class _SuffixPredictivePreplannedShrdluAgentMixin(_PredictivePreplannedShrdluAgentMixin):
    """Plan a full remaining suffix, verify along it, then replan from the first failure point."""

    def _run_agent_loop(self, request: str) -> str:
        initial_world_state = self._env.snapshot()
        initial_scene = copy.deepcopy(self._env.scene)
        initial_state = self._build_initial_property_state(
            initial_world_state,
            scene=initial_scene,
        )
        trace = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'model': self._model,
            'host': self._host,
            'max_steps': self._max_steps,
            'request': request,
            'planning_mode': 'suffix_predictive_preplanned',
            'max_branch_retries': self._max_branch_retries,
            'property_monitoring': self._property_monitoring_metadata(),
            'planning_tree': {
                'mode': 'suffix_predictive_preplanned',
                'max_steps': self._max_steps,
                'max_branch_retries': self._max_branch_retries,
                'properties': [
                    {
                        'id': item.get('id'),
                        'natural_language': item.get('natural_language'),
                    }
                    for item in self._property_verifier.properties
                ],
                'nodes': [],
                'feasible': False,
                'accepted_plan': [],
            },
            'steps': [],
        }
        action_help = self._env.action_help()
        trace['planning_tree']['initial_state'] = initial_state
        trace['planning_tree']['initial_world_state'] = initial_world_state
        trace['planning_tree']['action_help'] = action_help
        trace['status'] = 'planning'
        trace_path = self._start_trace_session(trace)

        result = self._search_plan_suffix(
            request=request,
            current_state=initial_state,
            current_world_state=initial_world_state,
            accepted_trace=[],
            depth=0,
            planning_tree=trace['planning_tree'],
            action_help=action_help,
            parent_node_id=None,
            inherited_failures=[],
            trace=trace,
            trace_path=trace_path,
        )

        trace['planning_tree']['feasible'] = bool(result.get('success'))
        trace['planning_tree']['accepted_plan'] = result.get('plan', []) if result.get('success') else []
        trace['planning_tree']['finish_response'] = result.get('finish_response')
        trace['planning_tree']['planning_response'] = result.get('planning_response')
        if result.get('failure'):
            trace['planning_tree']['failure'] = result['failure']

        if not result.get('success'):
            final_message = self._normalize_response_text(
                result.get('finish_response', 'No feasible property-satisfying plan found.'),
                is_finish=True,
            )
            trace['status'] = 'infeasible'
            trace['final_message'] = final_message
            trace_path = self._write_trace(trace, trace_path)
            return self._append_trace_notice(final_message, trace_path)

        plan = result['plan']
        response_text = self._normalize_response_text(
            result.get('planning_response', 'Verified plan ready.'),
            is_finish=not plan,
        )
        finish_response = self._normalize_response_text(
            result.get('finish_response', 'Done.'),
            is_finish=True,
        )

        if not plan:
            trace['status'] = 'finished'
            trace['final_message'] = finish_response
            self._write_trace(trace, trace_path)
            return finish_response if response_text == finish_response else self._format_reply(
                response_text,
                finish_response,
            )

        previous_property_status = None
        trace['status'] = 'executing'
        self._checkpoint_trace(trace, trace_path)
        for step_index, action in enumerate(plan):
            step_trace = {
                'step_index': step_index,
                'planned_action': action,
            }
            pre_state = self._env.snapshot()
            pre_scene = copy.deepcopy(self._env.scene)
            try:
                result_text = self._env.execute_action(action)
            except Exception as exc:
                result_text = "ERROR: %s" % exc
            post_state = self._env.snapshot()
            property_trace, previous_property_status = self._monitor_transition_properties(
                pre_state,
                action,
                post_state,
                pre_scene=pre_scene,
                post_scene=self._env.scene,
                previous_property_status=previous_property_status,
            )
            step_trace.update({
                'action_result': result_text,
                'property_verification': property_trace,
                'observation_after': self._env.snapshot_text(),
            })
            trace['steps'].append(step_trace)
            self._checkpoint_trace(trace, trace_path)
            if isinstance(result_text, str) and result_text.startswith('ERROR:'):
                final_message = self._format_reply(
                    response_text + "\n\nPlan execution failed.",
                    "Executed %s.\nResult: %s" % (self._format_action(action), result_text),
                )
                trace['status'] = 'error'
                trace['final_message'] = final_message
                trace_path = self._write_trace(trace, trace_path)
                return self._append_trace_notice(final_message, trace_path)

        final_message = finish_response
        if response_text != finish_response:
            final_message = self._format_reply(response_text, finish_response)
        trace['status'] = 'finished'
        trace['final_message'] = final_message
        self._write_trace(trace, trace_path)
        return final_message

    def _search_plan_suffix(
        self,
        *,
        request: str,
        current_state: Dict[str, object],
        current_world_state: Dict[str, object],
        accepted_trace: List[Dict[str, object]],
        depth: int,
        planning_tree: Dict[str, object],
        action_help: str,
        parent_node_id: Optional[int],
        inherited_failures: List[Dict[str, object]],
        trace: Dict[str, object],
        trace_path: Optional[str],
    ) -> Dict[str, object]:
        if depth >= self._max_steps:
            return {
                'success': False,
                'failure': {
                    'type': 'max_depth',
                    'depth': depth,
                    'message': 'Planning exceeded the max step budget before reaching finish.',
                },
                'finish_response': 'No feasible property-satisfying plan found.',
            }

        node_id = len(planning_tree['nodes'])
        node = {
            'node_id': node_id,
            'parent_node_id': parent_node_id,
            'depth': depth,
            'accepted_trace': copy.deepcopy(accepted_trace),
            'current_state': copy.deepcopy(current_state),
            'current_world_state': copy.deepcopy(current_world_state),
            'attempts': [],
            'children': [],
            'result': 'searching',
        }
        planning_tree['nodes'].append(node)
        self._checkpoint_trace(trace, trace_path)
        failed_attempts = list(inherited_failures)

        for retry_index in range(self._max_branch_retries):
            plan_prompt = self._build_suffix_plan_prompt(
                request=request,
                action_help=action_help,
                current_state=current_state,
                current_world_state=current_world_state,
                accepted_trace=accepted_trace,
                failed_attempts=failed_attempts,
            )
            history = [
                {'role': 'system', 'content': SUFFIX_PREDICTIVE_PLAN_SYSTEM_PROMPT},
                {'role': 'user', 'content': plan_prompt},
            ]
            try:
                content, plan_bundle, attempts = self._request_suffix_plan(history)
            except Exception as exc:
                failure = {
                    'type': 'planning_error',
                    'depth': depth,
                    'retry_index': retry_index,
                    'message': str(exc),
                }
                node['attempts'].append({
                    'retry_index': retry_index,
                    'planner_prompt': plan_prompt,
                    'error': str(exc),
                })
                self._checkpoint_trace(trace, trace_path)
                failed_attempts.append(failure)
                continue

            response_text = self._normalize_response_text(
                plan_bundle.get('response', ''),
                is_finish=not plan_bundle['plan'],
            )
            finish_response = self._normalize_response_text(
                plan_bundle.get('finish_response', 'Done.'),
                is_finish=True,
            )
            attempt_trace = {
                'retry_index': retry_index,
                'planner_prompt': plan_prompt,
                'planner_attempts': attempts,
                'planner_response': content,
                'planner_decision': plan_bundle,
            }

            if not plan_bundle['plan']:
                attempt_trace['accepted'] = True
                attempt_trace['finish'] = True
                node['attempts'].append(attempt_trace)
                node['result'] = 'finish'
                node['finish_response'] = finish_response
                self._checkpoint_trace(trace, trace_path)
                return {
                    'success': True,
                    'plan': [],
                    'planning_response': response_text,
                    'finish_response': finish_response,
                    'node_id': node_id,
                }

            if depth + len(plan_bundle['plan']) > self._max_steps:
                failure = {
                    'type': 'plan_too_long',
                    'depth': depth,
                    'retry_index': retry_index,
                    'planned_length': len(plan_bundle['plan']),
                    'remaining_budget': self._max_steps - depth,
                    'message': 'Planned suffix exceeds the remaining step budget.',
                }
                attempt_trace['accepted'] = False
                attempt_trace['failure_feedback'] = failure
                node['attempts'].append(attempt_trace)
                self._checkpoint_trace(trace, trace_path)
                failed_attempts.append(failure)
                continue

            rollout = self._verify_planned_suffix(
                plan=plan_bundle['plan'],
                current_state=current_state,
                current_world_state=current_world_state,
            )
            attempt_trace.update({
                'predicted_rollout': rollout.get('steps', []),
            })

            if rollout.get('success'):
                attempt_trace['accepted'] = True
                node['attempts'].append(attempt_trace)
                node['result'] = 'accepted'
                node['accepted_suffix_plan'] = plan_bundle['plan']
                node['accepted_predicted_state'] = rollout.get('final_state')
                node['accepted_predicted_world_state'] = rollout.get('final_world_state')
                self._checkpoint_trace(trace, trace_path)
                return {
                    'success': True,
                    'plan': plan_bundle['plan'],
                    'planning_response': response_text,
                    'finish_response': finish_response,
                    'node_id': node_id,
                }
            failure = rollout['failure']
            attempt_trace['accepted'] = False
            attempt_trace['failure_feedback'] = failure

            replan_prefix = rollout.get('verified_prefix', [])
            replan_state = rollout.get('last_valid_state') if replan_prefix else None
            replan_world_state = rollout.get('last_valid_world_state') if replan_prefix else None

            if replan_prefix and replan_state is not None and replan_world_state is not None:
                child_result = self._search_plan_suffix(
                    request=request,
                    current_state=replan_state,
                    current_world_state=replan_world_state,
                    accepted_trace=accepted_trace + replan_prefix,
                    depth=depth + len(replan_prefix),
                    planning_tree=planning_tree,
                    action_help=action_help,
                    parent_node_id=node_id,
                    inherited_failures=[failure],
                    trace=trace,
                    trace_path=trace_path,
                )
                attempt_trace['child_node_id'] = child_result.get('node_id')
                if child_result.get('node_id') is not None:
                    node['children'].append(child_result['node_id'])
                if child_result.get('success'):
                    node['attempts'].append(attempt_trace)
                    node['result'] = 'accepted_with_replan'
                    node['accepted_prefix'] = replan_prefix
                    self._checkpoint_trace(trace, trace_path)
                    return {
                        'success': True,
                        'plan': replan_prefix + child_result.get('plan', []),
                        'planning_response': child_result.get('planning_response', response_text),
                        'finish_response': child_result.get('finish_response', finish_response),
                        'node_id': node_id,
                    }
                attempt_trace['child_failure'] = child_result.get('failure')

            node['attempts'].append(attempt_trace)
            self._checkpoint_trace(trace, trace_path)
            failed_attempts.append(failure)

        node['result'] = 'backtracked'
        failure = {
            'type': 'branch_exhausted',
            'depth': depth,
            'node_id': node_id,
            'failed_attempts': failed_attempts,
            'message': 'All suffix replanning retries at this predicted state were exhausted.',
        }
        node['failure'] = failure
        self._checkpoint_trace(trace, trace_path)
        return {
            'success': False,
            'failure': failure,
            'finish_response': 'No feasible property-satisfying plan found.',
            'node_id': node_id,
        }

    def _verify_planned_suffix(
        self,
        *,
        plan: List[Dict[str, object]],
        current_state: Dict[str, object],
        current_world_state: Dict[str, object],
    ) -> Dict[str, object]:
        state = copy.deepcopy(current_state)
        world_state = copy.deepcopy(current_world_state)
        verified_prefix = []
        rollout_steps = []

        for suffix_index, action in enumerate(plan):
            prediction_prompt = self._build_property_state_prediction_prompt(
                current_state=state,
                current_world_state=world_state,
                action=action,
            )
            history = [
                {'role': 'system', 'content': PROPERTY_STATE_PREDICTION_SYSTEM_PROMPT},
                {'role': 'user', 'content': prediction_prompt},
            ]
            try:
                prediction_response, prediction_bundle, prediction_attempts = self._request_property_state_prediction(
                    history,
                    state,
                    world_state,
                )
            except Exception as exc:
                return {
                    'success': False,
                    'verified_prefix': verified_prefix,
                    'last_valid_state': state,
                    'last_valid_world_state': world_state,
                    'steps': rollout_steps,
                    'failure': {
                        'type': 'prediction_error',
                        'suffix_index': suffix_index,
                        'action': action,
                        'message': str(exc),
                    },
                }

            predicted_state = prediction_bundle['predicted_state']
            predicted_world_state = prediction_bundle.get('predicted_world_state', world_state)
            current_property_status = self._property_status_from_state(predicted_state)
            rollout_steps.append({
                'suffix_index': suffix_index,
                'action': action,
                'prediction_prompt': prediction_prompt,
                'prediction_response': prediction_response,
                'prediction_attempts': prediction_attempts,
                'prediction_summary': prediction_bundle.get('response', ''),
                'predicted_state': predicted_state,
                'predicted_world_state': predicted_world_state,
                'predicted_property_changes': self._diff_property_states(state, predicted_state),
                'property_verification': predicted_state,
            })

            if not predicted_state.get('all_satisfied', False):
                return {
                    'success': False,
                    'verified_prefix': verified_prefix,
                    'last_valid_state': state,
                    'last_valid_world_state': world_state,
                    'steps': rollout_steps,
                    'failure': {
                        'type': 'property_violation',
                        'suffix_index': suffix_index,
                        'action': action,
                        'violations': predicted_state.get('violations', []),
                        'changed_properties': self._diff_property_states(state, predicted_state),
                        'verified_prefix_length': len(verified_prefix),
                    },
                }

            verified_prefix.append(action)
            state = predicted_state
            world_state = predicted_world_state

        return {
            'success': True,
            'verified_prefix': verified_prefix,
            'final_state': state,
            'final_world_state': world_state,
            'steps': rollout_steps,
        }

    def _build_suffix_plan_prompt(
        self,
        *,
        request: str,
        action_help: str,
        current_state: Dict[str, object],
        current_world_state: Dict[str, object],
        accepted_trace: List[Dict[str, object]],
        failed_attempts: List[Dict[str, object]],
    ) -> str:
        current_property_bools = {
            item['id']: item['satisfied']
            for item in current_state.get('property_results', [])
            if isinstance(item.get('id'), str) and isinstance(item.get('satisfied'), bool)
        }
        return SUFFIX_PLAN_USER_PROMPT_TEMPLATE.format(
            request=request,
            grounding_verdict=self._grounding_verdict_text(current_world_state, request),
            current_property_bools_json=self._snapshot_json(current_property_bools),
            current_world_state_json=self._snapshot_json(current_world_state),
            planning_state_summary=self._planning_state_summary(
                current_world_state,
                accepted_trace,
                request=request,
            ),
            accepted_trace_json=self._json_or_none(accepted_trace),
            property_text=self._property_text,
            action_help=action_help,
            failed_attempts_json=self._json_or_none(failed_attempts[-5:]) if failed_attempts else 'None',
        )

    def _build_property_state_prediction_prompt(
        self,
        *,
        current_state: Dict[str, object],
        current_world_state: Dict[str, object],
        action: Dict[str, object],
    ) -> str:
        current_property_bools = {
            item['id']: item['satisfied']
            for item in current_state.get('property_results', [])
            if isinstance(item.get('id'), str) and isinstance(item.get('satisfied'), bool)
        }
        return PROPERTY_STATE_PREDICTION_PROMPT_TEMPLATE.format(
            current_property_bools_json=self._snapshot_json(current_property_bools),
            current_world_state_json=self._snapshot_json(current_world_state),
            action_json=self._json_or_none(action),
            property_text=self._property_text,
        )

    def _request_property_state_prediction(
        self,
        history: List[Dict[str, str]],
        current_state: Dict[str, object],
        current_world_state: Dict[str, object],
    ):
        content = self._chat(list(history), schema=PROPERTY_STATE_SCHEMA).strip()
        bundle = self._force_parse_property_state_prediction(content, current_state, current_world_state)
        attempt_log = [{'raw_content': content, 'parsed_prediction': bundle}]
        return content, bundle, attempt_log

    def _force_parse_property_state_prediction(
        self,
        content: str,
        current_state: Dict[str, object],
        current_world_state: Dict[str, object],
    ) -> Dict[str, object]:
        """Parse the predicted property state with a guaranteed result.

        Extraction order: JSON parse → regex scan → fallback to current state.
        Any property id not recovered from the model output is copied from
        current_state so the result is always complete.
        """
        known_ids = {str(spec.get('id')) for spec in self._property_verifier.properties}
        current_status = self._property_status_from_state(current_state)

        extracted: Dict[str, bool] = {}
        decision = {}
        predicted_world_state = None
        response_text = ''

        # 1. Try JSON parse via the standard extractor.
        try:
            json_content = OllamaShrdluAgent._extract_json_object(content)
            decision = json.loads(json_content)
            if isinstance(decision, dict):
                nested_decision = self._parse_nested_prediction_response(decision.get('response'))
                if nested_decision:
                    for key in ('predicted_state', 'predicted_world_state'):
                        if key not in decision and key in nested_decision:
                            decision[key] = nested_decision[key]
                    if nested_decision.get('response'):
                        decision['response'] = nested_decision['response']
                pr = decision.get('predicted_state', {}).get('property_results')
                if isinstance(pr, list):
                    for item in pr:
                        pid = item.get('id')
                        sat = item.get('satisfied')
                        if isinstance(pid, str) and pid in known_ids and isinstance(sat, bool):
                            extracted[pid] = sat
                predicted_world_state = decision.get('predicted_world_state')
                response_text = str(decision.get('response', ''))
        except (json.JSONDecodeError, ValueError):
            pass

        # 2. Regex scan for any ids still missing.
        if len(extracted) < len(known_ids):
            for pid in known_ids - set(extracted):
                pattern = r'"id"\s*:\s*"' + re.escape(pid) + r'"[^}]*?"satisfied"\s*:\s*(true|false)'
                m = re.search(pattern, content)
                if not m:
                    pattern = r'"satisfied"\s*:\s*(true|false)[^}]*?"id"\s*:\s*"' + re.escape(pid) + r'"'
                    m = re.search(pattern, content)
                if m:
                    extracted[pid] = m.group(1) == 'true'

        # 3. Fill remaining gaps from current_state (assume no change).
        for pid in known_ids - set(extracted):
            extracted[pid] = current_status.get(pid, False)

        property_results = [{'id': pid, 'satisfied': extracted[pid]} for pid in known_ids]
        normalized_state = self._property_state_from_results(property_results)

        normalized_world_state = copy.deepcopy(current_world_state)
        if isinstance(predicted_world_state, dict):
            normalized_world_state = self._merge_predicted_state(current_world_state, predicted_world_state)

        return {
            'response': response_text,
            'predicted_state': normalized_state,
            'predicted_world_state': normalized_world_state,
        }

    @staticmethod
    def _parse_nested_prediction_response(response_value) -> Optional[Dict[str, object]]:
        if not isinstance(response_value, str):
            return None
        response_value = response_value.strip()
        if not response_value:
            return None
        try:
            nested_content = OllamaShrdluAgent._extract_json_object(response_value)
            nested_decision = json.loads(nested_content)
        except (json.JSONDecodeError, ValueError):
            return None
        return nested_decision if isinstance(nested_decision, dict) else None

    def _build_initial_property_state(
        self,
        world_state: Dict[str, object],
        *,
        scene,
    ) -> Dict[str, object]:
        verification = self._property_verifier.verify_transition(
            world_state,
            {'name': 'initial_state', 'args': {}},
            world_state,
            pre_scene=scene,
            post_scene=scene,
        )
        return self._property_state_from_results(verification.get('property_results', []))

    def _property_state_from_results(self, property_results: List[Dict[str, object]]) -> Dict[str, object]:
        normalized_results = []
        for spec in self._property_verifier.properties:
            property_id = spec.get('id')
            match = next(
                (item for item in property_results if item.get('id') == property_id),
                None,
            )
            satisfied = bool(match.get('satisfied', False)) if match is not None else False
            normalized_results.append({
                'id': property_id,
                'natural_language': spec.get('natural_language'),
                'satisfied': satisfied,
            })
        violations = [item for item in normalized_results if not item['satisfied']]
        return {
            'all_satisfied': not violations,
            'violations': copy.deepcopy(violations),
            'property_results': normalized_results,
        }

    @staticmethod
    def _property_status_from_state(state: Dict[str, object]) -> Dict[str, bool]:
        status = {}
        for item in state.get('property_results', []):
            if isinstance(item, dict) and 'id' in item:
                status[str(item['id'])] = bool(item.get('satisfied', False))
        return status

    @classmethod
    def _diff_property_states(
        cls,
        previous_state: Dict[str, object],
        current_state: Dict[str, object],
    ) -> List[Dict[str, object]]:
        previous_status = cls._property_status_from_state(previous_state)
        current_status = cls._property_status_from_state(current_state)
        changed_properties = []
        for property_id, current_value in current_status.items():
            previous_value = previous_status.get(property_id)
            if previous_value is None or previous_value == current_value:
                continue
            changed_properties.append({
                'id': property_id,
                'before': previous_value,
                'after': current_value,
            })
        return changed_properties

    @staticmethod
    def _parse_plan(content: str) -> Dict[str, object]:
        _ALLOWED_PRIMITIVE_NAMES = frozenset(
            spec['name'] for spec in ShrdluBlocksEnv.ACTION_SPECS
        )
        content = OllamaShrdluAgent._extract_json_object(content)
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Model did not return valid JSON: %s" % content) from exc
        if not isinstance(decision, dict):
            raise ValueError("Model reply must be a JSON object.")
        if 'plan' not in decision:
            raise ValueError("Model reply must include a plan array.")
        raw_plan = decision.get('plan', [])
        if not isinstance(raw_plan, list):
            raise ValueError("Model reply must include a plan array.")
        plan = []
        for item in raw_plan:
            if not isinstance(item, dict):
                raise ValueError("Each planned action must be a JSON object.")
            normalized = OllamaShrdluAgent._normalize_action({'action': item})
            name = str(normalized.get('name', '')).strip()
            if name not in _ALLOWED_PRIMITIVE_NAMES:
                raise ValueError(
                    "Action name %r is not a primitive action. Allowed names: %s"
                    % (name, ', '.join(sorted(_ALLOWED_PRIMITIVE_NAMES)))
                )
            plan.append({
                'name': name,
                'args': normalized.get('args', {}) if isinstance(normalized.get('args', {}), dict) else {},
            })
        return {
            'response': str(decision.get('response', '')),
            'finish_response': str(decision.get('finish_response', '')),
            'plan': plan,
        }

    def _request_suffix_plan(self, history: List[Dict[str, str]]):
        attempts = list(history)
        errors = []
        attempt_log = []
        for attempt_index in range(2):
            content = self._chat(attempts, schema=PLAN_SCHEMA).strip()
            try:
                plan_bundle = self._parse_plan(content)
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
                        'content': SUFFIX_PLAN_REPAIR_PROMPT_TEMPLATE.format(error=exc),
                    },
                ])
                continue
            attempt_log.append({
                'attempt_index': attempt_index,
                'raw_content': content,
                'parsed_plan': plan_bundle,
            })
            return content, plan_bundle, attempt_log
        raise ValueError('Invalid plan reply after retry: %s' % errors[-1])


class SuffixPredictivePreplannedOllamaShrdluAgent(
        _SuffixPredictivePreplannedShrdluAgentMixin, OllamaShrdluAgent):
    """Suffix-replanning predictive preplanned agent over Ollama."""

    def __init__(self, env: ShrdluBlocksEnv, model: str = DEFAULT_MODEL,
                 host: str = 'http://127.0.0.1:11434', max_steps: int = DEFAULT_MAX_STEPS,
                 trace_dir: Optional[str] = DEFAULT_TRACE_DIR,
                 max_branch_retries: int = 3):
        super().__init__(
            env,
            model=model,
            host=host,
            max_steps=max_steps,
            trace_dir=trace_dir,
        )
        self._init_predictive_preplanned(max_branch_retries=max_branch_retries)


class SuffixPredictivePreplannedOpenAICompatibleShrdluAgent(
        _SuffixPredictivePreplannedShrdluAgentMixin, OpenAICompatibleShrdluAgent):
    """Suffix-replanning predictive preplanned agent over a local OpenAI API."""

    def __init__(self, env: ShrdluBlocksEnv, model: str = DEFAULT_OPENAI_MODEL,
                 base_url: str = DEFAULT_OPENAI_BASE_URL,
                 api_key: str = DEFAULT_OPENAI_API_KEY,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 trace_dir: Optional[str] = DEFAULT_TRACE_DIR,
                 temperature: float = 0.2,
                 max_tokens: int = 512,
                 client=None,
                 max_branch_retries: int = 3):
        super().__init__(
            env,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_steps=max_steps,
            trace_dir=trace_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
        )
        self._init_predictive_preplanned(max_branch_retries=max_branch_retries)
