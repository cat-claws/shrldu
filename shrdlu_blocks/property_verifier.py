"""Property verification utilities for SHRDLU transition snapshots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from shrdlu_blocks.scenes import Scene

__all__ = ['PROPERTY_FILE', 'TransitionPropertyVerifier']


PROPERTY_FILE = Path(__file__).resolve().parent.parent / 'SHRDLU_PROPERTIES_AST.json'

_OBJECT_CAN_SUPPORT_RE = re.compile(r'^object_(\d+)_can_support$')
_OBJECT_HIGHLIGHTED_RE = re.compile(r'^object_(\d+)_highlighted$')
_OBJECT_ABOVE_RE = re.compile(r'^object_(\d+)_above_object_(\d+)$')


@dataclass(frozen=True)
class _TransitionContext:
    pre_state: Dict[str, object]
    action: Dict[str, object]
    post_state: Dict[str, object]
    pre_scene: Optional[Scene] = None
    post_scene: Optional[Scene] = None


class TransitionPropertyVerifier:
    """Evaluate transition properties over simulator snapshots."""

    def __init__(self, properties: Iterable[Dict[str, object]]):
        self._properties = list(properties)

    @classmethod
    def from_file(cls, path: Path = PROPERTY_FILE) -> 'TransitionPropertyVerifier':
        payload = json.loads(path.read_text(encoding='utf-8'))
        return cls(payload.get('properties', []))

    @property
    def properties(self) -> List[Dict[str, object]]:
        return list(self._properties)

    def verify_transition(
        self,
        pre_state: Dict[str, object],
        action: Dict[str, object],
        post_state: Dict[str, object],
        *,
        pre_scene: Optional[Scene] = None,
        post_scene: Optional[Scene] = None,
    ) -> Dict[str, object]:
        context = _TransitionContext(
            pre_state=pre_state,
            action=action,
            post_state=post_state,
            pre_scene=pre_scene,
            post_scene=post_scene,
        )
        ap_cache: Dict[str, bool] = {}
        property_results = []
        for spec in self._properties:
            satisfied = self._eval_ast(spec['ast'], context, ap_cache)
            property_results.append({
                'id': spec.get('id'),
                'natural_language': spec.get('natural_language'),
                'satisfied': satisfied,
            })
        violations = [result for result in property_results if not result['satisfied']]
        return {
            'all_satisfied': not violations,
            'violations': violations,
            'property_results': property_results,
            'derived_aps': dict(sorted(ap_cache.items())),
        }

    def _eval_ast(
        self,
        node: Dict[str, object],
        context: _TransitionContext,
        ap_cache: Dict[str, bool],
    ) -> bool:
        node_type = node.get('type')
        if node_type == 'globally':
            return self._eval_ast(node['operand'], context, ap_cache)
        if node_type == 'implies':
            return (not self._eval_ast(node['left'], context, ap_cache)) or self._eval_ast(
                node['right'], context, ap_cache
            )
        if node_type == 'and':
            return all(self._eval_ast(arg, context, ap_cache) for arg in node.get('args', []))
        if node_type == 'or':
            return any(self._eval_ast(arg, context, ap_cache) for arg in node.get('args', []))
        if node_type == 'not':
            return not self._eval_ast(node['operand'], context, ap_cache)
        if node_type == 'ap':
            name = str(node['name'])
            if name not in ap_cache:
                ap_cache[name] = self._eval_ap(name, context)
            return ap_cache[name]
        raise ValueError('Unsupported AST node type: %r' % (node_type,))

    def _eval_ap(self, name: str, context: _TransitionContext) -> bool:
        action_name = str(context.action.get('name', '')).strip()
        if name.startswith('last_action_'):
            return action_name == name[len('last_action_'):]
        if name == 'pre_grasper_lowered':
            return bool(context.pre_state.get('grasper_lowered', False))
        if name == 'pre_grasper_closed':
            return bool(context.pre_state.get('grasper_closed', False))
        if name == 'pre_grasper_holding':
            return context.pre_state.get('grasped_object') is not None
        if name == 'pre_held_object_resting_on_object':
            held = self._get_object(context.pre_state, context.pre_state.get('grasped_object'))
            return held is not None and held.get('resting_on') is not None
        if name == 'pre_support_can_support_held':
            return self._pre_support_can_support_held(context)
        if name == 'pre_grasper_resting_on_graspable':
            grasper = self._get_default_grasper(context.pre_state)
            if grasper is None:
                return False
            support = self._get_object(context.pre_state, grasper.get('resting_on'))
            return support is not None and bool(support.get('graspable', False))
        if name == 'post_grasper_holding':
            return context.post_state.get('grasped_object') is not None
        if name == 'post_grasper_closed':
            return bool(context.post_state.get('grasper_closed', False))
        if name == 'post_grasper_lowered':
            return bool(context.post_state.get('grasper_lowered', False))

        object_match = _OBJECT_CAN_SUPPORT_RE.match(name)
        if object_match:
            obj = self._get_object(context.post_state, int(object_match.group(1)))
            return obj is not None and bool(obj.get('can_support', False))

        object_match = _OBJECT_HIGHLIGHTED_RE.match(name)
        if object_match:
            obj = self._get_object(context.post_state, int(object_match.group(1)))
            if obj is None:
                return False
            return bool(obj.get('tags', {}).get('highlight', False))

        object_match = _OBJECT_ABOVE_RE.match(name)
        if object_match:
            upper_id = int(object_match.group(1))
            lower_id = int(object_match.group(2))
            return self._object_above_object(context, upper_id, lower_id)

        raise ValueError('Unsupported atomic proposition: %s' % name)

    def _pre_support_can_support_held(self, context: _TransitionContext) -> bool:
        held_id = context.pre_state.get('grasped_object')
        held = self._get_object(context.pre_state, held_id)
        if held is None:
            return False
        support_id = held.get('resting_on')
        if support_id is None:
            return False
        if context.pre_scene is not None:
            held_obj = self._get_scene_object(context.pre_scene, int(held_id))
            support_obj = self._get_scene_object(context.pre_scene, int(support_id))
            if held_obj is not None and support_obj is not None:
                return support_obj.can_support(held_obj)
        support = self._get_object(context.pre_state, support_id)
        return support is not None and bool(support.get('can_support', False))

    def _object_above_object(
        self,
        context: _TransitionContext,
        upper_id: int,
        lower_id: int,
    ) -> bool:
        scene = context.post_scene or context.pre_scene
        if scene is not None:
            upper_obj = self._get_scene_object(scene, upper_id)
            lower_obj = self._get_scene_object(scene, lower_id)
            if upper_obj is not None and lower_obj is not None:
                upper_bottom = upper_obj.position.z
                lower_top = lower_obj.find_highest_point().z
                return upper_bottom > lower_top
        upper = self._get_object(context.post_state, upper_id) or self._get_object(context.pre_state, upper_id)
        lower = self._get_object(context.post_state, lower_id) or self._get_object(context.pre_state, lower_id)
        if upper is None or lower is None:
            return False
        return float(upper['position']['z']) > float(lower['position']['z'])

    @staticmethod
    def _get_object(snapshot: Dict[str, object], obj_id: object) -> Optional[Dict[str, object]]:
        if obj_id is None:
            return None
        for obj in snapshot.get('objects', []):
            if obj.get('obj_id') == obj_id:
                return obj
        return None

    @classmethod
    def _get_default_grasper(cls, snapshot: Dict[str, object]) -> Optional[Dict[str, object]]:
        default_grasper = snapshot.get('default_grasper')
        if default_grasper is not None:
            grasper = cls._get_object(snapshot, default_grasper)
            if grasper is not None:
                return grasper
        for obj in snapshot.get('objects', []):
            if obj.get('kind') == 'grasper':
                return obj
        return None

    @staticmethod
    def _get_scene_object(scene: Scene, obj_id: int):
        for obj in scene.find_objects(obj_id=obj_id):
            return obj
        return None
