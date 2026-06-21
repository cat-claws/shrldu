"""Symbolic state prediction for SHRDLU primitive actions."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from shrdlu_blocks.geometry import Point, make_block, make_box, make_grasper, make_pyramid, make_table
from shrdlu_blocks.scenes import PhysicalObject
from shrdlu_blocks.typedefs import Color

__all__ = ['predict_world_state_after_actions']


def predict_world_state_after_actions(
    init_world_state: Dict[str, object],
    actions: List[Dict[str, object]],
) -> Tuple[Dict[str, object], List[str]]:
    """Replay primitive simulator effects into a predicted world snapshot."""
    state = copy.deepcopy(init_world_state)
    notes = []
    for action in actions:
        notes.append(_apply_symbolic_action(state, action))
    return state, notes


def _apply_symbolic_action(
    state: Dict[str, object],
    action: Dict[str, object],
) -> str:
    name = action.get('name', '')
    args = action.get('args', {}) or {}
    grasper = _symbolic_grasper(state)
    grasped_object = state.get('grasped_object')

    if name == 'move_grasper':
        if state.get('grasper_lowered') is True:
            return 'move_grasper: precondition failed; grasper is lowered.'
        pos = grasper.setdefault('position', {})
        if 'x' in args:
            pos['x'] = args.get('x')
        if 'y' in args:
            pos['y'] = args.get('y')
        if grasped_object is not None:
            held = _symbolic_object(state, grasped_object)
            if held is not None:
                held_pos = held.setdefault('position', {})
                held_pos['x'] = pos.get('x')
                held_pos['y'] = pos.get('y')
        return 'move_grasper: moved to (%s, %s).' % (pos.get('x'), pos.get('y'))

    if name == 'lower_grasper':
        if state.get('grasper_lowered') is True:
            return 'lower_grasper: precondition failed; grasper is already lowered.'
        target, target_height = _find_symbolic_object_below_grasper(
            state,
            exclude_obj_id=grasped_object,
        )
        target_id = target.get('obj_id') if target else None
        state['grasper_lowered'] = True
        grasper.setdefault('tags', {})['lowered'] = True
        if grasped_object is None:
            _set_position_z(grasper, target_height)
            grasper['resting_on'] = target_id
            grasper.setdefault('tags', {})['resting_on'] = target_id
        else:
            held = _symbolic_object(state, grasped_object)
            if held is not None:
                held_height = _object_height(held)
                _set_position_z(grasper, target_height + held_height)
                held['resting_on'] = target_id
                held.setdefault('tags', {})['resting_on'] = target_id
                held_pos = held.setdefault('position', {})
                grasper_pos = grasper.get('position', {})
                held_pos['x'] = grasper_pos.get('x')
                held_pos['y'] = grasper_pos.get('y')
                held_pos['z'] = target_height
        return 'lower_grasper: lowered onto obj %s.' % target_id

    if name == 'raise_grasper':
        if state.get('grasper_lowered') is not True:
            return 'raise_grasper: precondition failed; grasper is already raised.'
        state['grasper_lowered'] = False
        grasper.setdefault('tags', {})['lowered'] = False
        if grasped_object is not None:
            held = _symbolic_object(state, grasped_object)
            if held is not None:
                minimum_height = _find_highest_stable_point(state) + 0.1
                _set_position_z(held, minimum_height)
                _set_position_z(grasper, minimum_height + _object_height(held))
                held['resting_on'] = None
                held.setdefault('tags', {})['resting_on'] = None
        else:
            grasper['resting_on'] = None
            grasper.setdefault('tags', {})['resting_on'] = None
            _set_position_z(grasper, _find_highest_stable_point(state) + 0.1)
        return 'raise_grasper: raised.'

    if name == 'close_grasper':
        if state.get('grasper_closed') is True:
            return 'close_grasper: precondition failed; grasper is already closed.'
        state['grasper_closed'] = True
        grasper.setdefault('tags', {})['closed'] = True
        if state.get('grasper_lowered') is True:
            target_id = grasper.get('resting_on')
            target = _symbolic_object(state, target_id)
            if target is not None and target.get('graspable'):
                state['grasped_object'] = target_id
                grasper.setdefault('tags', {})['grasped'] = target_id
                target['grasped_by'] = grasper.get('obj_id')
                target.setdefault('tags', {})['grasped_by'] = grasper.get('obj_id')
        return 'close_grasper: closed.'

    if name == 'open_grasper':
        if state.get('grasper_closed') is not True:
            return 'open_grasper: precondition failed; grasper is already open.'
        if grasped_object is not None:
            held = _symbolic_object(state, grasped_object)
            support_id = held.get('resting_on') if held else None
            support = _symbolic_object(state, support_id)
            if state.get('grasper_lowered') is not True:
                return 'open_grasper: precondition failed; grasper is raised while holding obj %s.' % grasped_object
            if support is None or not _can_support(support, held):
                return 'open_grasper: precondition failed; held obj %s is not on a valid support.' % grasped_object
            if held is not None:
                held['grasped_by'] = None
                held.setdefault('tags', {})['grasped_by'] = None
            state['grasped_object'] = None
            grasper.setdefault('tags', {})['grasped'] = None
        state['grasper_closed'] = False
        grasper.setdefault('tags', {})['closed'] = False
        return 'open_grasper: opened.'

    if name in ('highlight_object', 'unhighlight_object'):
        return '%s: visual-only action; no world state change.' % name

    return '%s: unknown action; no state change.' % name


def _symbolic_grasper(state: Dict[str, object]) -> Dict[str, object]:
    default_id = state.get('default_grasper')
    if default_id is None:
        default_id = 0
    grasper = _symbolic_object(state, default_id)
    if grasper is None:
        raise ValueError('No grasper object found in world state.')
    return grasper


def _symbolic_object(
    state: Dict[str, object],
    obj_id: object,
) -> Optional[Dict[str, object]]:
    for obj in state.get('objects', []):
        if isinstance(obj, dict) and obj.get('obj_id') == obj_id:
            return obj
    return None


def _find_symbolic_object_below_grasper(
    state: Dict[str, object],
    *,
    exclude_obj_id: object = None,
) -> Tuple[Optional[Dict[str, object]], float]:
    grasper = _symbolic_grasper(state)
    grasper_point = _point_from_position(grasper.get('position', {}))
    if grasper_point is None:
        return None, 0.0

    target = None
    target_height = 0.0
    grasper_id = grasper.get('obj_id')
    for obj in state.get('objects', []):
        if not isinstance(obj, dict):
            continue
        oid = obj.get('obj_id')
        if oid in (grasper_id, exclude_obj_id):
            continue
        phys_obj = _physical_object_from_snapshot(obj)
        if phys_obj is None or not phys_obj.is_below_point(grasper_point):
            continue
        highest = phys_obj.find_highest_point()
        if highest is None:
            continue
        if target_height <= highest.z:
            target = obj
            target_height = highest.z
    return target, target_height


def _find_highest_stable_point(state: Dict[str, object]) -> float:
    result = 0.0
    for obj in state.get('objects', []):
        if not isinstance(obj, dict):
            continue
        if obj.get('kind') == 'grasper' or obj.get('grasped_by') is not None:
            continue
        phys_obj = _physical_object_from_snapshot(obj)
        if phys_obj is None:
            continue
        highest = phys_obj.find_highest_point()
        if highest is not None and highest.z > result:
            result = highest.z
    return result


def _can_support(support: Dict[str, object], held: Optional[Dict[str, object]]) -> bool:
    if held is None:
        return False
    support_obj = _physical_object_from_snapshot(support)
    held_obj = _physical_object_from_snapshot(held)
    if support_obj is None or held_obj is None:
        return bool(support.get('can_support'))
    return support_obj.can_support(held_obj)


def _object_height(obj: Dict[str, object]) -> float:
    phys_obj = _physical_object_from_snapshot(obj)
    if phys_obj is None:
        return 0.0
    highest = phys_obj.find_highest_point()
    pos = _point_from_position(obj.get('position', {}))
    if highest is None or pos is None:
        return 0.0
    return highest.z - pos.z


def _set_position_z(obj: Dict[str, object], z: float) -> None:
    pos = obj.setdefault('position', {})
    pos['z'] = z


def _physical_object_from_snapshot(obj: Dict[str, object]) -> Optional[PhysicalObject]:
    shape = _shape_from_snapshot(obj)
    pos = _point_from_position(obj.get('position', {}))
    if shape is None or pos is None:
        return None
    return PhysicalObject(shape, Color(0, 0, 0), pos, dict(obj))


def _shape_from_snapshot(obj: Dict[str, object]):
    kind = obj.get('kind')
    if kind == 'grasper':
        return make_grasper(0.05, 1)
    if kind == 'table':
        return make_table(1, 1)
    if kind == 'box':
        return make_box(0.35, 0.35, 0.2)
    if kind == 'block':
        width, depth, height = _block_dimensions(obj)
        return make_block(width, depth, height)
    if kind == 'pyramid':
        width, depth, height = _pyramid_dimensions(obj)
        return make_pyramid(width, depth, height)
    return None


def _block_dimensions(obj: Dict[str, object]) -> Tuple[float, float, float]:
    size = obj.get('size') or obj.get('tags', {}).get('size')
    height = obj.get('height') or obj.get('tags', {}).get('height')
    width = obj.get('width') or obj.get('tags', {}).get('width')
    color = obj.get('color') or obj.get('tags', {}).get('color')
    if size == 'small':
        return 0.1, 0.1, 0.08
    if size == 'medium':
        return 0.15, 0.15, 0.1
    if size == 'big' and color == 'green' and width == 'wide':
        return 0.2, 0.2, 0.15
    if size == 'big' and height == 'tall' and width == 'medium':
        return 0.15, 0.15, 0.2
    return 0.15, 0.15, 0.15


def _pyramid_dimensions(obj: Dict[str, object]) -> Tuple[float, float, float]:
    size = obj.get('size') or obj.get('tags', {}).get('size')
    height = obj.get('height') or obj.get('tags', {}).get('height')
    if size == 'small':
        return 0.1, 0.1, 0.08
    if height == 'tall':
        return 0.1, 0.1, 0.2
    return 0.15, 0.15, 0.1


def _point_from_position(pos: Dict[str, object]) -> Optional[Point]:
    try:
        return Point(float(pos.get('x')), float(pos.get('y')), float(pos.get('z', 0.0) or 0.0))
    except (TypeError, ValueError):
        return None
