"""Environment wrapper for the SHRDLU blocks world."""

from typing import Dict, List, Optional

from shrdlu_blocks.commands import ControllerCommandExecutor
from shrdlu_blocks.control import Controller
from shrdlu_blocks.scenes import Scene, make_standard_scene

__all__ = ['ShrdluBlocksEnv']


class ShrdluBlocksEnv:
    """A small reusable environment API for agent-driven interaction."""

    ACTION_SPECS = (
        {
            'name': 'move_grasper',
            'args': {'x': 'float', 'y': 'float'},
            'description': 'Move the grasper to an x,y coordinate while it is raised.',
        },
        {
            'name': 'lower_grasper',
            'args': {},
            'description': 'Lower the grasper until it contacts an object or the table.',
        },
        {
            'name': 'raise_grasper',
            'args': {},
            'description': 'Raise the grasper high enough to clear the scene.',
        },
        {
            'name': 'close_grasper',
            'args': {},
            'description': 'Close the grasper. If aligned and lowered, it may grasp an object.',
        },
        {
            'name': 'open_grasper',
            'args': {},
            'description': 'Open the grasper. If the held object is supported, it will be released.',
        },
        {
            'name': 'highlight_object',
            'args': {'obj_id': 'int'},
            'description': 'Highlight an object in the viewer.',
        },
        {
            'name': 'unhighlight_object',
            'args': {'obj_id': 'int'},
            'description': 'Remove highlighting from an object.',
        },
    )

    def __init__(self, scene: Scene = None):
        self._scene = scene or make_standard_scene()
        self._controller = Controller(self._scene)
        self._executor = ControllerCommandExecutor(self._controller)

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def controller(self) -> Controller:
        return self._controller

    @property
    def executor(self) -> ControllerCommandExecutor:
        return self._executor

    def reset(self) -> Dict[str, object]:
        """Reset the scene to the standard initial state."""
        self._scene = make_standard_scene()
        self._controller = Controller(self._scene)
        self._executor = ControllerCommandExecutor(self._controller)
        return self.snapshot()

    def execute_command(self, text: str) -> Optional[str]:
        """Execute a controller command without going through the GUI."""
        return self._executor.execute(text)

    def command_help(self) -> str:
        """Return the public controller command list."""
        return self._executor.describe_commands()

    def action_help(self) -> str:
        """Return the validated agent action schema."""
        lines = [
            "Allowed actions:",
            'Return JSON as {"response": "...", "action": {"name": "...", "args": {...}}}',
            'Use {"response": "...", "action": {"name": "finish", "args": {}}} when done.',
        ]
        for spec in self.ACTION_SPECS:
            lines.append(
                "  {name} args={args} - {description}".format(
                    name=spec['name'],
                    args=spec['args'],
                    description=spec['description'],
                )
            )
        return "\n".join(lines)

    def execute_action(self, action: Dict[str, object]) -> Optional[str]:
        """Execute a validated high-level action object."""
        name = str(action.get('name', '')).strip()
        args = action.get('args', {})
        if name == 'finish':
            return None
        if not isinstance(args, dict):
            raise ValueError('Action args must be a JSON object.')

        if name == 'move_grasper':
            x = self._require_float(args, 'x')
            y = self._require_float(args, 'y')
            return self._controller_result(self._controller.move_grasper(x, y))
        if name == 'lower_grasper':
            return self._controller_result(self._controller.lower_grasper())
        if name == 'raise_grasper':
            return self._controller_result(self._controller.raise_grasper())
        if name == 'close_grasper':
            return self._controller_result(self._controller.close_grasper())
        if name == 'open_grasper':
            return self._controller_result(self._controller.open_grasper())
        if name == 'highlight_object':
            obj_id = self._require_int(args, 'obj_id')
            return self._controller_result(self._controller.highlight_object(obj_id))
        if name == 'unhighlight_object':
            obj_id = self._require_int(args, 'obj_id')
            return self._controller_result(self._controller.unhighlight_object(obj_id))
        raise ValueError('Unsupported action: %s' % name)

    def snapshot(self) -> Dict[str, object]:
        """Return a structured symbolic snapshot of the current world state."""
        object_summaries: List[Dict[str, object]] = []
        for obj_id in self._controller.find_objects():
            tags = dict(self._controller.iter_object_tags(obj_id))
            position = self._controller.get_object_position(obj_id)
            object_summaries.append({
                'obj_id': int(obj_id),
                'kind': tags.get('kind'),
                'color': tags.get('color'),
                'graspable': bool(tags.get('graspable', False)),
                'can_support': bool(tags.get('can_support', False)),
                'resting_on': self._normalize_value(tags.get('resting_on')),
                'grasped_by': self._normalize_value(tags.get('grasped_by')),
                'position': {
                    'x': position.x,
                    'y': position.y,
                    'z': position.z,
                },
                'tags': {
                    key: self._normalize_value(value)
                    for key, value in sorted(tags.items())
                },
            })
        return {
            'default_grasper': self._normalize_value(self._controller.default_grasper),
            'grasper_closed': self._controller.grasper_is_closed(),
            'grasper_lowered': self._controller.grasper_is_lowered(),
            'grasped_object': self._normalize_value(self._controller.get_grasped_object()),
            'objects': object_summaries,
        }

    def snapshot_text(self) -> str:
        """Return a compact text snapshot for prompting an LLM."""
        state = self.snapshot()
        lines = [
            "World state:",
            "default_grasper=%r" % (state['default_grasper'],),
            "grasper_closed=%r" % (state['grasper_closed'],),
            "grasper_lowered=%r" % (state['grasper_lowered'],),
            "grasped_object=%r" % (state['grasped_object'],),
            "objects:",
        ]
        for obj in state['objects']:
            lines.append(
                "  id={obj_id} kind={kind} color={color} graspable={graspable} support={support} "
                "resting_on={resting_on} grasped_by={grasped_by} pos=({x:.3f}, {y:.3f}, {z:.3f})".format(
                    obj_id=obj['obj_id'],
                    kind=obj['kind'],
                    color=obj['color'],
                    graspable=obj['graspable'],
                    support=obj['can_support'],
                    resting_on=obj['resting_on'],
                    grasped_by=obj['grasped_by'],
                    x=obj['position']['x'],
                    y=obj['position']['y'],
                    z=obj['position']['z'],
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _normalize_value(value):
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, 'x') and hasattr(value, 'y') and hasattr(value, 'z'):
            return {'x': value.x, 'y': value.y, 'z': value.z}
        return value

    @staticmethod
    def _controller_result(result) -> Optional[str]:
        if result is None:
            return 'OK'
        return str(result)

    @staticmethod
    def _require_float(args: Dict[str, object], key: str) -> float:
        value = args.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError('%s must be a number.' % key)
        return float(value)

    @staticmethod
    def _require_int(args: Dict[str, object], key: str) -> int:
        value = args.get(key)
        if not isinstance(value, int):
            raise ValueError('%s must be an integer.' % key)
        return value
