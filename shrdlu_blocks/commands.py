"""Reusable controller command parsing and execution."""

import ast
import io
from typing import Callable, Iterable, List, Optional

from shrdlu_blocks.control import Controller
from shrdlu_blocks.scenes import PhysicalObject
from shrdlu_blocks.typedefs import ObjectID, UnmetConditionError

__all__ = [
    'ControllerCommandError',
    'ControllerCommandExecutor',
]


class ControllerCommandError(ValueError):
    """Raised when a controller command cannot be parsed or executed."""


class ControllerCommandExecutor:
    """Parse and execute public controller commands."""

    def __init__(self, controller: Controller):
        self._controller = controller

    @property
    def controller(self) -> Controller:
        return self._controller

    def list_commands(self) -> List[str]:
        """Return the public controller commands accepted by the executor."""
        commands = []
        for name in dir(self._controller):
            if name.startswith('_'):
                continue
            if callable(getattr(self._controller, name)) or isinstance(
                    getattr(type(self._controller), name, None), property):
                commands.append(name)
        return sorted(commands)

    def describe_commands(self) -> str:
        """Return a text block listing the available commands."""
        output_buffer = io.StringIO()
        print("Commands:", file=output_buffer)
        print("    help", file=output_buffer)
        for name in self.list_commands():
            print("    " + name, file=output_buffer)
        return output_buffer.getvalue().rstrip()

    def execute(self, text: str) -> Optional[str]:
        """Execute a public controller command from a single line of text."""
        command_text = (text or '').strip()
        if not command_text:
            return None
        if command_text == 'help':
            return self.describe_commands()
        pieces = command_text.split()
        command_name = pieces.pop(0)
        if '.' in command_name or command_name.startswith('_'):
            raise ControllerCommandError('Invalid command.')
        if command_name not in dir(self._controller):
            raise ControllerCommandError('Invalid command.')

        args = [self._parse_arg(piece) for piece in pieces]
        attribute = getattr(self._controller, command_name)

        try:
            if callable(attribute) or args:
                result = attribute(*args)
            else:
                result = attribute
        except UnmetConditionError as exc:
            return str(exc)
        except Exception as exc:
            raise ControllerCommandError(str(exc)) from exc

        return self._format_result(command_name, result)

    def try_execute(self, text: str) -> Optional[str]:
        """Execute a command and return formatted errors as text."""
        try:
            return self.execute(text)
        except ControllerCommandError as exc:
            return "ERROR: %s" % exc

    def _format_result(self, command_name: str, result) -> Optional[str]:
        if result is None:
            return None
        if isinstance(result, str) or not hasattr(result, '__iter__'):
            return repr(result)

        output_buffer = io.StringIO()
        object_count = len(list(self._controller.find_objects()))
        for item in result:
            if ('objects' in command_name and isinstance(item, int) and
                    0 <= item < object_count):
                tags = dict(self._controller.iter_object_tags(ObjectID(item)))
                # Construct a lightweight object wrapper to reuse the string representation.
                mock_obj = PhysicalObject(None, None, None, tags)
                print(str(mock_obj), file=output_buffer)
            else:
                print(repr(item), file=output_buffer)
        return output_buffer.getvalue().rstrip() or None

    @staticmethod
    def _parse_arg(piece: str):
        try:
            return ast.literal_eval(piece)
        except (SyntaxError, ValueError):
            return piece
