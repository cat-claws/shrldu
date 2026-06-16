"""
A simple demo of the environment.

Usage:
    python3 -m shrdlu_blocks.demo

The environment will be displayed in a graphics window. The user can type
various commands into the graphics window to query the scene and control the
grasper. Type `help` to get a list of commands.
"""

import logging
import sys

import pygame.display
from shrdlu_blocks.commands import ControllerCommandExecutor
from shrdlu_blocks.control import Controller
from shrdlu_blocks.viewer import Viewer

__all__ = ['demo']


def demo_callback(controller: Controller, command: str) -> str:
    """Parse and execute the command."""
    if command == 'exit':
        pygame.quit()
        sys.exit(0)
    executor = ControllerCommandExecutor(controller)
    return executor.try_execute(command)


def demo():
    """
    Let the user play around with the standard scene using programmatic
    instructions passed directly to the controller.

    The environment will be displayed in a graphics window. The user can type
    various commands into the graphics window to query the scene and control
    the grasper. Type `help` to get a list of commands.
    """

    pygame.init()

    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    screen = pygame.display.set_mode((screen_width // 2, screen_height // 2))

    Viewer(screen, "SHRDLU Blocks Demo", demo_callback,
           initial_output='Type "help" for a list of available commands.').run()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    demo()
