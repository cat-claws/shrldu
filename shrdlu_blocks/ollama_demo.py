"""Natural-language GUI demo backed by an Ollama agent."""

import logging
import sys

import pygame

from shrdlu_blocks.agent import OllamaShrdluAgent
from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.viewer import Viewer

__all__ = ['demo']


def demo(model: str = 'qwen3:14b', host: str = 'http://127.0.0.1:11434'):
    """Run the GUI demo with natural-language input routed through Ollama."""
    pygame.init()

    env = ShrdluBlocksEnv()
    agent = OllamaShrdluAgent(env, model=model, host=host)

    screen_info = pygame.display.Info()
    screen = pygame.display.set_mode((screen_info.current_w // 2, screen_info.current_h // 2))
    viewer = Viewer(
        screen,
        "SHRDLU Blocks VLA Demo",
        callback=lambda controller, text: agent.handle_user_input(text),
        initial_output=(
            'Type a natural-language instruction.\n'
            'Use /command <controller command> for direct execution.\n'
            'Use /reset to reset the environment.'
        ),
    )
    viewer.scene = env.scene
    viewer.run()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    demo()
