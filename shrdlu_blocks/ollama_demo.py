"""Natural-language GUI demo backed by an Ollama agent."""

import logging
import os
import sys

import pygame

from shrdlu_blocks.agent import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TRACE_DIR,
    OllamaShrdluAgent,
)
from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.viewer import Viewer

__all__ = ['demo']


def demo(model: str = DEFAULT_MODEL, host: str = 'http://127.0.0.1:11434',
         max_steps: int = DEFAULT_MAX_STEPS, trace_dir: str = DEFAULT_TRACE_DIR):
    """Run the GUI demo with natural-language input routed through Ollama."""
    pygame.init()

    env = ShrdluBlocksEnv()
    agent = OllamaShrdluAgent(
        env,
        model=model,
        host=host,
        max_steps=max_steps,
        trace_dir=trace_dir,
    )

    screen_info = pygame.display.Info()
    screen = pygame.display.set_mode((screen_info.current_w // 2, screen_info.current_h // 2))
    viewer = Viewer(
        screen,
        "SHRDLU Blocks VLA Demo",
        callback=lambda controller, text: agent.handle_user_input(text),
        initial_output=(
            'Type a natural-language instruction.\n'
            'Use /command <controller command> for direct execution.\n'
            'Use /reset to reset the environment.\n'
        ),
    )
    viewer.scene = env.scene
    viewer.run()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    demo(
        model=os.environ.get('SHRDLU_OLLAMA_MODEL', DEFAULT_MODEL),
        host=os.environ.get('SHRDLU_OLLAMA_HOST', 'http://127.0.0.1:11434'),
        max_steps=int(os.environ.get('SHRDLU_AGENT_MAX_STEPS', DEFAULT_MAX_STEPS)),
        trace_dir=os.environ.get('SHRDLU_AGENT_TRACE_DIR', DEFAULT_TRACE_DIR),
    )
