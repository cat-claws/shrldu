"""Natural-language GUI demo backed by a predictive preplanned Ollama agent."""

import logging
import os
import sys

import pygame

from shrdlu_blocks.agent import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TRACE_DIR,
)
from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.predictive_preplanned_agent import PredictivePreplannedOllamaShrdluAgent
from shrdlu_blocks.viewer import Viewer

__all__ = ['demo']


def demo(model: str = DEFAULT_MODEL, host: str = 'http://127.0.0.1:11434',
         max_steps: int = DEFAULT_MAX_STEPS, trace_dir: str = DEFAULT_TRACE_DIR,
         max_branch_retries: int = 3):
    """Run the GUI demo with a predictive preplanned Ollama agent."""
    pygame.init()

    env = ShrdluBlocksEnv()
    agent = PredictivePreplannedOllamaShrdluAgent(
        env,
        model=model,
        host=host,
        max_steps=max_steps,
        trace_dir=trace_dir,
        max_branch_retries=max_branch_retries,
    )

    screen_info = pygame.display.Info()
    screen = pygame.display.set_mode((screen_info.current_w // 2, screen_info.current_h // 2))
    viewer = None

    def handle_text(controller, text):
        del controller
        result = agent.handle_user_input(text)
        viewer.scene = env.scene
        return result

    viewer = Viewer(
        screen,
        "SHRDLU Blocks Predictive Preplanned Ollama Demo",
        callback=handle_text,
        initial_output=(
            'Type a natural-language instruction.\n'
            'This demo plans one predicted step at a time, verifies properties, then executes once.\n'
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
        max_branch_retries=int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3')),
    )
