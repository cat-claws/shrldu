"""Natural-language GUI demo backed by a predictive preplanned OpenAI-compatible agent."""

import logging
import os
import sys

import pygame

from shrdlu_blocks.agent import (
    DEFAULT_MAX_STEPS,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TRACE_DIR,
)
from shrdlu_blocks.env import ShrdluBlocksEnv
from shrdlu_blocks.predictive_preplanned_agent import PredictivePreplannedOpenAICompatibleShrdluAgent
from shrdlu_blocks.viewer import Viewer

__all__ = ['demo']


def demo(model: str = DEFAULT_OPENAI_MODEL,
         base_url: str = DEFAULT_OPENAI_BASE_URL,
         api_key: str = DEFAULT_OPENAI_API_KEY,
         max_steps: int = DEFAULT_MAX_STEPS,
         trace_dir: str = DEFAULT_TRACE_DIR,
         temperature: float = 0.2,
         max_tokens: int = 512,
         max_branch_retries: int = 3):
    """Run the GUI demo with a predictive preplanned OpenAI-compatible agent."""
    pygame.init()

    env = ShrdluBlocksEnv()
    agent = PredictivePreplannedOpenAICompatibleShrdluAgent(
        env,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_steps=max_steps,
        trace_dir=trace_dir,
        temperature=temperature,
        max_tokens=max_tokens,
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
        "SHRDLU Blocks Predictive Preplanned OpenAI-Compatible Demo",
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
        model=os.environ.get('SHRDLU_OPENAI_MODEL', DEFAULT_OPENAI_MODEL),
        base_url=os.environ.get('SHRDLU_OPENAI_BASE_URL', DEFAULT_OPENAI_BASE_URL),
        api_key=os.environ.get('SHRDLU_OPENAI_API_KEY', DEFAULT_OPENAI_API_KEY),
        max_steps=int(os.environ.get('SHRDLU_AGENT_MAX_STEPS', DEFAULT_MAX_STEPS)),
        trace_dir=os.environ.get('SHRDLU_AGENT_TRACE_DIR', DEFAULT_TRACE_DIR),
        temperature=float(os.environ.get('SHRDLU_OPENAI_TEMPERATURE', '0.2')),
        max_tokens=int(os.environ.get('SHRDLU_OPENAI_MAX_TOKENS', '512')),
        max_branch_retries=int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3')),
    )
