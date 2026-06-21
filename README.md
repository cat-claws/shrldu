# SHRDLU Blocks VLA

`SHRDLUBlocksVLA` is a fork of `SHRDLUBlocks` that keeps the original block-world
simulator and adds a source-run VLA-style interface on top of it.

The main pieces are:
- `shrdlu_blocks.demo`: the original low-level command demo
- `shrdlu_blocks.ollama_demo`: the natural-language GUI backed by Ollama
- `shrdlu_blocks.preplanned_ollama_demo`: the natural-language GUI that plans the full action sequence first
- `shrdlu_blocks.openai_demo`: the natural-language GUI backed by an OpenAI-compatible API
- `shrdlu_blocks.preplanned_openai_demo`: the plan-first OpenAI-compatible GUI
- `shrdlu_blocks.predictive_preplanned_ollama_demo`: the tree-search predictive Ollama GUI
- `shrdlu_blocks.predictive_preplanned_openai_demo`: the tree-search predictive OpenAI-compatible GUI
- `shrdlu_blocks.suffix_predictive_preplanned_ollama_demo`: the suffix-replanning predictive Ollama GUI
- `shrdlu_blocks.suffix_predictive_preplanned_openai_demo`: the suffix-replanning predictive OpenAI-compatible GUI
- `shrdlu_blocks.env`: a reusable in-process environment wrapper
- `shrdlu_blocks.agent`: the Ollama-backed agent loop

## Requirements

Before running, install:

```bash
python3 -m pip install -r requirements.txt
```

Ollama itself is expected to already be running separately on `127.0.0.1:11434`.

## Run From Source

From the `SHRDLUBlocksVLA` folder, run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.ollama_demo
```

This starts:
- the pygame GUI
- the in-memory simulator environment
- the Ollama-backed agent client

You do not need to build or compile the project first.

## Preplanned Ollama Demo

If you want a plan-first agent instead of the default step-by-step replanning loop, run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.preplanned_ollama_demo
```

This variant reads the scene once, plans the entire action sequence up front, and then
executes that stored plan without asking the model again after each action.

## Predictive Preplanned Demo

If you want a stronger planning mode that predicts the next state, verifies the properties,
and retries or backtracks before any real execution, run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.predictive_preplanned_ollama_demo
```

This variant:
- plans one next action at a time
- asks the model to predict the resulting symbolic next state
- checks the predicted transition against the property set
- retries or backtracks during planning if the predicted step violates properties
- executes only after a full property-satisfying plan is found

You can tune branch retries with:

```bash
export SHRDLU_AGENT_MAX_BRANCH_RETRIES=3
```

If you want the same predictive planning mode over the OpenAI-compatible backend, run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.predictive_preplanned_openai_demo
```

## Suffix Predictive Preplanned Demo

If you want a predictive planner that proposes the full remaining suffix, predicts along
that suffix, and replans from the first predicted violation point, run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.suffix_predictive_preplanned_ollama_demo
```

This variant:
- plans the full remaining suffix instead of only the next action
- predicts symbolic state changes step by step along that suffix
- stops at the first predicted violation
- keeps the verified prefix and replans the rest from the last valid predicted state
- executes only after a full property-satisfying plan is found

If you want the same suffix predictive mode over the OpenAI-compatible backend, run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.suffix_predictive_preplanned_openai_demo
```

## OpenAI-Compatible Demo

To use a local OpenAI-compatible server like the example below:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="None",
)
```

run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.openai_demo
```

The default OpenAI-compatible settings are:
- base URL: `http://127.0.0.1:30000/v1`
- API key: `EMPTY`
- model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- temperature: `0.2`
- max tokens per chat call: `512`

If you want the same OpenAI-compatible backend but with a full plan computed up front, run:

```bash
source .venv/bin/activate
python3 -m shrdlu_blocks.preplanned_openai_demo
```

You can override them with environment variables:

```bash
export SHRDLU_OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export SHRDLU_OPENAI_API_KEY=EMPTY
export SHRDLU_OPENAI_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
export SHRDLU_OPENAI_TEMPERATURE=0.2
export SHRDLU_OPENAI_MAX_TOKENS=512
python3 -m shrdlu_blocks.openai_demo
```

## GUI Usage

Inside the GUI text box:
- type natural-language instructions like `move the grasper over the blue block`
- use `/command ...` for direct simulator commands such as `/command move_grasper -0.1 0.4`
- use `/reset` to reset the world

The default model is `qwen3.5:27b`, the default agent budget is `50` steps, and the
agent calls Ollama over HTTP.

If `qwen3.5:27b` is not installed locally yet, pull it first:

```bash
ollama pull qwen3.5:27b
```

Every natural-language request now saves a trace JSON file in `agent_traces/` with:
- the request
- each prompt sent to the model
- raw model replies and retry details
- parsed actions and simulator results

You can override runtime settings with environment variables:

```bash
export SHRDLU_OLLAMA_MODEL=llama3.3:latest
export SHRDLU_AGENT_MAX_STEPS=50
export SHRDLU_AGENT_TRACE_DIR=agent_traces
python3 -m shrdlu_blocks.ollama_demo
```

## Original Command Demo

If you want the low-level command-shell version instead of the agent-driven one:

```bash
python3 -m shrdlu_blocks.demo
```

## Project Structure

- `shrdlu_blocks/control.py`: low-level controller API
- `shrdlu_blocks/scenes.py`: world objects and default scene
- `shrdlu_blocks/viewer.py`: pygame viewer and text input box
- `shrdlu_blocks/commands.py`: reusable direct command execution
- `shrdlu_blocks/env.py`: environment wrapper for agent use
- `shrdlu_blocks/agent.py`: validated-action agent loop over Ollama HTTP
- `shrdlu_blocks/ollama_demo.py`: natural-language GUI entry point
- `shrdlu_blocks/preplanned_ollama_demo.py`: plan-first Ollama GUI entry point
- `shrdlu_blocks/openai_demo.py`: OpenAI-compatible GUI entry point
- `shrdlu_blocks/preplanned_openai_demo.py`: plan-first OpenAI-compatible GUI entry point
- `shrdlu_blocks/predictive_preplanned_ollama_demo.py`: stepwise predictive Ollama GUI entry point
- `shrdlu_blocks/predictive_preplanned_openai_demo.py`: stepwise predictive OpenAI-compatible GUI entry point
- `shrdlu_blocks/suffix_predictive_preplanned_ollama_demo.py`: suffix-replanning predictive Ollama GUI entry point
- `shrdlu_blocks/suffix_predictive_preplanned_openai_demo.py`: suffix-replanning predictive OpenAI-compatible GUI entry point

## Notes

- Running from source is the intended workflow for this fork.
- Packaging metadata is still present, but it is not required for normal use.
- If `pygame` is missing, install it in the Python environment you plan to use for running the demo.
