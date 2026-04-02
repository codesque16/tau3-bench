# Getting Started

This guide walks you through installing τ-bench, configuring API keys, and running your first evaluation.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and project manager
- Python 3.12+ (uv will download it automatically if not present)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

### 2. Install τ-bench

```bash
uv sync                        # core only (text-mode: airline, retail, telecom, mock)
```

This creates a virtual environment, installs core dependencies from the lockfile, and enables the `tau2` command. The Python version is pinned via `.python-version` (3.12) — uv will download it automatically if needed.

#### Optional extras

Install the extras you need:

```bash
uv sync --extra voice          # + voice/audio-native features
uv sync --extra knowledge      # + banking_knowledge domain (retrieval pipeline)
uv sync --extra gym            # + gymnasium RL interface
uv sync --extra dev            # + pytest, ruff, pre-commit (required for contributing)
uv sync --extra experiments    # + plotting libs for src/experiments/
uv sync --all-extras           # everything
```

#### System dependencies (voice only)

If using voice features (`--extra voice`), install:

**macOS:**
```bash
brew install portaudio ffmpeg
```

> **Note:** If you install without `-e` mode (e.g., `uv pip install .`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:
> ```bash
> export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
> ```

### 3. Verify your installation

```bash
uv run tau2 check-data
```

This checks that your data directory is correctly configured and all required files are present.

## Setting Up API Keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

Copy `.env.example` as `.env` and edit it to include your API keys:

```bash
cp .env.example .env
```

### Voice API Keys (for voice-enabled features)

If you're using voice features, add the following to your `.env` file:
- `ELEVENLABS_API_KEY` — for voice synthesis
- `DEEPGRAM_API_KEY` — for voice transcription

### Vertex Agent (google.genai SDK, optional)

If you want to run the text agent with Vertex AI through the Google GenAI SDK (not LiteLLM), set:
- `VERTEXAI_PROJECT=<your_gcp_project_id>`
- `VERTEXAI_LOCATION=global` (recommended for Gemini 3 models)

Then run:

```bash
tau2 run --domain retail --agent vertex_agent --agent-llm gemini-2.5-flash \
  --user-llm gpt-4.1 --num-trials 1 --num-tasks 5
```

To run both sides without LiteLLM, use:

```bash
tau2 run --domain retail \
  --agent vertex_agent --agent-llm gemini-3.1-flash-lite-preview \
  --user vertex_user_simulator --user-llm gemini/gemini-3.1-pro \
  --num-trials 1 --num-tasks 5
```

### YAML config run (optional)

You can also keep run settings in a YAML file and execute with a single command:

```bash
uv run tau2config --config examples/retail_vertex_text.yaml
```

This is equivalent to passing the same knobs via `tau2 run`.
Use snake_case or kebab-case keys in YAML (for example, `max_concurrency` or `max-concurrency`).

`tau2config` supports both:
- single-run YAML (top-level keys are one run), and
- multi-run YAML using `runs` + top-level `enabled_run_ids`.

Examples:

```yaml
domain: retail
seed: 626729
enabled_run_ids: ["run_a"]
runs:
  run_a:
    num_trials: 4
    trial_concurrency: 2
    max_concurrency: 20
    save_to: retail_a
    fresh: true # -> retail_a_<MM_DD_HH_MIN_SEC>
    # task_ids omitted -> run all task ids in the split
    agent: vertex_agent
    agent_llm: gemini-3.1-flash-lite-preview
    agent_pricing:
      # Example: Gemini 3.1 Flash-Lite Preview (Standard, <=200K context)
      input_cost_per_million: 0.25
      cached_input_cost_per_million: 0.03
      output_cost_per_million: 1.50
    user: vertex_user_simulator
    user_llm: gemini-3.1-pro-preview
    user_pricing:
      # Example: Gemini 3.1 Pro Preview (Standard, <=200K context)
      input_cost_per_million: 2.00
      cached_input_cost_per_million: 0.20
      output_cost_per_million: 12.00
  run_b:
    enabled: false
    task_ids: ["0", "1", "2"]
    save_to: retail_b
```

You can also override selection from CLI:

```bash
uv run tau2config --config examples/retail_vertex_text.yaml --run-ids run_a run_b
```

When using Vertex models and you want run-level cost telemetry, set both
`agent_pricing` and `user_pricing` explicitly per run using the current values from the
Vertex pricing page.

## Running Your First Evaluation

### Standard text-based evaluation (half-duplex)

```bash
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 \
  --num-trials 1 --num-tasks 5
```

Results are saved in `data/simulations/`.

### Audio native mode (voice full-duplex)

```bash
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs
```

See the [Audio Native Documentation](../src/tau2/voice/audio_native/README.md) for provider configuration and all options.

### Knowledge retrieval evaluation

```bash
tau2 run --domain banking_knowledge --retrieval-config bm25 \
  --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 5
```

See the [Knowledge Retrieval Documentation](../src/tau2/knowledge/README.md) for retrieval configuration options.


> **tip**: for full agent evaluation that matches the original τ-bench methodology, remove `--num-tasks` to evaluate on the complete task set (the `base` split is used by default).

## Simulation Output Structure

Results are stored in one of two formats, chosen automatically based on modality:

### Text runs — monolithic JSON

Text-based simulations produce a single file containing all data:

```
data/simulations/<run_name>/
└── results.json             # Metadata, tasks, and all simulation data
```

### Voice runs — directory-based format

Voice simulations contain large tick-level data, so they use a directory-based format that splits simulation data into individual files for efficient checkpointing and streaming:

```
data/simulations/<run_name>/
├── results.json                        # Metadata and task definitions only
├── simulations/                        # Individual simulation data files
│   ├── sim_0.json
│   ├── sim_1.json
│   └── ...
└── artifacts/                          # Runtime artifacts (with --verbose-logs)
    └── task_<task_id>/
        └── sim_<uuid>/
            ├── sim_status.json         # Simulation status
            ├── task.log                # Per-task log
            ├── audio/
            │   ├── both.wav            # Full conversation audio (stereo)
            │   ├── assistant_labels.txt # Audacity labels for agent speech
            │   ├── user_labels.txt     # Audacity labels for user speech
            │   ├── assistant_tool_calls_labels.txt  # Audacity labels for agent tool calls (when present)
            │   └── user_tool_calls_labels.txt       # Audacity labels for user tool calls (when present)
            └── llm_debug/
                └── *.json              # LLM call logs
```

### Format conversion

You can convert between formats using `tau2 convert-results`:

```bash
# Convert a monolithic JSON to directory format
tau2 convert-results data/simulations/my_run --to dir

# Convert a directory format back to monolithic JSON
tau2 convert-results data/simulations/my_run --to json
```

Both formats are fully supported by `Results.load()`, which auto-detects the format on disk.

## Viewing Results

```bash
tau2 view
```

This allows you to browse simulation files, view agent performance metrics, inspect individual simulations, and view task details. Works for both standard text and audio native runs.

## Configuration

The framework is configured via [`src/tau2/config.py`](../src/tau2/config.py).

### LLM Call Caching

LLM call caching is disabled by default. To enable it:

1. Install the `redis` Python package: `uv pip install redis`
2. Make sure a Redis server is running
3. Update the redis config in `config.py` if necessary
4. Set `LLM_CACHE_ENABLED` to `True` in `config.py`

## Cleanup

To remove all generated files and the virtual environment:

```bash
make clean
```

## Next Steps

- [CLI Reference](cli-reference.md) — all `tau2` commands and options
- [Agent Developer Guide](../src/tau2/agent/README.md) — build and evaluate your own agent
- [Domain Documentation](../src/tau2/domains/README.md) — understand the available domains
- [Communication Modes](../src/tau2/orchestrator/README.md) — half-duplex and full-duplex orchestration
- [Knowledge Retrieval](../src/tau2/knowledge/README.md) — retrieval pipeline setup and configuration for banking_knowledge domain
- [Voice (Full-Duplex)](../src/tau2/voice/README.md) — providers, speech complexity, and CLI options for voice evaluation
- [Gym/RL Interface](../src/tau2/gym/README.md) — Gymnasium-compatible environment for RL training
