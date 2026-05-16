# Track 2 Codex Python-Call Agent Under Test

This package is a Track 2 reference implementation for a Python-call DSL
harness. It uses Codex Pro-backed inference and defaults to
`gpt-5.3-codex-spark`, the fast executor served on Cerebras infrastructure.

1. Codex Spark returns ordinary chat text plus one fenced `python` action block.
2. The adapter extracts that block and parses it with Python's built-in `ast`
   module.
3. Parsed calls are converted into normal A2A text Parts or tool-call data
   Parts for the evaluator.

The generated Python is never executed. This is inspired by programmatic tool
calling, but it is not true code execution and it does not add hidden tools.
The evaluator remains the only component that executes CAR-bench tools.

This README only describes what is special about the Python-call DSL variant.
Read the [base Codex agent README](../track_2_agent_under_test_codex/README.md)
first for the shared technical setup: what `codex app-server` is, how Codex is
started and authenticated, how A2A messages are converted into Codex turns, and
which benchmark boundaries the adapter enforces.

Like the direct Codex agent, this variant warms the shared `codex app-server`
process during A2A server startup, then sends each benchmark step as an
explicit transcript-based Codex turn. The difference is only the action syntax
Codex is asked to produce and the parser used before returning normal A2A
Parts.

For the shared A2A turn contract, start with the
[development guide](../../docs/development-guide.md). For allowed advanced
harnessing boundaries and additional Track 2 patterns, see
[agent-under-test-harnessing.md](../../docs/agent-under-test-harnessing.md) and
[codex-harness-patterns.md](../../docs/codex-harness-patterns.md).

## Accepted DSL

Codex may include a short private note before the action block:

````text
The request needs a specific percentage before changing the shade.

```python
respond("What percentage should I set it to?")
```
````

The code block itself may contain:

```python
respond("Sure, what percentage should I set it to?")
```

```python
open_close_sunshade(percentage=50)
```

```python
get_user_preferences(preference_categories={"vehicle_settings": {"vehicle_settings": True}})
open_close_sunshade(percentage=50)
```

Only top-level direct calls are accepted. Imports, assignments, variables,
attributes, loops, conditionals, comprehensions, positional tool arguments, and
non-literal arguments are rejected.

The code block should contain either `respond(...)` or tool calls. If Codex
emits a valid tool call followed by a premature `respond("Done")`, the parser
keeps the tool call and ignores the response. The evaluator will send the tool
result back, and the agent can confirm completion on the next turn.

The older `{"python_code": "..."}` JSON envelope is still accepted by the
parser for compatibility, but the default prompt uses fenced Python because it
is closer to Codex's natural code-proposal behavior.

## Local Run

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_codex_python/local_smoke.toml --show-logs
```

## Docker Run

```bash
uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_codex_python/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_2_agent_under_test_codex_python/docker-compose.yml up --abort-on-container-exit
```

Set `CODEX_HOME_HOST` in `.env` to an absolute host path containing Codex auth.
Prefer creating a dedicated benchmark home with
`CODEX_HOME="$HOME/.codex-car-bench" codex login` instead of mounting your
everyday Codex desktop/app state.
