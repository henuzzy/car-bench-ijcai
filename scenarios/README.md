# Scenario TOML Run Configs

Scenario files are the run configs for local and Docker CAR-bench evaluations.
They define which evaluator and agent-under-test to run, which task subset to
evaluate, how many trials to execute, and which environment variables or Docker
images/builds are used.

## Directory Layout

Scenario directories mirror the reference agent package names under `src/`.
Participant agents should follow the same pattern when adding their own
scenarios.

| Agent Package | Scenario Directory |
| --- | --- |
| `src/track_1_agent_under_test/` | `scenarios/track_1_agent_under_test/` |
| `src/track_2_agent_under_test_codex/` | `scenarios/track_2_agent_under_test_codex/` |
| `src/track_2_agent_under_test_codex_planner/` | `scenarios/track_2_agent_under_test_codex_planner/` |
| `src/track_2_agent_under_test_codex_python/` | `scenarios/track_2_agent_under_test_codex_python/` |

Each directory contains the same six-file matrix:

| Scenario File | Mode | Task Selection |
| --- | --- | --- |
| `local_smoke.toml` | Local Python | Train split, one task from each task type, one trial |
| `local_test_set.toml` | Local Python | Public CAR-bench test split, all tasks from each task type, three trials |
| `local_docker_smoke.toml` | Docker local build | Train split, one task from each task type, one trial |
| `local_docker_test_set.toml` | Docker local build | Public CAR-bench test split, all tasks from each task type, three trials |
| `ghcr_smoke.toml` | Docker published image | Train split, one task from each task type, one trial |
| `ghcr_test_set.toml` | Docker published image | Public CAR-bench test split, all tasks from each task type, three trials |

The public test-set scenarios are development validation only. Official final
evaluation is run by the organizers on a hidden test set.

## TOML Structure

Every scenario has three main tables:

```toml
[evaluator]
# local Python: endpoint + cmd
# Docker: build or image, env, volumes, optional command_args

[agent_under_test]
# local Python: endpoint + cmd
# Docker: build or image, env, volumes, optional command_args
# optional result labels: name, result_label, result_model, result_reasoning_effort

[config]
# CAR-bench task/trial selection
```

### `[evaluator]`

The evaluator wraps CAR-bench and owns the simulated user, tools, environment,
and scoring.

For local Python scenarios, provide:

```toml
[evaluator]
endpoint = "http://127.0.0.1:8081"
cmd = "python src/evaluator/server.py --host 127.0.0.1 --port 8081"
```

For Docker scenarios, provide either `build` or `image`:

```toml
[evaluator]
build = { context = ".", dockerfile = "src/evaluator/Dockerfile.evaluator" }
env = { GEMINI_API_KEY = "${GEMINI_API_KEY:?Set GEMINI_API_KEY in .env}" }
volumes = ["./third_party/car-bench:/workspace/third_party/car-bench:ro"]
```

### `[agent_under_test]`

This is the participant or reference agent being evaluated.

For local Python scenarios:

```toml
[agent_under_test]
endpoint = "http://127.0.0.1:8080"
cmd = "python src/track_1_agent_under_test/server.py --host 127.0.0.1 --port 8080"
```

For Docker local-build scenarios:

```toml
[agent_under_test]
build = { context = ".", dockerfile = "src/track_1_agent_under_test/Dockerfile.track-1-agent-under-test" }
env = { AGENT_LLM = "${AGENT_LLM:-gemini/gemini-2.5-flash}" }
```

For GHCR scenarios:

```toml
[agent_under_test]
image = "ghcr.io/yourusername/your-agent:latest"
env = { AGENT_LLM = "${AGENT_LLM:-gemini/gemini-2.5-flash}" }
```

Optional result-label fields help make output filenames and metadata easier to
read when your harness routes through multiple models:

```toml
[agent_under_test]
name = "my-agent"
result_model = "my-model-or-harness-label"
result_reasoning_effort = "medium"
```

### `[config]`

`[config]` maps to CAR-bench evaluation options:

```toml
[config]
num_trials = 3
task_split = "test"         # "train" or "test"
max_steps = 50

tasks_base_num_tasks = -1
tasks_hallucination_num_tasks = -1
tasks_disambiguation_num_tasks = -1

# Optional exact task filters:
# tasks_base_task_id_filter = ["base_0"]
# tasks_hallucination_task_id_filter = ["hallucination_0"]
# tasks_disambiguation_task_id_filter = ["disambiguation_0"]
```

Use `-1` for all tasks in a task type. Smoke scenarios use `1` for each task
type so you can quickly validate the full loop.

## Environment Variables

Docker scenario env values use Compose-style interpolation:

| Syntax | Meaning |
| --- | --- |
| `${VAR}` | Substitute `VAR` from `.env` or the shell. |
| `${VAR:-default}` | Use `default` when `VAR` is unset or blank. |
| `${VAR:?message}` | Fail early with `message` when `VAR` is missing. |

Keep secret values in `.env` or your deployment secret manager. Do not commit
real API keys.

## Docker Compose Generation

Generate Docker Compose from any `local_docker_*.toml` or `ghcr_*.toml` file:

```bash
uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit
```

`generate_compose.py` writes two ignored files into the selected scenario
folder:

- `docker-compose.yml`: starts evaluator, agent-under-test, and the A2A client.
- `a2a-scenario.toml`: internal scenario consumed by the A2A client inside
  Docker.

Results are written under `output/<agent-name>/`.
