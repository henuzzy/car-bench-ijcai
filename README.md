# IJCAI-ECAI 2026 Competition
# CAR-bench: Building Reliable LLM Agents Under Real-World Uncertainty

[![Paper](https://img.shields.io/badge/Paper-2601.22027-b31b1b.svg)](https://arxiv.org/abs/2601.22027)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![A2A](https://img.shields.io/badge/A2A-Protocol-blue.svg)](https://a2a-protocol.org)
[![Website](https://img.shields.io/badge/Website-CAR--bench-blue)](https://car-bench.github.io/car-bench/)

Dockerized A2A starter kit for the CAR-bench Challenge at IJCAI-ECAI 2026.

[Overview](#overview) | [Setup](#setup) | [Build An Agent](#build-an-agent) | [Validate](#validate-your-agent) | [Submit](#submission-instructions) | [Evaluation](#evaluation-summary) | [Read More](#read-more)

---

## Overview

[CAR-bench](https://github.com/CAR-bench/car-bench) evaluates whether tool-using
LLM agents behave reliably in realistic, uncertain, user-facing settings. The
benchmark is instantiated as an in-car voice assistant domain with ambiguous
requests, mutable vehicle/environment state, domain policies, and unavailable
capabilities.

This repository turns CAR-bench into a competition-ready A2A evaluation harness.
Participants build a dockerized **agent under test**. The evaluator sends the
agent policy context, user messages, tool definitions, and tool results. The
agent returns either user-facing text or tool calls. The evaluator remains the
only component that executes CAR-bench tools and computes scores.

The official competition has two tracks:

| Track | Goal | Starter |
| --- | --- | --- |
| **Track 1: Open Track** | Use any model, provider, framework, or architecture to maximize reliability. The Best Innovation Award focuses on agent harnessing and reliability design. | [`src/track_1_agent_under_test/`](src/track_1_agent_under_test/) |
| **Track 2: Cerebras Fast-Reasoning with Codex Pro** | Use Codex Pro-backed agents around Cerebras-hosted `gpt-5.3-codex-spark` to turn fast inference into better reliability under the official time budget. Participation is limited to 15 teams. | [`src/track_2_agent_under_test_codex/`](src/track_2_agent_under_test_codex/) and variants |

Final ranking is performed by the organizers on a hidden test set. The local
`local_test_set.toml`, `local_docker_test_set.toml`, and `ghcr_test_set.toml`
scenarios are for development validation only.

CAR-bench includes:

| Dimension | Details |
| --- | --- |
| Tools | 58 interconnected navigation, vehicle-control, charging, weather, and productivity tools |
| Policies | 19 domain-specific policies agents must follow |
| User model | LLM-simulated multi-turn user |
| Tasks | 254 public CAR-bench tasks across Base, Hallucination, and Disambiguation categories |
| Main reliability metric | `Pass^3`: a task must pass all 3 independent trials |

---

## Architecture

```text
Participant development
  -> choose Track 1 or Track 2 starter
  -> build an A2A agent under test
  -> validate locally and in Docker
  -> publish a GHCR Docker image + config
  -> organizers run hidden evaluation

Runtime exchange

CAR-bench evaluator
  - owns simulated user, tools, environment state, trajectories, and scoring
  - sends A2A messages to the submitted agent
  - executes only the tool calls returned by the agent

        A2A text/data messages
              <---->

Agent under test
  - receives policy/user text, tool definitions, and tool results
  - maintains its own conversation state per context_id
  - returns user-facing text and/or tool-call data
```

The benchmark boundary matters: agents may do internal planning, verification,
reranking, memory, or multi-pass reasoning, but must use only evaluator-provided
inputs and must not inspect hidden evaluator/task state or execute CAR-bench
tools directly.

---

## Setup

### Common Setup

For serious participation, create a fork of this starter repository first, then
clone your fork. That gives you a clean place for your agent code, Dockerfile,
scenario configs, and technical-report notes. Official submission will still be
a GHCR Docker image plus config, not a pull request to this repository.

```bash
git clone https://github.com/YOUR_ORG_OR_USERNAME/car-bench-ijcai.git
cd car-bench-ijcai

python3.11 -m venv .venv
source .venv/bin/activate

./scripts/setup_car_bench.sh
cp .env.example .env
```

`./scripts/setup_car_bench.sh` clones the original CAR-bench repository into
`third_party/car-bench/`. That checkout is ignored by git and treated as a local
dependency for evaluator runs.

Set at least the evaluator key in `.env`:

```bash
GEMINI_API_KEY=...
```

### Track 1 Setup

Track 1 can use any model provider. Install the Track 1 template dependencies:

```bash
uv sync --extra track-1-agent --extra car-bench-evaluator
```

Then add whatever provider keys your agent implementation needs to `.env`, for
example:

```bash
AGENT_LLM=anthropic/claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=...
```

The Track 1 starter is documented in
[`src/track_1_agent_under_test/README.md`](src/track_1_agent_under_test/README.md).

### Track 2 Setup

Track 2 uses Codex Pro-backed inference. The reference agents default to
`gpt-5.3-codex-spark` as the fast executor and pin `@openai/codex@0.130.0` in
Docker builds for reproducibility.

The Track 2 reference agents share the same Python dependencies; their main
runtime dependency is the external Codex CLI:

```bash
uv sync --extra car-bench-evaluator
```

For local Codex runs, install and authenticate the Codex CLI. Installation
packages, platform requirements, and installation methods may change, so follow
the official [Codex CLI installation
instructions](https://developers.openai.com/codex/cli). For Docker runs, use a
dedicated Codex home instead of mounting your everyday Codex app state:

```bash
mkdir -p "$HOME/.codex-car-bench"
CODEX_HOME="$HOME/.codex-car-bench" codex login
CODEX_HOME="$HOME/.codex-car-bench" codex login status
```

Then set the absolute path in `.env`:

```bash
CODEX_HOME_HOST=/Users/yourname/.codex-car-bench
GEMINI_API_KEY=...
LOGURU_LEVEL=DEBUG
```

Track 2 details live in the agent READMEs:

| Reference | README |
| --- | --- |
| Direct Codex JSON agent | [`src/track_2_agent_under_test_codex/README.md`](src/track_2_agent_under_test_codex/README.md) |
| Planner/executor agent | [`src/track_2_agent_under_test_codex_planner/README.md`](src/track_2_agent_under_test_codex_planner/README.md) |
| Python-call DSL agent | [`src/track_2_agent_under_test_codex_python/README.md`](src/track_2_agent_under_test_codex_python/README.md) |

---

## Build An Agent

The most important contract is simple: each turn, the evaluator sends your agent
the information it is allowed to use, and your agent sends back one
benchmark-visible response.

| Turn situation | Evaluator sends | Your agent returns |
| --- | --- | --- |
| First turn of a task | text Part with `System: ... User: ...` plus data Part with `{"tools": [...]}` | text Part for a user response, data Part with `{"tool_calls": [...]}`, or both |
| After your agent called tools | data Part with `{"tool_results": [...]}` | another text response and/or tool-call data |
| After your agent spoke to the user | next simulated user message as a text Part | another text response and/or tool-call data |
| Any turn | same `context_id` for the task | maintain your own per-context conversation state |

The evaluator executes all CAR-bench tools. Your agent should only request tool
calls by returning data like:

```json
{"tool_calls": [{"tool_name": "get_weather", "arguments": {"location_or_poi_id": "loc_123"}}]}
```

For exact A2A shapes, protobuf helper usage, metadata, and code references, read:

- [Inbound messages: what your agent receives](docs/development-guide.md#inbound-messages--what-your-agent-receives)
- [Outbound messages: what your agent should return](docs/development-guide.md#outbound-messages--what-your-agent-should-return)
- [Agent executor contract](docs/development-guide.md#agent-executor-contract)

### Reference Agents

| Agent | Package | Scenario Directory | Best For |
| --- | --- | --- | --- |
| Track 1 template | [`src/track_1_agent_under_test/`](src/track_1_agent_under_test/) | [`scenarios/track_1_agent_under_test/`](scenarios/track_1_agent_under_test/) | Building your own provider/model integration |
| Track 2 Codex JSON | [`src/track_2_agent_under_test_codex/`](src/track_2_agent_under_test_codex/) | [`scenarios/track_2_agent_under_test_codex/`](scenarios/track_2_agent_under_test_codex/) | Simple Spark next-action baseline |
| Track 2 planner/executor | [`src/track_2_agent_under_test_codex_planner/`](src/track_2_agent_under_test_codex_planner/) | [`scenarios/track_2_agent_under_test_codex_planner/`](scenarios/track_2_agent_under_test_codex_planner/) | Larger planner plus Spark executor |
| Track 2 Python-call DSL | [`src/track_2_agent_under_test_codex_python/`](src/track_2_agent_under_test_codex_python/) | [`scenarios/track_2_agent_under_test_codex_python/`](scenarios/track_2_agent_under_test_codex_python/) | Programmatic-tool-calling-style action blocks |

---

## Validate Your Agent

Use the same progression for either track: first local smoke tests, then local
Docker builds, then GHCR image validation.

### Scenario Files

Scenario TOML files are the run configs for CAR-bench evaluations. They choose
which evaluator and agent to start or pull, which environment variables to pass,
which task split and task counts to run, and how many trials to execute. Each
agent directory under `scenarios/` has the same six-file matrix:

| Scenario File | Purpose |
| --- | --- |
| `local_smoke.toml` | Local Python, train split, one task from each task type, one trial |
| `local_test_set.toml` | Local Python, public CAR-bench test split, three trials |
| `local_docker_smoke.toml` | Local Docker build, train smoke |
| `local_docker_test_set.toml` | Local Docker build, public CAR-bench test split |
| `ghcr_smoke.toml` | Published image, train smoke |
| `ghcr_test_set.toml` | Published image, public CAR-bench test split |

### A. Local Smoke And Debug

Fastest way to iterate on code. Agents run as local Python processes.

| Track | Command |
| --- | --- |
| Track 1 | `uv run car-bench-run scenarios/track_1_agent_under_test/local_smoke.toml --show-logs` |
| Track 2 Codex JSON | `uv run car-bench-run scenarios/track_2_agent_under_test_codex/local_smoke.toml --show-logs` |
| Track 2 planner/executor | `uv run car-bench-run scenarios/track_2_agent_under_test_codex_planner/local_smoke.toml --show-logs` |
| Track 2 Python-call DSL | `uv run car-bench-run scenarios/track_2_agent_under_test_codex_python/local_smoke.toml --show-logs` |

Use the corresponding `local_test_set.toml` only after the smoke scenario works.
Local test-set runs are development validation, not official final evaluation.

### B. Docker Local Build

Use this before publishing. It verifies that your Dockerfile and runtime
environment work without local Python process assumptions.

| Track | Generate Compose | Run |
| --- | --- | --- |
| Track 1 | `uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/local_docker_smoke.toml` | `docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit` |
| Track 2 Codex JSON | `uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_codex/local_docker_smoke.toml` | `docker compose --env-file .env -f scenarios/track_2_agent_under_test_codex/docker-compose.yml up --abort-on-container-exit` |

For the Track 2 planner/executor or Python-call agents, use the same commands
with their scenario directories. `generate_compose.py` writes
`docker-compose.yml` and `a2a-scenario.toml` next to the selected Docker
scenario; those generated files are ignored by git.

### C. GHCR Image Validation

Use this to test the same kind of image/config that organizers will run.

Build and push an `linux/amd64` image:

```bash
docker build --platform linux/amd64 \
  -f src/track_1_agent_under_test/Dockerfile.track-1-agent-under-test \
  -t ghcr.io/yourusername/your-agent:latest .

docker push ghcr.io/yourusername/your-agent:latest
```

Then update your `ghcr_smoke.toml` or `ghcr_test_set.toml` image reference and
validate it:

```bash
uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/ghcr_smoke.toml
docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit
```

For Track 2, use the matching Track 2 scenario directory and Dockerfile.

If you intentionally want CI publishing, this repo includes a disabled opt-in
template at
[`.github/workflows/publish-ghcr.yml.disabled`](.github/workflows/publish-ghcr.yml.disabled).
It has no push/tag trigger; rename and configure it only if you want to use it.

Results are written under `output/<agent-name>/` with filenames that include
timestamp, scenario, task selection, trial count, and reliable model/reasoning
hints when the scenario exposes them.

---

## Submission Instructions

Detailed submission mechanics will be announced by the organizers. The expected
submission shape is:

1. A registered GHCR Docker image for your agent under test, preferably pinned
   by digest.
2. The scenario/config file needed to run that image.
3. Required environment variable or secret names, excluding secret values.
4. Track selection: Track 1, Track 2, or both.

The organizers will run submitted Docker agents and configs on controlled
evaluation infrastructure. Final ranking is determined from hidden test set
evaluation, not from local public development scenarios.

---

## Evaluation Summary

CAR-bench evaluates different reliability failures across three task types:

| Task Type | Public CAR-bench Tasks | What It Tests |
| --- | ---: | --- |
| Base | 100 | Correct tool use, final state, intermediate state, and policy compliance |
| Hallucination | 98 | Whether the agent acknowledges missing capabilities/data instead of fabricating |
| Disambiguation | 56 | Whether the agent resolves ambiguity through preferences or clarification before acting |

Each task receives fine-grained metric scores such as action correctness,
required information-gathering tools, tool execution validity, policy
compliance, and user end-conversation behavior. A task reward is 1 only when all
required metrics for that task pass.

The competition reports consistency:

| Metric | Meaning |
| --- | --- |
| `Pass^3` | Task passes in all 3 independent trials. This is the main deployment-readiness score. |
| `Pass@3` | Task passes in at least 1 of 3 trials. This measures latent capability. |

For implementation details, see the original CAR-bench reward calculators in
`third_party/car-bench/car_bench/envs/reward_calculators.py` after running
`./scripts/setup_car_bench.sh`.

---

## Project Structure

```text
src/
  agentbeats/                         inherited internal A2A runner helpers
  evaluator/                          CAR-bench evaluator A2A server
  track_1_agent_under_test/           Track 1 minimal template
  track_2_agent_under_test_codex/     Track 2 direct Codex JSON agent
  track_2_agent_under_test_codex_planner/
                                      Track 2 planner/executor agent
  track_2_agent_under_test_codex_python/
                                      Track 2 Python-call DSL agent
  tool_call_types.py                  shared tool-call data models
  turn_metrics.py                     shared metadata keys

scenarios/
  track_1_agent_under_test/
  track_2_agent_under_test_codex/
  track_2_agent_under_test_codex_planner/
  track_2_agent_under_test_codex_python/

docs/
  development-guide.md                detailed A2A turn contract
  agent-under-test-harnessing.md      allowed harness boundaries
  codex-harness-patterns.md           Track 2 model/harness patterns
```

---

## Read More

Use this reading path when building your own agent:

1. **Pick a starter**
   - Track 1: [`src/track_1_agent_under_test/README.md`](src/track_1_agent_under_test/README.md)
   - Track 2 direct Codex: [`src/track_2_agent_under_test_codex/README.md`](src/track_2_agent_under_test_codex/README.md)
   - Track 2 planner/executor: [`src/track_2_agent_under_test_codex_planner/README.md`](src/track_2_agent_under_test_codex_planner/README.md)
   - Track 2 Python-call DSL: [`src/track_2_agent_under_test_codex_python/README.md`](src/track_2_agent_under_test_codex_python/README.md)
2. **Understand the turn contract**
   - [`docs/development-guide.md`](docs/development-guide.md)
3. **Design a more sophisticated harness**
   - [`docs/agent-under-test-harnessing.md`](docs/agent-under-test-harnessing.md)
   - [`docs/codex-harness-patterns.md`](docs/codex-harness-patterns.md)
4. **Protocol background**
   - [`docs/a2a-introduction.md`](docs/a2a-introduction.md)

Rules of thumb:

- Preserve the A2A boundary.
- Maintain conversation state per `context_id`.
- Return benchmark-visible text and tool calls in message parts, not hidden metadata.
- Let the evaluator execute CAR-bench tools.
- Do not inspect hidden task/evaluator state or add private vehicle-capability tools.

---

## Citation

If you use CAR-bench in your research, please cite:

```bibtex
@misc{kirmayr2026carbenchevaluatingconsistencylimitawareness,
      title={CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty},
      author={Johannes Kirmayr and Lukas Stappen and Elisabeth Andre},
      year={2026},
      eprint={2601.22027},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.22027},
}
```

---

## Important Links

- Original CAR-bench: [github.com/CAR-bench/car-bench](https://github.com/CAR-bench/car-bench)
- Competition website: [car-bench.github.io/car-bench](https://car-bench.github.io/car-bench/)
- Paper: [arxiv.org/abs/2601.22027](https://arxiv.org/abs/2601.22027)
- A2A Protocol: [a2a-protocol.org](https://a2a-protocol.org)
