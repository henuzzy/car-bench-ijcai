# Track 2 Codex Pro / Cerebras Spark Harness Patterns

Track 2 agents use Codex Pro-backed inference, with `gpt-5.3-codex-spark` as
the practical fast executor served on Cerebras infrastructure. The reference
Codex agent under test is intentionally a small next-action adapter, but Track 2
participants can replace the internal harness as long as the submitted A2A
agent stays within the official time budget.

## Reference Agent Map

| Agent | Package | Local Scenario | Internal Strategy |
|-------|---------|----------------|-------------------|
| Codex JSON agent | [`src/track_2_agent_under_test_codex/`](../src/track_2_agent_under_test_codex/) | [`scenarios/track_2_agent_under_test_codex/local_smoke.toml`](../scenarios/track_2_agent_under_test_codex/local_smoke.toml) | Spark returns schema-constrained next-action JSON. |
| Codex planner/executor | [`src/track_2_agent_under_test_codex_planner/`](../src/track_2_agent_under_test_codex_planner/) | [`scenarios/track_2_agent_under_test_codex_planner/local_smoke.toml`](../scenarios/track_2_agent_under_test_codex_planner/local_smoke.toml) | Larger planner runs once after a user message; Spark executor reuses the private plan across tool-result turns. |
| Codex Python-call DSL | [`src/track_2_agent_under_test_codex_python/`](../src/track_2_agent_under_test_codex_python/) | [`scenarios/track_2_agent_under_test_codex_python/local_smoke.toml`](../scenarios/track_2_agent_under_test_codex_python/local_smoke.toml) | Spark emits a fenced Python-call action block that is parsed, never executed. |

## Model Selection

Use `CODEX_MODEL` for the default model used by the Codex agent:

```env
CODEX_MODEL=gpt-5.3-codex-spark
CODEX_REASONING_EFFORT=medium
```

`gpt-5.3-codex-spark` is the recommended practical default because Track 2 has
a time budget and Spark is substantially faster. It is not the only Codex model
that can appear inside a harness. Participants can use larger Codex models for
selected planner, verifier, or condenser steps if the total agent still fits the
official time budget, which will be announced before official evaluation.

Ways to change the model:

- Local run: edit `CODEX_MODEL` in `.env`.
- Docker local build: edit `CODEX_MODEL` in `.env`; `scenarios/track_2_agent_under_test_codex/local_docker_smoke.toml`
  forwards it into the container.
- Scenario-specific local run: add `--model <model-id>` to the participant
  command in `scenarios/track_2_agent_under_test_codex/local_smoke.toml`.
- Code-level advanced harness: pass `model=` to `CodexAppServerClient.generate`
  for individual internal calls.

Keep model identifiers in the format accepted by your installed Codex CLI.

## App-Server Stability For The Competition

`codex app-server` is useful here because it keeps a warm Codex runtime behind a
small JSON-RPC client, but the CLI labels the command experimental. Treat it as
a pinned runtime dependency, not as a floating platform API.

The reference agent is deliberately conservative:

- It does not opt in to the app-server experimental API surface.
- It uses only a small stable subset: initialize, `thread/start`, `turn/start`,
  item notifications, and `turn/completed`.
- It keeps protocol handling behind `src/track_2_agent_under_test_codex/codex_client.py`.
- Docker builds pin `@openai/codex@0.130.0` by default.
- `CODEX_APP_SERVER_CMD` lets participants point at a specific local Codex
  binary if they need to reproduce an exact run.

For a three-month benchmark window, the safest operating model is:

1. Publish GHCR images built with the pinned Codex CLI using an explicit manual
   build/push or the disabled opt-in workflow template.
2. Record the Codex CLI version in run logs. The current reference pin is
   `codex-cli 0.130.0`.
3. Before accepting any CLI upgrade, run the smoke scenarios for the direct,
   planner/executor, and Python-call agents.
4. Generate the app-server schema from the candidate CLI and verify the fields
   used by `codex_client.py` still exist.
5. Keep a fallback direct-API or `codex exec` adapter behind the same A2A
   renderer if app-server changes unexpectedly.

## Pattern 1: Spark Next-Action Baseline

This is the current implementation. Each CAR-bench assistant step becomes one
Codex turn:

```text
A2A input from the evaluator
  -> build transcript and task-filtered tool prompt
  -> Codex Spark next-action JSON
  -> parse JSON
  -> return text Part or data Part(tool_calls) to the evaluator
```

This is the lowest-latency and easiest-to-debug pattern. It is the best first
target before trying multi-pass harnesses.

### Conversation State And Prompt Caching

The reference agents keep the Codex app-server process warm, but they do not
reuse a hidden Codex thread as benchmark conversation memory. Each Codex turn is
an ephemeral app-server thread that receives the complete CAR-bench transcript
and task-filtered tools. This follows the OpenAI Responses API pattern of
manual conversation-state management: model requests are stateless unless you
send prior messages or use an explicit conversation/previous-response handle.

For CAR-bench, manual state is the safer default because the evaluator
trajectory is the source of truth. It keeps malformed-output retries, tool
observations, and hallucination scoring visible in the prompt rather than split
between A2A history and hidden model-side state. Participants may experiment
with persistent thread state, but should verify that it does not duplicate
messages, hide observations, or make trajectories harder to reproduce.

The prompt layout is cache-friendly: stable task rules, tool definitions, and
schemas appear before the dynamic conversation transcript. OpenAI prompt
caching works on matching prompt prefixes, so keeping static content first can
reduce latency without giving up explicit CAR-bench state.

References:

- [OpenAI conversation state guide](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses)
- [OpenAI prompt caching guide](https://platform.openai.com/docs/guides/prompt-caching)

## Pattern 2: Planner Plus Spark Executor

Use a larger model only to write a compact plan, then let Spark produce the final
benchmark action. The plan must be internal. The evaluator should only receive the final
text Part or tool-call data Part.

The reference planner runs once after a user message. If the executor returns
tool calls, the evaluator executes them and sends tool observations back; those
continuation turns reuse the active private plan and call only the Spark
executor. The planner is cleared when the executor finally returns a user-facing
response. The loop is:

```text
user message -> planner -> executor -> tool call
tool result  -> executor -> tool call
tool result  -> executor -> response
next user    -> planner -> executor
```

This keeps the expensive planner off tool-result turns and forces the private
plan to be useful across multiple executor steps. If the executor needs extra
reasoning and CAR-bench exposes `planning_tool`, it can still return
`planning_tool` as a normal benchmark-visible tool call.

This repository includes a working reference implementation in
`src/track_2_agent_under_test_codex_planner/`. It uses a private
`planning_tool`-shaped JSON object because CAR-bench already defines a
`planning_tool` for multi-step reasoning. The reference implementation does not
send that private plan to the evaluator for execution. Participants may replace this
with their own private plan tool, planning mode, planner sub-agent, or framework
primitive as long as it only uses benchmark-visible inputs.

Run it locally with:

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_codex_planner/local_smoke.toml --show-logs
```

```python
PLANNER_MODEL = "gpt-5.5"
EXECUTOR_MODEL = "gpt-5.3-codex-spark"


PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["planning_tool", "notes", "risk_flags"],
    "properties": {
        "planning_tool": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command", "plan_id", "title", "steps"],
            "properties": {
                "command": {"type": "string", "enum": ["create"]},
                "plan_id": {"type": "string"},
                "title": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["step_description", "step_dependent_on"],
                        "properties": {
                            "step_description": {"type": "string"},
                            "step_dependent_on": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        },
        "notes": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
}


def next_action_with_planner(client, context_state, transcript_prompt, executor_prompt):
    if context_state.latest_message_role == "user":
        context_state.private_plan = client.generate(
            prompt=transcript_prompt,
            output_schema=PLAN_SCHEMA,
            developer_instructions=(
                "Make a short private planning_tool-shaped plan for the "
                "current user request. It must be useful across later tool "
                "observation turns. Do not execute tools. Do not answer the user."
            ),
            model=PLANNER_MODEL,
            reasoning_effort="medium",
        ).text

    final_prompt = (
        executor_prompt
        + "\n\nPrivate planning_tool guidance:\n"
        + context_state.private_plan
        + "\n\nReturn only the final next-action JSON."
    )
    result = client.generate(
        prompt=final_prompt,
        output_schema=NEXT_ACTION_OUTPUT_SCHEMA,
        developer_instructions=CODEX_DEVELOPER_INSTRUCTIONS,
        model=EXECUTOR_MODEL,
        reasoning_effort="medium",
    )
    parsed = parse_next_action(result.text)
    if parsed["action"] == "respond":
        context_state.private_plan = None
    return result
```

Use this sparingly. A good version gates the planner behind risk signals such as
ambiguous user requests, confirmation-sensitive actions, removed-tool tasks, or
previous malformed outputs.

## Pattern 3: Spark Ensemble Plus Larger Condenser

Ask Spark for multiple candidate next actions, then use a larger model to select
or condense them into one final action.

```python
SPARK_MODEL = "gpt-5.3-codex-spark"
CONDENSER_MODEL = "gpt-5.5"


def next_action_with_ensemble(client, prompt):
    candidates = []
    for variant in ["strict", "policy_first", "minimal_tools"]:
        candidates.append(
            client.generate(
                prompt=prompt + f"\n\nCandidate style: {variant}",
                output_schema=NEXT_ACTION_OUTPUT_SCHEMA,
                developer_instructions=CODEX_DEVELOPER_INSTRUCTIONS,
                model=SPARK_MODEL,
                reasoning_effort="medium",
            ).text
        )

    condenser_prompt = {
        "task": "Choose one benchmark-safe next action from these candidates.",
        "candidates": candidates,
        "rules": [
            "Return only valid next-action JSON.",
            "Do not invent tools or parameters.",
            "Prefer policy-compliant actions over conversational shortcuts.",
        ],
    }
    return client.generate(
        prompt=json.dumps(condenser_prompt),
        output_schema=NEXT_ACTION_OUTPUT_SCHEMA,
        developer_instructions=CODEX_DEVELOPER_INSTRUCTIONS,
        model=CONDENSER_MODEL,
        reasoning_effort="medium",
    )
```

The current `CodexAppServerClient` serializes calls through one warm app-server
process to avoid protocol races. If you want true parallel ensemble inference,
use separate app-server clients/processes and measure the quota and latency
impact before relying on it.

## Pattern 4: Budget-Gated Hybrid

For most turns, use Spark only. Escalate to a larger planner or condenser only
when the turn looks risky:

```python
def needs_extra_reasoning(messages, tools):
    latest = messages[-1].get("content", "").lower()
    risky_words = ["confirm", "which", "where", "before", "instead"]
    return (
        any(word in latest for word in risky_words)
        or any("removed" in str(tool).lower() for tool in tools)
        or len(messages) > 6
    )


if needs_extra_reasoning(messages, tools):
    result = next_action_with_planner(client, planner_prompt, executor_prompt)
else:
    result = client.generate(
        prompt=executor_prompt,
        output_schema=NEXT_ACTION_OUTPUT_SCHEMA,
        developer_instructions=CODEX_DEVELOPER_INSTRUCTIONS,
        model="gpt-5.3-codex-spark",
        reasoning_effort="medium",
    )
```

This pattern usually gives the best tradeoff: Spark handles the common case, and
larger models are reserved for turns where policy or ambiguity failures are more
likely.

## Pattern 5: Python-Call DSL

The Python-call reference agent in `src/track_2_agent_under_test_codex_python/`
lets Spark answer in a more Codex-native chat style: optional brief private
text plus exactly one fenced Python action block:

````text
The user has supplied the target percentage, so the next action is a tool call.

```python
open_close_sunshade(percentage=50)
```
````

The adapter parses only the fenced Python block. Prose outside the block is
ignored for the benchmark trajectory; it is useful only for debugging and for
participants experimenting with harness styles. The only way to speak to the
simulated user is still a `respond(...)` call inside the block:

```python
respond("Sure, what percentage should I set it to?")
```

This is inspired by programmatic tool calling, but it is not true code
execution. Codex is not given shell, file, network, or hidden vehicle tools.
The generated Python is parsed with Python's built-in `ast` module, never
executed, and then mapped back into the normal A2A text response or
`tool_calls` data Part that the evaluator already understands.

The older structured envelope is still accepted as a parser fallback:

```json
{"python_code": "open_close_sunshade(percentage=50)"}
```

However, the default prompt no longer requests that shape because the fenced
code block is closer to how Codex naturally proposes small pieces of code.

Run it locally with:

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_codex_python/local_smoke.toml --show-logs
```

Accepted examples:

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

The parser accepts only top-level direct calls. It rejects imports, assignments,
variables, attributes, loops, conditionals, comprehensions, helper functions,
positional tool arguments, and non-literal arguments. Unknown tool names and
unknown parameters still pass through as parsed tool calls so CAR-bench can
score hallucination behavior normally.

The code block should choose either `respond(...)` or tool calls. If a model
emits a valid tool call and then a premature `respond("Done")` in the same
block, the reference parser keeps the tool call and ignores the response. This
matches the CAR-bench loop: the evaluator executes the tool, sends the observation back,
and the agent can produce the user-facing confirmation on the following turn.

## Non-Negotiable Boundary

All patterns must still return exactly one benchmark-compatible A2A response to
the evaluator for each assistant step. Internal planners, executors, ensembles,
condensers, and Python-call parsers must not execute vehicle tools, inspect
hidden CAR-bench state, add private tools, browse the network, or perform
file/shell side effects during benchmark inference.

If a participant chooses to return CAR-bench's real `planning_tool` as an A2A
tool call, the evaluator will execute and record it like any other benchmark tool call.
That can be valid, but it is no longer private planning. The provided
planner/executor agent keeps planning private and only returns the executor's
final user response or environment tool calls.
