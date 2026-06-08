"""
CAR-bench evaluator that runs CAR-bench evaluation on an agent under test.

This agent:
1. Sets up CAR-bench voice assistant environments
2. Sends task prompts to the agent under test
(wrapped in a RemoteA2AAgent that communicates via A2A protocol)
3. Parses the agent under test's tool-call responses
4. Steps through the environment and collects metrics
"""
import argparse
import asyncio
import functools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import nest_asyncio
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.helpers.proto_helpers import (
    new_text_part,
    new_data_part,
    new_text_message,
)
from google.protobuf.json_format import MessageToDict

from agentbeats.evaluator_executor import EvaluatorAgent
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from turn_metrics import (
    TURN_METRICS_KEY, SOURCE_KEY, SOURCE_USER, SOURCE_ENVIRONMENT,
    extract_turn_metrics, AVG_LLM_CALL_TIME_MS, NUM_LLM_CALLS, COST,
)
try:
    from car_bench_paths import CAR_BENCH_REPO
except ImportError:
    from .car_bench_paths import CAR_BENCH_REPO
sys.path.pop(0)

# Import run.py from car-bench repo root
sys.path.insert(0, str(CAR_BENCH_REPO))
from run import run as run_benchmark
sys.path.pop(0)

# Import from car_bench package
from car_bench.types import Action, EnvRunResult

nest_asyncio.apply()
logger = configure_logger(role="evaluator", context="-")

RESPOND_ACTION_NAME = "respond"


def _progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    completed = max(0, min(completed, total))
    filled = width if completed >= total else int(width * completed / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def format_progress_update(progress: Dict[str, Any]) -> str:
    """Format benchmark progress for the CLI status stream."""
    event = str(progress.get("event") or "progress")
    task_type = str(progress.get("task_type") or "task")
    task_split = str(progress.get("task_split") or "split")
    total_tasks = int(progress.get("total_tasks") or 0)
    total_trials = int(progress.get("total_trials") or 0)
    total_runs = int(progress.get("total_runs") or 0)
    completed_runs = int(progress.get("completed_runs") or 0)
    remaining_runs = int(progress.get("remaining_runs") or max(total_runs - completed_runs, 0))

    label_by_event = {
        "split_start": "start",
        "trial_start": "trial start",
        "task_start": "task start",
        "task_done": "task done",
        "trial_done": "trial done",
        "split_done": "done",
    }
    label = label_by_event.get(event, event.replace("_", " "))

    if event in {"split_start", "trial_start"}:
        details = [label]
        if total_tasks and total_trials:
            details.append(f"{total_tasks} tasks x {total_trials} trials")
        trial_number = progress.get("trial_number")
        if trial_number and total_trials:
            details.append(f"trial {trial_number}/{total_trials}")
        return f"[Progress] {task_type}/{task_split} | " + ", ".join(details)

    if event == "task_start" and total_runs:
        current_run = min(completed_runs + 1, total_runs)
        header = (
            f"[Progress] {task_type}/{task_split} "
            f"{_progress_bar(completed_runs, total_runs)} "
            f"running task-run {current_run}/{total_runs}, "
            f"{completed_runs} completed, {remaining_runs} remaining"
        )
    else:
        header = (
            f"[Progress] {task_type}/{task_split} "
            f"{_progress_bar(completed_runs, total_runs)} "
            f"{completed_runs}/{total_runs} task-runs completed, {remaining_runs} remaining"
        )

    task_id = progress.get("task_id")
    task_position = progress.get("task_position")
    trial_number = progress.get("trial_number")
    details = [label]
    if task_id:
        if task_position and total_tasks:
            task_label = f"task {task_position}/{total_tasks} {task_id}"
        else:
            task_label = f"task {task_id}"
        details.append(task_label)
    elif total_tasks and total_trials:
        details.append(f"{total_tasks} tasks x {total_trials} trials")
    if trial_number and total_trials:
        details.append(f"trial {trial_number}/{total_trials}")
    if "reward" in progress:
        try:
            details.append(f"reward {float(progress['reward']):.2f}")
        except (TypeError, ValueError):
            details.append(f"reward {progress['reward']}")

    return header + " | " + ", ".join(details)


def create_remote_agent_factory(agent_url: str):
    """Create a factory that produces RemoteA2AAgent instances.

    Each agent gets its own ToolProvider to avoid threading issues.
    """
    def factory(tools_info, wiki, args):
        # Import Agent base class and types
        from car_bench.agents.base import Agent
        from car_bench.types import AgentState

        # Create an agent that delegates to the remote agent under test via A2A.
        class RemoteA2AAgent(Agent):
            def __init__(self, agent_url: str):
                self.agent_url = agent_url
                self.tool_provider = ToolProvider()
                self._is_first_message = True

            def get_init_state(self, system_prompt: str, initial_observation: str) -> AgentState:
                """Initialize agent state with system prompt and initial observation."""
                self._is_first_message = True
                return AgentState(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_observation},
                    ]
                )

            def generate_next_message(self, state: AgentState, tools_info: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], AgentState]:
                """Generate next message by calling remote agent under test."""
                import asyncio

                # Collect trailing tool result messages (there may be multiple from parallel tool calls)
                tool_result_messages = []
                for msg in reversed(state.messages):
                    if msg.get("role") == "tool":
                        tool_result_messages.insert(0, msg)
                    else:
                        break

                # Extract last user/tool message content
                last_user_msg = state.messages[-1]["content"]

                # Handle empty messages - replace with placeholder to avoid LLM errors
                if not last_user_msg or not last_user_msg.strip():
                    logger.warning(
                        "Empty user message detected, using placeholder 'none'",
                        message_index=len(state.messages) - 1
                    )
                    last_user_msg = "none"

                # Build proper A2A message with Parts (protobuf)
                if self._is_first_message:
                    # First message: combine system prompt and user message in one text Part,
                    # send tools as separate data Part
                    parts = []

                    # Combine system prompt and user message into single text Part
                    system_prompt = state.messages[0]["content"] if state.messages[0]["role"] == "system" else ""
                    prompt_text = f"System: {system_prompt}\n\nUser: {last_user_msg}" if system_prompt else last_user_msg
                    parts.append(new_text_part(prompt_text))

                    # Add tools as data Part (structured data)
                    if tools_info:
                        parts.append(new_data_part({"tools": tools_info}))

                    source_tag = SOURCE_USER
                elif len(tool_result_messages) > 0:
                    # Tool result turn: send individual results as structured data Part
                    # so the agent under test can match each result to its tool_call_id
                    tool_results_data = [
                        {
                            "tool_name": msg.get("name", ""),
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                        for msg in tool_result_messages
                    ]
                    parts = [new_data_part({"tool_results": tool_results_data})]
                    source_tag = SOURCE_ENVIRONMENT
                else:
                    # Regular user message
                    parts = [new_text_part(last_user_msg)]
                    source_tag = SOURCE_USER

                outbound_metadata = {SOURCE_KEY: source_tag}

                # Call remote agent via A2A
                # Use synchronous call since we're in a thread pool executor
                is_new_conversation = self._is_first_message
                self._is_first_message = False

                # Build content preview for outbound log
                if source_tag == SOURCE_USER:
                    outbound_preview = last_user_msg[:120] if last_user_msg else ""
                else:
                    # Environment: show raw tool results as compact JSON
                    tool_summaries = []
                    for msg in tool_result_messages:
                        name = msg.get("name", "?")
                        tool_summaries.append({"tool": name, "result": msg.get("content", "")})
                    outbound_preview = json.dumps(tool_summaries, separators=(",", ":"))

                msg_logger = logger.bind(role=source_tag, context="-")
                msg_logger.debug(
                    "Sending to agent",
                    content_preview=outbound_preview,
                    new_conversation=is_new_conversation,
                )

                # Use synchronous method to avoid event loop issues in thread pool
                turn_start = time.perf_counter()
                response = self.tool_provider.talk_to_agent_with_parts_sync(
                    parts=parts,
                    url=self.agent_url,
                    new_conversation=is_new_conversation,
                    metadata=outbound_metadata,
                )
                turn_time_ms = (time.perf_counter() - turn_start) * 1000.0

                msg_logger.debug(
                    "Received response",
                    turn_time_ms=round(turn_time_ms, 1),
                )

                # Parse response into standard message format
                next_message = self._parse_response(response)

                # Extract turn_metrics from response metadata (only on final responses)
                response_metadata = getattr(response, "metadata", None)
                turn_metrics = extract_turn_metrics(response_metadata)

                # Add evaluator-measured turn_time_ms
                turn_metrics["turn_time_ms"] = round(turn_time_ms, 1)

                # Attach metrics to the message if this is a final response (no tool calls)
                if not next_message.get("tool_calls") and turn_metrics.get(NUM_LLM_CALLS, 0) > 0:
                    next_message["turn_metrics"] = turn_metrics

                # Update AgentState cost/latency totals
                additional_cost = turn_metrics.get(COST, 0.0)
                additional_llm_latency = (
                    turn_metrics.get(AVG_LLM_CALL_TIME_MS, 0.0) * turn_metrics.get(NUM_LLM_CALLS, 0)
                )

                # Update state
                updated_state = AgentState(
                    messages=state.messages + [next_message],
                    total_cost=state.total_cost + additional_cost,
                    total_llm_induced_latency_ms=state.total_llm_induced_latency_ms + additional_llm_latency,
                    turn_counter=state.turn_counter,
                    least_prompt_tokens=state.least_prompt_tokens,
                    latest_prompt_tokens=turn_metrics.get("prompt_tokens", state.latest_prompt_tokens),
                )

                return next_message, updated_state

            def _parse_response(self, response) -> Dict[str, Any]:
                """Parse the A2A Message response into standard agent message format.

                Handles both protobuf Message (v1.0) and Pydantic Message (v0.3 compat) formats.
                """
                try:
                    content = None
                    tool_calls = None
                    reasoning_content = None

                    # Get parts from response
                    if hasattr(response, 'parts'):
                        parts = response.parts
                    else:
                        # Fallback: try parsing as JSON string
                        parsed = json.loads(response)
                        parts = parsed.get("parts", [])

                    # Process each part
                    for part in parts:
                        # Handle protobuf Part (v1.0) — has WhichOneof
                        if hasattr(part, 'WhichOneof'):
                            content_type = part.WhichOneof("content")
                            if content_type == "text":
                                content = part.text
                            elif content_type == "data":
                                data = MessageToDict(part.data)
                                if "tool_calls" in data:
                                    tool_calls = self._parse_tool_calls(data["tool_calls"])
                                elif "reasoning_content" in data:
                                    reasoning_content = data["reasoning_content"]
                        # Handle Pydantic Part (v0.3 compat) — has .root attribute
                        elif hasattr(part, 'root'):
                            if hasattr(part.root, 'text') and part.root.text is not None:
                                content = part.root.text
                            elif hasattr(part.root, 'data') and part.root.data is not None:
                                data = part.root.data
                                if "tool_calls" in data:
                                    tool_calls = self._parse_tool_calls(data["tool_calls"])
                                elif "reasoning_content" in data:
                                    reasoning_content = data["reasoning_content"]
                        # Handle dict representation
                        elif isinstance(part, dict):
                            part_data = part.get("root", part)
                            if "text" in part_data and part_data["text"]:
                                content = part_data["text"]
                            elif "data" in part_data and part_data["data"]:
                                data = part_data["data"]
                                if "tool_calls" in data:
                                    tool_calls = self._parse_tool_calls(data["tool_calls"])
                                elif "reasoning_content" in data:
                                    reasoning_content = data["reasoning_content"]

                    parsed_msg = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }

                    # Include reasoning_content for debugging if present
                    if reasoning_content:
                        parsed_msg["reasoning_content"] = reasoning_content

                    logger.debug(
                        "Parsed agent response",
                        has_tool_calls=bool(tool_calls),
                        num_tool_calls=len(tool_calls) if tool_calls else 0,
                    )

                    return parsed_msg

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse agent response: {e}")
                    # If parsing fails, treat as plain text response
                    return {
                        "role": "assistant",
                        "content": response,
                        "tool_calls": None,
                    }

            @staticmethod
            def _parse_tool_calls(tool_calls_data: list) -> list:
                """Parse tool calls from structured data into LLM format."""
                return [
                    {
                        "id": f"call_{hash(json.dumps(tc)) % 100000000:08x}",
                        "type": "function",
                        "function": {
                            "name": tc.get("tool_name", tc.get("toolName", "")),
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in tool_calls_data
                ]

        return RemoteA2AAgent(agent_url=agent_url)

    return factory


def calculate_evaluation_results(
    results_by_split: Dict[str, List[EnvRunResult]],
    time_used: float,
    *,
    include_detailed_results: bool = False,
    include_trajectories: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Calculate comprehensive evaluation results and format summary.

    Args:
        results_by_split: Results organized by task split (base, hallucination, disambiguation)
        time_used: Total evaluation time in seconds

    Returns:
        Tuple of (result_data dict, summary string)
    """
    # Import analysis functions from car-bench repo root
    sys.path.insert(0, str(CAR_BENCH_REPO))
    try:
        from analyze_results_v2 import (
            organize_data_by_task_and_trial,
            calculate_pass_power_k_scores,
            calculate_pass_at_k_scores,
        )
    finally:
        sys.path.pop(0)

    # Flatten all results
    all_results = [r for results in results_by_split.values() for r in results]
    total_reward = sum(r.reward for r in all_results)
    num_completed = len(all_results)
    pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0

    # Split task rewards by task type
    task_rewards_by_split = {
        split: {str(r.task_id): r.reward for r in results}
        for split, results in results_by_split.items()
        if results
    }

    # Calculate metrics for each split separately
    pass_power_k_scores_by_split = {}
    pass_at_k_scores_by_split = {}
    max_trials = 1

    for split, results in results_by_split.items():
        if not results:
            continue

        # Convert results to format expected by analyze_results.py
        analysis_data = [
            {
                "task_id": result.task_id,
                "reward": result.reward,
                "info": result.info,
                "trial": result.trial,
            }
            for result in results
        ]

        # Organize data and calculate metrics for this split
        organized_data = organize_data_by_task_and_trial(analysis_data)
        split_max_trials = (
            max(len(trials) for trials in organized_data.values())
            if organized_data else 1
        )
        max_trials = max(max_trials, split_max_trials)

        pass_power_k_scores_by_split[split] = calculate_pass_power_k_scores(organized_data, split_max_trials)
        pass_at_k_scores_by_split[split] = calculate_pass_at_k_scores(organized_data, split_max_trials)

    # Calculate overall metrics as average across splits
    pass_power_k_scores, pass_at_k_scores = calculate_average_metrics_across_splits(
        pass_power_k_scores_by_split,
        pass_at_k_scores_by_split,
        max_trials
    )
    weighted_pass_power_k_scores, weighted_pass_at_k_scores = calculate_weighted_metrics(
        all_results,
        max_trials,
        organize_data_by_task_and_trial,
        calculate_pass_power_k_scores,
        calculate_pass_at_k_scores,
    )
    pass_summary = build_pass_summary(
        pass_power_k_scores=pass_power_k_scores,
        pass_power_k_scores_by_split=pass_power_k_scores_by_split,
    )
    weighted_pass_summary = build_pass_summary(
        pass_power_k_scores=weighted_pass_power_k_scores,
        pass_power_k_scores_by_split=pass_power_k_scores_by_split,
    )
    task_pass3_summary = build_task_pass3_summary(results_by_split)

    # Keep A2A artifacts compact by default. Full trajectories remain in the
    # checkpoint files and can be enabled for smaller debugging runs.
    results_by_split_compact = {}

    for split, results in results_by_split.items():
        if not results:
            continue

        split_records = []
        for result in results:
            record = {
                "task_id": result.task_id,
                "reward": result.reward,
                "trial": result.trial,
                "reward_info": result.info.get("reward_info", {}),
                "error": result.info.get("error", None),
                "traceback": result.info.get("traceback", None),
                "user_cost": result.info.get("user_cost", 0),
                "total_agent_cost": result.info.get("total_agent_cost", 0),
                "total_llm_latency_ms": result.info.get("total_llm_induced_latency_ms", 0),
            }
            if include_detailed_results:
                record["task"] = result.info.get("task", {})
            if include_trajectories:
                record["trajectory"] = [
                    msg for msg in result.traj
                    if msg.get("role") != "system"
                ]
            split_records.append(record)
        results_by_split_compact[split] = split_records

    # Format task results for display by split
    task_results_by_split_str = []
    for split in ["base", "hallucination", "disambiguation"]:
        if split in results_by_split and results_by_split[split]:
            results = results_by_split[split]
            split_results = "\n".join(
                f"    Task {r.task_id}: {'✓' if r.reward >= 0.99 else '✗'} ({r.reward:.2f})"
                for r in results
            )
            split_reward = sum(r.reward for r in results)
            split_count = len(results)
            split_pass_rate = (split_reward / split_count * 100) if split_count > 0 else 0
            task_results_by_split_str.append(
                f"  {split.capitalize()}: {split_pass_rate:.1f}% ({split_reward:.1f}/{split_count})\n{split_results}"
            )

    task_results_str = "\n\n".join(task_results_by_split_str)

    # Format Pass^k and Pass@k scores
    pass_scores_str = [
        f"  Pass^{k}: {pass_power_k_scores.get(f'Pass^{k}', 0) * 100:.1f}%  |  Pass@{k}: {pass_at_k_scores.get(f'Pass@{k}', 0) * 100:.1f}%"
        for k in range(1, max_trials + 1)
    ]
    pass_scores_display = "\n".join(pass_scores_str)

    # Build result data
    result_data = {
        "summary": {
            "pass_rate": pass_rate,
            "score": total_reward,
            "max_score": num_completed,
            "pass_summary": pass_summary,
            "weighted_pass_summary": weighted_pass_summary,
            "task_pass3_summary": task_pass3_summary,
        },
        "score": total_reward,
        "max_score": num_completed,
        "pass_rate": pass_rate,
        "task_rewards_by_split": task_rewards_by_split,
        "time_used": time_used,
        "pass_power_k_scores": pass_power_k_scores,
        "pass_at_k_scores": pass_at_k_scores,
        "weighted_pass_power_k_scores": weighted_pass_power_k_scores,
        "weighted_pass_at_k_scores": weighted_pass_at_k_scores,
        "pass_power_k_scores_by_split": pass_power_k_scores_by_split,
        "pass_at_k_scores_by_split": pass_at_k_scores_by_split,
        "pass_summary": pass_summary,
        "weighted_pass_summary": weighted_pass_summary,
        "task_pass3_summary": task_pass3_summary,
        "max_trials": max_trials,
        "results_by_split": results_by_split_compact,
    }

    task_pass3_display = format_task_pass3_summary(task_pass3_summary)

    # Build summary string
    summary = f"""CAR-bench Results
Tasks: {num_completed}
Overall Pass Rate: {pass_rate:.1f}% ({total_reward:.1f}/{num_completed})
Time: {time_used:.1f}s

Task-level Pass^3:
{task_pass3_display}

Pass Scores:
{pass_scores_display}"""

    return result_data, summary


def build_task_pass3_summary(
    results_by_split: Dict[str, List[EnvRunResult]],
) -> Dict[str, Any]:
    """Build strict per-task and total Pass^3 details."""

    task_records: dict[str, dict[str, Any]] = {}
    for split, results in results_by_split.items():
        for result in results:
            task_id = str(result.task_id or result.task_index)
            record = task_records.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "split": split,
                    "trials": {},
                    "pass^3": None,
                    "num_trials": 0,
                },
            )
            record["trials"][str(result.trial)] = result.reward

    by_split: dict[str, dict[str, Any]] = {}
    passed_tasks = 0
    eligible_tasks = 0
    incomplete_tasks = 0

    for record in task_records.values():
        trials = record["trials"]
        required_keys = ["0", "1", "2"]
        has_three = all(key in trials for key in required_keys)
        record["num_trials"] = len(trials)
        if has_three:
            eligible_tasks += 1
            passed = all(_is_full_reward(trials[key]) for key in required_keys)
            record["pass^3"] = passed
            if passed:
                passed_tasks += 1
        else:
            incomplete_tasks += 1
            record["pass^3"] = None

        split = record["split"]
        split_summary = by_split.setdefault(
            split,
            {
                "total_tasks": 0,
                "eligible_tasks": 0,
                "passed_tasks": 0,
                "incomplete_tasks": 0,
                "pass^3": None,
            },
        )
        split_summary["total_tasks"] += 1
        if record["pass^3"] is None:
            split_summary["incomplete_tasks"] += 1
        else:
            split_summary["eligible_tasks"] += 1
            if record["pass^3"]:
                split_summary["passed_tasks"] += 1

    for split_summary in by_split.values():
        eligible = split_summary["eligible_tasks"]
        split_summary["pass^3"] = (
            split_summary["passed_tasks"] / eligible
            if eligible else None
        )

    return {
        "overall": {
            "total_tasks": len(task_records),
            "eligible_tasks": eligible_tasks,
            "passed_tasks": passed_tasks,
            "incomplete_tasks": incomplete_tasks,
            "pass^3": passed_tasks / eligible_tasks if eligible_tasks else None,
        },
        "by_split": by_split,
        "by_task": {
            task_id: task_records[task_id]
            for task_id in sorted(task_records)
        },
    }


def format_task_pass3_summary(summary: Dict[str, Any]) -> str:
    """Format strict Pass^3 as total plus compact per-task status."""

    overall = summary.get("overall", {})
    lines = [
        "  Overall: "
        f"{overall.get('passed_tasks', 0)}/{overall.get('eligible_tasks', 0)} "
        f"tasks passed, Pass^3 {_format_score(overall.get('pass^3'))}"
    ]
    incomplete = overall.get("incomplete_tasks", 0)
    if incomplete:
        lines[0] += f", incomplete {incomplete}"

    for split in ["base", "hallucination", "disambiguation"]:
        split_summary = summary.get("by_split", {}).get(split)
        if not split_summary:
            continue
        lines.append(
            f"  {split.capitalize()}: "
            f"{split_summary.get('passed_tasks', 0)}/{split_summary.get('eligible_tasks', 0)} "
            f"tasks passed, Pass^3 {_format_score(split_summary.get('pass^3'))}"
        )

    task_parts = []
    for task_id, record in summary.get("by_task", {}).items():
        value = record.get("pass^3")
        if value is True:
            status = "PASS"
        elif value is False:
            status = "FAIL"
        else:
            status = "INCOMPLETE"
        task_parts.append(f"{task_id}:{status}")

    if task_parts:
        lines.append("  Tasks: " + ", ".join(task_parts))
    return "\n".join(lines)


def _is_full_reward(value: Any) -> bool:
    try:
        return abs(float(value) - 1.0) <= 1e-6
    except (TypeError, ValueError):
        return False


def build_pass_summary(
    *,
    pass_power_k_scores: Dict[str, float],
    pass_power_k_scores_by_split: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Build explicit Pass^1/Pass^3 summary for JSON artifacts."""

    return {
        "overall": {
            "pass^1": _score_or_none(pass_power_k_scores, "Pass^1"),
            "pass^3": _score_or_none(pass_power_k_scores, "Pass^3"),
        },
        "by_split": {
            split: {
                "pass^1": _score_or_none(scores, "Pass^1"),
                "pass^3": _score_or_none(scores, "Pass^3"),
            }
            for split, scores in pass_power_k_scores_by_split.items()
        },
    }


def format_pass_summary(pass_summary: Dict[str, Any]) -> str:
    """Format overall and per-split Pass^1/Pass^3 metrics for display."""

    lines = [
        "  Overall: "
        f"Pass^1 {_format_score(pass_summary['overall'].get('pass^1'))}, "
        f"Pass^3 {_format_score(pass_summary['overall'].get('pass^3'))}"
    ]
    for split in ["base", "hallucination", "disambiguation"]:
        split_scores = pass_summary.get("by_split", {}).get(split)
        if not split_scores:
            continue
        lines.append(
            f"  {split.capitalize()}: "
            f"Pass^1 {_format_score(split_scores.get('pass^1'))}, "
            f"Pass^3 {_format_score(split_scores.get('pass^3'))}"
        )
    return "\n".join(lines)


def _score_or_none(scores: Dict[str, float], key: str) -> Optional[float]:
    return scores[key] if key in scores else None


def _format_score(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    return f"{score * 100:.1f}%"


def calculate_average_metrics_across_splits(
    pass_power_k_scores_by_split: Dict[str, Dict[str, float]],
    pass_at_k_scores_by_split: Dict[str, Dict[str, float]],
    max_trials: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate average metrics across splits (not weighted by task count).

    Returns:
        Tuple of (pass_power_k_scores, pass_at_k_scores)
    """
    num_splits = len(pass_power_k_scores_by_split)
    if num_splits == 0:
        return {}, {}

    # Average Pass^k and Pass@k scores across splits
    pass_power_k_scores = {}
    pass_at_k_scores = {}

    for k in range(1, max_trials + 1):
        pass_power_key = f"Pass^{k}"
        pass_at_key = f"Pass@{k}"

        # Sum scores across splits
        pass_power_sum = sum(
            scores.get(pass_power_key, 0.0)
            for scores in pass_power_k_scores_by_split.values()
            if pass_power_key in scores
        )
        pass_at_sum = sum(
            scores.get(pass_at_key, 0.0)
            for scores in pass_at_k_scores_by_split.values()
            if pass_at_key in scores
        )

        pass_power_k_scores[pass_power_key] = pass_power_sum / num_splits
        pass_at_k_scores[pass_at_key] = pass_at_sum / num_splits

    return pass_power_k_scores, pass_at_k_scores


def calculate_weighted_metrics(
    all_results: List[EnvRunResult],
    max_trials: int,
    organize_data_by_task_and_trial,
    calculate_pass_power_k_scores,
    calculate_pass_at_k_scores,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate total Pass^k/Pass@k over all task IDs as one weighted pool."""

    if not all_results:
        return {}, {}

    analysis_data = [
        {
            "task_id": result.task_id or str(result.task_index),
            "reward": result.reward,
            "info": result.info,
            "trial": result.trial,
        }
        for result in all_results
    ]
    organized_data = organize_data_by_task_and_trial(analysis_data)
    weighted_max_trials = (
        max(len(trials) for trials in organized_data.values())
        if organized_data else max_trials
    )
    metric_trials = max(max_trials, weighted_max_trials)
    return (
        calculate_pass_power_k_scores(organized_data, metric_trials),
        calculate_pass_at_k_scores(organized_data, metric_trials),
    )


def build_args_from_config(config: dict, task_type: str) -> argparse.Namespace:
    """Convert evaluation config to run() arguments for a specific task type."""
    return argparse.Namespace(
        env="car_voice_assistant",
        task_type=task_type,
        task_split=config.get("task_split", "test"),
        num_tasks=config.get(f"tasks_{task_type}_num_tasks", -1),
        task_id_filter=config.get(f"tasks_{task_type}_task_id_filter", None),
        num_trials=config.get("num_trials", 1),
        max_steps=config.get("max_steps", 40),
        max_concurrency=1,  # Sequential to avoid overloading agent under test
        # User simulator settings
        user_strategy="llm",
        user_model=config.get("user_model", "gemini/gemini-2.5-flash"),
        user_model_provider=config.get("user_provider", "gemini"),
        user_thinking=config.get("user_thinking", True),
        # Policy evaluator settings
        policy_evaluator_strategy="llm",
        policy_evaluator_model=config.get("policy_evaluator_model", "gemini/gemini-2.5-flash"),
        policy_evaluator_model_provider=config.get("policy_evaluator_provider", "gemini"),
        evaluate_policy=True,
        score_tool_execution_errors=True,
        score_policy_errors=True,
        # Agent settings (NOT USED for custom agent factory, but required by some code paths)
        agent_strategy="tool-calling",  # Default strategy if factory not used
        model="remote-agent",  # Placeholder, not used for remote agents
        model_provider="a2a", # not used
        temperature=0.0, # not used
        thinking=False, # not used
        interleaved_thinking=False, # not used
        reasoning_effort="none", # not used
        # =======
        use_user_as_a_tool_tools=False,
        planning_and_thinking_tool=True,
        remove_non_standard_fields_from_tools=False,
        few_shot_displays_path=None,
        seed=10,
        shuffle=False,
    )


class CARBenchEvaluator(EvaluatorAgent):
    """Evaluator that runs CAR-bench against one agent under test."""

    def __init__(self):
        self._required_config_keys = []
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        eval_logger = logger.bind(role="evaluator", context="eval")
        eval_logger.info(
            "Starting CAR-bench evaluation",
            agent_url=str(req.agent_under_test),
            num_trials=req.config.get("num_trials", 1)
        )
        start_time = time.time()

        agent_url = str(req.agent_under_test)

        # Create agent factory
        agent_factory = create_remote_agent_factory(agent_url)

        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message("Starting evaluation of CAR-bench tasks")
        )

        all_results: List[EnvRunResult] = []
        results_by_split: Dict[str, List[EnvRunResult]] = {
            "base": [],
            "hallucination": [],
            "disambiguation": []
        }

        try:
            # Run each task type (base, hallucination, disambiguation)
            for task_type in ["base", "hallucination", "disambiguation"]:
                num_tasks_key = f"tasks_{task_type}_num_tasks"
                task_id_filter_key = f"tasks_{task_type}_task_id_filter"

                # Skip if not configured
                if num_tasks_key not in req.config and task_id_filter_key not in req.config:
                    eval_logger.info(
                        "Skipping task type (not configured)",
                        task_type=task_type
                    )
                    continue

                split_logger = logger.bind(role="evaluator", context=f"type:{task_type}")

                # Build args for this task type
                args = build_args_from_config(req.config, task_type)

                # Log task configuration
                task_desc = f"{task_type} tasks (split={args.task_split}"
                if args.task_id_filter:
                    task_desc += f", ids={args.task_id_filter}"
                elif args.num_tasks > 0:
                    task_desc += f", first {args.num_tasks} tasks"
                else:
                    task_desc += ", all tasks"
                task_desc += ")"

                split_logger.info(
                    "Starting task type evaluation",
                    task_type=task_type,
                    task_split=args.task_split,
                    num_tasks=args.num_tasks,
                    task_id_filter=args.task_id_filter,
                    num_trials=req.config.get("num_trials", 1)
                )

                await updater.update_status(
                    TaskState.TASK_STATE_WORKING,
                    new_text_message(f"Starting evaluation: {task_desc}")
                )

                # Build checkpoint path
                ckpt_path = f"/tmp/car_bench_eval_{task_type}_{args.task_split}.json"

                # Clean up any existing checkpoint file to avoid JSON parse errors
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    eval_logger.debug("Removed existing checkpoint file", path=ckpt_path)

                # Run in executor to avoid blocking async event loop.
                # Progress callbacks are emitted from the benchmark worker
                # thread, then scheduled back onto the A2A event loop.
                loop = asyncio.get_event_loop()
                last_progress_text: Dict[str, str] = {"value": ""}

                def progress_callback(progress: Dict[str, Any]) -> None:
                    progress_text = format_progress_update(progress)
                    if progress_text == last_progress_text["value"]:
                        return
                    last_progress_text["value"] = progress_text
                    split_logger.info("Benchmark progress", progress=progress_text)
                    asyncio.run_coroutine_threadsafe(
                        updater.update_status(
                            TaskState.TASK_STATE_WORKING,
                            new_text_message(progress_text),
                        ),
                        loop,
                    )

                benchmark_call = functools.partial(
                    run_benchmark,
                    args,
                    ckpt_path,
                    agent_factory,
                    progress_callback,
                )
                results = await loop.run_in_executor(
                    None,
                    benchmark_call,
                )

                all_results.extend(results)
                results_by_split[task_type].extend(results)

                # Log completion with summary stats
                split_reward = sum(r.reward for r in results)
                split_logger.info(
                    "Completed task type",
                    task_type=task_type,
                    num_tasks=len(results),
                    total_reward=split_reward,
                    pass_rate=f"{(split_reward / len(results) * 100) if results else 0:.1f}%"
                )

                # Emit intermediate artifact with per-split results so far
                # This allows crash recovery and live progress tracking
                intermediate_time = time.time() - start_time
                intermediate_data, intermediate_summary = calculate_evaluation_results(
                    {k: v for k, v in results_by_split.items() if v},
                    intermediate_time
                )
                await updater.add_artifact(
                    parts=[
                        new_text_part(f"[Intermediate] {intermediate_summary}"),
                        new_data_part(intermediate_data),
                    ],
                    name=f"intermediate_{task_type}",
                )

            # Calculate metrics and format results
            time_used = time.time() - start_time
            result_data, summary = calculate_evaluation_results(results_by_split, time_used)

            await updater.add_artifact(
                parts=[
                    new_text_part(summary),
                    new_data_part(result_data),
                ],
                name="Result",
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            # Let EvaluatorExecutor publish the terminal FAILED state. Marking
            # it here as well makes a2a-sdk reject the outer failure update.
            raise
