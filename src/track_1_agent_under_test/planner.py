"""Planner/coordinator for the Track 1 multi-agent harness."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from litellm import completion

try:
    from .agent_tool import AgentTool, SubagentOutputError
    from .context_manager import ContextManager
    from .guards import PolicyGuard, SchemaGuard
    from .multi_agent_types import LLMCallMetrics, PlannerResult, SubagentProposal
    from .task_memory import TASK_TOOL_SCHEMAS, TaskMemoryStore
    from .task_guard import TaskGuard
except ImportError:
    from agent_tool import AgentTool, SubagentOutputError
    from context_manager import ContextManager
    from guards import PolicyGuard, SchemaGuard
    from multi_agent_types import LLMCallMetrics, PlannerResult, SubagentProposal
    from task_memory import TASK_TOOL_SCHEMAS, TaskMemoryStore
    from task_guard import TaskGuard


GENERIC_SUBAGENT_NAME = "PrivateSubagent"
GENERIC_SUBAGENT_DOMAIN = "general"
GENERIC_SUBAGENT_POLICY = (
    "Assist the main planner with the current CAR-bench step only. Use available "
    "tools, recent facts, and policy constraints. Do not re-plan the whole task."
)
MAIN_PLANNER_NAME = "MainPlannerAgent"
MAIN_PLANNER_POLICY = (
    "Own the full task plan, maintain task memory, and decide the single next "
    "benchmark-visible action. Use private internal tools before external actions "
    "when task tracking or subtask analysis is needed."
)
MAIN_PLANNER_SYSTEM_PROMPT = """You are the main private planner for a CAR-bench in-car assistant.
You own the task state. You may use private internal tools, but only your final
external_action is sent to the evaluator.

Return exactly one JSON object and no extra text.

Use TaskCreate / TaskUpdate / TaskList to maintain structured task memory for
multi-step work, state-changing actions with preconditions, navigation, charging,
calendar/email, ambiguity, confirmation, or multiple user intents.
Use CreateSubagent only for the current subtask. A subagent cannot execute tools,
cannot update task memory, and cannot talk to the evaluator.

Task memory rules:
- Create tasks for work that needs three or more steps, multiple tool calls,
  planning, confirmation, disambiguation, or verification.
- Mark exactly one active step in_progress before working on it.
- Mark completed immediately after relevant tool results or user confirmations.
- Do not say a task is done while pending/in_progress tasks remain.

External action rules:
- Use only tool names and parameters from external_tools.
- Never invent route IDs, POI IDs, contact details, plug IDs, vehicle state, or
  missing tool result fields.
- For calendar, weather, navigation, and email text, express times only in
  24-hour HH:MM format. Never use AM/PM.
- If the user confirms a proposed action, execute that exact action once. Do not
  ask for the same confirmation again, and do not switch to unrelated vehicle
  controls.
- If a relevant external tool is present, do not claim the service or capability
  is unavailable. Re-check external_tools and gather missing IDs/results first.
- Before state-changing actions, satisfy required information-gathering and policy
  preconditions.
- If a required tool, parameter, or result field is unavailable, respond with a
  concise limitation instead of fabricating.

Output schema:
{
  "summary": "brief private summary",
  "internal_tool_calls": [
    {"name": "TaskCreate | TaskUpdate | TaskList | CreateSubagent", "arguments": {}}
  ],
  "external_action": {
    "action": "tool_calls",
    "tool_calls": [{"tool_name": "exact_external_tool_name", "arguments": {}}]
  }
}
For a spoken response, use:
{"external_action": {"action": "respond", "content": "short spoken sentence"}}
If you use internal_tool_calls, external_action must be null."""

CREATE_SUBAGENT_TOOL_SCHEMA = {
    "name": "CreateSubagent",
    "description": (
        "Create one private subagent for a current subtask. The subagent returns "
        "analysis and a recommendation to the main planner only."
    ),
    "parameters": {
        "type": "object",
        "required": ["subtask"],
        "properties": {
            "subtask": {"type": "string"},
            "domain": {"type": "string"},
            "required_output": {"type": "string"},
            "relevant_tool_names": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
}


class Track1Planner:
    """Coordinates private planner/subagent calls and returns one benchmark-visible action."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        interleaved_thinking: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.interleaved_thinking = interleaved_thinking
        self.context_manager = ContextManager()
        self.task_memory = TaskMemoryStore()
        self.agent_tool = AgentTool(
            model=model,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            interleaved_thinking=interleaved_thinking,
        )
        self.schema_guard = SchemaGuard()
        self.policy_guard = PolicyGuard()
        self.task_guard = TaskGuard()
        self.timeout_seconds = float(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "75"))
        self.num_retries = int(os.getenv("AGENT_LLM_NUM_RETRIES", "1"))
        self.max_internal_steps = int(os.getenv("AGENT_MAX_INTERNAL_STEPS", "4"))

    def reset(self, context_id: str) -> None:
        self.context_manager.reset(context_id)
        self.task_memory.reset(context_id)

    def choose_next_action(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger,
    ) -> PlannerResult:
        metrics = LLMCallMetrics()
        self.context_manager.observe_messages(context_id, messages)
        self.task_memory.observe_messages(context_id, messages)

        debug: dict[str, Any] = {
            "planner": MAIN_PLANNER_NAME,
            "repair_used": False,
            "native_fallback_used": False,
        }

        stop_decision = self.task_guard.finish_after_stop_signal(messages=messages)
        if stop_decision.action:
            return PlannerResult(
                next_action=stop_decision.action,
                metrics=metrics,
                internal_calls=0,
                debug={
                    **debug,
                    "task_guard_warnings": stop_decision.warnings,
                    "terminal_stop_signal": True,
                },
            )

        finish_decision = self.task_guard.finish_after_successful_state_change(
            messages=messages
        )
        if finish_decision.action:
            return PlannerResult(
                next_action=finish_decision.action,
                metrics=metrics,
                internal_calls=0,
                debug={
                    **debug,
                    "task_guard_warnings": finish_decision.warnings,
                    "terminal_after_state_change": True,
                },
            )

        preemptive_decision = self.task_guard.preempt(messages=messages, tools=tools)
        if preemptive_decision.action:
            validation = self.schema_guard.validate(preemptive_decision.action, tools)
            action = validation.normalized_action or preemptive_decision.action
            if validation.valid:
                return PlannerResult(
                    next_action=action,
                    metrics=metrics,
                    internal_calls=0,
                    debug={**debug, "task_guard_warnings": preemptive_decision.warnings},
                )

        internal_observations: list[dict[str, Any]] = []
        for internal_step in range(self.max_internal_steps):
            try:
                decision, call_metrics = self._run_main_planner(
                    context_id=context_id,
                    messages=messages,
                    tools=tools,
                    internal_observations=internal_observations,
                    metrics=metrics,
                    ctx_logger=ctx_logger,
                )
                metrics.add(call_metrics)
            except Exception as exc:
                ctx_logger.warning("Main planner call failed", error=str(exc))
                break

            internal_tool_calls = _normalize_internal_tool_calls(decision)
            if internal_tool_calls:
                for call in internal_tool_calls:
                    observation = self._execute_internal_tool(
                        context_id=context_id,
                        call=call,
                        messages=messages,
                        tools=tools,
                        metrics=metrics,
                        ctx_logger=ctx_logger,
                    )
                    internal_observations.append(observation)
                continue

            action = _external_action_from_decision(decision)
            if action:
                return self._finalize_visible_action(
                    action=action,
                    tools=tools,
                    messages=messages,
                    metrics=metrics,
                    debug={
                        **debug,
                        "internal_steps": internal_step + 1,
                        "task_memory": self.task_memory.snapshot(context_id),
                    },
                )

            internal_observations.append(
                {
                    "tool": "ActionVerifier",
                    "ok": False,
                    "error": "Planner returned neither internal_tool_calls nor external_action.",
                }
            )

        fallback = self._native_fallback(messages=messages, tools=tools, ctx_logger=ctx_logger)
        metrics.add(fallback.metrics)
        return PlannerResult(
            next_action=fallback.next_action,
            metrics=metrics,
            internal_calls=max(metrics.num_calls, 1),
            debug={**debug, **fallback.debug, "native_fallback_used": True},
        )

    def _run_main_planner(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        internal_observations: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        ctx_logger,
    ) -> tuple[dict[str, Any], LLMCallMetrics]:
        context_bundle = self.context_manager.build_bundle(
            context_id=context_id,
            messages=messages,
            tools=tools,
            subagent_name=MAIN_PLANNER_NAME,
            subagent_policy=MAIN_PLANNER_POLICY,
            relevant_tools=tools,
            summary_callback=lambda payload: self._summary_callback(
                payload=payload,
                metrics=metrics,
                ctx_logger=ctx_logger,
            ),
        )
        payload = {
            "task": "Choose the next private planner step or one benchmark-visible action.",
            "context": context_bundle,
            "task_memory": self.task_memory.snapshot(context_id),
            "task_reminders": self.task_memory.reminders(context_id),
            "internal_tools": [*TASK_TOOL_SCHEMAS, CREATE_SUBAGENT_TOOL_SCHEMA],
            "external_tools": tools,
            "internal_observations": internal_observations[-8:],
            "rules": [
                "If you need to create/update/list tasks, use internal_tool_calls only.",
                "If you need subtask help, use CreateSubagent only.",
                "If you can safely act now, return exactly one external_action.",
                "Do not include both internal_tool_calls and external_action.",
            ],
        }
        start = time.perf_counter()
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": MAIN_PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self.temperature,
            timeout=self.timeout_seconds,
            num_retries=self.num_retries,
            **self._reasoning_kwargs(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        call_metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        if not content and hasattr(message, "model_dump"):
            content = message.model_dump(exclude_unset=True).get("content") or ""
        decision = _extract_json_object(content)
        if not isinstance(decision, dict):
            raise ValueError("main planner output must be a JSON object")
        ctx_logger.debug(
            "Main planner call completed",
            elapsed_ms=round(elapsed_ms, 1),
            prompt_tokens=call_metrics.prompt_tokens,
            completion_tokens=call_metrics.completion_tokens,
        )
        return decision, call_metrics

    def _execute_internal_tool(
        self,
        *,
        context_id: str,
        call: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        ctx_logger,
    ) -> dict[str, Any]:
        name = str(call.get("name") or call.get("tool_name") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if name in {"TaskCreate", "TaskUpdate", "TaskList"}:
            result = self.task_memory.execute(context_id, name, arguments)
            return {"tool": name, "arguments": arguments, "result": result}
        if name == "CreateSubagent":
            result = self._execute_create_subagent(
                context_id=context_id,
                arguments=arguments,
                messages=messages,
                tools=tools,
                metrics=metrics,
                ctx_logger=ctx_logger,
            )
            return {"tool": name, "arguments": arguments, "result": result}
        return {
            "tool": name or "unknown",
            "arguments": arguments,
            "result": {"ok": False, "error": f"Unknown internal tool: {name}"},
        }

    def _execute_create_subagent(
        self,
        *,
        context_id: str,
        arguments: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        ctx_logger,
    ) -> dict[str, Any]:
        relevant_names = {
            str(name)
            for name in arguments.get("relevant_tool_names", [])
            if isinstance(name, str)
        }
        relevant_tools = [
            tool
            for tool in tools
            if not relevant_names or _tool_name(tool) in relevant_names
        ]
        if not relevant_tools:
            relevant_tools = tools
        domain = str(arguments.get("domain") or GENERIC_SUBAGENT_DOMAIN)
        subtask = str(arguments.get("subtask") or "Analyze the current planner step.")
        domain_policy = (
            str(arguments.get("required_output") or "")
            or GENERIC_SUBAGENT_POLICY
        )
        bundle = self.context_manager.build_bundle(
            context_id=context_id,
            messages=messages,
            tools=tools,
            subagent_name=GENERIC_SUBAGENT_NAME,
            subagent_policy=domain_policy,
            relevant_tools=relevant_tools,
        )
        bundle["task_memory"] = self.task_memory.snapshot(context_id)
        bundle["subtask"] = subtask
        proposal = self._run_subagent_with_one_retry(
            agent_name=GENERIC_SUBAGENT_NAME,
            domain=domain,
            domain_policy=domain_policy,
            bundle=bundle,
            relevant_tools=relevant_tools,
            metrics=metrics,
            ctx_logger=ctx_logger,
        )
        if proposal is None:
            return {"ok": False, "error": "Subagent failed to return a usable proposal."}
        return {
            "ok": True,
            "proposal": {
                "agent": proposal.agent,
                "understood_intent": proposal.understood_intent,
                "recommended_tool_calls": proposal.proposed_tool_calls,
                "ask_user": proposal.ask_user,
                "final_response": proposal.final_response,
                "required_facts": proposal.required_facts,
                "policy_risks": proposal.policy_risks,
                "confidence": proposal.confidence,
            },
        }

    def _finalize_visible_action(
        self,
        *,
        action: dict[str, Any],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        debug: dict[str, Any],
    ) -> PlannerResult:
        validation = self.schema_guard.validate(action, tools)
        if not validation.valid:
            return PlannerResult(
                next_action=self._response_for_schema_failure(validation.errors),
                metrics=metrics,
                internal_calls=max(metrics.num_calls, 1),
                debug={**debug, "schema_errors": validation.errors},
            )

        action = validation.normalized_action or action
        policy_result = self.policy_guard.apply(action=action, tools=tools, messages=messages)
        if not policy_result.allowed and policy_result.replacement_action:
            action = policy_result.replacement_action
        task_guard_result = self.task_guard.postprocess(
            action=action,
            tools=tools,
            messages=messages,
        )
        if task_guard_result.action:
            validation = self.schema_guard.validate(task_guard_result.action, tools)
            if validation.valid:
                action = validation.normalized_action or task_guard_result.action
            else:
                action = task_guard_result.action
        debug["policy_warnings"] = policy_result.warnings
        debug["task_guard_warnings"] = task_guard_result.warnings
        return PlannerResult(
            next_action=action,
            metrics=metrics,
            internal_calls=max(metrics.num_calls, 1),
            debug=debug,
        )

    def _run_subagent_with_one_retry(
        self,
        *,
        agent_name: str,
        domain: str,
        domain_policy: str,
        bundle: dict[str, Any],
        relevant_tools: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        ctx_logger,
        correction: str | None = None,
    ) -> SubagentProposal | None:
        try:
            proposal, call_metrics = self.agent_tool.run_subagent(
                agent_name=agent_name,
                domain=domain,
                domain_policy=domain_policy,
                context_bundle=bundle,
                available_tools=relevant_tools,
                correction=correction,
                ctx_logger=ctx_logger,
            )
            metrics.add(call_metrics)
            return proposal
        except (SubagentOutputError, json.JSONDecodeError, ValueError) as exc:
            ctx_logger.warning(
                "Subagent output malformed; retrying once",
                subagent=agent_name,
                error=str(exc),
            )
            try:
                proposal, call_metrics = self.agent_tool.run_subagent(
                    agent_name=agent_name,
                    domain=domain,
                    domain_policy=domain_policy,
                    context_bundle=bundle,
                    available_tools=relevant_tools,
                    correction=(
                        "Your previous output was invalid. Return only one JSON "
                        f"object matching the schema. Error: {exc}"
                    ),
                    ctx_logger=ctx_logger,
                )
                metrics.add(call_metrics)
                return proposal
            except Exception as retry_exc:
                ctx_logger.warning(
                    "Subagent retry failed",
                    subagent=agent_name,
                    error=str(retry_exc),
                )
                return None
        except Exception as exc:
            ctx_logger.warning(
                "Subagent call failed",
                subagent=agent_name,
                error=str(exc),
            )
            return None

    def _native_fallback(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger,
    ) -> PlannerResult:
        start = time.perf_counter()
        try:
            response = completion(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
                num_retries=self.num_retries,
                **self._reasoning_kwargs(),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
            message = response.choices[0].message
            assistant_content = (
                message.model_dump(exclude_unset=True)
                if hasattr(message, "model_dump")
                else {"content": getattr(message, "content", "")}
            )
            tool_calls = assistant_content.get("tool_calls") or []
            if tool_calls:
                normalized = []
                for call in tool_calls:
                    function = call.get("function", {})
                    normalized.append(
                        {
                            "tool_name": function.get("name"),
                            "arguments": _parse_arguments(function.get("arguments")),
                        }
                    )
                action = {"action": "tool_calls", "tool_calls": normalized}
                validation = self.schema_guard.validate(action, tools)
                if validation.valid:
                    action = validation.normalized_action or action
                else:
                    action = {
                        "action": "respond",
                        "content": "I cannot safely complete that with the available tools.",
                    }
            else:
                action = {
                    "action": "respond",
                    "content": assistant_content.get("content") or "Done.",
                }
            return PlannerResult(
                next_action=action,
                metrics=metrics,
                internal_calls=max(metrics.num_calls, 1),
                debug={"native_fallback_used": True},
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            ctx_logger.error("Native fallback failed", error=str(exc))
            return PlannerResult(
                next_action={
                    "action": "respond",
                    "content": "I cannot safely complete that with the available tools.",
                },
                metrics=LLMCallMetrics(elapsed_ms=elapsed_ms),
                internal_calls=1,
                debug={"native_fallback_used": True, "native_error": str(exc)},
            )

    @staticmethod
    def _response_for_schema_failure(errors: list[str]) -> dict[str, Any]:
        joined = " ".join(errors).lower()
        if "unknown tool" in joined:
            return {
                "action": "respond",
                "content": "I cannot complete that because the required capability is not available right now.",
            }
        if "missing required argument" in joined:
            return {
                "action": "respond",
                "content": "I cannot complete that because the available tool is missing a required parameter for this request.",
            }
        return {
            "action": "respond",
            "content": "I cannot safely complete that with the available tools.",
        }

    def _summary_callback(
        self,
        *,
        payload: dict[str, Any],
        metrics: LLMCallMetrics,
        ctx_logger,
    ) -> str:
        summary, call_metrics = self.agent_tool.summarize_context(
            payload=payload,
            ctx_logger=ctx_logger,
        )
        metrics.add(call_metrics)
        return summary

    def _reasoning_kwargs(self) -> dict[str, Any]:
        if not self.thinking:
            return {}
        if self.reasoning_effort in {"none", "disable", "low", "medium", "high"}:
            kwargs: dict[str, Any] = {"reasoning_effort": self.reasoning_effort}
        else:
            try:
                kwargs = {"thinking": {"type": "enabled", "budget_tokens": int(self.reasoning_effort)}}
            except ValueError:
                kwargs = {"reasoning_effort": "medium"}
        if self.interleaved_thinking:
            kwargs["extra_headers"] = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
        return kwargs


def _normalize_internal_tool_calls(decision: dict[str, Any]) -> list[dict[str, Any]]:
    calls = decision.get("internal_tool_calls")
    if calls is None:
        call = decision.get("internal_tool_call")
        calls = [call] if isinstance(call, dict) else []
    if not isinstance(calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        arguments = call.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = _parse_arguments(arguments)
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def _external_action_from_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    action = decision.get("external_action")
    if not isinstance(action, dict):
        return None
    if action.get("action") == "tool_calls":
        calls = action.get("tool_calls") or []
        if not isinstance(calls, list):
            calls = []
        normalized_calls: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("tool_name") or call.get("name")
            arguments = call.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = _parse_arguments(arguments)
            if isinstance(tool_name, str) and tool_name:
                normalized_calls.append({"tool_name": tool_name, "arguments": arguments})
        return {"action": "tool_calls", "tool_calls": normalized_calls}
    if action.get("action") == "respond":
        return {"action": "respond", "content": str(action.get("content") or "")}
    if "tool_calls" in action:
        return _external_action_from_decision({"external_action": {"action": "tool_calls", **action}})
    if "content" in action:
        return {"action": "respond", "content": str(action.get("content") or "")}
    return None


def _extract_json_object(text: str) -> Any:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in planner output: {text[:200]}")
        return json.loads(stripped[start : end + 1])


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or tool.get("name") or "")


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
