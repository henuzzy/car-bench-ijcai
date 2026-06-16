"""LangGraph-based planner for the Track 1 CAR-bench agent."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from litellm import completion

try:
    from .approved_plan import (
        ApprovedPlan,
        CriticVerdict,
        normalize_approved_plan,
        normalize_critic_verdict,
        steps_for_current_phase,
    )
    from .langgraph_workflow import Track1LangGraphWorkflow
    from .multi_agent_types import LLMCallMetrics, PlannerResult
except ImportError:
    from approved_plan import (
        ApprovedPlan,
        CriticVerdict,
        normalize_approved_plan,
        normalize_critic_verdict,
        steps_for_current_phase,
    )
    from langgraph_workflow import Track1LangGraphWorkflow
    from multi_agent_types import LLMCallMetrics, PlannerResult


LANGGRAPH_APPROVED_PLAN_SYSTEM_PROMPT = """You are the planner node inside a LangGraph in-car voice assistant.

Return exactly one JSON object and no extra text.

You receive:
- langgraph_memory: structured state derived inside the graph,
- context_bundle: compact recent conversation and tool-result context,
- external_tools: the only tools that may be called.

Produce one ApprovedPlan for the current turn:
{
  "summary": "brief private plan summary",
  "task_feasible": true,
  "infeasible_reason": null,
  "phase": "get | execute | done",
  "allowed_tools": ["tool names allowed now"],
  "forbidden_tools": ["tool names forbidden now"],
  "action_plan": [
    {"tool": "exact_tool_name", "arguments": {}, "phase": "get | execute", "purpose": "why this call is needed"}
  ],
  "response": null
}

Rules:
- Use only tools and parameters from external_tools.
- Do not invent IDs, contacts, weather, route IDs, vehicle facts, or missing result fields.
- phase=get gathers read-only facts. phase=execute performs approved state changes. phase=done responds.
- Do not repeat a successful identical read-only call unless the latest user message changes the need.
- Before state-changing actions, gather required preconditions first.
- If the requested action needs a missing required tool/parameter/result, set task_feasible=false or phase=done with a concise limitation.
- For email/calendar/weather text, use 24-hour HH:MM time. Never use AM/PM.
- If a tool description starts with REQUIRES_CONFIRMATION, ask for confirmation in phase=done unless the latest user confirmed the exact action.
"""

LANGGRAPH_PLAN_CRITIC_SYSTEM_PROMPT = """You are the critic node inside a LangGraph in-car voice assistant.

Audit one ApprovedPlan before the graph executes it. Return exactly one JSON object:
{
  "verdict": "PASS | REVISE",
  "violations": ["brief issue"],
  "recommended_changes": ["concrete plan change"],
  "reasoning": "brief private reason"
}

Use REVISE when the plan repeats completed work, calls unavailable tools, uses unsupported parameters,
skips clear preconditions, fabricates IDs or result fields, asks for unnecessary confirmation, or performs
extra state changes outside the user request.
"""


class Track1Planner:
    """Planner facade whose runtime state and flow live in LangGraph."""

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
        self.timeout_seconds = float(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "75"))
        self.num_retries = int(os.getenv("AGENT_LLM_NUM_RETRIES", "1"))
        self.max_critic_revisions = int(os.getenv("AGENT_MAX_CRITIC_REVISIONS", "1"))
        self.turn_budget_seconds = float(os.getenv("AGENT_TURN_BUDGET_SECONDS", "45"))
        self.langgraph_workflow = Track1LangGraphWorkflow(self)

    def reset(self, context_id: str) -> None:
        self.langgraph_workflow.reset(context_id)

    def choose_next_action(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger,
    ) -> PlannerResult:
        return self.langgraph_workflow.invoke(
            context_id=context_id,
            messages=messages,
            tools=tools,
            ctx_logger=ctx_logger,
        )

    def _run_langgraph_approved_planner(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        graph_memory: dict[str, Any],
        context_bundle: dict[str, Any],
        critic_feedback: str,
        deadline: float,
        ctx_logger,
    ) -> tuple[ApprovedPlan, LLMCallMetrics]:
        available_tool_names = {_tool_name(tool) for tool in tools if _tool_name(tool)}
        payload = {
            "task": "Produce an ApprovedPlan for the current LangGraph turn.",
            "context_id": context_id,
            "langgraph_memory": graph_memory,
            "context_bundle": context_bundle,
            "external_tools": tools,
            "available_tool_names": sorted(available_tool_names),
            "critic_feedback": critic_feedback,
        }
        start = time.perf_counter()
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": LANGGRAPH_APPROVED_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self.temperature,
            timeout=self._remaining_timeout(deadline),
            num_retries=self.num_retries,
            **self._reasoning_kwargs(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        plan = normalize_approved_plan(
            _extract_json_object(_message_content(response)),
            available_tool_names,
        )
        ctx_logger.debug(
            "LangGraph planner node completed",
            elapsed_ms=round(elapsed_ms, 1),
            phase=plan.phase,
            allowed_tools=plan.allowed_tools,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
        )
        return plan, metrics

    def _run_langgraph_plan_critic(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        graph_memory: dict[str, Any],
        context_bundle: dict[str, Any],
        plan: ApprovedPlan,
        deadline: float,
        ctx_logger,
    ) -> tuple[CriticVerdict, LLMCallMetrics]:
        payload = {
            "context_id": context_id,
            "conversation": messages[-10:],
            "langgraph_memory": graph_memory,
            "context_bundle": context_bundle,
            "approved_plan": plan.to_dict(),
            "external_tools": tools,
            "available_tool_names": sorted(_tool_name(tool) for tool in tools if _tool_name(tool)),
        }
        start = time.perf_counter()
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": LANGGRAPH_PLAN_CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            timeout=self._remaining_timeout(deadline),
            num_retries=self.num_retries,
            **self._reasoning_kwargs(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        verdict = normalize_critic_verdict(_extract_json_object(_message_content(response)))
        ctx_logger.debug(
            "LangGraph critic node completed",
            elapsed_ms=round(elapsed_ms, 1),
            verdict=verdict.verdict,
            violations=verdict.violations,
        )
        return verdict, metrics

    def _execute_approved_plan(
        self,
        approved_plan: ApprovedPlan,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not approved_plan.task_feasible:
            return {
                "action": "respond",
                "content": (
                    approved_plan.infeasible_reason
                    or "I cannot complete that with the available vehicle capabilities."
                ),
            }
        if approved_plan.phase == "done":
            return {"action": "respond", "content": approved_plan.response or "Done."}

        available_tool_names = {_tool_name(tool) for tool in tools if _tool_name(tool)}
        steps = steps_for_current_phase(approved_plan, available_tool_names)
        if not steps:
            if approved_plan.response:
                return {"action": "respond", "content": approved_plan.response}
            return None
        return {
            "action": "tool_calls",
            "tool_calls": [
                {"tool_name": step.tool, "arguments": step.arguments}
                for step in steps
            ],
        }

    def _native_fallback(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger,
        deadline: float | None = None,
    ) -> PlannerResult:
        start = time.perf_counter()
        try:
            response = completion(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=self.temperature,
                timeout=(
                    self._remaining_timeout(deadline)
                    if deadline is not None
                    else self.timeout_seconds
                ),
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
                action = {
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "tool_name": call.get("function", {}).get("name"),
                            "arguments": _parse_arguments(
                                call.get("function", {}).get("arguments")
                            ),
                        }
                        for call in tool_calls
                    ],
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

    def _remaining_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self.timeout_seconds
        return max(1.0, min(self.timeout_seconds, deadline - time.perf_counter()))

    def _reasoning_kwargs(self) -> dict[str, Any]:
        if not self.thinking:
            return {}
        if self.reasoning_effort in {"none", "disable", "low", "medium", "high"}:
            kwargs: dict[str, Any] = {"reasoning_effort": self.reasoning_effort}
        else:
            try:
                kwargs = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": int(self.reasoning_effort),
                    }
                }
            except ValueError:
                kwargs = {"reasoning_effort": "medium"}
        if self.interleaved_thinking:
            kwargs["extra_headers"] = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
        return kwargs


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


def _message_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", None) or ""
    if not content and hasattr(message, "model_dump"):
        content = message.model_dump(exclude_unset=True).get("content") or ""
    return str(content or "")


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
