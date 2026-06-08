"""Private subagent runner used by the Track 1 planner."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from litellm import completion

try:
    from .multi_agent_types import LLMCallMetrics, SubagentProposal
except ImportError:
    from multi_agent_types import LLMCallMetrics, SubagentProposal


SUBAGENT_SYSTEM_PROMPT = """You are a private subagent inside a CAR-bench in-car assistant harness.
You do not execute tools. You do not call the evaluator. You only propose the next assistant action.
Return exactly one JSON object and no extra text.
Use only tool names and parameters in available_tools.
If information is missing and a safe tool can retrieve it, propose that tool call.
For disambiguation, exhaust internal resolution first: policies, explicit user text, preferences, context/state tools, and prior tool results. Ask the user only when ambiguity remains unresolved.
For hallucination/limit-awareness cases, if a required tool, parameter, or result field is unavailable, do not fabricate it or proceed incorrectly. Put a concise acknowledgement of the limitation in final_response.
Before state-changing actions, include required information-gathering and policy precondition tools. For example, sunroof opening requires weather checking and sunshade policy handling.
Never invent IDs. Use location IDs, route IDs, contact details, charging plug IDs, and POI details only after a tool result provides them.
When a state-changing action depends on a tool result, propose the information-gathering call first; do not bundle the dependent state change in the same proposal.
CAR-bench high-frequency policies:
- Sunroof opening needs weather condition and the sunshade fully open or opened in parallel.
- AC on needs climate status, window status, windows above 20% closed, and fan speed at least 1.
- Defrost on needs climate/window status, AC on, fan at least 2, and airflow including WINDSHIELD.
- Fog lights need weather, exterior light status, low beams on, and high beams off when required.
- High beams cannot be on with fog lights and normally need confirmation.
- Active navigation must be edited with navigation edit tools; set_new_navigation is only for inactive navigation.
- Charging/range reasoning needs charging specs/status. Route charging-station searches need a grounded route and at_kilometer when required.
- Contact calls/emails need retrieved phone_number/email; send_email requires confirmation.
If ambiguity remains after available facts and policies, put a short spoken clarification in ask_user.
If no tool is needed, put the short spoken answer in final_response.
Keep user-facing text natural, brief, and suitable for text-to-speech."""


class SubagentOutputError(ValueError):
    """Raised when a private subagent returns malformed JSON."""


class AgentTool:
    """Spawns one private subagent model call with isolated context."""

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

    def run_subagent(
        self,
        *,
        agent_name: str,
        domain: str,
        domain_policy: str,
        context_bundle: dict[str, Any],
        available_tools: list[dict[str, Any]],
        correction: str | None = None,
        ctx_logger: Any = None,
    ) -> tuple[SubagentProposal, LLMCallMetrics]:
        payload = _build_subagent_payload(
            agent_name=agent_name,
            domain=domain,
            domain_policy=domain_policy,
            context_bundle=context_bundle,
            available_tools=available_tools,
            correction=correction,
        )
        result_text, metrics = self._complete_text(
            messages=[
                {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            ctx_logger=ctx_logger,
            call_label=agent_name,
        )
        proposal = parse_subagent_proposal(result_text, agent_name)
        return proposal, metrics

    def summarize_context(
        self,
        *,
        payload: dict[str, Any],
        ctx_logger: Any = None,
    ) -> tuple[str, LLMCallMetrics]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize CAR-bench conversation context into the required "
                    "sections. Preserve task state, pending work, confirmations, "
                    "tool facts, policy constraints, and next step. Return plain text."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return self._complete_text(
            messages=messages,
            ctx_logger=ctx_logger,
            call_label="ContextCompactAgent",
        )

    def _complete_text(
        self,
        *,
        messages: list[dict[str, Any]],
        ctx_logger: Any,
        call_label: str,
    ) -> tuple[str, LLMCallMetrics]:
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "num_retries": self.num_retries,
        }
        if self.thinking:
            if self.reasoning_effort in {"none", "disable", "low", "medium", "high"}:
                completion_kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                try:
                    completion_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": int(self.reasoning_effort),
                    }
                except ValueError:
                    completion_kwargs["reasoning_effort"] = "medium"
            if self.interleaved_thinking:
                completion_kwargs["extra_headers"] = {
                    "anthropic-beta": "interleaved-thinking-2025-05-14"
                }

        start = time.perf_counter()
        response = completion(**completion_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        if not content and hasattr(message, "model_dump"):
            content = message.model_dump(exclude_unset=True).get("content") or ""
        if ctx_logger is not None:
            ctx_logger.debug(
                "Internal subagent call completed",
                subagent=call_label,
                elapsed_ms=round(elapsed_ms, 1),
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
            )
        return content, metrics


def parse_subagent_proposal(text: str, default_agent: str) -> SubagentProposal:
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        raise SubagentOutputError("subagent output must be a JSON object")

    proposed_tool_calls = payload.get("proposed_tool_calls", [])
    if proposed_tool_calls is None:
        proposed_tool_calls = []
    if not isinstance(proposed_tool_calls, list):
        raise SubagentOutputError("proposed_tool_calls must be a list")

    normalized_calls: list[dict[str, Any]] = []
    for item in proposed_tool_calls:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name") or item.get("name")
        arguments = item.get("arguments")
        if arguments is None and item.get("arguments_json") is not None:
            arguments = _parse_arguments_json(item["arguments_json"])
        if arguments is None:
            arguments = {}
        if not isinstance(tool_name, str) or not tool_name:
            continue
        if not isinstance(arguments, dict):
            raise SubagentOutputError("tool arguments must be an object")
        normalized_calls.append({"tool_name": tool_name, "arguments": arguments})

    confidence = payload.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0

    return SubagentProposal(
        agent=str(payload.get("agent") or default_agent),
        understood_intent=str(payload.get("understood_intent") or ""),
        proposed_tool_calls=normalized_calls,
        ask_user=_optional_string(payload.get("ask_user")),
        final_response=_optional_string(payload.get("final_response")),
        required_facts=_string_list(payload.get("required_facts")),
        policy_risks=_string_list(payload.get("policy_risks")),
        confidence=max(0.0, min(confidence_value, 1.0)),
        raw=payload,
    )


def _build_subagent_payload(
    *,
    agent_name: str,
    domain: str,
    domain_policy: str,
    context_bundle: dict[str, Any],
    available_tools: list[dict[str, Any]],
    correction: str | None,
) -> dict[str, Any]:
    payload = {
        "task": (
            "Propose exactly one next CAR-bench assistant action for the planner. "
            "Never execute tools yourself."
        ),
        "subagent": {
            "name": agent_name,
            "domain": domain,
            "domain_policy_summary": domain_policy,
        },
        "context": context_bundle,
        "available_tools": available_tools,
        "output_schema": {
            "agent": agent_name,
            "understood_intent": "short private summary",
            "proposed_tool_calls": [
                {"tool_name": "exact_available_tool_name", "arguments": {}}
            ],
            "ask_user": None,
            "final_response": None,
            "required_facts": [],
            "policy_risks": [],
            "confidence": 0.0,
        },
        "rules": [
            "If proposed_tool_calls is non-empty, ask_user and final_response must be null.",
            "If a user clarification is needed, proposed_tool_calls must be empty and ask_user must be a short spoken sentence.",
            "If the task is complete, proposed_tool_calls must be empty and final_response must be a short spoken sentence.",
            "Do not invent tool names, enum values, IDs, tool results, or vehicle state.",
            "Respect REQUIRES_CONFIRMATION tool descriptions and policy constraints.",
            "If a required capability is absent from available_tools or a required parameter is absent from its schema, explicitly state that the request cannot be fulfilled.",
            "If a required tool result field is unknown or missing, do not infer it before a state-changing action.",
            "For ambiguous tool arguments, call get_user_preferences or relevant state/context tools before asking the user.",
            "For route, POI, contact, charging, and vehicle-state dependent actions, propose the get/search tool first unless the needed result is already in recent_tool_facts.",
            "Do not use set_new_navigation when recent navigation state says navigation_active is true; use navigation edit tools if available.",
        ],
    }
    if correction:
        payload["correction"] = correction
    return payload


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise SubagentOutputError(f"No JSON object found in output: {text[:200]}")
        return json.loads(stripped[start : end + 1])


def _parse_arguments_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise SubagentOutputError("arguments_json must decode to an object")
    return parsed


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
