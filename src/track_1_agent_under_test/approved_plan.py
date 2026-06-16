"""Structured PEC-lite plan types for the Track 1 agent.

The planner produces an ApprovedPlan first.  A critic audits that plan, and the
executor only emits tool calls that belong to the current approved phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PLAN_PHASES = {"get", "execute", "done"}
STEP_PHASE_ALIASES = {"set": "execute", "act": "execute", "respond": "done"}


@dataclass
class ApprovedStep:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    phase: str = "get"
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovedPlan:
    task_feasible: bool = True
    infeasible_reason: str | None = None
    phase: str = "get"
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    action_plan: list[ApprovedStep] = field(default_factory=list)
    response: str | None = None
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_feasible": self.task_feasible,
            "infeasible_reason": self.infeasible_reason,
            "phase": self.phase,
            "allowed_tools": self.allowed_tools,
            "forbidden_tools": self.forbidden_tools,
            "action_plan": [step.to_dict() for step in self.action_plan],
            "response": self.response,
            "summary": self.summary,
        }


@dataclass
class CriticVerdict:
    verdict: str = "PASS"
    violations: list[str] = field(default_factory=list)
    recommended_changes: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict.upper() == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "violations": self.violations,
            "recommended_changes": self.recommended_changes,
            "reasoning": self.reasoning,
        }


def normalize_approved_plan(payload: Any, available_tool_names: set[str]) -> ApprovedPlan:
    if not isinstance(payload, dict):
        raise ValueError("ApprovedPlan payload must be a JSON object")

    feasible = bool(payload.get("task_feasible", True))
    phase = _normalize_phase(payload.get("phase"))
    steps = _normalize_steps(payload.get("action_plan"), available_tool_names)
    allowed_tools = _string_list(payload.get("allowed_tools"))
    forbidden_tools = _string_list(payload.get("forbidden_tools"))

    if not allowed_tools:
        allowed_tools = _tools_for_phase(steps, phase)

    if phase == "done" and not payload.get("response") and feasible:
        payload["response"] = "Done."

    return ApprovedPlan(
        task_feasible=feasible,
        infeasible_reason=_optional_string(payload.get("infeasible_reason")),
        phase=phase,
        allowed_tools=_dedupe([name for name in allowed_tools if name in available_tool_names]),
        forbidden_tools=_dedupe([name for name in forbidden_tools if name in available_tool_names]),
        action_plan=steps,
        response=_optional_string(payload.get("response")),
        summary=str(payload.get("summary") or payload.get("plan_reasoning") or ""),
        raw=payload,
    )


def normalize_critic_verdict(payload: Any) -> CriticVerdict:
    if not isinstance(payload, dict):
        raise ValueError("Critic payload must be a JSON object")
    verdict = str(payload.get("verdict") or "PASS").upper()
    if verdict not in {"PASS", "REVISE"}:
        verdict = "REVISE" if verdict in {"BLOCK", "FAIL"} else "PASS"
    return CriticVerdict(
        verdict=verdict,
        violations=_string_list(payload.get("violations")),
        recommended_changes=_string_list(payload.get("recommended_changes")),
        reasoning=str(payload.get("reasoning") or ""),
        raw=payload,
    )


def steps_for_current_phase(plan: ApprovedPlan, available_tool_names: set[str]) -> list[ApprovedStep]:
    if plan.phase == "done" or not plan.task_feasible:
        return []
    allowed = set(plan.allowed_tools) if plan.allowed_tools else available_tool_names
    forbidden = set(plan.forbidden_tools)
    steps: list[ApprovedStep] = []
    for step in plan.action_plan:
        if step.phase != plan.phase:
            continue
        if step.tool not in available_tool_names:
            continue
        if step.tool in forbidden:
            continue
        if step.tool not in allowed:
            continue
        steps.append(step)
    return steps


def _normalize_steps(raw_steps: Any, available_tool_names: set[str]) -> list[ApprovedStep]:
    if not isinstance(raw_steps, list):
        return []
    steps: list[ApprovedStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("tool_name") or item.get("name") or "")
        if not tool or tool not in available_tool_names:
            continue
        arguments = item.get("arguments", item.get("args", {}))
        if not isinstance(arguments, dict):
            arguments = {}
        steps.append(
            ApprovedStep(
                tool=tool,
                arguments=arguments,
                phase=_normalize_step_phase(item.get("phase")),
                purpose=str(item.get("purpose") or ""),
            )
        )
    return steps


def _normalize_phase(value: Any) -> str:
    phase = str(value or "get").strip().lower()
    phase = STEP_PHASE_ALIASES.get(phase, phase)
    return phase if phase in PLAN_PHASES else "get"


def _normalize_step_phase(value: Any) -> str:
    phase = str(value or "get").strip().lower()
    phase = STEP_PHASE_ALIASES.get(phase, phase)
    if phase == "done":
        return "execute"
    return phase if phase in {"get", "execute"} else "get"


def _tools_for_phase(steps: list[ApprovedStep], phase: str) -> list[str]:
    if phase == "done":
        return []
    return _dedupe(step.tool for step in steps if step.phase == phase)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float)) and str(item)]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
