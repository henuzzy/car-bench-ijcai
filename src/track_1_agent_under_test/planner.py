"""Planner/coordinator for the Track 1 multi-agent harness."""

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
    from .agent_tool import AgentTool, SubagentOutputError
    from .completion_verifier import CompletionVerifier
    from .context_manager import ContextManager
    from .failure_guard import FailureGuard
    from .guards import PolicyGuard, SchemaGuard
    from .langgraph_workflow import Track1LangGraphWorkflow
    from .multi_agent_types import LLMCallMetrics, PlannerResult, SubagentProposal
    from .plan_state import PlanStateStore
    from .skills import SkillRegistry
    from .task_memory import TASK_TOOL_SCHEMAS, TaskMemoryStore
    from .task_guard import TaskGuard
    from .training_insights import TrainingInsightStore, default_training_insights
except ImportError:
    from approved_plan import (
        ApprovedPlan,
        CriticVerdict,
        normalize_approved_plan,
        normalize_critic_verdict,
        steps_for_current_phase,
    )
    from agent_tool import AgentTool, SubagentOutputError
    from completion_verifier import CompletionVerifier
    from context_manager import ContextManager
    from failure_guard import FailureGuard
    from guards import PolicyGuard, SchemaGuard
    from langgraph_workflow import Track1LangGraphWorkflow
    from multi_agent_types import LLMCallMetrics, PlannerResult, SubagentProposal
    from plan_state import PlanStateStore
    from skills import SkillRegistry
    from task_memory import TASK_TOOL_SCHEMAS, TaskMemoryStore
    from task_guard import TaskGuard
    from training_insights import TrainingInsightStore, default_training_insights


GENERIC_SUBAGENT_NAME = "PrivateSubagent"
GENERIC_SUBAGENT_DOMAIN = "general"
GENERIC_SUBAGENT_POLICY = (
    "Assist the main planner with the current CAR-bench step only. Use available "
    "tools, recent facts, training-derived generic recipes, and policy "
    "constraints. Do not re-plan the whole task."
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

You will also receive plan_state. Treat plan_state as authoritative structured
state for the current evaluator context. Do not re-decide finished steps from
scratch. Follow plan_state.next_allowed_tools when present and do not take
plan_state.forbidden_actions. If tool results change the plan, update task memory
or choose the next allowed action instead of repeating old calls.

You may also receive training_insights derived from public base training tasks.
Use them only as abstract tool-order priors and policy reminders. Never infer,
copy, or memorize concrete task ids, location ids, route ids, contact details,
POI ids, or exact arguments from training data.

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
- Do not repeat an identical read-only tool call after it already returned a
  result unless the user supplied new information changing the arguments.
- For POI navigation, a poi_* id is the destination. A loc_* city id is only a
  location used to search POIs or route to the city itself.
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

APPROVED_PLAN_SYSTEM_PROMPT = """You are the Planner in a PEC-lite in-car voice assistant agent.

Your job is to produce one ApprovedPlan for the current turn. The ApprovedPlan
does not directly execute tools. A separate executor will only execute tool
calls that are in the current phase and in allowed_tools.

Return exactly one JSON object and no extra text.

ApprovedPlan schema:
{
  "summary": "brief private plan summary",
  "task_feasible": true,
  "infeasible_reason": null,
  "phase": "get | execute | done",
  "allowed_tools": ["tool names that may be used now"],
  "forbidden_tools": ["tool names that must not be used now"],
  "action_plan": [
    {"tool": "exact_tool_name", "arguments": {}, "phase": "get | execute", "purpose": "why this call is needed"}
  ],
  "response": null
}

Phase rules:
- phase=get: gather only immediately executable read-only information needed now.
- phase=execute: perform only the approved state-changing or user-facing tool calls now.
- phase=done: respond to the user using response. Do not include tool calls.
- action_plan must contain only calls that can be safely executed in the current
  turn with arguments grounded in the conversation/tool results. Do not include
  future calls whose arguments depend on missing results.

Gather-first and prerequisite rules:
- Before any state-changing call, include all relevant read-only prerequisite
  tools first.
- open_close_sunroof -> get_weather + get_sunroof_and_sunshade_position first.
- open_close_sunshade -> get_sunroof_and_sunshade_position first.
- set_air_conditioning -> get_climate_settings + get_vehicle_window_positions first.
- open_close_window -> get_climate_settings + get_vehicle_window_positions first.
- set_window_defrost -> get_climate_settings + get_vehicle_window_positions first.
- set_fog_lights -> get_weather + get_exterior_lights_status first.
- set_head_lights_high_beams or set_head_lights_low_beams -> get_exterior_lights_status first.
- send_email -> gather contact/calendar/weather facts required by the user message first.
- navigation changes -> resolve exact location/POI IDs and route IDs first. For POI
  requests, the destination must be the poi_* id, not only a loc_* city id.
- ambiguous values such as percentage, color, level, or temperature -> use
  get_user_preferences before asking the user.

Policy and format rules:
- Use only tools and parameters from external_tools.
- Never invent IDs, route IDs, contacts, weather, calendar fields, or missing tool
  result fields.
- Keep the minimum action principle: no extra state changes beyond the user's
  request or policy requirements.
- For spoken/calendar/email/weather text, use 24-hour HH:MM time. Never use AM/PM.
- If a tool description starts with REQUIRES_CONFIRMATION, phase must be done with
  a confirmation request unless the latest user message already explicitly
  confirms the exact action.
- If a required tool/parameter/result field is unavailable, set task_feasible=false
  or phase=done with a concise limitation. Do not fabricate.

PlanState rules:
- You will receive plan_state. Treat it as authoritative per-context structured
  memory. Use plan_state.next_allowed_tools as strong phase guidance when present.
- Do not repeat plan_state.repeated_actions or forbidden_tools unless the latest
  user message changes the task.
- If a prior get call produced the needed fact, advance to the next phase instead
  of repeating the same get call.

Training insights are abstract public-train-derived priors only. Never copy or
infer concrete train/test answers, ids, routes, contacts, or exact arguments from
them."""

PLAN_CRITIC_SYSTEM_PROMPT = """You are an independent Critic for a CAR-bench in-car assistant plan.

Audit the ApprovedPlan before any tool is executed. You are not scoring the run
and must not use reward metrics. Check only general policy, feasibility, tool
ordering, grounding, and consistency with the user request.

Return exactly one JSON object:
{
  "verdict": "PASS | REVISE",
  "violations": ["brief issue"],
  "recommended_changes": ["concrete plan change"],
  "reasoning": "brief private reason"
}

Use REVISE when:
- a state-changing tool lacks required get prerequisites,
- the plan asks the user before checking available preferences/context,
- current phase contains a tool outside allowed_tools or a forbidden tool,
- the plan includes extra state changes not requested,
- arguments contain invented or unsupported IDs/fields,
- confirmation is required but not requested/confirmed,
- time format may use AM/PM instead of HH:MM,
- the plan repeats a completed/failed action instead of advancing.

Use PASS only when the current phase is safe to execute as written."""

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
        training_insights: TrainingInsightStore | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.interleaved_thinking = interleaved_thinking
        self.context_manager = ContextManager()
        self.task_memory = TaskMemoryStore()
        self.plan_state = PlanStateStore()
        self.training_insights = training_insights or default_training_insights()
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
        self.skill_registry = SkillRegistry()
        self.completion_verifier = CompletionVerifier(self.skill_registry)
        self.failure_guard = FailureGuard()
        self.timeout_seconds = float(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "75"))
        self.num_retries = int(os.getenv("AGENT_LLM_NUM_RETRIES", "1"))
        self.max_internal_steps = int(os.getenv("AGENT_MAX_INTERNAL_STEPS", "4"))
        self.max_critic_revisions = int(os.getenv("AGENT_MAX_CRITIC_REVISIONS", "1"))
        self.turn_budget_seconds = float(os.getenv("AGENT_TURN_BUDGET_SECONDS", "45"))
        self.langgraph_workflow = Track1LangGraphWorkflow(self)

    def reset(self, context_id: str) -> None:
        self.context_manager.reset(context_id)
        self.task_memory.reset(context_id)
        self.plan_state.reset(context_id)

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

    def _build_approved_plan(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        deadline: float,
        ctx_logger,
    ) -> tuple[ApprovedPlan, CriticVerdict, LLMCallMetrics, dict[str, Any]]:
        pec_metrics = LLMCallMetrics()
        critic_feedback = ""
        latest_plan: ApprovedPlan | None = None
        latest_critic = CriticVerdict()
        revisions = 0

        for attempt in range(self.max_critic_revisions + 1):
            plan, plan_metrics = self._run_approved_planner(
                context_id=context_id,
                messages=messages,
                tools=tools,
                metrics=metrics,
                critic_feedback=critic_feedback,
                deadline=deadline,
                ctx_logger=ctx_logger,
            )
            pec_metrics.add(plan_metrics)
            latest_plan = plan
            if self._remaining_timeout(deadline) <= 1.0:
                break

            critic, critic_metrics = self._run_plan_critic(
                context_id=context_id,
                messages=messages,
                tools=tools,
                plan=plan,
                deadline=deadline,
                ctx_logger=ctx_logger,
            )
            pec_metrics.add(critic_metrics)
            latest_critic = critic
            if critic.passed:
                break
            revisions += 1
            critic_feedback = (
                "The previous ApprovedPlan must be revised.\nViolations:\n- "
                + "\n- ".join(critic.violations or ["critic requested revision"])
                + "\nRecommended changes:\n- "
                + "\n- ".join(critic.recommended_changes or ["produce a safer phase plan"])
            )
            ctx_logger.info(
                "Critic requested ApprovedPlan revision",
                attempt=attempt + 1,
                violations=critic.violations,
                recommended_changes=critic.recommended_changes,
            )

        if latest_plan is None:
            raise RuntimeError("Approved planner did not return a plan")
        return latest_plan, latest_critic, pec_metrics, {
            "pec_lite": True,
            "critic_revisions": revisions,
        }

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
        call_metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        plan = normalize_approved_plan(
            _extract_json_object(_message_content(response)),
            available_tool_names,
        )
        ctx_logger.debug(
            "LangGraph planner node completed",
            elapsed_ms=round(elapsed_ms, 1),
            phase=plan.phase,
            allowed_tools=plan.allowed_tools,
            prompt_tokens=call_metrics.prompt_tokens,
            completion_tokens=call_metrics.completion_tokens,
        )
        return plan, call_metrics

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
        call_metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        verdict = normalize_critic_verdict(_extract_json_object(_message_content(response)))
        ctx_logger.debug(
            "LangGraph critic node completed",
            elapsed_ms=round(elapsed_ms, 1),
            verdict=verdict.verdict,
            violations=verdict.violations,
        )
        return verdict, call_metrics

    def _run_approved_planner(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        critic_feedback: str,
        deadline: float,
        ctx_logger,
    ) -> tuple[ApprovedPlan, LLMCallMetrics]:
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
        plan_state_snapshot = self.plan_state.snapshot(context_id)
        training_hints = self._training_hints_for(
            messages=messages,
            completed_tools=plan_state_snapshot.get("completed_tools") or [],
            tools=tools,
        )
        available_tool_names = {_tool_name(tool) for tool in tools if _tool_name(tool)}
        payload = {
            "task": "Produce an ApprovedPlan for the current turn.",
            "context": context_bundle,
            "task_memory": self.task_memory.snapshot(context_id),
            "task_reminders": self.task_memory.reminders(context_id),
            "plan_state": plan_state_snapshot,
            "plan_state_guidance": [
                *self.plan_state.guidance(context_id),
                *training_hints.get("policy_hints", []),
                *training_hints.get("transition_hints", []),
            ],
            "training_insights": training_hints,
            "external_tools": tools,
            "available_tool_names": sorted(available_tool_names),
            "critic_feedback": critic_feedback,
        }
        start = time.perf_counter()
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": APPROVED_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self.temperature,
            timeout=self._remaining_timeout(deadline),
            num_retries=self.num_retries,
            **self._reasoning_kwargs(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        call_metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        content = _message_content(response)
        plan = normalize_approved_plan(_extract_json_object(content), available_tool_names)
        ctx_logger.debug(
            "Approved planner call completed",
            elapsed_ms=round(elapsed_ms, 1),
            phase=plan.phase,
            allowed_tools=plan.allowed_tools,
            prompt_tokens=call_metrics.prompt_tokens,
            completion_tokens=call_metrics.completion_tokens,
        )
        return plan, call_metrics

    def _run_plan_critic(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        plan: ApprovedPlan,
        deadline: float,
        ctx_logger,
    ) -> tuple[CriticVerdict, LLMCallMetrics]:
        plan_state_snapshot = self.plan_state.snapshot(context_id)
        payload = {
            "conversation": messages[-10:],
            "plan_state": plan_state_snapshot,
            "approved_plan": plan.to_dict(),
            "external_tools": tools,
            "available_tool_names": sorted(_tool_name(tool) for tool in tools if _tool_name(tool)),
        }
        start = time.perf_counter()
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": PLAN_CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            timeout=self._remaining_timeout(deadline),
            num_retries=self.num_retries,
            **self._reasoning_kwargs(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        call_metrics = LLMCallMetrics.from_litellm_response(response, elapsed_ms)
        verdict = normalize_critic_verdict(_extract_json_object(_message_content(response)))
        ctx_logger.debug(
            "Plan critic call completed",
            elapsed_ms=round(elapsed_ms, 1),
            verdict=verdict.verdict,
            violations=verdict.violations,
        )
        return verdict, call_metrics

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
            return {
                "action": "respond",
                "content": approved_plan.response or "Done.",
            }

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
        plan_state_snapshot = self.plan_state.snapshot(context_id)
        training_hints = self._training_hints_for(
            messages=messages,
            completed_tools=plan_state_snapshot.get("completed_tools") or [],
            tools=tools,
        )
        payload = {
            "task": "Choose the next private planner step or one benchmark-visible action.",
            "context": context_bundle,
            "task_memory": self.task_memory.snapshot(context_id),
            "task_reminders": self.task_memory.reminders(context_id),
            "plan_state": plan_state_snapshot,
            "plan_state_guidance": [
                *self.plan_state.guidance(context_id),
                *training_hints.get("policy_hints", []),
                *training_hints.get("transition_hints", []),
            ],
            "training_insights": training_hints,
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
        bundle["plan_state"] = self.plan_state.snapshot(context_id)
        bundle["training_insights"] = self._training_hints_for(
            messages=messages,
            completed_tools=bundle["plan_state"].get("completed_tools") or [],
            tools=relevant_tools,
        )
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
        context_id: str,
        action: dict[str, Any],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        metrics: LLMCallMetrics,
        debug: dict[str, Any],
        internal_calls_floor: int = 1,
    ) -> PlannerResult:
        validation = self.schema_guard.validate(action, tools)
        if not validation.valid:
            return PlannerResult(
                next_action=self._response_for_schema_failure(validation.errors),
                metrics=metrics,
                internal_calls=max(metrics.num_calls, internal_calls_floor),
                debug={**debug, "schema_errors": validation.errors},
            )

        action = validation.normalized_action or action
        failure_guard_warnings: list[str] = []
        failure_guard_evidence: dict[str, Any] = {}

        def apply_failure_guard(current_action: dict[str, Any], stage: str) -> dict[str, Any]:
            nonlocal failure_guard_evidence
            failure_guard_result = self.failure_guard.apply(
                action=current_action,
                messages=messages,
                tools=tools,
            )
            if failure_guard_result.warnings:
                failure_guard_warnings.extend(failure_guard_result.warnings)
                if failure_guard_result.evidence:
                    failure_guard_evidence = failure_guard_result.evidence
            if not failure_guard_result.action:
                return current_action
            guarded_validation = self.schema_guard.validate(
                failure_guard_result.action,
                tools,
            )
            if guarded_validation.valid:
                return guarded_validation.normalized_action or failure_guard_result.action
            failure_guard_warnings.append(
                f"{stage} failure guard replacement failed schema validation: "
                + "; ".join(guarded_validation.errors)
            )
            return current_action

        action = apply_failure_guard(action, "initial")
        policy_result = self.policy_guard.apply(action=action, tools=tools, messages=messages)
        if not policy_result.allowed and policy_result.replacement_action:
            action = policy_result.replacement_action
            action = apply_failure_guard(action, "policy")
        task_guard_result = self.task_guard.postprocess(
            action=action,
            tools=tools,
            messages=messages,
        )
        if task_guard_result.action:
            validation = self.schema_guard.validate(task_guard_result.action, tools)
            if validation.valid:
                action = validation.normalized_action or task_guard_result.action
                action = apply_failure_guard(action, "task guard")
            else:
                task_guard_result.warnings.append(
                    "task guard replacement failed schema validation: "
                    + "; ".join(validation.errors)
                )
        plan_state_result = self.plan_state.postprocess_action(context_id, action, tools)
        if plan_state_result.action:
            validation = self.schema_guard.validate(plan_state_result.action, tools)
            if validation.valid:
                action = validation.normalized_action or plan_state_result.action
                action = apply_failure_guard(action, "plan state")
            else:
                plan_state_result.warnings.append(
                    "plan state replacement failed schema validation: "
                    + "; ".join(validation.errors)
                )

        skill_tool_warnings: list[str] = []
        skill_response_warnings: list[str] = []
        completion_verifier_warnings: list[str] = []
        completion_verifier_evidence: dict[str, Any] = {}
        completion_source_action = str(action.get("action") or "")
        completion_verdict = self.completion_verifier.verify(
            action=action,
            messages=messages,
            tools=tools,
        )
        if completion_verdict.warnings:
            completion_verifier_warnings.extend(completion_verdict.warnings)
            completion_verifier_evidence = completion_verdict.evidence
            if completion_source_action == "tool_calls":
                skill_tool_warnings.extend(completion_verdict.warnings)
            elif completion_source_action == "respond":
                skill_response_warnings.extend(completion_verdict.warnings)
        if completion_verdict.action:
            validation = self.schema_guard.validate(completion_verdict.action, tools)
            if validation.valid:
                action = validation.normalized_action or completion_verdict.action
                guarded = self.task_guard.postprocess(
                    action=action,
                    tools=tools,
                    messages=messages,
                )
                if guarded.action:
                    guarded_validation = self.schema_guard.validate(guarded.action, tools)
                    if guarded_validation.valid:
                        action = guarded_validation.normalized_action or guarded.action
                        task_guard_result.warnings.extend(guarded.warnings)
                    else:
                        task_guard_result.warnings.append(
                            "completion verifier replacement failed task guard schema validation: "
                            + "; ".join(guarded_validation.errors)
                        )
                gated_plan_state = self.plan_state.postprocess_action(context_id, action, tools)
                if gated_plan_state.action:
                    gated_validation = self.schema_guard.validate(gated_plan_state.action, tools)
                    if gated_validation.valid:
                        action = gated_validation.normalized_action or gated_plan_state.action
                        plan_state_result.warnings.extend(gated_plan_state.warnings)
                        action = apply_failure_guard(action, "completion verifier")
                    else:
                        plan_state_result.warnings.append(
                            "completion verifier replacement failed plan state schema validation: "
                            + "; ".join(gated_validation.errors)
                        )
            else:
                completion_verifier_warnings.append(
                    "completion verifier replacement failed schema validation: "
                    + "; ".join(validation.errors)
                )
        debug["policy_warnings"] = policy_result.warnings
        debug["task_guard_warnings"] = task_guard_result.warnings
        debug["plan_state_warnings"] = plan_state_result.warnings
        debug["failure_guard_warnings"] = failure_guard_warnings
        debug["failure_guard_evidence"] = failure_guard_evidence
        debug["skill_tool_warnings"] = skill_tool_warnings
        debug["skill_response_warnings"] = skill_response_warnings
        debug["completion_verifier_warnings"] = completion_verifier_warnings
        debug["completion_verifier_evidence"] = completion_verifier_evidence
        debug["plan_state"] = self.plan_state.snapshot(context_id)
        return PlannerResult(
            next_action=action,
            metrics=metrics,
            internal_calls=max(metrics.num_calls, internal_calls_floor),
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

    def _remaining_timeout(self, deadline: float) -> float:
        return max(1.0, min(self.timeout_seconds, deadline - time.perf_counter()))

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

    def _training_hints_for(
        self,
        *,
        messages: list[dict[str, Any]],
        completed_tools: list[str],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        user_text = " ".join(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
            and str(message.get("content") or "").strip().lower() != "###stop###"
        )
        available_tools = {_tool_name(tool) for tool in tools if _tool_name(tool)}
        return self.training_insights.hints_for(
            user_text=user_text,
            completed_tools=completed_tools,
            available_tools=available_tools,
        )

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
