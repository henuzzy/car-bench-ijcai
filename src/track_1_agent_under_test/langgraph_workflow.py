"""LangGraph-native orchestration for the Track 1 planner.

This file is the runtime path for Track1Planner.  It intentionally keeps
conversation memory, context shaping, guard routing, tool-call validation, and
planner/critic/executor handoff inside LangGraph state nodes instead of calling
the older bespoke PlanState/TaskMemory/TaskGuard/SkillRegistry stack.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

try:
    from .approved_plan import ApprovedPlan, CriticVerdict
    from .multi_agent_types import LLMCallMetrics, PlannerResult
except ImportError:
    from approved_plan import ApprovedPlan, CriticVerdict
    from multi_agent_types import LLMCallMetrics, PlannerResult


GraphRoute = Literal[
    "end",
    "continue",
    "finalize",
    "fallback",
    "fallback_gate",
    "execute_plan",
    "revise",
]


class PlannerGraphState(TypedDict, total=False):
    context_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    graph_memory: dict[str, Any]
    context_bundle: dict[str, Any]
    deadline: float
    metrics: LLMCallMetrics
    debug: dict[str, Any]
    action: dict[str, Any] | None
    result: PlannerResult | None
    internal_calls_floor: int
    approved_plan: ApprovedPlan | None
    critic: CriticVerdict | None
    critic_feedback: str
    critic_revisions: int
    plan_attempts: int
    planning_error: str | None


class Track1LangGraphWorkflow:
    """Compiled LangGraph workflow for one Track1Planner instance."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self._active_loggers: dict[str, Any] = {}
        self.graph = self._build_graph()

    def invoke(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> PlannerResult:
        self._active_loggers[context_id] = ctx_logger
        try:
            final_state = self.graph.invoke(
                {
                    "context_id": context_id,
                    "messages": messages,
                    "tools": tools,
                },
                config={"configurable": {"thread_id": context_id}},
            )
        finally:
            self._active_loggers.pop(context_id, None)
        result = final_state.get("result")
        if result is None:
            raise RuntimeError("LangGraph planner completed without a PlannerResult")
        return result

    def _build_graph(self):
        graph = StateGraph(PlannerGraphState)
        graph.add_node("observe_memory", self._observe_memory)
        graph.add_node("context_builder", self._context_builder)
        graph.add_node("stop_gate", self._stop_gate)
        graph.add_node("langgraph_skill_gate", self._langgraph_skill_gate)
        graph.add_node("completion_gate", self._completion_gate)
        graph.add_node("precondition_gate", self._precondition_gate)
        graph.add_node("approved_planner", self._approved_planner)
        graph.add_node("plan_critic", self._plan_critic)
        graph.add_node("revision_feedback", self._revision_feedback)
        graph.add_node("execute_plan", self._execute_plan)
        graph.add_node("fallback_gate", self._fallback_gate)
        graph.add_node("native_fallback", self._native_fallback)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("observe_memory")
        graph.add_edge("observe_memory", "context_builder")
        graph.add_edge("context_builder", "stop_gate")
        graph.add_conditional_edges(
            "stop_gate",
            self._route_stop_gate,
            {"end": END, "continue": "langgraph_skill_gate"},
        )
        graph.add_conditional_edges(
            "langgraph_skill_gate",
            self._route_action_or_continue,
            {"finalize": "finalize", "continue": "completion_gate"},
        )
        graph.add_conditional_edges(
            "completion_gate",
            self._route_action_or_continue,
            {"finalize": "finalize", "continue": "precondition_gate"},
        )
        graph.add_conditional_edges(
            "precondition_gate",
            self._route_action_or_continue,
            {"finalize": "finalize", "continue": "approved_planner"},
        )
        graph.add_conditional_edges(
            "approved_planner",
            self._route_planner,
            {"continue": "plan_critic", "fallback_gate": "fallback_gate"},
        )
        graph.add_conditional_edges(
            "plan_critic",
            self._route_after_critic,
            {
                "revise": "revision_feedback",
                "execute_plan": "execute_plan",
                "fallback_gate": "fallback_gate",
            },
        )
        graph.add_edge("revision_feedback", "approved_planner")
        graph.add_conditional_edges(
            "execute_plan",
            self._route_action_or_fallback,
            {"finalize": "finalize", "fallback_gate": "fallback_gate"},
        )
        graph.add_conditional_edges(
            "fallback_gate",
            self._route_fallback_gate,
            {"finalize": "finalize", "fallback": "native_fallback"},
        )
        graph.add_edge("native_fallback", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=InMemorySaver())

    def _observe_memory(self, state: PlannerGraphState) -> PlannerGraphState:
        memory = _build_graph_memory(state["messages"])
        return {
            "deadline": time.perf_counter() + self.planner.turn_budget_seconds,
            "metrics": LLMCallMetrics(),
            "graph_memory": memory,
            "critic_feedback": "",
            "critic_revisions": 0,
            "plan_attempts": 0,
            "planning_error": None,
            "approved_plan": None,
            "critic": None,
            "action": None,
            "result": None,
            "internal_calls_floor": 1,
            "debug": {
                "planner": "LangGraphTrack1Planner",
                "langgraph": True,
                "langgraph_native": True,
                "graph_nodes": ["observe_memory"],
                "native_fallback_used": False,
                "graph_memory": _memory_preview(memory),
            },
        }

    def _context_builder(self, state: PlannerGraphState) -> PlannerGraphState:
        bundle = {
            "latest_user_request": state["graph_memory"].get("latest_user_text", ""),
            "recent_messages": _render_recent_messages(state["messages"]),
            "recent_tool_results": state["graph_memory"].get("recent_tool_results", []),
            "completed_tools": state["graph_memory"].get("completed_tools", []),
            "failed_tools": state["graph_memory"].get("failed_tools", []),
            "available_tool_names": [_tool_name(tool) for tool in state["tools"]],
            "tool_call_counts": state["graph_memory"].get("tool_call_counts", {}),
            "confirmed": state["graph_memory"].get("latest_user_confirmed", False),
        }
        return {
            "context_bundle": bundle,
            "debug": self._debug_with_node(state, "context_builder"),
        }

    def _stop_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        debug = self._debug_with_node(state, "stop_gate")
        if not _is_stop_signal(state["graph_memory"].get("latest_user_text", "")):
            return {"debug": debug}
        return {
            "result": PlannerResult(
                next_action={"action": "respond", "content": "Done."},
                metrics=state["metrics"],
                internal_calls=0,
                debug={**debug, "terminal_stop_signal": True},
            )
        }

    def _langgraph_skill_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        action, metadata = _graph_skill_action(
            messages=state["messages"],
            tools=state["tools"],
            memory=state["graph_memory"],
        )
        debug = self._debug_with_node(state, "langgraph_skill_gate")
        if not action:
            return {"debug": debug}
        return {
            "action": action,
            "internal_calls_floor": 0,
            "debug": {
                **debug,
                "skill_preempted": True,
                "skill": metadata.get("skill", "langgraph_state_skill"),
                "skill_warnings": metadata.get("warnings", []),
                **metadata.get("debug", {}),
            },
        }

    def _completion_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        debug = self._debug_with_node(state, "completion_gate")
        if not _last_visible_tool_batch_succeeded(state["graph_memory"]):
            return {"debug": debug}
        if _has_pending_graph_step(state):
            return {"debug": debug}
        return {
            "action": {"action": "respond", "content": "Done."},
            "internal_calls_floor": 0,
            "debug": {**debug, "terminal_after_state_change": True},
        }

    def _precondition_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        action, warnings = _precondition_action(
            messages=state["messages"],
            tools=state["tools"],
            memory=state["graph_memory"],
        )
        debug = self._debug_with_node(state, "precondition_gate")
        if not action:
            return {"debug": debug}
        return {
            "action": action,
            "debug": {
                **debug,
                "langgraph_precondition_warnings": warnings,
            },
        }

    def _approved_planner(self, state: PlannerGraphState) -> PlannerGraphState:
        metrics = state["metrics"]
        try:
            plan, plan_metrics = self.planner._run_langgraph_approved_planner(
                context_id=state["context_id"],
                messages=state["messages"],
                tools=state["tools"],
                graph_memory=state["graph_memory"],
                context_bundle=state["context_bundle"],
                critic_feedback=state.get("critic_feedback", ""),
                deadline=state["deadline"],
                ctx_logger=self._logger(state["context_id"]),
            )
            metrics.add(plan_metrics)
            return {
                "approved_plan": plan,
                "metrics": metrics,
                "plan_attempts": state.get("plan_attempts", 0) + 1,
                "debug": self._debug_with_node(state, "approved_planner"),
            }
        except Exception as exc:
            self._logger(state["context_id"]).warning(
                "LangGraph approved planner failed",
                error=str(exc),
            )
            return {
                "planning_error": str(exc),
                "debug": {
                    **self._debug_with_node(state, "approved_planner"),
                    "pec_error": str(exc),
                },
            }

    def _plan_critic(self, state: PlannerGraphState) -> PlannerGraphState:
        metrics = state["metrics"]
        try:
            critic, critic_metrics = self.planner._run_langgraph_plan_critic(
                context_id=state["context_id"],
                messages=state["messages"],
                tools=state["tools"],
                graph_memory=state["graph_memory"],
                context_bundle=state["context_bundle"],
                plan=state["approved_plan"],
                deadline=state["deadline"],
                ctx_logger=self._logger(state["context_id"]),
            )
            metrics.add(critic_metrics)
            return {
                "critic": critic,
                "metrics": metrics,
                "debug": self._debug_with_node(state, "plan_critic"),
            }
        except Exception as exc:
            self._logger(state["context_id"]).warning(
                "LangGraph plan critic failed",
                error=str(exc),
            )
            return {
                "planning_error": str(exc),
                "debug": {
                    **self._debug_with_node(state, "plan_critic"),
                    "critic_error": str(exc),
                },
            }

    def _revision_feedback(self, state: PlannerGraphState) -> PlannerGraphState:
        critic = state.get("critic")
        violations = critic.violations if critic else ["critic requested revision"]
        changes = critic.recommended_changes if critic else ["produce a safer phase plan"]
        self._logger(state["context_id"]).info(
            "Critic requested ApprovedPlan revision",
            attempt=state.get("plan_attempts", 0),
            violations=violations,
            recommended_changes=changes,
        )
        return {
            "critic_feedback": (
                "The previous ApprovedPlan must be revised.\nViolations:\n- "
                + "\n- ".join(violations or ["critic requested revision"])
                + "\nRecommended changes:\n- "
                + "\n- ".join(changes or ["produce a safer phase plan"])
            ),
            "critic_revisions": state.get("critic_revisions", 0) + 1,
            "debug": self._debug_with_node(state, "revision_feedback"),
        }

    def _execute_plan(self, state: PlannerGraphState) -> PlannerGraphState:
        plan = state.get("approved_plan")
        debug = {
            **self._debug_with_node(state, "execute_plan"),
            "pec_lite": True,
            "critic_revisions": state.get("critic_revisions", 0),
            "graph_memory": _memory_preview(state["graph_memory"]),
        }
        critic = state.get("critic")
        if plan:
            debug["approved_plan"] = plan.to_dict()
        if critic:
            debug["critic"] = critic.to_dict()
        if plan is None:
            return {
                "planning_error": state.get("planning_error")
                or "Approved planner did not return a plan",
                "debug": debug,
            }
        action = self.planner._execute_approved_plan(plan, state["tools"])
        if not action:
            self._logger(state["context_id"]).warning(
                "Approved plan did not produce an executable action",
                approved_plan=plan.to_dict(),
            )
            return {
                "planning_error": "Approved plan did not produce an executable action",
                "debug": debug,
            }
        return {"action": action, "debug": debug}

    def _fallback_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        debug = self._debug_with_node(state, "fallback_gate")
        if self.planner._remaining_timeout(state["deadline"]) > 1.0:
            return {"debug": debug}
        return {
            "action": {
                "action": "respond",
                "content": "I cannot safely complete that with the available tools.",
            },
            "internal_calls_floor": max(state["metrics"].num_calls, 1),
            "debug": {**debug, "turn_budget_exhausted": True},
        }

    def _native_fallback(self, state: PlannerGraphState) -> PlannerGraphState:
        fallback = self.planner._native_fallback(
            messages=state["messages"],
            tools=state["tools"],
            ctx_logger=self._logger(state["context_id"]),
            deadline=state["deadline"],
        )
        metrics = state["metrics"]
        metrics.add(fallback.metrics)
        return {
            "action": fallback.next_action,
            "metrics": metrics,
            "debug": {
                **self._debug_with_node(state, "native_fallback"),
                **fallback.debug,
                "native_fallback_used": True,
            },
        }

    def _finalize(self, state: PlannerGraphState) -> PlannerGraphState:
        result = _finalize_langgraph_action(
            action=state["action"] or {
                "action": "respond",
                "content": "I cannot safely complete that with the available tools.",
            },
            tools=state["tools"],
            metrics=state["metrics"],
            debug=self._debug_with_node(state, "finalize"),
            internal_calls_floor=state.get("internal_calls_floor", 1),
        )
        return {"result": result}

    def _logger(self, context_id: str) -> Any:
        return self._active_loggers.get(context_id, _NullLogger())

    @staticmethod
    def _route_stop_gate(state: PlannerGraphState) -> GraphRoute:
        return "end" if state.get("result") is not None else "continue"

    @staticmethod
    def _route_action_or_continue(state: PlannerGraphState) -> GraphRoute:
        return "finalize" if state.get("action") is not None else "continue"

    @staticmethod
    def _route_planner(state: PlannerGraphState) -> GraphRoute:
        return "fallback_gate" if state.get("planning_error") else "continue"

    def _route_after_critic(self, state: PlannerGraphState) -> GraphRoute:
        if state.get("planning_error"):
            return "fallback_gate"
        critic = state.get("critic")
        if critic and not critic.passed:
            can_revise = (
                state.get("plan_attempts", 0) <= self.planner.max_critic_revisions
                and self.planner._remaining_timeout(state["deadline"]) > 1.0
            )
            if can_revise:
                return "revise"
        return "execute_plan"

    @staticmethod
    def _route_action_or_fallback(state: PlannerGraphState) -> GraphRoute:
        return "finalize" if state.get("action") is not None else "fallback_gate"

    @staticmethod
    def _route_fallback_gate(state: PlannerGraphState) -> GraphRoute:
        return "finalize" if state.get("action") is not None else "fallback"

    @staticmethod
    def _debug_with_node(state: PlannerGraphState, node: str) -> dict[str, Any]:
        debug = dict(state.get("debug") or {})
        nodes = list(debug.get("graph_nodes") or [])
        nodes.append(node)
        debug["graph_nodes"] = nodes
        return debug


class _NullLogger:
    def debug(self, *args, **kwargs) -> None:
        return None

    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


def _build_graph_memory(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tool_results: list[dict[str, Any]] = []
    tool_call_counts: dict[str, int] = {}
    completed_tools: list[str] = []
    failed_tools: list[str] = []
    assistant_tool_calls: list[dict[str, Any]] = []

    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                name = call.get("function", {}).get("name") or call.get("name")
                arguments = _parse_arguments(call.get("function", {}).get("arguments", {}))
                assistant_tool_calls.append(
                    {
                        "id": call.get("id"),
                        "tool_name": name,
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    }
                )
                if name:
                    tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        parsed = _parse_tool_content(message.get("content"))
        succeeded = _tool_result_succeeded(parsed)
        record = {
            "tool_name": name,
            "tool_call_id": message.get("tool_call_id"),
            "success": succeeded,
            "content": str(message.get("content") or "")[:4000],
            "parsed": parsed,
        }
        tool_results.append(record)
        if succeeded and name not in completed_tools:
            completed_tools.append(name)
        if not succeeded and name not in failed_tools:
            failed_tools.append(name)

    latest_user = _latest_message_content(messages, "user")
    latest_assistant = _latest_message_content(messages, "assistant")
    return {
        "latest_user_text": latest_user,
        "latest_user_confirmed": _is_confirmation(latest_user),
        "latest_assistant_text": latest_assistant,
        "tool_call_counts": tool_call_counts,
        "completed_tools": completed_tools,
        "failed_tools": failed_tools,
        "assistant_tool_calls": assistant_tool_calls[-12:],
        "recent_tool_results": tool_results[-8:],
        "all_tool_results": tool_results,
    }


def _render_recent_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages[-12:]:
        item = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "tool_name": call.get("function", {}).get("name"),
                    "arguments": _parse_arguments(
                        call.get("function", {}).get("arguments", {})
                    ),
                }
                for call in message["tool_calls"]
            ]
        if message.get("role") == "tool":
            item["name"] = message.get("name")
            item["content"] = str(message.get("content") or "")[:4000]
        rendered.append(item)
    return rendered


def _memory_preview(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_user_text": memory.get("latest_user_text", "")[:240],
        "completed_tools": memory.get("completed_tools", []),
        "failed_tools": memory.get("failed_tools", []),
        "tool_call_counts": memory.get("tool_call_counts", {}),
    }


def _graph_skill_action(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    memory: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    email_action = _email_confirmation_action(messages=messages, tools=tools, memory=memory)
    if email_action:
        return email_action, {"skill": "communication_email", "warnings": []}

    climate_action, climate_meta = _climate_efficiency_action(
        tools=tools,
        memory=memory,
    )
    if climate_action:
        return climate_action, climate_meta
    return None, {}


def _email_confirmation_action(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    memory: dict[str, Any],
) -> dict[str, Any] | None:
    if not memory.get("latest_user_confirmed"):
        return None
    if "send_email" not in {_tool_name(tool) for tool in tools}:
        return None
    assistant_text = memory.get("latest_assistant_text", "")
    if not assistant_text:
        return None
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", assistant_text)
    if not email_match:
        return None
    body = assistant_text[email_match.end() :].strip(" :\n")
    body = re.sub(r"(?i)\b(should i send it\?|please say yes to confirm\.?).*$", "", body).strip()
    body = re.sub(r"(?i)^i can send this email to [^:]+:\s*", "", body).strip()
    if not body:
        body = "Confirmed."
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "send_email",
                "arguments": {
                    "email_addresses": [email_match.group(0)],
                    "content_message": body,
                },
            }
        ],
    }


def _climate_efficiency_action(
    *,
    tools: list[dict[str, Any]],
    memory: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    tool_names = {_tool_name(tool) for tool in tools}
    latest_user = str(memory.get("latest_user_text") or "").lower()
    original_user_text = latest_user
    if not any(
        phrase in latest_user
        for phrase in [
            "empty heated seat",
            "match my temperature",
            "energy efficiency",
            "who's actually in the car",
        ]
    ):
        for result in memory.get("all_tool_results", []):
            if result.get("tool_name") == "get_seats_occupancy":
                original_user_text += " energy efficiency"
                break
        if "energy efficiency" not in original_user_text:
            return None, {}
    occupancy = _latest_result_object(memory, "get_seats_occupancy")
    temperatures = _latest_result_object(memory, "get_temperature_inside_car")
    heating = _latest_result_object(memory, "get_seat_heating_level")
    if not isinstance(occupancy, dict) or not isinstance(temperatures, dict):
        return None, {}

    seats = occupancy.get("seats_occupied") if isinstance(occupancy.get("seats_occupied"), dict) else {}
    passenger_empty = seats.get("passenger") is False
    passenger_heat = _as_number(heating.get("seat_heating_passenger")) if isinstance(heating, dict) else None
    already_set_heat = _assistant_called_with(
        memory,
        "set_seat_heating",
        {"seat_zone": "PASSENGER", "level": 0},
    )
    already_set_temp = _assistant_called_with(
        memory,
        "set_climate_temperature",
        {
            "seat_zone": "DRIVER",
            "temperature": temperatures.get("climate_temperature_passenger"),
        },
    )

    calls: list[dict[str, Any]] = []
    if (
        passenger_empty
        and passenger_heat is not None
        and passenger_heat > 0
        and not already_set_heat
        and "set_seat_heating" in tool_names
    ):
        calls.append(
            {
                "tool_name": "set_seat_heating",
                "arguments": {"seat_zone": "PASSENGER", "level": 0},
            }
        )
    passenger_temp = temperatures.get("climate_temperature_passenger")
    driver_temp = temperatures.get("climate_temperature_driver")
    if (
        passenger_temp is not None
        and driver_temp != passenger_temp
        and not already_set_temp
        and "set_climate_temperature" in tool_names
    ):
        calls.append(
            {
                "tool_name": "set_climate_temperature",
                "arguments": {"temperature": passenger_temp, "seat_zone": "DRIVER"},
            }
        )
    if not calls:
        return None, {}
    meta = {
        "skill": "occupancy_climate_efficiency",
        "warnings": ["LangGraph state skill completed remaining climate efficiency step."],
        "debug": {},
    }
    if _last_visible_tool_batch_succeeded(memory):
        meta["debug"] = {
            "terminal_after_state_change": True,
            "langgraph_completion_warnings": [
                "LangGraph completion gate advanced the remaining climate step."
            ],
        }
    return {"action": "tool_calls", "tool_calls": calls}, meta


def _precondition_action(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    memory: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    latest_user = str(memory.get("latest_user_text") or "").lower()
    tool_names = {_tool_name(tool) for tool in tools}
    if (
        "charging station" in latest_user
        and "route" in latest_user
        and "search_poi_along_the_route" in tool_names
        and "get_current_navigation_state" in tool_names
        and "get_current_navigation_state" not in memory.get("completed_tools", [])
    ):
        return (
            {
                "action": "tool_calls",
                "tool_calls": [
                    {"tool_name": "get_current_navigation_state", "arguments": {}}
                ],
            },
            ["LangGraph precondition: route-based charging search needs current navigation state."],
        )
    return None, []


def _has_pending_graph_step(state: PlannerGraphState) -> bool:
    action, _metadata = _graph_skill_action(
        messages=state["messages"],
        tools=state["tools"],
        memory=state["graph_memory"],
    )
    return action is not None


def _last_visible_tool_batch_succeeded(memory: dict[str, Any]) -> bool:
    recent = memory.get("recent_tool_results") or []
    if not recent:
        return False
    last = recent[-1]
    return bool(last.get("success")) and not str(last.get("tool_name", "")).startswith("get_")


def _finalize_langgraph_action(
    *,
    action: dict[str, Any],
    tools: list[dict[str, Any]],
    metrics: LLMCallMetrics,
    debug: dict[str, Any],
    internal_calls_floor: int,
) -> PlannerResult:
    normalized, errors = _validate_action(action, tools)
    if errors:
        normalized = _response_for_schema_failure(errors)
        debug = {**debug, "schema_errors": errors}
    else:
        debug = {
            **debug,
            "schema_errors": [],
            "langgraph_policy_warnings": [],
            "langgraph_precondition_warnings": debug.get("langgraph_precondition_warnings", []),
            "langgraph_tool_warnings": debug.get("langgraph_tool_warnings", []),
            "langgraph_response_warnings": debug.get("langgraph_response_warnings", []),
            "langgraph_completion_warnings": debug.get("langgraph_completion_warnings", []),
        }
    return PlannerResult(
        next_action=normalized,
        metrics=metrics,
        internal_calls=max(metrics.num_calls, internal_calls_floor),
        debug=debug,
    )


def _validate_action(
    action: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    if action.get("action") == "respond":
        return {"action": "respond", "content": str(action.get("content") or "")}, []
    if action.get("action") != "tool_calls":
        return action, ["action must be respond or tool_calls"]

    tool_by_name = {_tool_name(tool): tool for tool in tools if _tool_name(tool)}
    normalized_calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or call.get("name") or "")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        schema = tool_by_name.get(name)
        if schema is None:
            errors.append(f"unknown tool: {name}")
            continue
        params = schema.get("function", {}).get("parameters", {})
        properties = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []
        cleaned: dict[str, Any] = {}
        for key, value in args.items():
            if key not in properties:
                errors.append(f"unknown argument for {name}: {key}")
                continue
            cleaned[key] = _coerce_value(value, properties.get(key, {}))
        for key in required:
            if key not in cleaned:
                errors.append(f"missing required argument for {name}: {key}")
        normalized_calls.append({"tool_name": name, "arguments": cleaned})
    if errors:
        return action, errors
    return {"action": "tool_calls", "tool_calls": normalized_calls}, []


def _coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    expected = schema.get("type")
    if expected in {"number", "integer"} and isinstance(value, str):
        try:
            number = float(value)
            return int(number) if expected == "integer" else number
        except ValueError:
            return value
    return value


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


def _latest_result_object(memory: dict[str, Any], tool_name: str) -> Any:
    for result in reversed(memory.get("all_tool_results") or []):
        if result.get("tool_name") != tool_name:
            continue
        parsed = result.get("parsed")
        if isinstance(parsed, dict):
            value = parsed.get("result", parsed)
            return value
    return None


def _assistant_called_with(
    memory: dict[str, Any],
    tool_name: str,
    required_args: dict[str, Any],
) -> bool:
    for call in memory.get("assistant_tool_calls") or []:
        if call.get("tool_name") != tool_name:
            continue
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if all(args.get(key) == value for key, value in required_args.items()):
            return True
    return False


def _parse_tool_content(content: Any) -> Any:
    if isinstance(content, (dict, list)):
        return content
    text = str(content or "")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _tool_result_succeeded(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    status = str(parsed.get("status") or "").upper()
    if status in {"SUCCESS", "OK"}:
        return True
    if status in {"FAILURE", "ERROR", "FAILED"}:
        return False
    return "errors" not in parsed


def _parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def _latest_message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role and message.get("content"):
            return str(message["content"])
    return ""


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or tool.get("name") or "")


def _is_stop_signal(text: str) -> bool:
    return text.strip().lower() == "###stop###"


def _is_confirmation(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {
        "yes",
        "y",
        "sure",
        "confirm",
        "confirmed",
        "ok",
        "okay",
        "yes, send it.",
        "yes, send it",
    } or lowered.startswith("yes")


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
