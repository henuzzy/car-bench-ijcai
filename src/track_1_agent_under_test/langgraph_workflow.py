"""LangGraph orchestration for the Track 1 planner.

The existing Track1Planner owns the domain logic: guards, task memory,
PlanState, skill recipes, LLM calls, and final validation.  This module owns
the control flow.  Each turn is represented as a LangGraph state graph so the
planner/critic/executor loop is explicit and testable.
"""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

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
    "fallback_gate",
    "fallback",
    "execute_plan",
    "revise",
]


class PlannerGraphState(TypedDict, total=False):
    context_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    ctx_logger: Any
    deadline: float
    metrics: LLMCallMetrics
    debug: dict[str, Any]
    action: dict[str, Any]
    result: PlannerResult
    internal_calls_floor: int
    approved_plan: ApprovedPlan
    critic: CriticVerdict
    critic_feedback: str
    critic_revisions: int
    plan_attempts: int
    planning_error: str


class Track1LangGraphWorkflow:
    """One compiled LangGraph workflow shared by a Track1Planner instance."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.graph = self._build_graph()

    def invoke(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger: Any,
    ) -> PlannerResult:
        final_state = self.graph.invoke(
            {
                "context_id": context_id,
                "messages": messages,
                "tools": tools,
                "ctx_logger": ctx_logger,
            }
        )
        result = final_state.get("result")
        if result is None:
            raise RuntimeError("LangGraph planner completed without a PlannerResult")
        return result

    def _build_graph(self):
        graph = StateGraph(PlannerGraphState)
        graph.add_node("observe_memory", self._observe_memory)
        graph.add_node("stop_gate", self._stop_gate)
        graph.add_node("skill_gate", self._skill_gate)
        graph.add_node("finish_gate", self._finish_gate)
        graph.add_node("preempt_gate", self._preempt_gate)
        graph.add_node("approved_planner", self._approved_planner)
        graph.add_node("plan_critic", self._plan_critic)
        graph.add_node("revision_feedback", self._revision_feedback)
        graph.add_node("execute_plan", self._execute_plan)
        graph.add_node("fallback_gate", self._fallback_gate)
        graph.add_node("native_fallback", self._native_fallback)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("observe_memory")
        graph.add_edge("observe_memory", "stop_gate")
        graph.add_conditional_edges(
            "stop_gate",
            self._route_stop_gate,
            {"end": END, "continue": "skill_gate"},
        )
        graph.add_conditional_edges(
            "skill_gate",
            self._route_action_or_continue,
            {"finalize": "finalize", "continue": "finish_gate"},
        )
        graph.add_conditional_edges(
            "finish_gate",
            self._route_action_or_continue,
            {"finalize": "finalize", "continue": "preempt_gate"},
        )
        graph.add_conditional_edges(
            "preempt_gate",
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
        return graph.compile()

    def _observe_memory(self, state: PlannerGraphState) -> PlannerGraphState:
        planner = self.planner
        context_id = state["context_id"]
        messages = state["messages"]
        tools = state["tools"]
        planner.context_manager.observe_messages(context_id, messages)
        planner.task_memory.observe_messages(context_id, messages)
        planner.plan_state.observe_messages(context_id, messages, tools)
        metrics = LLMCallMetrics()
        return {
            "deadline": time.perf_counter() + planner.turn_budget_seconds,
            "metrics": metrics,
            "critic_feedback": "",
            "critic_revisions": 0,
            "plan_attempts": 0,
            "debug": {
                "planner": "LangGraphTrack1Planner",
                "langgraph": True,
                "graph_nodes": ["observe_memory"],
                "repair_used": False,
                "native_fallback_used": False,
                "plan_state": planner.plan_state.snapshot(context_id),
            },
        }

    def _stop_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        decision = self.planner.task_guard.finish_after_stop_signal(
            messages=state["messages"]
        )
        debug = self._debug_with_node(state, "stop_gate")
        if not decision.action:
            return {"debug": debug}
        return {
            "result": PlannerResult(
                next_action=decision.action,
                metrics=state["metrics"],
                internal_calls=0,
                debug={
                    **debug,
                    "task_guard_warnings": decision.warnings,
                    "terminal_stop_signal": True,
                },
            )
        }

    def _skill_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        decision = self.planner.skill_registry.preempt(
            messages=state["messages"],
            tools=state["tools"],
        )
        debug = self._debug_with_node(state, "skill_gate")
        if not decision.action:
            return {"debug": debug}
        return {
            "action": decision.action,
            "internal_calls_floor": 0,
            "debug": {
                **debug,
                "skill_preempted": True,
                "skill": decision.skill,
                "skill_warnings": decision.warnings,
            },
        }

    def _finish_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        decision = self.planner.task_guard.finish_after_successful_state_change(
            messages=state["messages"]
        )
        debug = self._debug_with_node(state, "finish_gate")
        if not decision.action:
            return {"debug": debug}
        return {
            "action": decision.action,
            "internal_calls_floor": 0,
            "debug": {
                **debug,
                "task_guard_warnings": decision.warnings,
                "terminal_after_state_change": True,
            },
        }

    def _preempt_gate(self, state: PlannerGraphState) -> PlannerGraphState:
        decision = self.planner.task_guard.preempt(
            messages=state["messages"],
            tools=state["tools"],
        )
        debug = self._debug_with_node(state, "preempt_gate")
        if not decision.action:
            return {"debug": debug}
        return {
            "action": decision.action,
            "debug": {
                **debug,
                "task_guard_warnings": decision.warnings,
                "task_guard_preempted": True,
            },
        }

    def _approved_planner(self, state: PlannerGraphState) -> PlannerGraphState:
        metrics = state["metrics"]
        try:
            plan, plan_metrics = self.planner._run_approved_planner(
                context_id=state["context_id"],
                messages=state["messages"],
                tools=state["tools"],
                metrics=metrics,
                critic_feedback=state.get("critic_feedback", ""),
                deadline=state["deadline"],
                ctx_logger=state["ctx_logger"],
            )
            metrics.add(plan_metrics)
            return {
                "approved_plan": plan,
                "metrics": metrics,
                "plan_attempts": state.get("plan_attempts", 0) + 1,
                "debug": self._debug_with_node(state, "approved_planner"),
            }
        except Exception as exc:
            state["ctx_logger"].warning(
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
            critic, critic_metrics = self.planner._run_plan_critic(
                context_id=state["context_id"],
                messages=state["messages"],
                tools=state["tools"],
                plan=state["approved_plan"],
                deadline=state["deadline"],
                ctx_logger=state["ctx_logger"],
            )
            metrics.add(critic_metrics)
            return {
                "critic": critic,
                "metrics": metrics,
                "debug": self._debug_with_node(state, "plan_critic"),
            }
        except Exception as exc:
            state["ctx_logger"].warning(
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
        state["ctx_logger"].info(
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
            "task_memory": self.planner.task_memory.snapshot(state["context_id"]),
        }
        critic = state.get("critic")
        if plan:
            debug["approved_plan"] = plan.to_dict()
        if critic:
            debug["critic"] = critic.to_dict()
        if plan is None:
            return {
                "planning_error": state.get("planning_error") or "Approved planner did not return a plan",
                "debug": debug,
            }
        action = self.planner._execute_approved_plan(plan, state["tools"])
        if not action:
            state["ctx_logger"].warning(
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
            ctx_logger=state["ctx_logger"],
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
        result = self.planner._finalize_visible_action(
            context_id=state["context_id"],
            action=state["action"],
            tools=state["tools"],
            messages=state["messages"],
            metrics=state["metrics"],
            debug=self._debug_with_node(state, "finalize"),
            internal_calls_floor=state.get("internal_calls_floor", 1),
        )
        return {"result": result}

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
