"""Track 1 CAR-bench agent under test with a private multi-agent planner."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from a2a.helpers.proto_helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Role
from google.protobuf.json_format import MessageToDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import (
    AVG_LLM_CALL_TIME_MS,
    COMPLETION_TOKENS,
    COST,
    MODEL,
    NUM_LLM_CALLS,
    NUM_PASSES,
    PROMPT_TOKENS,
    THINKING_TOKENS,
    TURN_METRICS_KEY,
)
sys.path.pop(0)

try:
    from .multi_agent_types import LLMCallMetrics
    from .planner import Track1Planner
except ImportError:
    from multi_agent_types import LLMCallMetrics
    from planner import Track1Planner


logger = configure_logger(role="agent_under_test", context="-")

DEFAULT_MAX_CONTEXTS = 64
DEFAULT_CONTEXT_TTL_SECONDS = 6 * 60 * 60


class CARBenchAgentExecutor(AgentExecutor):
    """A2A executor that delegates next-action choice to a private planner."""

    def __init__(
        self,
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
        self.planner = Track1Planner(
            model=model,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            interleaved_thinking=interleaved_thinking,
        )
        self.ctx_id_to_messages: dict[str, list[dict[str, Any]]] = {}
        self.ctx_id_to_tools: dict[str, list[dict[str, Any]]] = {}
        self.ctx_id_to_turn_metrics: dict[str, dict[str, Any]] = {}
        self.ctx_id_last_seen: OrderedDict[str, float] = OrderedDict()
        self.max_contexts = int(os.getenv("AGENT_MAX_CONTEXTS", DEFAULT_MAX_CONTEXTS))
        self.context_ttl_seconds = int(
            os.getenv("AGENT_CONTEXT_TTL_SECONDS", DEFAULT_CONTEXT_TTL_SECONDS)
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        ctx_logger = logger.bind(
            role="agent_under_test",
            context=f"ctx:{context.context_id[:8]}",
        )
        self._mark_context_seen(context.context_id)
        messages = self.ctx_id_to_messages.setdefault(context.context_id, [])
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        try:
            user_message_text, incoming_tool_results, tools_from_message = self._parse_inbound_parts(
                inbound_message=context.message,
                context=context,
                messages=messages,
            )
            if tools_from_message is not None:
                tools = tools_from_message
                self.ctx_id_to_tools[context.context_id] = tools

            self._append_inbound_to_history(
                messages=messages,
                user_message_text=user_message_text,
                incoming_tool_results=incoming_tool_results,
            )

            ctx_logger.info(
                "Received message",
                turn=len(messages),
                has_tools=bool(tools),
                num_tools=len(tools),
                has_tool_results=bool(incoming_tool_results),
                message_preview=(
                    user_message_text[:120]
                    if user_message_text
                    else f"[{len(incoming_tool_results or [])} tool results]"
                ),
            )

            planner_result = self.planner.choose_next_action(
                context_id=context.context_id,
                messages=messages,
                tools=tools,
                ctx_logger=ctx_logger,
            )
            parts, assistant_history = self._build_a2a_response_parts(
                planner_result.next_action
            )
            messages.append(assistant_history)
            self._record_turn_metrics(
                context_id=context.context_id,
                metrics=planner_result.metrics,
                internal_calls=planner_result.internal_calls,
            )

            response_message = new_message(
                parts=parts,
                context_id=context.context_id,
                role=Role.ROLE_AGENT,
            )
            has_tool_calls = bool(assistant_history.get("tool_calls"))
            if not has_tool_calls and context.context_id in self.ctx_id_to_turn_metrics:
                response_message.metadata.update(
                    {
                        TURN_METRICS_KEY: self._public_turn_metrics(
                            self.ctx_id_to_turn_metrics.pop(context.context_id)
                        )
                    }
                )

            ctx_logger.info(
                "Sending response",
                action=planner_result.next_action.get("action"),
                num_parts=len(parts),
                internal_calls=planner_result.internal_calls,
                debug=planner_result.debug,
            )
            await event_queue.enqueue_event(response_message)
            if planner_result.debug.get("terminal_after_state_change") or planner_result.debug.get(
                "terminal_stop_signal"
            ):
                self._drop_context(context.context_id)
            self._prune_context_cache(current_context_id=context.context_id)

        except Exception as exc:
            ctx_logger.error("Agent execution failed", error=str(exc), exc_info=True)
            response_message = new_message(
                parts=[new_text_part(f"Error processing request: {exc}")],
                context_id=context.context_id,
                role=Role.ROLE_AGENT,
            )
            await event_queue.enqueue_event(response_message)
            self._prune_context_cache(current_context_id=context.context_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}").info(
            "Canceling context"
        )
        self.ctx_id_to_messages.pop(context.context_id, None)
        self.ctx_id_to_tools.pop(context.context_id, None)
        self.ctx_id_to_turn_metrics.pop(context.context_id, None)
        self.ctx_id_last_seen.pop(context.context_id, None)
        self.planner.reset(context.context_id)

    def _drop_context(self, context_id: str) -> None:
        self.ctx_id_to_messages.pop(context_id, None)
        self.ctx_id_to_tools.pop(context_id, None)
        self.ctx_id_to_turn_metrics.pop(context_id, None)
        self.ctx_id_last_seen.pop(context_id, None)
        self.planner.reset(context_id)

    def _mark_context_seen(self, context_id: str) -> None:
        self.ctx_id_last_seen.pop(context_id, None)
        self.ctx_id_last_seen[context_id] = time.time()

    def _prune_context_cache(self, *, current_context_id: str) -> None:
        now = time.time()
        evict: list[str] = []
        for context_id, last_seen in list(self.ctx_id_last_seen.items()):
            if context_id == current_context_id:
                continue
            if now - last_seen > self.context_ttl_seconds:
                evict.append(context_id)

        while len(self.ctx_id_last_seen) - len(evict) > self.max_contexts:
            context_id = next(
                (
                    candidate
                    for candidate in self.ctx_id_last_seen
                    if candidate != current_context_id and candidate not in evict
                ),
                None,
            )
            if context_id is None:
                break
            evict.append(context_id)

        for context_id in evict:
            self.ctx_id_to_messages.pop(context_id, None)
            self.ctx_id_to_tools.pop(context_id, None)
            self.ctx_id_to_turn_metrics.pop(context_id, None)
            self.ctx_id_last_seen.pop(context_id, None)
            self.planner.reset(context_id)

    @staticmethod
    def _parse_inbound_parts(
        *,
        inbound_message,
        context: RequestContext,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        user_message_text = None
        incoming_tool_results = None
        tools_from_message = None

        for part in inbound_message.parts:
            content_type = part.WhichOneof("content")
            if content_type == "text":
                text = part.text
                if "System:" in text and "\n\nUser:" in text:
                    system_part, user_part = text.split("\n\nUser:", 1)
                    system_prompt = system_part.replace("System:", "", 1).strip()
                    user_message_text = user_part.strip()
                    if not messages:
                        messages.append({"role": "system", "content": system_prompt})
                else:
                    user_message_text = text
            elif content_type == "data":
                data = MessageToDict(part.data)
                if "tools" in data:
                    tools_from_message = data["tools"]
                elif "tool_results" in data:
                    incoming_tool_results = data["tool_results"]

        if not user_message_text and not incoming_tool_results:
            user_message_text = context.get_user_input()

        return user_message_text, incoming_tool_results, tools_from_message

    @staticmethod
    def _append_inbound_to_history(
        *,
        messages: list[dict[str, Any]],
        user_message_text: str | None,
        incoming_tool_results: list[dict[str, Any]] | None,
    ) -> None:
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            messages.extend(
                _format_tool_results(
                    prev_tool_calls=messages[-1]["tool_calls"],
                    incoming_tool_results=incoming_tool_results,
                    fallback_text=user_message_text,
                )
            )
        else:
            messages.append({"role": "user", "content": user_message_text or ""})

    @staticmethod
    def _build_a2a_response_parts(
        action: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        if action.get("action") == "respond":
            content = str(action.get("content") or "")
            return [new_text_part(content)], {"role": "assistant", "content": content}

        tool_calls_for_history = []
        tool_calls_data = []
        for tool_call in action.get("tool_calls") or []:
            call_id = f"call_{uuid4().hex[:12]}"
            name = tool_call["tool_name"]
            arguments = tool_call.get("arguments") or {}
            tool_calls_for_history.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }
            )
            tool_calls_data.append(ToolCall(tool_name=name, arguments=arguments))

        return [
            new_data_part(
                ToolCallsData(tool_calls=tool_calls_data).model_dump()
            )
        ], {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_for_history,
        }

    def _record_turn_metrics(
        self,
        *,
        context_id: str,
        metrics: LLMCallMetrics,
        internal_calls: int,
    ) -> None:
        turn_metrics = self.ctx_id_to_turn_metrics.setdefault(
            context_id,
            {
                PROMPT_TOKENS: 0,
                COMPLETION_TOKENS: 0,
                THINKING_TOKENS: 0,
                COST: 0.0,
                MODEL: self.model,
                NUM_LLM_CALLS: 0,
                "_total_llm_time_ms": 0.0,
            },
        )
        calls = max(metrics.num_calls, internal_calls)
        if (
            calls <= 0
            and metrics.prompt_tokens == 0
            and metrics.completion_tokens == 0
            and metrics.thinking_tokens == 0
            and metrics.cost == 0.0
        ):
            return
        turn_metrics[PROMPT_TOKENS] += metrics.prompt_tokens
        turn_metrics[COMPLETION_TOKENS] += metrics.completion_tokens
        turn_metrics[THINKING_TOKENS] += metrics.thinking_tokens
        turn_metrics[COST] += metrics.cost
        turn_metrics[NUM_LLM_CALLS] += calls
        turn_metrics["_total_llm_time_ms"] += metrics.elapsed_ms

    @staticmethod
    def _public_turn_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        public = dict(metrics)
        total_time = public.pop("_total_llm_time_ms", 0.0)
        num_calls = public.get(NUM_LLM_CALLS, 0) or 1
        public[AVG_LLM_CALL_TIME_MS] = round(total_time / num_calls, 1)
        public[NUM_PASSES] = num_calls
        return public


def _format_tool_results(
    *,
    prev_tool_calls: list[dict[str, Any]],
    incoming_tool_results: list[dict[str, Any]] | None,
    fallback_text: str | None,
) -> list[dict[str, Any]]:
    if not incoming_tool_results:
        return [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call.get("function", {}).get("name"),
                "content": fallback_text or "",
            }
            for call in prev_tool_calls
        ]

    tool_call_by_name: dict[str, list[dict[str, Any]]] = {}
    for call in prev_tool_calls:
        name = call.get("function", {}).get("name", "")
        tool_call_by_name.setdefault(name, []).append(call)

    tool_results = []
    for result in incoming_tool_results:
        if not isinstance(result, dict):
            result = MessageToDict(result)
        result_name = result.get("tool_name", result.get("toolName", ""))
        matching_calls = tool_call_by_name.get(result_name, [])
        if matching_calls:
            matched_call = matching_calls.pop(0)
            tool_call_id = matched_call["id"]
        else:
            tool_call_id = result.get("tool_call_id", result.get("toolCallId", f"unknown_{result_name}"))
        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": result_name,
                "content": result.get("content", ""),
            }
        )
    return tool_results
