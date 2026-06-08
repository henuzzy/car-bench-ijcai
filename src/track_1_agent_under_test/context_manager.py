"""Context construction and compression for the Track 1 planner harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


MODEL_CONTEXT_WINDOW = 200_000
COMPACT_MAX_OUTPUT_TOKENS = 20_000
EFFECTIVE_INPUT_WINDOW = MODEL_CONTEXT_WINDOW - COMPACT_MAX_OUTPUT_TOKENS
AUTOCOMPACT_BUFFER = 13_000
WARNING_BUFFER = 20_000
MANUAL_COMPACT_BUFFER = 3_000
WARNING_THRESHOLD = EFFECTIVE_INPUT_WINDOW - AUTOCOMPACT_BUFFER - WARNING_BUFFER
AUTO_COMPACT_THRESHOLD = EFFECTIVE_INPUT_WINDOW - AUTOCOMPACT_BUFFER
BLOCKING_LIMIT = EFFECTIVE_INPUT_WINDOW - MANUAL_COMPACT_BUFFER

TOOL_SCHEMA_FILTERING_TRIGGER = MODEL_CONTEXT_WINDOW // 10
SINGLE_TOOL_RESULT_CHAR_LIMIT = 50_000
AGGREGATE_TOOL_RESULTS_CHAR_LIMIT = 200_000
RECENT_RAW_TOOL_RESULTS_TO_KEEP = 5
SESSION_COMPACT_MIN_TOKENS = 10_000
SESSION_COMPACT_MAX_TOKENS = 40_000
SESSION_COMPACT_MIN_TEXT_MESSAGES = 5
MODEL_SUMMARY_TARGET_TOKENS = 40_000
MAX_ARCHIVED_TOOL_RESULTS_PER_CONTEXT = 64
MAX_ARCHIVED_TOOL_RESULT_CHARS = 500_000


@dataclass
class ContextMemory:
    """Per-conversation compression state retained outside model context."""

    compact_summary: str = ""
    confirmed_facts: list[str] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)
    archived_tool_results: dict[str, str] = field(default_factory=dict)
    compact_count: int = 0


class ContextManager:
    """Builds compact, subagent-specific context bundles."""

    def __init__(self) -> None:
        self._memory_by_context: dict[str, ContextMemory] = {}

    def reset(self, context_id: str) -> None:
        self._memory_by_context.pop(context_id, None)

    def observe_messages(self, context_id: str, messages: list[dict[str, Any]]) -> None:
        memory = self._memory_by_context.setdefault(context_id, ContextMemory())
        latest_user = _latest_message_content(messages, "user")
        if not latest_user:
            return
        lowered = latest_user.strip().lower()
        if lowered in {"yes", "y", "sure", "confirm", "confirmed", "ok", "okay"}:
            _append_unique(memory.confirmed_facts, "The user gave explicit confirmation.")
        elif lowered in {"no", "nope", "cancel", "stop"}:
            _append_unique(memory.confirmed_facts, "The user rejected or cancelled the pending action.")

    def build_bundle(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        subagent_name: str,
        subagent_policy: str,
        relevant_tools: list[dict[str, Any]],
        summary_callback: Callable[[dict[str, Any]], str] | None = None,
    ) -> dict[str, Any]:
        memory = self._memory_by_context.setdefault(context_id, ContextMemory())
        rendered_messages = self._render_messages_for_model(
            context_id=context_id,
            messages=messages,
            memory=memory,
        )
        token_estimate = estimate_tokens(
            {
                "messages": rendered_messages,
                "tools": relevant_tools,
                "summary": memory.compact_summary,
            }
        )

        if token_estimate >= WARNING_THRESHOLD:
            rendered_messages = self._fold_old_tool_results(rendered_messages)
            token_estimate = estimate_tokens(
                {
                    "messages": rendered_messages,
                    "tools": relevant_tools,
                    "summary": memory.compact_summary,
                }
            )

        if token_estimate >= AUTO_COMPACT_THRESHOLD:
            rendered_messages = self._fold_old_dialogue(
                rendered_messages,
                memory,
            )
            token_estimate = estimate_tokens(
                {
                    "messages": rendered_messages,
                    "tools": relevant_tools,
                    "summary": memory.compact_summary,
                }
            )

        if token_estimate >= BLOCKING_LIMIT:
            summary = self._summarize_with_model_or_fallback(
                rendered_messages=rendered_messages,
                memory=memory,
                summary_callback=summary_callback,
            )
            memory.compact_summary = summary
            memory.compact_count += 1
            rendered_messages = self._tail_messages(rendered_messages)
            token_estimate = estimate_tokens(
                {
                    "messages": rendered_messages,
                    "tools": relevant_tools,
                    "summary": memory.compact_summary,
                }
            )

        latest_user = _latest_message_content(messages, "user")
        recent_tool_facts = _recent_tool_messages(rendered_messages, limit=5)

        return {
            "subagent_name": subagent_name,
            "latest_user_request": latest_user,
            "compact_summary": memory.compact_summary,
            "confirmed_facts": memory.confirmed_facts,
            "known_facts": memory.known_facts,
            "recent_messages": rendered_messages[-12:],
            "recent_tool_facts": recent_tool_facts,
            "subagent_policy": subagent_policy,
            "available_tool_names": [_tool_name(tool) for tool in relevant_tools],
            "all_tool_names": [_tool_name(tool) for tool in tools],
            "compression_state": {
                "model_context_window": MODEL_CONTEXT_WINDOW,
                "warning_threshold": WARNING_THRESHOLD,
                "auto_compact_threshold": AUTO_COMPACT_THRESHOLD,
                "blocking_limit": BLOCKING_LIMIT,
                "estimated_tokens": token_estimate,
                "compact_count": memory.compact_count,
            },
        }

    def _render_messages_for_model(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        memory: ContextMemory,
    ) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        aggregate_tool_chars = 0
        tool_result_index = 0
        for message in messages:
            item: dict[str, Any] = {
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
                item["tool_call_id"] = message.get("tool_call_id")
                if message.get("name"):
                    item["name"] = message.get("name")
                content = str(message.get("content") or "")
                archive_id = f"{context_id}:{message.get('tool_call_id', tool_result_index)}"
                _archive_tool_result(memory, archive_id, content)
                aggregate_tool_chars += len(content)
                if len(content) > SINGLE_TOOL_RESULT_CHAR_LIMIT:
                    item["content"] = _preview_tool_result(content, archive_id)
                elif aggregate_tool_chars > AGGREGATE_TOOL_RESULTS_CHAR_LIMIT:
                    item["content"] = _preview_tool_result(content, archive_id)
                else:
                    item["content"] = content
                tool_result_index += 1
            rendered.append(item)
        return rendered

    def _fold_old_tool_results(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_indexes = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "tool"
        ]
        keep_indexes = set(tool_indexes[-RECENT_RAW_TOOL_RESULTS_TO_KEEP:])
        folded: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if message.get("role") == "tool" and index not in keep_indexes:
                folded.append(
                    {
                        **message,
                        "content": (
                            "[Older tool result folded. Key fact should be read from "
                            "compact_summary or recent_tool_facts if still relevant.]"
                        ),
                    }
                )
            else:
                folded.append(message)
        return folded

    def _fold_old_dialogue(
        self,
        messages: list[dict[str, Any]],
        memory: ContextMemory,
    ) -> list[dict[str, Any]]:
        tail = self._tail_messages(messages)
        head = messages[: max(0, len(messages) - len(tail))]
        if head:
            memory.compact_summary = _merge_summary(
                memory.compact_summary,
                _mechanical_summary(head),
            )
        return tail

    def _tail_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        token_count = 0
        text_messages = 0
        for message in reversed(messages):
            message_tokens = estimate_tokens(message)
            if kept and token_count + message_tokens > SESSION_COMPACT_MAX_TOKENS:
                break
            kept.append(message)
            token_count += message_tokens
            if message.get("role") in {"user", "assistant"} and message.get("content"):
                text_messages += 1
            if (
                token_count >= SESSION_COMPACT_MIN_TOKENS
                and text_messages >= SESSION_COMPACT_MIN_TEXT_MESSAGES
            ):
                break
        return list(reversed(kept)) or messages[-4:]

    def _summarize_with_model_or_fallback(
        self,
        *,
        rendered_messages: list[dict[str, Any]],
        memory: ContextMemory,
        summary_callback: Callable[[dict[str, Any]], str] | None,
    ) -> str:
        payload = {
            "target_tokens": MODEL_SUMMARY_TARGET_TOKENS,
            "required_sections": [
                "Primary Request and Intent",
                "Pending Tasks",
                "Confirmed Facts",
                "Known Vehicle/User/Environment Facts",
                "Performed Actions",
                "Tool Results Summary",
                "Policy Constraints",
                "Current Work",
                "Optional Next Step",
            ],
            "existing_summary": memory.compact_summary,
            "messages": rendered_messages,
        }
        if summary_callback is not None:
            try:
                summary = summary_callback(payload)
                if summary:
                    return summary[: MODEL_SUMMARY_TARGET_TOKENS * 4]
            except Exception:
                pass
        return _merge_summary(memory.compact_summary, _mechanical_summary(rendered_messages))


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, (len(text) + 3) // 4)


def _latest_message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role and message.get("content"):
            return str(message["content"])
    return ""


def _recent_tool_messages(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "tool"][-limit:]


def _preview_tool_result(content: str, archive_id: str) -> str:
    head = content[:24_000]
    tail = content[-4_000:] if len(content) > 28_000 else ""
    return (
        f"[Large tool result truncated; archive_id={archive_id}; "
        f"original_chars={len(content)}]\n{head}"
        + (f"\n...[middle omitted]...\n{tail}" if tail else "")
    )


def _archive_tool_result(memory: ContextMemory, archive_id: str, content: str) -> None:
    memory.archived_tool_results[archive_id] = content[:MAX_ARCHIVED_TOOL_RESULT_CHARS]
    while len(memory.archived_tool_results) > MAX_ARCHIVED_TOOL_RESULTS_PER_CONTEXT:
        oldest_key = next(iter(memory.archived_tool_results), None)
        if oldest_key is None:
            break
        memory.archived_tool_results.pop(oldest_key, None)


def _mechanical_summary(messages: list[dict[str, Any]]) -> str:
    lines = [
        "Primary Request and Intent: preserve the latest CAR-bench user goal.",
        "Pending Tasks: continue any unresolved user request from the transcript.",
        "Confirmed Facts: keep explicit confirmations or refusals from the user.",
        "Known Vehicle/User/Environment Facts:",
    ]
    for message in messages[-20:]:
        role = message.get("role")
        content = str(message.get("content") or "")
        if not content:
            continue
        content = content.replace("\n", " ")
        lines.append(f"- {role}: {content[:500]}")
    lines.extend(
        [
            "Performed Actions: infer only from assistant tool_calls and tool results above.",
            "Tool Results Summary: older tool results may be folded; rely on recent facts.",
            "Policy Constraints: obey CAR-bench system/wiki policy, confirmations, and disambiguation.",
            "Current Work: choose the next safe assistant action.",
            "Optional Next Step: ask the user only when ambiguity cannot be resolved.",
        ]
    )
    return "\n".join(lines)


def _merge_summary(old: str, new: str) -> str:
    if not old:
        return new
    return f"{old}\n\n[Additional compacted context]\n{new}"


def _parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or "")


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)
