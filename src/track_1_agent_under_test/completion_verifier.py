"""Completion gate for Track 1 visible actions.

The planner may propose a user-facing response, but this verifier treats that
as a proposal.  It checks grounded deterministic task-family skills before the
action is returned to the evaluator, so a premature "Done." can be converted
into the next pending tool call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompletionVerdict:
    action: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class CompletionVerifier:
    """Deterministically verifies that completion claims are not premature."""

    def __init__(self, skill_registry: Any, checklist_path: Path | None = None) -> None:
        self.skill_registry = skill_registry
        self.checklist_path = checklist_path or Path(__file__).with_name(
            "feature_checklist.json"
        )
        self.checklist = _load_checklist(self.checklist_path)

    def verify(
        self,
        *,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CompletionVerdict:
        if action.get("action") == "respond":
            return self._verify_response(action=action, messages=messages, tools=tools)
        if action.get("action") == "tool_calls":
            return self._verify_tool_calls(action=action, messages=messages, tools=tools)
        return CompletionVerdict()

    def _verify_response(
        self,
        *,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CompletionVerdict:
        response_content = str(action.get("content") or "")
        skill_decision = self.skill_registry.before_response(
            messages=messages,
            tools=tools,
            response_content=response_content,
        )
        pending_action, skipped = _pending_action_from_skill_decision(
            skill_decision.action,
            messages,
        )
        if pending_action:
            return CompletionVerdict(
                action=pending_action,
                warnings=[
                    *skill_decision.warnings,
                    "completion verifier blocked premature response",
                ],
                evidence={
                    "skill": skill_decision.skill,
                    "skipped_completed_calls": skipped,
                    "checklists": _checklist_ids(self.checklist),
                },
            )
        if skipped:
            return CompletionVerdict(
                warnings=[
                    "completion verifier allowed response after all checklist calls were already completed"
                ],
                evidence={
                    "skill": skill_decision.skill,
                    "skipped_completed_calls": skipped,
                    "checklists": _checklist_ids(self.checklist),
                },
            )
        return CompletionVerdict()

    def _verify_tool_calls(
        self,
        *,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CompletionVerdict:
        skill_decision = self.skill_registry.before_tool_calls(
            messages=messages,
            tools=tools,
            proposed_action=action,
        )
        pending_action, skill_skipped = _pending_action_from_skill_decision(
            skill_decision.action,
            messages,
        )
        if pending_action:
            return CompletionVerdict(
                action=pending_action,
                warnings=[
                    *skill_decision.warnings,
                    "completion verifier replaced partial tool batch",
                ],
                evidence={
                    "skill": skill_decision.skill,
                    "skipped_completed_calls": skill_skipped,
                    "checklists": _checklist_ids(self.checklist),
                },
            )

        filtered_action, skipped = _drop_completed_calls(action, messages)
        if filtered_action is None:
            return CompletionVerdict(
                action={"action": "respond", "content": "Done."},
                warnings=[
                    "completion verifier replaced already-completed tool batch with final acknowledgement"
                ],
                evidence={
                    "skipped_completed_calls": skipped,
                    "checklists": _checklist_ids(self.checklist),
                },
            )
        if skipped:
            return CompletionVerdict(
                action=filtered_action,
                warnings=[
                    "completion verifier removed already-completed checklist calls"
                ],
                evidence={
                    "skipped_completed_calls": skipped,
                    "checklists": _checklist_ids(self.checklist),
                },
            )
        return CompletionVerdict()


def _pending_action_from_skill_decision(
    action: dict[str, Any] | None,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not action or action.get("action") != "tool_calls":
        return None, []
    return _drop_completed_calls(action, messages)


def _drop_completed_calls(
    action: dict[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    calls = list(action.get("tool_calls") or [])
    if not calls:
        return None, []

    completed = _successful_call_fingerprints(messages)
    pending: list[dict[str, Any]] = []
    skipped: list[str] = []
    for call in calls:
        name = str(call.get("tool_name") or call.get("name") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        fingerprint = _fingerprint(name, arguments)
        if fingerprint in completed:
            skipped.append(name)
            continue
        pending.append({"tool_name": name, "arguments": arguments})

    if not pending:
        return None, skipped
    if len(pending) == len(calls):
        return action, skipped
    return {"action": "tool_calls", "tool_calls": pending}, skipped


def _successful_call_fingerprints(messages: list[dict[str, Any]]) -> set[str]:
    by_call_id: dict[str, tuple[str, dict[str, Any]]] = {}
    completed: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "")
                function = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
                name = str(function.get("name") or call.get("tool_name") or "")
                arguments = _parse_arguments(function.get("arguments", call.get("arguments", {})))
                if call_id and name:
                    by_call_id[call_id] = (name, arguments)
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id not in by_call_id:
                continue
            if not _tool_result_success(str(message.get("content") or "")):
                continue
            name, arguments = by_call_id[call_id]
            completed.add(_fingerprint(name, arguments))
    return completed


def _tool_result_success(content: str) -> bool:
    data = _parse_json(content)
    if isinstance(data, dict):
        status = _find_key(data, "status")
        if isinstance(status, str):
            return status.strip().lower() not in {"failure", "failed", "error"}
        error = _find_key(data, "error")
        if error not in (None, "", []):
            return False
    lowered = content.lower()
    return not any(piece in lowered for piece in ("failure", "failed", "error"))


def _fingerprint(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json(content: str) -> Any:
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_key(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_key(nested, key)
            if found is not None:
                return found
    return None


def _load_checklist(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _checklist_ids(checklist: dict[str, Any]) -> list[str]:
    items = checklist.get("task_family_checklists")
    if not isinstance(items, list):
        return []
    return [
        str(item.get("id"))
        for item in items
        if isinstance(item, dict) and item.get("id")
    ]
