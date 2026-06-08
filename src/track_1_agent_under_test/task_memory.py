"""Per-context task memory for the Track 1 private planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TASK_STATUSES = {"pending", "in_progress", "completed", "blocked"}


@dataclass
class TaskRecord:
    id: str
    subject: str
    description: str = ""
    active_form: str | None = None
    status: str = "pending"
    owner: str | None = "main_planner"
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextTaskMemory:
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    highest_id: int = 0
    turn_index: int = 0
    last_task_update_turn: int = 0
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    seen_tool_result_keys: set[str] = field(default_factory=set)


TASK_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "TaskCreate",
        "description": (
            "Create a structured task for the current CAR-bench conversation. "
            "Use for multi-step work, state-changing work with preconditions, "
            "navigation, charging, calendar/email, ambiguity, or confirmation."
        ),
        "parameters": {
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "activeForm": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "blocked"],
                },
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "TaskUpdate",
        "description": (
            "Update one task in the current conversation. Keep exactly one task "
            "in_progress when work is active. Mark tasks completed immediately "
            "after their step is done."
        ),
        "parameters": {
            "type": "object",
            "required": ["taskId"],
            "properties": {
                "taskId": {"type": "string"},
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "activeForm": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "blocked"],
                },
                "owner": {"type": "string"},
                "addBlocks": {"type": "array", "items": {"type": "string"}},
                "addBlockedBy": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "TaskList",
        "description": "List tasks for the current conversation only.",
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
            "additionalProperties": False,
        },
    },
]


class TaskMemoryStore:
    """Stores structured tasks scoped to one A2A context_id."""

    def __init__(self) -> None:
        self._memory_by_context: dict[str, ContextTaskMemory] = {}

    def reset(self, context_id: str) -> None:
        self._memory_by_context.pop(context_id, None)

    def observe_messages(self, context_id: str, messages: list[dict[str, Any]]) -> None:
        memory = self._memory(context_id)
        memory.turn_index += 1
        for message in messages:
            if message.get("role") != "tool":
                continue
            key = str(message.get("tool_call_id") or f"{message.get('name')}:{len(memory.tool_history)}")
            if key in memory.seen_tool_result_keys:
                continue
            memory.seen_tool_result_keys.add(key)
            memory.tool_history.append(
                {
                    "tool_call_id": message.get("tool_call_id"),
                    "tool_name": message.get("name"),
                    "content": str(message.get("content") or "")[:4000],
                    "turn": memory.turn_index,
                }
            )
            memory.tool_history = memory.tool_history[-40:]

    def snapshot(self, context_id: str) -> dict[str, Any]:
        memory = self._memory(context_id)
        tasks = [task.to_dict() for task in sorted(memory.tasks.values(), key=lambda item: int(item.id))]
        return {
            "task_list_id": context_id,
            "turn_index": memory.turn_index,
            "last_task_update_turn": memory.last_task_update_turn,
            "turns_since_task_update": max(0, memory.turn_index - memory.last_task_update_turn),
            "tasks": tasks,
            "active_tasks": [
                task
                for task in tasks
                if task.get("status") in {"pending", "in_progress", "blocked"}
            ],
            "tool_history": memory.tool_history[-12:],
        }

    def reminders(self, context_id: str) -> list[str]:
        snapshot = self.snapshot(context_id)
        tasks = snapshot["tasks"]
        reminders: list[str] = []
        active = [task for task in tasks if task["status"] in {"pending", "in_progress"}]
        in_progress = [task for task in tasks if task["status"] == "in_progress"]
        if len(in_progress) > 1:
            reminders.append("Only one task may be in_progress. Call TaskUpdate to fix task statuses.")
        if active and snapshot["turns_since_task_update"] >= 5:
            reminders.append(
                "Task memory has not been updated recently. Review TaskList and call TaskUpdate before continuing."
            )
        if active:
            reminders.append(
                "There are unfinished tasks. Do not final-response as done until pending/in_progress tasks are resolved."
            )
        return reminders

    def execute(self, context_id: str, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        arguments = arguments or {}
        if name == "TaskCreate":
            return self._create(context_id, arguments)
        if name == "TaskUpdate":
            return self._update(context_id, arguments)
        if name == "TaskList":
            return {"ok": True, "tasks": self.snapshot(context_id)["tasks"]}
        return {"ok": False, "error": f"Unknown internal task tool: {name}"}

    def _memory(self, context_id: str) -> ContextTaskMemory:
        return self._memory_by_context.setdefault(context_id, ContextTaskMemory())

    def _create(self, context_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        memory = self._memory(context_id)
        memory.highest_id += 1
        task_id = str(memory.highest_id)
        status = _status(arguments.get("status"), default="pending")
        task = TaskRecord(
            id=task_id,
            subject=str(arguments.get("subject") or "Untitled task"),
            description=str(arguments.get("description") or ""),
            active_form=_optional_string(arguments.get("activeForm")),
            status=status,
            metadata=_dict(arguments.get("metadata")),
        )
        if task.status == "in_progress":
            _clear_other_in_progress(memory, task_id)
        memory.tasks[task_id] = task
        memory.last_task_update_turn = memory.turn_index
        return {"ok": True, "task": task.to_dict()}

    def _update(self, context_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        memory = self._memory(context_id)
        task_id = str(arguments.get("taskId") or "")
        task = memory.tasks.get(task_id)
        if task is None:
            return {"ok": False, "error": f"Task {task_id!r} does not exist."}
        if "subject" in arguments:
            task.subject = str(arguments.get("subject") or task.subject)
        if "description" in arguments:
            task.description = str(arguments.get("description") or "")
        if "activeForm" in arguments:
            task.active_form = _optional_string(arguments.get("activeForm"))
        if "status" in arguments:
            task.status = _status(arguments.get("status"), default=task.status)
            if task.status == "in_progress":
                _clear_other_in_progress(memory, task.id)
                if not task.owner:
                    task.owner = "main_planner"
        if "owner" in arguments:
            task.owner = _optional_string(arguments.get("owner"))
        for value in _string_list(arguments.get("addBlocks")):
            if value not in task.blocks:
                task.blocks.append(value)
        for value in _string_list(arguments.get("addBlockedBy")):
            if value not in task.blocked_by:
                task.blocked_by.append(value)
        if isinstance(arguments.get("metadata"), dict):
            task.metadata.update(arguments["metadata"])
        memory.last_task_update_turn = memory.turn_index
        return {"ok": True, "task": task.to_dict()}


def _status(value: Any, *, default: str) -> str:
    text = str(value or default)
    return text if text in TASK_STATUSES else default


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _clear_other_in_progress(memory: ContextTaskMemory, keep_task_id: str) -> None:
    for task in memory.tasks.values():
        if task.id != keep_task_id and task.status == "in_progress":
            task.status = "pending"
