"""Schema and lightweight policy guards for planner-selected actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    normalized_action: dict[str, Any] | None = None


@dataclass
class PolicyResult:
    allowed: bool
    replacement_action: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


class SchemaGuard:
    """Validates planner tool calls against evaluator-provided schemas."""

    def validate(self, action: dict[str, Any], tools: list[dict[str, Any]]) -> ValidationResult:
        if action.get("action") == "respond":
            content = action.get("content", "")
            if not isinstance(content, str):
                return ValidationResult(False, ["respond content must be a string"])
            return ValidationResult(True, normalized_action={"action": "respond", "content": content})

        if action.get("action") != "tool_calls":
            return ValidationResult(False, ["action must be respond or tool_calls"])

        tool_calls = action.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return ValidationResult(False, ["tool_calls action requires at least one tool call"])

        tool_map = {_tool_name(tool): tool for tool in tools}
        normalized_calls: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                errors.append(f"tool call {index} must be an object")
                continue
            name = call.get("tool_name") or call.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"tool call {index} missing tool_name")
                continue
            if name not in tool_map:
                errors.append(f"unknown tool: {name}")
                continue
            arguments = call.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                errors.append(f"{name} arguments must be an object")
                continue
            normalized_args, arg_errors = _validate_arguments(name, arguments, tool_map[name])
            errors.extend(arg_errors)
            normalized_calls.append({"tool_name": name, "arguments": normalized_args})

        if errors:
            return ValidationResult(False, errors, {"action": "tool_calls", "tool_calls": normalized_calls})
        return ValidationResult(True, normalized_action={"action": "tool_calls", "tool_calls": normalized_calls})


class PolicyGuard:
    """Applies high-signal policy checks that are cheap and deterministic."""

    def apply(
        self,
        *,
        action: dict[str, Any],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> PolicyResult:
        if action.get("action") != "tool_calls":
            return PolicyResult(True)

        tool_map = {_tool_name(tool): tool for tool in tools}
        warnings: list[str] = []
        for call in action.get("tool_calls") or []:
            name = call.get("tool_name")
            description = _tool_description(tool_map.get(name, {}))
            if description.startswith("REQUIRES_CONFIRMATION") and not _has_recent_confirmation(messages):
                detail = _format_action_detail(name, call.get("arguments") or {})
                return PolicyResult(
                    allowed=False,
                    replacement_action={
                        "action": "respond",
                        "content": (
                            f"I can do that, but I need your confirmation first: {detail}. "
                            "Please say yes to confirm."
                        ),
                    },
                    warnings=[f"{name} requires explicit confirmation"],
                )

            if name == "open_close_sunroof":
                percentage = (call.get("arguments") or {}).get("percentage")
                if isinstance(percentage, (int, float)) and percentage > 0:
                    has_parallel_sunshade_open = any(
                        other.get("tool_name") == "open_close_sunshade"
                        and (other.get("arguments") or {}).get("percentage") == 100
                        for other in action.get("tool_calls") or []
                    )
                    if not has_parallel_sunshade_open and not _recent_tool_text_contains(
                        messages,
                        ("sunshade", "100"),
                    ):
                        warnings.append(
                            "sunroof opening may require checking or opening the sunshade first"
                        )

        return PolicyResult(True, warnings=warnings)


def _validate_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tool: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    parameters = tool.get("function", {}).get("parameters", {})
    required = parameters.get("required", []) or []
    properties = parameters.get("properties", {}) or {}
    additional_allowed = parameters.get("additionalProperties", True)
    errors: list[str] = []
    normalized = dict(arguments)

    for key in required:
        if key not in normalized:
            errors.append(f"{tool_name} missing required argument: {key}")

    if additional_allowed is False:
        for key in list(normalized):
            if key not in properties:
                errors.append(f"{tool_name} has unknown argument: {key}")

    for key, schema in properties.items():
        if key not in normalized:
            continue
        value = normalized[key]
        expected_type = schema.get("type")
        enum = schema.get("enum")
        if enum and value not in enum:
            errors.append(f"{tool_name}.{key} must be one of {enum}; got {value!r}")
        if expected_type and not _matches_json_type(value, expected_type):
            coerced = _coerce_json_type(value, expected_type)
            if coerced is None:
                errors.append(f"{tool_name}.{key} must be {expected_type}; got {type(value).__name__}")
            else:
                normalized[key] = coerced
        if isinstance(normalized.get(key), (int, float)):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and normalized[key] < minimum:
                errors.append(f"{tool_name}.{key} must be >= {minimum}")
            if maximum is not None and normalized[key] > maximum:
                errors.append(f"{tool_name}.{key} must be <= {maximum}")

    return normalized, errors


def _matches_json_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def _coerce_json_type(value: Any, expected_type: Any) -> Any:
    if expected_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if expected_type == "integer":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric.is_integer():
            return int(numeric)
    if expected_type == "string" and value is not None:
        return str(value)
    return None


def _has_recent_confirmation(messages: list[dict[str, Any]]) -> bool:
    latest_user = ""
    previous_assistant = ""
    for message in reversed(messages):
        if not latest_user and message.get("role") == "user":
            latest_user = str(message.get("content") or "").strip().lower()
        elif latest_user and message.get("role") == "assistant":
            previous_assistant = str(message.get("content") or "").lower()
            break
    if not _is_confirmation_text(latest_user):
        return False
    return any(
        phrase in previous_assistant
        for phrase in (
            "confirm",
            "confirmation",
            "need your",
            "should i",
            "shall i",
            "do you want",
            "want me to",
            "send",
            "proceed",
            "go ahead",
        )
    )


def _is_confirmation_text(content: str) -> bool:
    content = content.strip().lower()
    if not content:
        return False
    yes_words = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "confirmed"}
    if content in yes_words:
        return True
    return any(
        phrase in content
        for phrase in (
            "yes please",
            "yes, please",
            "go ahead",
            "please proceed",
            "proceed",
            "do it",
            "send it",
            "sounds good",
            "that works",
        )
    )


def _recent_tool_text_contains(messages: list[dict[str, Any]], pieces: tuple[str, ...]) -> bool:
    text = " ".join(
        str(message.get("content") or "").lower()
        for message in messages[-8:]
        if message.get("role") == "tool"
    )
    return all(piece.lower() in text for piece in pieces)


def _format_action_detail(name: str, arguments: dict[str, Any]) -> str:
    if name == "send_email":
        addresses = arguments.get("email_addresses")
        if isinstance(addresses, list) and addresses:
            recipients = ", ".join(str(address) for address in addresses)
            return f"send_email to {recipients} with the gathered details"
        return "send_email with the gathered details"
    if not arguments:
        return str(name)
    arg_text = ", ".join(f"{key}={value}" for key, value in arguments.items())
    return f"{name} with {arg_text}"


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or tool.get("name") or "")


def _tool_description(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("description") or "")
