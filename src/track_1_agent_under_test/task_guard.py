"""Deterministic CAR-bench task gates for high-risk decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardDecision:
    """A planner-visible override plus private diagnostics."""

    action: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


class TaskGuard:
    """Applies cheap task-specific checks before and after model planning.

    These rules cover high-impact CAR-bench failure modes without reading gold
    task files: missing capabilities, unresolved ambiguity, and sunroof policy
    preconditions.
    """

    def finish_after_successful_state_change(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> GuardDecision:
        """Return a final spoken response after successful state-changing tools.

        Query tools still need model reasoning to turn their result into a user
        answer. State-changing tools usually only need a short completion
        acknowledgement, so this avoids an extra LLM call on many benchmark
        turns.
        """

        trailing_results = _trailing_tool_results(messages)
        if not trailing_results:
            return GuardDecision()
        if not all(_tool_result_status_is_success(message) for message in trailing_results):
            return GuardDecision()

        tool_calls = _latest_assistant_tool_calls(messages)
        if not tool_calls or len(trailing_results) < len(tool_calls):
            return GuardDecision()
        tool_names = [_history_tool_call_name(call) for call in tool_calls]
        if not any(_is_state_changing_tool_name(name) for name in tool_names):
            return GuardDecision()

        return GuardDecision(
            action=_cannot_do("Done."),
            warnings=["deterministic completion after successful state change"],
        )

    def finish_after_stop_signal(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> GuardDecision:
        """Short-circuit evaluator stop turns and let the executor free context."""

        latest_user = _latest_user_text(messages).strip().lower()
        if latest_user != "###stop###":
            return GuardDecision()
        return GuardDecision(
            action=_cannot_do("Done."),
            warnings=["terminal stop signal"],
        )

    def preempt(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> GuardDecision:
        tool_map = _tool_map(tools)
        communication_confirmation = _communication_confirmation_action(
            messages=messages,
            tool_map=tool_map,
        )
        if communication_confirmation.action:
            return communication_confirmation

        latest_user = _latest_actionable_user_text(messages)
        if not latest_user:
            return GuardDecision()

        text = latest_user.lower()
        missing_capability = _missing_requested_capability(text, tool_map)
        if missing_capability.action:
            return missing_capability

        policy_decision = _policy_precondition_action(
            text=text,
            messages=messages,
            tool_map=tool_map,
        )
        if policy_decision.action:
            return policy_decision

        knowledge_decision = _knowledge_precondition_action(
            text=text,
            messages=messages,
            tool_map=tool_map,
        )
        if knowledge_decision.action:
            return knowledge_decision

        direct_decision = _direct_simple_action(
            text=text,
            messages=messages,
            tool_map=tool_map,
        )
        if direct_decision.action:
            return direct_decision

        if _requests_sunshade_control(text):
            if not _tool_has_parameter(tool_map, "open_close_sunshade", "percentage"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot adjust the sunshade because the required sunshade control capability is unavailable."
                    ),
                    warnings=["sunshade control unavailable"],
                )

        if _requests_sunroof_open(text):
            if not _tool_has_parameter(tool_map, "open_close_sunroof", "percentage"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot open the sunroof because the required sunroof control capability is unavailable."
                    ),
                    warnings=["sunroof control unavailable"],
                )

            if (
                not _has_explicit_percentage(text)
                and "get_user_preferences" in tool_map
                and not _tool_called_after_latest_actionable_user(
                    messages, "get_user_preferences"
                )
            ):
                return GuardDecision(
                    action={
                        "action": "tool_calls",
                        "tool_calls": [
                            {
                                "tool_name": "get_user_preferences",
                                "arguments": {
                                    "preference_categories": {
                                        "vehicle_settings": {
                                            "vehicle_settings": True,
                                            "climate_control": True,
                                        }
                                    }
                                },
                            }
                        ],
                    },
                    warnings=["sunroof percentage ambiguous; checking preferences"],
                )

            sunshade_position = _latest_sunshade_position(messages)
            if (
                sunshade_position is None
                and "get_sunroof_and_sunshade_position" in tool_map
                and not _tool_called_after_latest_actionable_user(
                    messages, "get_sunroof_and_sunshade_position"
                )
            ):
                return GuardDecision(
                    action={
                        "action": "tool_calls",
                        "tool_calls": [
                            {
                                "tool_name": "get_sunroof_and_sunshade_position",
                                "arguments": {},
                            }
                        ],
                    },
                    warnings=["sunroof opening requires sunshade state"],
                )

            if (
                sunshade_position is not None
                and sunshade_position < 100
                and not _tool_has_parameter(tool_map, "open_close_sunshade", "percentage")
            ):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot open the sunroof because the sunshade must be opened first, but sunshade control is unavailable."
                    ),
                    warnings=["sunroof blocked by unavailable sunshade control"],
                )

        return GuardDecision()

    def postprocess(
        self,
        *,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> GuardDecision:
        if action.get("action") == "respond":
            content = str(action.get("content") or "")
            normalized = _normalize_24h_times(content)
            if _assistant_response_repeated(messages, normalized):
                lookup_call = _navigation_followup_lookup_call(
                    latest_user_text=_latest_actionable_user_text(messages),
                    tool_map=_tool_map(tools),
                )
                if lookup_call:
                    return GuardDecision(
                        action=_tool_calls([lookup_call]),
                        warnings=["changed strategy after repeated response"],
                    )
            return GuardDecision(
                action={
                    "action": "respond",
                    "content": normalized,
                }
            )

        if action.get("action") != "tool_calls":
            return GuardDecision(action=action)

        tool_map = _tool_map(tools)
        calls = list(action.get("tool_calls") or [])
        warnings: list[str] = []

        if _has_unknown_recent_result(messages) and _contains_state_change(calls):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot complete that reliably because the available tool result is missing required information."
                ),
                warnings=["blocked state change after unknown tool result"],
            )

        for guard in (
            _navigation_action_guard,
            _charging_action_guard,
            _temperature_action_guard,
            _communication_result_guard,
            _poi_action_guard,
        ):
            decision = guard(calls=calls, messages=messages, tool_map=tool_map)
            if decision.action:
                return decision

        policy_call_decision = _policy_action_from_tool_calls(
            calls=calls,
            messages=messages,
            tool_map=tool_map,
        )
        if policy_call_decision.action:
            return policy_call_decision

        if not any(_is_open_sunroof_call(call) for call in calls):
            return GuardDecision(action=action)

        if "get_weather" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot open the sunroof because I need to check the weather first, but weather information is unavailable."
                ),
                warnings=["sunroof weather check unavailable"],
            )

        weather_condition = _latest_weather_condition(messages)
        if weather_condition is None:
            weather_calls = [call for call in calls if call.get("tool_name") == "get_weather"]
            if weather_calls:
                return GuardDecision(
                    action={"action": "tool_calls", "tool_calls": weather_calls},
                    warnings=["split weather check before sunroof state change"],
                )
            if _tool_called_after_latest_actionable_user(messages, "get_weather"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot open the sunroof because the weather result is missing the required condition information."
                    ),
                    warnings=["sunroof blocked by missing weather result field"],
                )
            weather_args = _current_weather_args_from_system(messages, tool_map)
            if weather_args is not None:
                return GuardDecision(
                    action={
                        "action": "tool_calls",
                        "tool_calls": [
                            {
                                "tool_name": "get_weather",
                                "arguments": weather_args,
                            }
                        ],
                    },
                    warnings=["checking current weather before sunroof opening"],
                )
            return GuardDecision(
                action=_cannot_do(
                    "I cannot open the sunroof because I need current weather information first, but I cannot derive the required weather query."
                ),
                warnings=["blocked sunroof opening before weather check"],
            )

        if _is_adverse_sunroof_weather(weather_condition) and not _has_recent_confirmation(
            messages
        ):
            return GuardDecision(
                action={
                    "action": "respond",
                    "content": (
                        f"The weather is {weather_condition}, so opening the sunroof may be uncomfortable. "
                        "Do you still want me to proceed?"
                    ),
                },
                warnings=["sunroof opening requires weather confirmation"],
            )

        sunshade_position = _latest_sunshade_position(messages)
        has_parallel_sunshade = any(_is_full_sunshade_call(call) for call in calls)
        if sunshade_position is None and not has_parallel_sunshade:
            if "get_sunroof_and_sunshade_position" in tool_map:
                return GuardDecision(
                    action={
                        "action": "tool_calls",
                        "tool_calls": [
                            {
                                "tool_name": "get_sunroof_and_sunshade_position",
                                "arguments": {},
                            }
                        ],
                    },
                    warnings=["checking sunshade state before sunroof opening"],
                )
            warnings.append("sunshade state unknown before sunroof opening")

        if sunshade_position is not None and sunshade_position < 100 and not has_parallel_sunshade:
            if not _tool_has_parameter(tool_map, "open_close_sunshade", "percentage"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot open the sunroof because the sunshade must be fully opened first, but that capability is unavailable."
                    ),
                    warnings=["sunroof blocked by missing sunshade capability"],
                )
            calls = [
                {
                    "tool_name": "open_close_sunshade",
                    "arguments": {"percentage": 100},
                },
                *calls,
            ]
            warnings.append("prepended sunshade opening before sunroof")

        return GuardDecision(action={"action": "tool_calls", "tool_calls": calls}, warnings=warnings)


def _cannot_do(content: str) -> dict[str, Any]:
    return {"action": "respond", "content": content}


def _normalize_24h_times(content: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minutes = match.group(2) or "00"
        suffix = match.group(3).lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minutes}"

    return re.sub(
        r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([AaPp][Mm])\b",
        replace,
        content,
    )


def _assistant_response_repeated(messages: list[dict[str, Any]], content: str) -> bool:
    normalized = content.strip().lower()
    if not normalized:
        return False
    repeats = 0
    for message in reversed(messages[-12:]):
        if message.get("role") != "assistant":
            continue
        previous = _normalize_24h_times(str(message.get("content") or "")).strip().lower()
        if previous == normalized:
            repeats += 1
        if repeats >= 2:
            return True
    return False


def _tool_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"action": "tool_calls", "tool_calls": calls}


def _tool_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_tool_name(tool): tool for tool in tools if _tool_name(tool)}


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or tool.get("name") or "")


def _tool_has_parameter(
    tool_map: dict[str, dict[str, Any]], tool_name: str, parameter: str
) -> bool:
    tool = tool_map.get(tool_name)
    if not tool:
        return False
    properties = (
        tool.get("function", {})
        .get("parameters", {})
        .get("properties", {})
    )
    return parameter in properties


def _latest_actionable_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or content == "###STOP###":
            continue
        if _is_confirmation_text(content):
            continue
        return content
    return ""


def _requests_sunroof_open(text: str) -> bool:
    return "sunroof" in text and any(
        word in text
        for word in ("open", "fresh air", "air", "vent", "halfway", "half", "partially")
    )


def _requests_sunshade_control(text: str) -> bool:
    return "sunshade" in text and any(
        word in text
        for word in ("open", "close", "adjust", "block", "position", "partially", "fully")
    )


def _requests_navigation_status(text: str) -> bool:
    return _has_any(text, ("navigation status", "current navigation", "active route", "route details")) and _has_any(
        text, ("check", "show", "tell", "what", "status", "detail", "information")
    )


def _requests_navigation_edit(text: str) -> bool:
    return _has_any(text, ("navigation", "route", "destination", "waypoint", "stop")) and _has_any(
        text,
        (
            "replace",
            "change",
            "delete",
            "remove",
            "add",
            "skip",
            "cancel",
            "shorten",
            "reroute",
        ),
    )


def _requests_new_navigation_control(text: str) -> bool:
    if _requests_navigation_edit(text):
        return False
    if _has_any(text, ("do not actually set", "do not set", "don't set", "only gather", "just tell")):
        return False
    return _has_any(
        text,
        (
            "navigate me",
            "start navigation",
            "set up navigation",
            "set navigation",
            "guide me",
            "take me to",
            "drive me to",
        ),
    )


def _requests_delete_current_navigation(text: str) -> bool:
    return _has_any(
        text,
        (
            "delete current navigation",
            "cancel navigation",
            "cancel the navigation",
            "stop navigation",
            "stop the navigation",
            "end navigation",
            "clear navigation",
            "clear the current navigation",
            "remove current navigation",
        ),
    )


def _requests_only_delete_current_navigation(text: str) -> bool:
    if not _requests_delete_current_navigation(text):
        return False
    return not _has_any(
        text,
        (
            "set a new",
            "set new",
            "set the navigation",
            "navigate to",
            "navigation to",
            "new destination",
            "replace",
            "change",
            "then navigate",
            "then set",
        ),
    )


def _navigation_followup_lookup_call(
    *,
    latest_user_text: str,
    tool_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    text = latest_user_text.lower()
    if not _has_any(text, ("restaurant", "restaurants", "destination", "go to", "navigate to", "find")):
        return None
    if not _tool_has_parameter(tool_map, "get_location_id_by_location_name", "location"):
        return None
    location = _location_after_in_or_to(latest_user_text)
    if not location:
        return None
    return {
        "tool_name": "get_location_id_by_location_name",
        "arguments": {"location": location},
    }


def _location_after_in_or_to(text: str) -> str | None:
    cancelled = {
        match.strip()
        for match in re.findall(
            r"\bcancel(?:\s+\w+){0,3}\s+navigation\s+to\s+([A-Z][A-Za-zÀ-ÿ]*(?:\s+[A-Z][A-Za-zÀ-ÿ]*){0,3})",
            text,
            flags=re.IGNORECASE,
        )
    }
    in_matches = re.findall(
        r"\bin\s+([A-Z][A-Za-zÀ-ÿ]*(?:\s+[A-Z][A-Za-zÀ-ÿ]*){0,3})",
        text,
    )
    if in_matches:
        return in_matches[-1].strip()
    to_matches = re.findall(
        r"\bto\s+([A-Z][A-Za-zÀ-ÿ]*(?:\s+[A-Z][A-Za-zÀ-ÿ]*){0,3})",
        text,
    )
    for match in reversed(to_matches):
        candidate = match.strip()
        if candidate and candidate not in cancelled:
            return candidate
    return None


def _requests_ac_on(text: str) -> bool:
    if not _has_any(
        text,
        (
            "air conditioning",
            "a/c",
            "cool down",
            "cooler",
            "too hot",
            "hot in here",
            "stuffy",
            "fresh air",
            "air quality",
        ),
    ) and re.search(r"\bac\b", text) is None:
        return False
    if _has_any(
        text,
        (
            "do not want to turn on air conditioning",
            "don't want to turn on air conditioning",
            "do not turn on air conditioning",
            "don't turn on air conditioning",
            "no air conditioning",
            "turn off air conditioning",
            "disable air conditioning",
        ),
    ):
        return False
    return _has_any(
        text,
        (
            "turn on",
            "switch on",
            "enable",
            "start",
            "cool",
            "stuffy",
            "fresh air",
        ),
    )


def _requests_close_all_windows(text: str) -> bool:
    return "window" in text and _has_any(text, ("close all", "closed all", "close the windows", "completely closed", "fully close"))


def _requests_defrost_on(text: str) -> bool:
    return _has_any(text, ("defrost", "defog", "fogged", "fog up", "condensation")) and not _has_any(
        text, ("turn off", "disable", "stop")
    )


def _requests_fog_lights_on(text: str) -> bool:
    return _has_any(text, ("fog light", "fog lights", "reduced visibility")) and not _has_any(
        text, ("turn off", "disable", "switch off")
    )


def _requests_high_beams_on(text: str) -> bool:
    return _has_any(text, ("high beam", "high beams", "better visibility", "dark rural", "dark road")) and not _has_any(
        text, ("turn off", "disable", "switch off")
    )


def _requests_window_control(text: str) -> bool:
    return "window" in text and not _requests_defrost_on(text) and _has_control_verb(text)


def _requests_reading_light_control(text: str) -> bool:
    return "reading light" in text and _has_control_verb(text)


def _missing_requested_capability(
    text: str,
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    for rule in _MISSING_CAPABILITY_RULES:
        if not rule["predicate"](text):
            continue
        tool_name = rule["tool_name"]
        label = rule["label"]
        if tool_name not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    f"I cannot complete that because the required {label} capability is unavailable."
                ),
                warnings=[f"{tool_name} unavailable"],
            )
        for parameter in rule.get("essential_parameters", ()):
            if not _tool_has_parameter(tool_map, tool_name, parameter):
                return GuardDecision(
                    action=_cannot_do(
                        f"I cannot complete that because the {label} tool is missing the required {parameter} parameter."
                    ),
                    warnings=[f"{tool_name}.{parameter} unavailable"],
                )
    return GuardDecision()


def _policy_precondition_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if _requests_navigation_status(text):
        if "get_current_navigation_state" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot check the navigation state because that information capability is unavailable."
                ),
                warnings=["navigation status unavailable"],
            )
        if _tool_called_after_latest_actionable_user(
            messages, "get_current_navigation_state"
        ):
            return GuardDecision()
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "get_current_navigation_state",
                        "arguments": {"detailed_information": True},
                    }
                ]
            ),
            warnings=["direct navigation status lookup"],
        )

    if _requests_navigation_edit(text) and "get_current_navigation_state" not in tool_map:
        return GuardDecision(
            action=_cannot_do(
                "I cannot modify the current navigation route because the navigation state lookup capability is unavailable."
            ),
            warnings=["navigation edit blocked by unavailable current state lookup"],
        )

    if (
        _requests_navigation_edit(text)
        and "get_current_navigation_state" in tool_map
        and not _tool_called_after_latest_actionable_user(
            messages, "get_current_navigation_state"
        )
    ):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "get_current_navigation_state",
                        "arguments": {"detailed_information": True},
                    }
                ]
            ),
            warnings=["navigation edit requires current navigation state"],
        )

    for handler in (
        _fog_light_policy_action,
        _high_beam_policy_action,
        _defrost_policy_action,
        _ac_policy_action,
        _window_policy_action,
    ):
        decision = handler(text=text, messages=messages, tool_map=tool_map)
        if decision.action:
            return decision
    return GuardDecision()


def _knowledge_precondition_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if _requests_charging_knowledge(text):
        if "get_charging_specs_and_status" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot answer that charging or range question because charging status information is unavailable."
                ),
                warnings=["charging status lookup unavailable"],
            )
        if _tool_called_after_latest_actionable_user(
            messages, "get_charging_specs_and_status"
        ):
            if not _charging_status_has_required_fields(messages):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot answer that reliably because the charging status result is missing required battery information."
                    ),
                    warnings=["charging status result missing required fields"],
                )
            soc_distance = _soc_distance_precondition_action(
                text=text,
                messages=messages,
                tool_map=tool_map,
            )
            if soc_distance.action:
                return soc_distance
            return GuardDecision()
        return GuardDecision(
            action=_tool_calls(
                [{"tool_name": "get_charging_specs_and_status", "arguments": {}}]
            ),
            warnings=["checking charging status before battery/range reasoning"],
        )

    if _requests_relative_temperature_change(text):
        if "get_temperature_inside_car" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot adjust the temperature relatively because the current cabin temperature information is unavailable."
                ),
                warnings=["temperature lookup unavailable for relative change"],
            )
        if _tool_called_after_latest_actionable_user(
            messages, "get_temperature_inside_car"
        ):
            if not _temperature_result_has_required_fields(messages):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot adjust the temperature reliably because the temperature result is missing required cabin temperature information."
                    ),
                    warnings=["temperature result missing required fields"],
                )
            return GuardDecision()
        return GuardDecision(
            action=_tool_calls(
                [{"tool_name": "get_temperature_inside_car", "arguments": {}}]
            ),
            warnings=["checking cabin temperature before relative temperature change"],
        )

    return GuardDecision()


def _navigation_action_guard(
    *,
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    names = [str(call.get("tool_name") or "") for call in calls]
    state_changing_nav = [
        call for call in calls if str(call.get("tool_name") or "") in _NAVIGATION_STATE_CHANGE_TOOLS
    ]
    if not state_changing_nav:
        return GuardDecision()

    latest_user_text = _latest_actionable_user_text(messages)
    latest_user = latest_user_text.lower()
    if "delete_current_navigation" in names and not _requests_only_delete_current_navigation(latest_user):
        lookup_call = _navigation_followup_lookup_call(
            latest_user_text=latest_user_text,
            tool_map=tool_map,
        )
        if lookup_call:
            return GuardDecision(
                action=_tool_calls([lookup_call]),
                warnings=["redirected navigation deletion to new destination lookup"],
            )
        if "get_current_navigation_state" in tool_map and not _tool_called_after_latest_actionable_user(
            messages, "get_current_navigation_state"
        ):
            return GuardDecision(
                action=_tool_calls(
                    [
                        {
                            "tool_name": "get_current_navigation_state",
                            "arguments": {"detailed_information": True},
                        }
                    ]
                ),
                warnings=["blocked navigation deletion; checking current route"],
            )
        return GuardDecision(
            action=_cannot_do(
                "I cannot delete the current navigation unless you explicitly ask me to cancel navigation."
            ),
            warnings=["blocked unintended current navigation deletion"],
        )

    route_lookup_calls = [
        call
        for call in calls
        if call.get("tool_name")
        in {
            "get_location_id_by_location_name",
            "get_routes_from_start_to_destination",
            "search_poi_at_location",
            "search_poi_along_the_route",
        }
    ]
    if route_lookup_calls and state_changing_nav:
        return GuardDecision(
            action=_tool_calls(route_lookup_calls),
            warnings=["split navigation lookup before navigation state change"],
        )

    route_result = _latest_tool_json(messages, "get_routes_from_start_to_destination")
    if (
        _tool_called_after_latest_actionable_user(
            messages, "get_routes_from_start_to_destination"
        )
        and _find_key(route_result, "routes") is None
    ):
        return GuardDecision(
            action=_cannot_do(
                "I cannot set or modify navigation because the route lookup result is missing route alternatives."
            ),
            warnings=["navigation blocked by missing route result field"],
        )

    if "set_new_navigation" in names:
        if not _tool_has_parameter(tool_map, "set_new_navigation", "route_ids"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot set navigation because the navigation tool is missing the required route_ids parameter."
                ),
                warnings=["set_new_navigation.route_ids unavailable"],
            )
        nav_active = _latest_navigation_active(messages)
        if nav_active is True:
            replacement_call = _active_navigation_replacement_call(
                calls=calls,
                messages=messages,
                latest_user=latest_user,
                tool_map=tool_map,
            )
            if replacement_call:
                return GuardDecision(
                    action=_tool_calls([replacement_call]),
                    warnings=["converted active set_new_navigation to final destination replacement"],
                )
            return GuardDecision(
                action=_cannot_do(
                    "I cannot replace the active navigation with a new route; current route edits must use navigation editing tools."
                ),
                warnings=["set_new_navigation blocked while navigation active"],
            )

    edit_calls = [
        call for call in state_changing_nav if call.get("tool_name") in _NAVIGATION_EDIT_TOOLS
    ]
    if not edit_calls:
        return GuardDecision()

    if "get_current_navigation_state" not in tool_map:
        return GuardDecision(
            action=_cannot_do(
                "I cannot modify the current navigation route because navigation state lookup is unavailable."
            ),
            warnings=["navigation edit blocked by unavailable current state lookup"],
        )
    if "get_current_navigation_state" in names:
        return GuardDecision(
            action=_tool_calls(
                [
                    call
                    for call in calls
                    if call.get("tool_name") == "get_current_navigation_state"
                ]
            ),
            warnings=["split current navigation state lookup before edit"],
        )
    if not _tool_called_after_latest_actionable_user(
        messages, "get_current_navigation_state"
    ):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "get_current_navigation_state",
                        "arguments": {"detailed_information": True},
                    }
                ]
            ),
            warnings=["navigation edit requires current navigation state"],
        )

    state = _latest_tool_json(messages, "get_current_navigation_state")
    nav_active = _find_bool(state, "navigation_active")
    waypoints = _find_key(state, "waypoints_id")
    routes = _find_key(state, "routes_to_final_destination_id")
    if nav_active is None or not isinstance(waypoints, list) or not isinstance(routes, list):
        return GuardDecision(
            action=_cannot_do(
                "I cannot modify navigation because the current navigation state result is missing required route information."
            ),
            warnings=["navigation edit blocked by missing current state field"],
        )
    if nav_active is False:
        return GuardDecision(
            action=_cannot_do(
                "I cannot modify the current navigation route because navigation is not active."
            ),
            warnings=["navigation edit blocked because navigation inactive"],
        )

    edit_tool_names = [str(call.get("tool_name") or "") for call in edit_calls]
    if len(edit_tool_names) > 1:
        return GuardDecision(
            action=_tool_calls([edit_calls[0]]),
            warnings=["split multiple navigation edits into sequential turns"],
        )

    unknown_route_args = _unknown_navigation_route_arguments(edit_calls, messages)
    if unknown_route_args:
        return GuardDecision(
            action=_cannot_do(
                "I cannot modify navigation because the required route id was not found in the available route results."
            ),
            warnings=[f"navigation route id not grounded: {', '.join(unknown_route_args)}"],
        )

    return GuardDecision()


def _charging_action_guard(
    *,
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    names = {str(call.get("tool_name") or "") for call in calls}
    if not names.intersection(_CHARGING_REASONING_TOOLS):
        return GuardDecision()

    if "get_charging_specs_and_status" in names:
        return GuardDecision(
            action=_tool_calls(
                [
                    call
                    for call in calls
                    if call.get("tool_name") == "get_charging_specs_and_status"
                ]
            ),
            warnings=["split charging status lookup before charging reasoning"],
        )
    if "get_charging_specs_and_status" not in tool_map:
        return GuardDecision(
            action=_cannot_do(
                "I cannot complete the charging or range calculation because charging status information is unavailable."
            ),
            warnings=["charging reasoning blocked by unavailable status lookup"],
        )
    if not _tool_called_after_latest_actionable_user(
        messages, "get_charging_specs_and_status"
    ):
        return GuardDecision(
            action=_tool_calls(
                [{"tool_name": "get_charging_specs_and_status", "arguments": {}}]
            ),
            warnings=["checking charging status before charging reasoning"],
        )
    if not _charging_status_has_required_fields(messages):
        return GuardDecision(
            action=_cannot_do(
                "I cannot complete the charging or range calculation because the charging status result is missing required battery information."
            ),
            warnings=["charging reasoning blocked by missing status fields"],
        )
    return GuardDecision()


def _temperature_action_guard(
    *,
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    temp_calls = [
        call for call in calls if call.get("tool_name") == "set_climate_temperature"
    ]
    if not temp_calls:
        return GuardDecision()

    if not (
        _tool_has_parameter(tool_map, "set_climate_temperature", "temperature")
        and _tool_has_parameter(tool_map, "set_climate_temperature", "seat_zone")
    ):
        return GuardDecision(
            action=_cannot_do(
                "I cannot set the climate temperature because the temperature control tool is missing required parameters."
            ),
            warnings=["set_climate_temperature parameter unavailable"],
        )

    latest_user = _latest_actionable_user_text(messages).lower()
    if _requests_relative_temperature_change(latest_user):
        if "get_temperature_inside_car" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot adjust the temperature relatively because current cabin temperature information is unavailable."
                ),
                warnings=["relative temperature change blocked by unavailable lookup"],
            )
        if "get_temperature_inside_car" in [call.get("tool_name") for call in calls]:
            return GuardDecision(
                action=_tool_calls(
                    [
                        call
                        for call in calls
                        if call.get("tool_name") == "get_temperature_inside_car"
                    ]
                ),
                warnings=["split temperature lookup before relative temperature change"],
            )
        if not _tool_called_after_latest_actionable_user(
            messages, "get_temperature_inside_car"
        ):
            return GuardDecision(
                action=_tool_calls(
                    [{"tool_name": "get_temperature_inside_car", "arguments": {}}]
                ),
                warnings=["checking cabin temperature before relative change"],
            )
        if not _temperature_result_has_required_fields(messages):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot adjust the temperature reliably because the temperature result is missing required cabin temperature information."
                ),
                warnings=["relative temperature blocked by missing result fields"],
            )

    if not _user_text_has_explicit_temperature(latest_user):
        return GuardDecision(
            action={
                "action": "respond",
                "content": "What temperature should I set?",
            },
            warnings=["climate temperature target ambiguous"],
        )

    return GuardDecision()


def _communication_result_guard(
    *,
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    names = {str(call.get("tool_name") or "") for call in calls}
    contact_info = _latest_tool_json(messages, "get_contact_information")
    if "send_email" in names and _tool_called_after_latest_actionable_user(
        messages, "get_contact_information"
    ):
        if _find_key(contact_info, "email") is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot send the email because the contact information result is missing an email address."
                ),
                warnings=["email blocked by missing contact email field"],
            )
    if "call_phone_by_number" in names and _tool_called_after_latest_actionable_user(
        messages, "get_contact_information"
    ):
        if _find_key(contact_info, "phone_number") is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot place the call because the contact information result is missing a phone number."
                ),
                warnings=["phone call blocked by missing contact phone field"],
            )
    return GuardDecision()


def _communication_confirmation_action(
    *,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    latest_user = _latest_user_text(messages)
    if not _is_confirmation_text(latest_user):
        return GuardDecision()
    if not (
        _tool_has_parameter(tool_map, "send_email", "email_addresses")
        and _tool_has_parameter(tool_map, "send_email", "content_message")
    ):
        return GuardDecision()

    assistant_text = _previous_assistant_text(messages)
    if not assistant_text or not _has_any(assistant_text.lower(), ("email", "send")):
        return GuardDecision()

    email_addresses = _extract_email_addresses(assistant_text)
    if not email_addresses:
        email_addresses = _latest_contact_email_addresses(messages)
    if not email_addresses:
        return GuardDecision()

    content_message = _extract_proposed_email_content(assistant_text)
    if not content_message:
        content_message = _fallback_email_content(
            _latest_actionable_user_text(messages),
            messages=messages,
        )

    return GuardDecision(
        action=_tool_calls(
            [
                {
                    "tool_name": "send_email",
                    "arguments": {
                        "email_addresses": email_addresses,
                        "content_message": _normalize_24h_times(content_message),
                    },
                }
            ]
        ),
        warnings=["sending email after user confirmation"],
    )


def _poi_action_guard(
    *,
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    for call in calls:
        if call.get("tool_name") != "search_poi_along_the_route":
            continue
        args = call.get("arguments") or {}
        if (
            args.get("category_poi") == "charging_stations"
            and "at_kilometer" not in args
        ):
            return GuardDecision(
                action={
                    "action": "respond",
                    "content": "At what kilometer along the route should I search for a charging station?",
                },
                warnings=["charging station route search missing at_kilometer"],
            )

    if any(call.get("tool_name") == "call_phone_by_number" for call in calls):
        poi_result = _latest_poi_result(messages)
        if poi_result is not None and _find_key(poi_result, "phone_number") is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot call the point of interest because the search result is missing a phone number."
                ),
                warnings=["POI call blocked by missing phone number"],
            )
    return GuardDecision()


def _policy_action_from_tool_calls(
    *,
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    latest_user = _latest_actionable_user_text(messages).lower()
    call_names = {str(call.get("tool_name") or "") for call in calls}
    if (
        _requests_ac_on(latest_user)
        and "set_air_conditioning" not in call_names
        and call_names.intersection({"open_close_window", "set_fan_speed", "set_air_circulation"})
    ):
        ac_decision = _ac_policy_action(text=latest_user, messages=messages, tool_map=tool_map)
        if ac_decision.action:
            ac_decision.warnings.append("completed missing AC action from user intent")
            return ac_decision

    if any(call.get("tool_name") == "set_air_conditioning" and (call.get("arguments") or {}).get("on") is True for call in calls):
        if "get_climate_settings" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_climate_settings"):
            return GuardDecision(
                action=_tool_calls([{"tool_name": "get_climate_settings", "arguments": {}}]),
                warnings=["split climate check before AC state change"],
            )
        if "get_vehicle_window_positions" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions"):
            return GuardDecision(
                action=_tool_calls([{"tool_name": "get_vehicle_window_positions", "arguments": {}}]),
                warnings=["split window check before AC state change"],
            )
        if "get_climate_settings" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the air conditioning safely because climate status information is unavailable."
                ),
                warnings=["AC blocked by unavailable climate status lookup"],
            )
        if "get_vehicle_window_positions" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the air conditioning safely because window position information is unavailable."
                ),
                warnings=["AC blocked by unavailable window position lookup"],
            )
        climate = _latest_tool_json(messages, "get_climate_settings")
        if _find_number(climate, "fan_speed") is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the air conditioning safely because the climate result is missing the fan speed."
                ),
                warnings=["AC blocked by missing climate result field"],
            )
        windows = _latest_tool_json(messages, "get_vehicle_window_positions")
        window_positions = _window_positions_from_result(windows)
        if not window_positions:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the air conditioning safely because the window position result is missing required window information."
                ),
                warnings=["AC blocked by missing window result field"],
            )
        fixed_calls: list[dict[str, Any]] = []
        if not _tool_has_parameter(tool_map, "open_close_window", "window") and any(
            percentage > 20 for percentage in window_positions.values()
        ):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the air conditioning efficiently because some windows must be closed, but window control is unavailable."
                ),
                warnings=["AC blocked by open windows and missing window control"],
            )
        for window, percentage in window_positions.items():
            if percentage > 20:
                fixed_calls.append(
                    {
                        "tool_name": "open_close_window",
                        "arguments": {"window": window, "percentage": 0},
                    }
                )
        fan_speed = _find_number(climate, "fan_speed")
        fixed_calls.append(
            {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
        )
        if fan_speed is not None and fan_speed < 1:
            if not _tool_has_parameter(tool_map, "set_fan_speed", "level"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot turn on the air conditioning safely because fan speed control is unavailable."
                    ),
                    warnings=["AC blocked by missing fan speed control"],
                )
            fixed_calls.append({"tool_name": "set_fan_speed", "arguments": {"level": 1}})
        _append_preserved_calls(
            fixed_calls,
            calls,
            skip_names={"set_air_conditioning", "open_close_window", "set_fan_speed"},
        )
        return GuardDecision(
            action=_tool_calls(_dedupe_tool_calls(fixed_calls)),
            warnings=["AC policy actions from proposed tool call"],
        )

    if any(call.get("tool_name") == "set_window_defrost" and (call.get("arguments") or {}).get("on") is True for call in calls):
        if "get_climate_settings" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_climate_settings"):
            return GuardDecision(
                action=_tool_calls([{"tool_name": "get_climate_settings", "arguments": {}}]),
                warnings=["split climate check before defrost state change"],
            )
        if "get_vehicle_window_positions" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions"):
            return GuardDecision(
                action=_tool_calls([{"tool_name": "get_vehicle_window_positions", "arguments": {}}]),
                warnings=["split window check before defrost state change"],
            )
        if "get_climate_settings" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate window defrost safely because climate status information is unavailable."
                ),
                warnings=["defrost blocked by unavailable climate status lookup"],
            )
        if "get_vehicle_window_positions" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate window defrost safely because window position information is unavailable."
                ),
                warnings=["defrost blocked by unavailable window position lookup"],
            )
        climate = _latest_tool_json(messages, "get_climate_settings")
        if (
            _find_number(climate, "fan_speed") is None
            or _find_string(climate, "fan_airflow_direction") is None
            or _find_bool(climate, "air_conditioning") is None
        ):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate window defrost safely because the climate result is missing required fields."
                ),
                warnings=["defrost blocked by missing climate result field"],
            )
        windows = _latest_tool_json(messages, "get_vehicle_window_positions")
        window_positions = _window_positions_from_result(windows)
        if not window_positions:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate window defrost safely because the window position result is missing required window information."
                ),
                warnings=["defrost blocked by missing window result field"],
            )
        fixed_calls: list[dict[str, Any]] = []
        if any(value > 20 for value in window_positions.values()):
            if not _tool_has_parameter(tool_map, "open_close_window", "window"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot activate defrost efficiently because the windows need to be closed, but window control is unavailable."
                    ),
                    warnings=["defrost blocked by missing window control"],
                )
            fixed_calls.append(
                {
                    "tool_name": "open_close_window",
                    "arguments": {"window": "ALL", "percentage": 0},
                }
            )
        proposed_defrost = next(
            call
            for call in calls
            if call.get("tool_name") == "set_window_defrost"
            and (call.get("arguments") or {}).get("on") is True
        )
        fixed_calls.append(proposed_defrost)
        fan_speed = _find_number(climate, "fan_speed")
        airflow = _find_string(climate, "fan_airflow_direction")
        ac_on = _find_bool(climate, "air_conditioning")
        if fan_speed is not None and fan_speed < 2:
            if not _tool_has_parameter(tool_map, "set_fan_speed", "level"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot activate defrost safely because fan speed control is unavailable."
                    ),
                    warnings=["defrost blocked by missing fan speed control"],
                )
            fixed_calls.append({"tool_name": "set_fan_speed", "arguments": {"level": 2}})
        if airflow is not None and "WINDSHIELD" not in airflow:
            if not _tool_has_parameter(tool_map, "set_fan_airflow_direction", "direction"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot activate defrost safely because airflow direction control is unavailable."
                    ),
                    warnings=["defrost blocked by missing airflow control"],
                )
            fixed_calls.append(
                {
                    "tool_name": "set_fan_airflow_direction",
                    "arguments": {"direction": _windshield_airflow_direction(airflow)},
                }
            )
        if ac_on is False:
            if not _tool_has_parameter(tool_map, "set_air_conditioning", "on"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot activate defrost safely because air conditioning control is unavailable."
                    ),
                    warnings=["defrost blocked by missing AC control"],
                )
            fixed_calls.append({"tool_name": "set_air_conditioning", "arguments": {"on": True}})
        _append_preserved_calls(
            fixed_calls,
            calls,
            skip_names={
                "set_window_defrost",
                "open_close_window",
                "set_fan_speed",
                "set_fan_airflow_direction",
                "set_air_conditioning",
            },
        )
        return GuardDecision(
            action=_tool_calls(_dedupe_tool_calls(fixed_calls)),
            warnings=["defrost policy actions from proposed tool call"],
        )

    if any(call.get("tool_name") == "set_fog_lights" and (call.get("arguments") or {}).get("on") is True for call in calls):
        if "get_weather" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights because current weather information is unavailable."
                ),
                warnings=["fog lights weather check unavailable"],
            )
        if "get_weather" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_weather"):
            weather_args = _current_weather_args_from_system(messages, tool_map)
            if weather_args is not None:
                return GuardDecision(
                    action=_tool_calls([{"tool_name": "get_weather", "arguments": weather_args}]),
                    warnings=["split weather check before fog lights"],
                )
        weather_condition = _latest_weather_condition(messages)
        if weather_condition is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights because the weather result is missing the required condition information."
                ),
                warnings=["fog lights blocked by missing weather condition"],
            )
        if "get_exterior_lights_status" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_exterior_lights_status"):
            return GuardDecision(
                action=_tool_calls([{"tool_name": "get_exterior_lights_status", "arguments": {}}]),
                warnings=["split exterior light check before fog lights"],
            )
        if "get_exterior_lights_status" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights safely because exterior light status is unavailable."
                ),
                warnings=["exterior light status unavailable"],
            )
        lights = _latest_tool_json(messages, "get_exterior_lights_status")
        low_on = _find_bool(lights, "head_lights_low_beams")
        high_on = _find_bool(lights, "head_lights_high_beams")
        if low_on is None or high_on is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights safely because the exterior light status is missing required headlight information."
                ),
                warnings=["fog lights blocked by missing exterior light field"],
            )
        if _fog_lights_require_confirmation(weather_condition) and not _has_recent_confirmation(messages):
            return GuardDecision(
                action={
                    "action": "respond",
                    "content": (
                        f"The current weather is {weather_condition}, so I need your confirmation "
                        "before turning on the fog lights. Should I proceed?"
                    ),
                },
                warnings=["fog lights require weather confirmation"],
            )
        fixed_calls: list[dict[str, Any]] = []
        if low_on is False:
            if not _tool_has_parameter(tool_map, "set_head_lights_low_beams", "on"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot turn on the fog lights because low beam headlights must be on first, but that control is unavailable."
                    ),
                    warnings=["fog lights blocked by missing low beam control"],
                )
            fixed_calls.append({"tool_name": "set_head_lights_low_beams", "arguments": {"on": True}})
        if high_on is True:
            if not _tool_has_parameter(tool_map, "set_head_lights_high_beams", "on"):
                return GuardDecision(
                    action=_cannot_do(
                        "I cannot turn on the fog lights because high beam headlights must be off first, but that control is unavailable."
                    ),
                    warnings=["fog lights blocked by missing high beam control"],
                )
            fixed_calls.append({"tool_name": "set_head_lights_high_beams", "arguments": {"on": False}})
        fixed_calls.append({"tool_name": "set_fog_lights", "arguments": {"on": True}})
        _append_preserved_calls(
            fixed_calls,
            calls,
            skip_names={
                "set_fog_lights",
                "set_head_lights_low_beams",
                "set_head_lights_high_beams",
            },
        )
        return GuardDecision(
            action=_tool_calls(_dedupe_tool_calls(fixed_calls)),
            warnings=["fog light policy actions from proposed tool call"],
        )

    if any(call.get("tool_name") == "set_head_lights_high_beams" and (call.get("arguments") or {}).get("on") is True for call in calls):
        if "get_exterior_lights_status" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_exterior_lights_status"):
            return GuardDecision(
                action=_tool_calls([{"tool_name": "get_exterior_lights_status", "arguments": {}}]),
                warnings=["split exterior light check before high beams"],
            )
        if "get_exterior_lights_status" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on high beam headlights safely because exterior light status is unavailable."
                ),
                warnings=["high beams blocked by unavailable exterior light status"],
            )
        lights = _latest_tool_json(messages, "get_exterior_lights_status")
        if lights is None or _find_bool(lights, "fog_lights") is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on high beam headlights safely because the exterior light status is missing fog light information."
                ),
                warnings=["high beams blocked by missing fog light field"],
            )
        if lights is not None and _find_bool(lights, "fog_lights") is True:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the high beam headlights while the fog lights are on."
                ),
                warnings=["high beams blocked by active fog lights"],
            )
        if not _has_recent_confirmation(messages):
            return GuardDecision(
                action={
                    "action": "respond",
                    "content": "I can turn on the high beam headlights, but I need your confirmation first. Should I proceed?",
                },
                warnings=["high beams require confirmation"],
            )

    return GuardDecision()


def _ac_policy_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if not _requests_ac_on(text):
        return GuardDecision()
    if not _tool_has_parameter(tool_map, "set_air_conditioning", "on"):
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the air conditioning because that control capability is unavailable."
            ),
            warnings=["air conditioning control unavailable"],
        )
    if "get_climate_settings" not in tool_map:
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the air conditioning safely because climate status information is unavailable."
            ),
            warnings=["AC blocked by unavailable climate status lookup"],
        )

    missing_gets = []
    if (
        "get_climate_settings" in tool_map
        and not _tool_called_after_latest_actionable_user(messages, "get_climate_settings")
    ):
        missing_gets.append({"tool_name": "get_climate_settings", "arguments": {}})
    if (
        "get_vehicle_window_positions" in tool_map
        and not _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions")
    ):
        missing_gets.append({"tool_name": "get_vehicle_window_positions", "arguments": {}})
    elif "get_vehicle_window_positions" not in tool_map and not _requests_close_all_windows(text):
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the air conditioning safely because window position information is unavailable."
            ),
            warnings=["AC blocked by unavailable window position lookup"],
        )
    if missing_gets:
        return GuardDecision(
            action=_tool_calls(missing_gets),
            warnings=["checking AC policy preconditions"],
        )

    climate = _latest_tool_json(messages, "get_climate_settings")
    windows = _latest_tool_json(messages, "get_vehicle_window_positions")
    fan_speed = _find_number(climate, "fan_speed")
    if _tool_called_after_latest_actionable_user(messages, "get_climate_settings") and fan_speed is None:
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the air conditioning safely because the climate result is missing the fan speed."
            ),
            warnings=["AC blocked by missing climate result field"],
        )

    window_positions = _window_positions_from_result(windows)
    if _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions") and not window_positions:
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the air conditioning safely because the window position result is missing required window information."
            ),
            warnings=["AC blocked by missing window result field"],
        )

    calls: list[dict[str, Any]] = []
    if _requests_close_all_windows(text) and not window_positions:
        if not _tool_has_parameter(tool_map, "open_close_window", "window"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot optimize the air conditioning because window control is unavailable."
                ),
                warnings=["AC blocked by missing window control"],
            )
        calls.append(
            {
                "tool_name": "open_close_window",
                "arguments": {"window": "ALL", "percentage": 0},
            }
        )
    else:
        for window, percentage in window_positions.items():
            if percentage > 20:
                if not _tool_has_parameter(tool_map, "open_close_window", "window"):
                    return GuardDecision(
                        action=_cannot_do(
                            "I cannot turn on the air conditioning efficiently because some windows must be closed, but window control is unavailable."
                        ),
                        warnings=["AC blocked by open windows and missing window control"],
                    )
                calls.append(
                    {
                        "tool_name": "open_close_window",
                        "arguments": {"window": window, "percentage": 0},
                    }
                )

    calls.append({"tool_name": "set_air_conditioning", "arguments": {"on": True}})
    if fan_speed is None or fan_speed < 1:
        if not _tool_has_parameter(tool_map, "set_fan_speed", "level"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the air conditioning safely because fan speed control is unavailable."
                ),
                warnings=["AC blocked by missing fan speed control"],
            )
        calls.append({"tool_name": "set_fan_speed", "arguments": {"level": 1}})

    air_mode = _requested_air_circulation_mode(text)
    if air_mode:
        if not _tool_has_parameter(tool_map, "set_air_circulation", "mode"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot set the requested air circulation mode because that capability is unavailable."
                ),
                warnings=["air circulation control unavailable"],
            )
        calls.append(
            {"tool_name": "set_air_circulation", "arguments": {"mode": air_mode}}
        )
    return GuardDecision(action=_tool_calls(_dedupe_tool_calls(calls)), warnings=["AC policy actions"])


def _defrost_policy_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if not _requests_defrost_on(text):
        return GuardDecision()
    if not (
        _tool_has_parameter(tool_map, "set_window_defrost", "on")
        and _tool_has_parameter(tool_map, "set_window_defrost", "defrost_window")
    ):
        return GuardDecision(
            action=_cannot_do(
                "I cannot activate window defrost because that control capability is unavailable."
            ),
            warnings=["defrost control unavailable"],
        )
    if "get_climate_settings" not in tool_map:
        return GuardDecision(
            action=_cannot_do(
                "I cannot activate window defrost safely because climate status information is unavailable."
            ),
            warnings=["defrost blocked by unavailable climate status lookup"],
        )

    missing_gets = []
    if (
        "get_climate_settings" in tool_map
        and not _tool_called_after_latest_actionable_user(messages, "get_climate_settings")
    ):
        missing_gets.append({"tool_name": "get_climate_settings", "arguments": {}})
    if (
        "get_vehicle_window_positions" in tool_map
        and not _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions")
        and not _requests_close_all_windows(text)
    ):
        missing_gets.append({"tool_name": "get_vehicle_window_positions", "arguments": {}})
    elif "get_vehicle_window_positions" not in tool_map and not _requests_close_all_windows(text):
        return GuardDecision(
            action=_cannot_do(
                "I cannot activate window defrost safely because window position information is unavailable."
            ),
            warnings=["defrost blocked by unavailable window position lookup"],
        )
    if missing_gets:
        return GuardDecision(
            action=_tool_calls(missing_gets),
            warnings=["checking defrost policy preconditions"],
        )

    climate = _latest_tool_json(messages, "get_climate_settings")
    fan_speed = _find_number(climate, "fan_speed")
    airflow = _find_string(climate, "fan_airflow_direction")
    ac_on = _find_bool(climate, "air_conditioning")
    if _tool_called_after_latest_actionable_user(messages, "get_climate_settings") and (
        fan_speed is None or airflow is None or ac_on is None
    ):
        return GuardDecision(
            action=_cannot_do(
                "I cannot activate window defrost safely because the climate result is missing required fields."
            ),
            warnings=["defrost blocked by missing climate result field"],
        )

    windows = _latest_tool_json(messages, "get_vehicle_window_positions")
    window_positions = _window_positions_from_result(windows)
    if (
        _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions")
        and not window_positions
        and not _requests_close_all_windows(text)
    ):
        return GuardDecision(
            action=_cannot_do(
                "I cannot activate window defrost safely because the window position result is missing required window information."
            ),
            warnings=["defrost blocked by missing window result field"],
        )

    defrost_window = _requested_defrost_window(text)
    if defrost_window is None:
        return GuardDecision(
            action={"action": "respond", "content": "Which window should I defrost, front, rear, or all?"},
            warnings=["defrost target ambiguous"],
        )

    calls: list[dict[str, Any]] = []
    if _requests_close_all_windows(text) or any(value > 20 for value in window_positions.values()):
        if not _tool_has_parameter(tool_map, "open_close_window", "window"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate defrost efficiently because the windows need to be closed, but window control is unavailable."
                ),
                warnings=["defrost blocked by missing window control"],
            )
        calls.append(
            {
                "tool_name": "open_close_window",
                "arguments": {"window": "ALL", "percentage": 0},
            }
        )

    calls.append(
        {
            "tool_name": "set_window_defrost",
            "arguments": {"on": True, "defrost_window": defrost_window},
        }
    )
    if fan_speed is None or fan_speed < 2:
        if not _tool_has_parameter(tool_map, "set_fan_speed", "level"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate defrost safely because fan speed control is unavailable."
                ),
                warnings=["defrost blocked by missing fan speed control"],
            )
        calls.append({"tool_name": "set_fan_speed", "arguments": {"level": 2}})
    if airflow is None or "WINDSHIELD" not in airflow:
        if not _tool_has_parameter(tool_map, "set_fan_airflow_direction", "direction"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate defrost safely because airflow direction control is unavailable."
                ),
                warnings=["defrost blocked by missing airflow control"],
            )
        calls.append(
            {
                "tool_name": "set_fan_airflow_direction",
                "arguments": {"direction": _windshield_airflow_direction(airflow)},
            }
        )
    if ac_on is None or not ac_on:
        if not _tool_has_parameter(tool_map, "set_air_conditioning", "on"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot activate defrost safely because air conditioning control is unavailable."
                ),
                warnings=["defrost blocked by missing AC control"],
            )
        calls.append({"tool_name": "set_air_conditioning", "arguments": {"on": True}})
    return GuardDecision(action=_tool_calls(_dedupe_tool_calls(calls)), warnings=["defrost policy actions"])


def _fog_light_policy_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if not _requests_fog_lights_on(text):
        return GuardDecision()
    if not _tool_has_parameter(tool_map, "set_fog_lights", "on"):
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the fog lights because that control capability is unavailable."
            ),
            warnings=["fog light control unavailable"],
        )

    weather_condition = _latest_weather_condition(messages)
    if weather_condition is None:
        if _tool_called_after_latest_actionable_user(messages, "get_weather"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights because the weather result is missing the required condition information."
                ),
                warnings=["fog lights blocked by missing weather condition"],
            )
        weather_args = _current_weather_args_from_system(messages, tool_map)
        if weather_args is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights because current weather information is unavailable."
                ),
                warnings=["fog lights weather check unavailable"],
            )
        return GuardDecision(
            action=_tool_calls([{"tool_name": "get_weather", "arguments": weather_args}]),
            warnings=["checking weather before fog lights"],
        )

    lights = _latest_tool_json(messages, "get_exterior_lights_status")
    if lights is None:
        if "get_exterior_lights_status" not in tool_map:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights safely because exterior light status is unavailable."
                ),
                warnings=["exterior light status unavailable"],
            )
        return GuardDecision(
            action=_tool_calls([{"tool_name": "get_exterior_lights_status", "arguments": {}}]),
            warnings=["checking exterior lights before fog lights"],
        )
    low_on = _find_bool(lights, "head_lights_low_beams")
    high_on = _find_bool(lights, "head_lights_high_beams")
    if low_on is None or high_on is None:
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the fog lights safely because the exterior light status is missing required headlight information."
            ),
            warnings=["fog lights blocked by missing exterior light field"],
        )

    if _fog_lights_require_confirmation(weather_condition) and not _has_recent_confirmation(messages):
        return GuardDecision(
            action={
                "action": "respond",
                "content": (
                    f"The current weather is {weather_condition}, so I need your confirmation "
                    "before turning on the fog lights. Should I proceed?"
                ),
            },
            warnings=["fog lights require weather confirmation"],
        )

    calls: list[dict[str, Any]] = []
    if not low_on:
        if not _tool_has_parameter(tool_map, "set_head_lights_low_beams", "on"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights because low beam headlights must be on first, but that control is unavailable."
                ),
                warnings=["fog lights blocked by missing low beam control"],
            )
        calls.append({"tool_name": "set_head_lights_low_beams", "arguments": {"on": True}})
    if high_on:
        if not _tool_has_parameter(tool_map, "set_head_lights_high_beams", "on"):
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the fog lights because high beam headlights must be off first, but that control is unavailable."
                ),
                warnings=["fog lights blocked by missing high beam control"],
            )
        calls.append({"tool_name": "set_head_lights_high_beams", "arguments": {"on": False}})
    calls.append({"tool_name": "set_fog_lights", "arguments": {"on": True}})
    return GuardDecision(action=_tool_calls(calls), warnings=["fog light policy actions"])


def _high_beam_policy_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if not _requests_high_beams_on(text):
        return GuardDecision()
    if not _tool_has_parameter(tool_map, "set_head_lights_high_beams", "on"):
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on the high beam headlights because that control capability is unavailable."
            ),
            warnings=["high beam control unavailable"],
        )
    lights = _latest_tool_json(messages, "get_exterior_lights_status")
    if lights is None and "get_exterior_lights_status" in tool_map:
        return GuardDecision(
            action=_tool_calls([{"tool_name": "get_exterior_lights_status", "arguments": {}}]),
            warnings=["checking exterior lights before high beams"],
        )
    if lights is None:
        return GuardDecision(
            action=_cannot_do(
                "I cannot turn on high beam headlights safely because exterior light status is unavailable."
            ),
            warnings=["high beams blocked by unavailable exterior light status"],
        )
    if lights is not None:
        fog_on = _find_bool(lights, "fog_lights")
        if fog_on is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on high beam headlights safely because the exterior light status is missing fog light information."
                ),
                warnings=["high beams blocked by missing fog light field"],
            )
        if fog_on:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot turn on the high beam headlights while the fog lights are on."
                ),
                warnings=["high beams blocked by active fog lights"],
            )
    if not _has_recent_confirmation(messages):
        return GuardDecision(
            action={
                "action": "respond",
                "content": "I can turn on the high beam headlights, but I need your confirmation first. Should I proceed?",
            },
            warnings=["high beams require confirmation"],
        )
    return GuardDecision(
        action=_tool_calls(
            [{"tool_name": "set_head_lights_high_beams", "arguments": {"on": True}}]
        ),
        warnings=["direct high beam action after confirmation"],
    )


def _window_policy_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if not _requests_window_control(text):
        return GuardDecision()
    window = _requested_window(text)
    percentage = _requested_percentage(text)
    if window is None or percentage is None:
        return GuardDecision()
    if not (
        _tool_has_parameter(tool_map, "open_close_window", "window")
        and _tool_has_parameter(tool_map, "open_close_window", "percentage")
    ):
        return GuardDecision(
            action=_cannot_do(
                "I cannot adjust the windows because that control capability is unavailable."
            ),
            warnings=["window control unavailable"],
        )
    if percentage > 25:
        climate = _latest_tool_json(messages, "get_climate_settings")
        if climate is None and "get_climate_settings" in tool_map:
            return GuardDecision(
                action=_tool_calls([{"tool_name": "get_climate_settings", "arguments": {}}]),
                warnings=["checking AC state before opening windows"],
            )
        if climate is None:
            return GuardDecision(
                action=_cannot_do(
                    "I cannot open the windows more than 25% safely because air conditioning status is unavailable."
                ),
                warnings=["window opening blocked by unavailable climate status"],
            )
        ac_on = _find_bool(climate, "air_conditioning")
        if ac_on and not _has_recent_confirmation(messages):
            return GuardDecision(
                action={
                    "action": "respond",
                    "content": (
                        "Opening the windows more than 25% while air conditioning is on is energy inefficient. "
                        "Do you still want me to proceed?"
                    ),
                },
                warnings=["window opening requires AC efficiency confirmation"],
            )
    return GuardDecision(
        action=_tool_calls(
            [
                {
                    "tool_name": "open_close_window",
                    "arguments": {"window": window, "percentage": percentage},
                }
            ]
        ),
        warnings=["direct window control"],
    )


def _direct_simple_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if _has_multiple_direct_control_intents(text):
        return GuardDecision()

    if _requests_trunk_control(text) and _tool_has_parameter(
        tool_map, "open_close_trunk_door", "action"
    ):
        action = "CLOSE" if _has_any(text, ("close", "shut")) else "OPEN"
        if not _has_recent_confirmation(messages):
            return GuardDecision(
                action=_cannot_do(
                    f"I can {action.lower()} the trunk door, but I need your confirmation first. Please say yes to confirm."
                ),
                warnings=["trunk action requires confirmation"],
            )
        return GuardDecision(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_trunk_door",
                        "arguments": {"action": action},
                    }
                ],
            },
            warnings=["direct trunk control"],
        )

    air_mode = _requested_air_circulation_mode(text)
    if air_mode and _tool_has_parameter(tool_map, "set_air_circulation", "mode"):
        return GuardDecision(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "set_air_circulation",
                        "arguments": {"mode": air_mode},
                    }
                ],
            },
            warnings=["direct air circulation control"],
        )

    airflow_direction = _requested_airflow_direction(text)
    if airflow_direction and _tool_has_parameter(
        tool_map, "set_fan_airflow_direction", "direction"
    ):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "set_fan_airflow_direction",
                        "arguments": {"direction": airflow_direction},
                    }
                ]
            ),
            warnings=["direct fan airflow control"],
        )

    fan_speed = _requested_fan_speed_level(text)
    if fan_speed is not None and _tool_has_parameter(tool_map, "set_fan_speed", "level"):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "set_fan_speed",
                        "arguments": {"level": fan_speed},
                    }
                ]
            ),
            warnings=["direct fan speed control"],
        )

    low_beam = _requested_low_beam_state(text)
    if low_beam is not None and _tool_has_parameter(
        tool_map, "set_head_lights_low_beams", "on"
    ):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "set_head_lights_low_beams",
                        "arguments": {"on": low_beam},
                    }
                ]
            ),
            warnings=["direct low beam control"],
        )

    reading_light = _requested_reading_light(text)
    if reading_light and (
        _tool_has_parameter(tool_map, "set_reading_light", "position")
        and _tool_has_parameter(tool_map, "set_reading_light", "on")
    ):
        return GuardDecision(
            action=_tool_calls(
                [{"tool_name": "set_reading_light", "arguments": reading_light}]
            ),
            warnings=["direct reading light control"],
        )

    steering_level = _requested_steering_wheel_heating_level(text)
    if steering_level is not None and _tool_has_parameter(
        tool_map, "set_steering_wheel_heating", "level"
    ):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "set_steering_wheel_heating",
                        "arguments": {"level": steering_level},
                    }
                ]
            ),
            warnings=["direct steering wheel heating control"],
        )

    climate_temperature = _requested_climate_temperature(text)
    climate_zone = _requested_temperature_zone(text)
    if (
        climate_temperature is not None
        and climate_zone is not None
        and _tool_has_parameter(tool_map, "set_climate_temperature", "temperature")
        and _tool_has_parameter(tool_map, "set_climate_temperature", "seat_zone")
    ):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "set_climate_temperature",
                        "arguments": {
                            "temperature": climate_temperature,
                            "seat_zone": climate_zone,
                        },
                    }
                ]
            ),
            warnings=["direct climate temperature control"],
        )

    seat_heating = _requested_seat_heating(text)
    if seat_heating and (
        _tool_has_parameter(tool_map, "set_seat_heating", "level")
        and _tool_has_parameter(tool_map, "set_seat_heating", "seat_zone")
    ):
        return GuardDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "set_seat_heating",
                        "arguments": seat_heating,
                    }
                ]
            ),
            warnings=["direct seat heating control"],
        )

    if _requests_ambient_control(text):
        if not (
            _tool_has_parameter(tool_map, "set_ambient_lights", "on")
            and _tool_has_parameter(tool_map, "set_ambient_lights", "lightcolor")
        ):
            return GuardDecision()
        color = _requested_ambient_color(text) or _ambient_color_from_preferences(messages)
        if color:
            return GuardDecision(
                action={
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "tool_name": "set_ambient_lights",
                            "arguments": {
                                "on": color != "NONE",
                                "lightcolor": color,
                            },
                        }
                    ],
                },
                warnings=["direct ambient light control"],
            )
        if (
            "get_user_preferences" in tool_map
            and not _tool_called_after_latest_actionable_user(
                messages, "get_user_preferences"
            )
        ):
            return GuardDecision(
                action={
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "tool_name": "get_user_preferences",
                            "arguments": {
                                "preference_categories": {
                                    "vehicle_settings": {
                                        "vehicle_settings": True,
                                        "climate_control": True,
                                    }
                                }
                            },
                        }
                    ],
                },
                warnings=["ambient light color ambiguous; checking preferences"],
            )

    return GuardDecision()


def _requests_trunk_control(text: str) -> bool:
    return "trunk" in text and (
        _has_control_verb(text)
        or _has_any(text, ("access", "put something", "load", "unload"))
    )


def _has_multiple_direct_control_intents(text: str) -> bool:
    checks = (
        lambda value: _requests_trunk_control(value),
        lambda value: _requested_air_circulation_mode(value) is not None,
        lambda value: _requested_airflow_direction(value) is not None,
        lambda value: _requested_fan_speed_level(value) is not None,
        lambda value: _requested_low_beam_state(value) is not None,
        lambda value: _requested_reading_light(value) is not None,
        lambda value: _requested_steering_wheel_heating_level(value) is not None,
        lambda value: _requested_climate_temperature(value) is not None,
        lambda value: _requested_seat_heating(value) is not None,
        lambda value: _requests_ambient_control(value),
    )
    return sum(1 for check in checks if check(text)) > 1


def _requested_air_circulation_mode(text: str) -> str | None:
    if _has_any(text, ("fresh air", "outside air")):
        return "FRESH_AIR"
    if _has_any(text, ("stuffy", "stale")) and _has_any(text, ("air circulation", "recirculation")):
        return "FRESH_AIR"
    if "recirculation" in text:
        if _has_any(text, ("turn off", "disable", "stop")):
            return "FRESH_AIR"
        if _has_any(text, ("turn on", "enable", "set", "use")):
            return "RECIRCULATION"
    if "auto" in text and _has_any(text, ("air circulation", "recirculation")):
        return "AUTO"
    return None


def _requested_airflow_direction(text: str) -> str | None:
    has_airflow_subject = _has_any(
        text,
        ("airflow", "fan direction", "air flow", "vents", "windshield", "feet"),
    ) or _has_word(text, "head")
    if not has_airflow_subject:
        return None
    mentions_head = _has_word(text, "head")
    if "windshield" in text and "feet" in text and mentions_head:
        return "WINDSHIELD_HEAD_FEET"
    if "windshield" in text and mentions_head:
        return "WINDSHIELD_HEAD"
    if "windshield" in text and "feet" in text:
        return "WINDSHIELD_FEET"
    if mentions_head and "feet" in text:
        return "HEAD_FEET"
    if "windshield" in text:
        return "WINDSHIELD"
    if "feet" in text:
        return "FEET"
    if mentions_head:
        return "HEAD"
    return None


def _requested_fan_speed_level(text: str) -> int | None:
    if "fan" not in text or not _has_any(text, ("speed", "level")):
        return None
    match = re.search(r"\b(?:level|speed)\s*(\d)\b", text)
    if not match:
        return None
    level = int(match.group(1))
    if 0 <= level <= 5:
        return level
    return None


def _requested_low_beam_state(text: str) -> bool | None:
    if "low beam" not in text:
        return None
    if _has_any(text, ("turn off", "switch off", "disable", "off")):
        return False
    if _has_any(text, ("turn on", "switch on", "enable", "on")):
        return True
    return None


def _requested_reading_light(text: str) -> dict[str, Any] | None:
    if not _requests_reading_light_control(text):
        return None
    if _has_any(text, ("turn off", "switch off", "disable", "off")):
        on = False
    elif _has_any(text, ("turn on", "switch on", "enable", "on")):
        on = True
    else:
        return None
    if "all" in text:
        position = "ALL"
    elif "driver rear" in text or "right rear" in text:
        position = "DRIVER_REAR"
    elif "passenger rear" in text or "left rear" in text:
        position = "PASSENGER_REAR"
    elif "passenger" in text:
        position = "PASSENGER"
    elif "driver" in text or "my reading light" in text:
        position = "DRIVER"
    else:
        return None
    return {"position": position, "on": on}


def _requested_steering_wheel_heating_level(text: str) -> int | None:
    if "steering wheel" not in text or not _has_any(text, ("heat", "warm")):
        return None
    if _has_any(text, ("turn off", "disable", "off")):
        return 0
    match = re.search(r"\blevel\s*([0-3])\b", text)
    if match:
        return int(match.group(1))
    return None


def _requested_climate_temperature(text: str) -> float | None:
    if not _has_any(text, ("temperature", "climate", "degree", "degrees", "celsius")):
        return None
    match = re.search(r"\b(1[6-9]|2[0-8])(?:\.(5))?\s*(?:degrees?|celsius|°c)?\b", text)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value += 0.5
    return value


def _requested_temperature_zone(text: str) -> str | None:
    if _has_any(text, ("all zones", "both zones", "whole cabin", "entire cabin", "cabin")):
        return "ALL_ZONES"
    if "driver" in text or "my side" in text or "my zone" in text:
        return "DRIVER"
    if "passenger" in text:
        return "PASSENGER"
    return None


def _requested_seat_heating(text: str) -> dict[str, Any] | None:
    if not _has_any(text, ("seat heating", "heated seat")):
        return None
    match = re.search(r"\blevel\s*([0-3])\b", text)
    if not match:
        return None
    if _has_any(text, ("both", "all", "driver and passenger")):
        zone = "ALL_ZONES"
    elif "passenger" in text:
        zone = "PASSENGER"
    elif "driver" in text or "my seat" in text:
        zone = "DRIVER"
    else:
        return None
    return {"level": int(match.group(1)), "seat_zone": zone}


def _requests_ambient_control(text: str) -> bool:
    return (
        _has_any(text, ("ambient", "surrounding light"))
        and _has_control_verb(text)
    )


def _requested_ambient_color(text: str) -> str | None:
    if _has_any(text, ("off", "turn off", "disable")):
        return "NONE"
    for color in _AMBIENT_COLORS:
        if color.lower() in text:
            return color
    return None


def _ambient_color_from_preferences(messages: list[dict[str, Any]]) -> str | None:
    content = _latest_tool_content(messages, "get_user_preferences")
    if not content:
        return None
    text = content.lower()
    for color in _AMBIENT_COLORS:
        if color.lower() in text:
            return color
    return None


def _requested_defrost_window(text: str) -> str | None:
    if "front" in text or "windshield" in text or "windscreen" in text:
        return "FRONT"
    if "rear" in text or "back window" in text:
        return "REAR"
    if "all" in text:
        return "ALL"
    if _has_any(text, ("fogged", "fog up", "condensation", "hard to see", "visibility")):
        return "FRONT"
    return None


def _windshield_airflow_direction(current: str | None) -> str:
    if not current:
        return "WINDSHIELD"
    directions = set(current.split("_"))
    directions.add("WINDSHIELD")
    order = ["WINDSHIELD", "HEAD", "FEET"]
    return "_".join(part for part in order if part in directions)


def _fog_lights_require_confirmation(condition: str) -> bool:
    allowed_without_confirmation = {"cloudy_and_thunderstorm", "cloudy_and_hail"}
    return condition not in allowed_without_confirmation


def _requested_window(text: str) -> str | None:
    if "all" in text:
        return "ALL"
    if "driver rear" in text or "right rear" in text:
        return "DRIVER_REAR"
    if "passenger rear" in text or "left rear" in text:
        return "PASSENGER_REAR"
    if "driver" in text:
        return "DRIVER"
    if "passenger" in text:
        return "PASSENGER"
    return None


def _requested_percentage(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*(?:%|percent)\b", text)
    if match:
        return max(0, min(100, int(match.group(1))))
    if _has_any(text, ("half", "halfway")):
        return 50
    if _has_any(text, ("fully open", "open all the way", "completely open")):
        return 100
    if _has_any(text, ("close", "closed", "shut")) and _has_any(text, ("fully", "completely", "all")):
        return 0
    return None


def _latest_tool_json(messages: list[dict[str, Any]], tool_name: str) -> Any:
    content = _latest_tool_content(messages, tool_name)
    if not content:
        return None
    return _parse_json(content)


def _find_number(value: Any, key: str) -> float | None:
    found = _find_key(value, key)
    if isinstance(found, (int, float)) and not isinstance(found, bool):
        return float(found)
    if isinstance(found, str):
        try:
            return float(found)
        except ValueError:
            return None
    return None


def _find_bool(value: Any, key: str) -> bool | None:
    found = _find_key(value, key)
    if isinstance(found, bool):
        return found
    if isinstance(found, str):
        lowered = found.lower()
        if lowered in {"true", "on", "yes"}:
            return True
        if lowered in {"false", "off", "no"}:
            return False
    return None


def _find_string(value: Any, key: str) -> str | None:
    found = _find_key(value, key)
    if isinstance(found, str) and found:
        return found
    return None


def _window_positions_from_result(value: Any) -> dict[str, int]:
    mapping = {
        "window_driver_position": "DRIVER",
        "window_passenger_position": "PASSENGER",
        "window_driver_rear_position": "DRIVER_REAR",
        "window_passenger_rear_position": "PASSENGER_REAR",
    }
    positions: dict[str, int] = {}
    for key, window in mapping.items():
        number = _find_number(value, key)
        if number is not None:
            positions[window] = int(number)
    return positions


def _latest_assistant_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return list(message.get("tool_calls") or [])
    return []


def _history_tool_call_name(call: dict[str, Any]) -> str:
    return str(
        call.get("tool_name")
        or call.get("name")
        or call.get("function", {}).get("name")
        or ""
    )


def _trailing_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        results.append(message)
    return list(reversed(results))


def _tool_result_status_is_success(message: dict[str, Any]) -> bool:
    data = _parse_json(str(message.get("content") or ""))
    status = _find_key(data, "status")
    return isinstance(status, str) and status.upper() == "SUCCESS"


def _is_state_changing_tool_name(name: str) -> bool:
    return name.startswith(("open_", "set_", "navigation_", "delete_", "call_", "send_"))


def _append_preserved_calls(
    target: list[dict[str, Any]],
    original: list[dict[str, Any]],
    *,
    skip_names: set[str],
) -> None:
    for call in original:
        name = str(call.get("tool_name") or "")
        if name in skip_names:
            continue
        target.append(call)


def _dedupe_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        key = json.dumps(call, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


_AMBIENT_COLORS = (
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "WHITE",
    "PINK",
    "ORANGE",
    "PURPLE",
    "CYAN",
    "NONE",
)

_NAVIGATION_EDIT_TOOLS = {
    "navigation_add_one_waypoint",
    "navigation_delete_waypoint",
    "navigation_replace_one_waypoint",
    "navigation_delete_destination",
    "navigation_replace_final_destination",
}

_NAVIGATION_ROUTE_DEPENDENT_TOOLS = {
    "set_new_navigation",
    "navigation_add_one_waypoint",
    "navigation_delete_waypoint",
    "navigation_replace_one_waypoint",
    "navigation_replace_final_destination",
}

_NAVIGATION_STATE_CHANGE_TOOLS = {
    "set_new_navigation",
    "delete_current_navigation",
    *_NAVIGATION_EDIT_TOOLS,
}

_CHARGING_REASONING_TOOLS = {
    "get_distance_by_soc",
    "calculate_charging_time_by_soc",
    "calculate_charging_soc_by_time",
}


def _has_any(text: str, pieces: tuple[str, ...]) -> bool:
    return any(piece in text for piece in pieces)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _has_control_verb(text: str) -> bool:
    return _has_any(
        text,
        (
            "open",
            "close",
            "turn",
            "set",
            "change",
            "adjust",
            "enable",
            "disable",
            "start",
            "stop",
            "increase",
            "decrease",
            "make",
        ),
    )


_MISSING_CAPABILITY_RULES = (
    {
        "predicate": lambda text: "trunk" in text and _has_control_verb(text),
        "tool_name": "open_close_trunk_door",
        "essential_parameters": ("action",),
        "label": "trunk door control",
    },
    {
        "predicate": lambda text: (
            _has_any(text, ("air circulation", "recirculation", "fresh air mode"))
            and _has_control_verb(text)
        ),
        "tool_name": "set_air_circulation",
        "essential_parameters": ("mode",),
        "label": "air circulation control",
    },
    {
        "predicate": lambda text: (
            _has_any(text, ("ambient", "surrounding light"))
            and _has_control_verb(text)
        ),
        "tool_name": "set_ambient_lights",
        "essential_parameters": ("on", "lightcolor"),
        "label": "ambient light control",
    },
    {
        "predicate": lambda text: (
            _has_any(text, ("airflow", "fan direction", "windshield", "feet vents", "head vents"))
            and _has_control_verb(text)
        ),
        "tool_name": "set_fan_airflow_direction",
        "essential_parameters": ("direction",),
        "label": "fan airflow direction control",
    },
    {
        "predicate": lambda text: (
            "fan" in text and _has_any(text, ("speed", "level", "increase", "decrease", "turn"))
        ),
        "tool_name": "set_fan_speed",
        "essential_parameters": ("level",),
        "label": "fan speed control",
    },
    {
        "predicate": lambda text: _has_any(text, ("air conditioning", "a/c", " ac ")) and _has_control_verb(text),
        "tool_name": "set_air_conditioning",
        "essential_parameters": ("on",),
        "label": "air conditioning control",
    },
    {
        "predicate": lambda text: _has_any(text, ("defrost", "defog")) and _has_control_verb(text),
        "tool_name": "set_window_defrost",
        "essential_parameters": ("on", "defrost_window"),
        "label": "window defrost control",
    },
    {
        "predicate": lambda text: _has_any(text, ("seat heating", "heated seat")) and _has_control_verb(text),
        "tool_name": "set_seat_heating",
        "essential_parameters": ("seat_zone", "level"),
        "label": "seat heating control",
    },
    {
        "predicate": lambda text: "steering wheel" in text and _has_any(text, ("heat", "warm")) and _has_control_verb(text),
        "tool_name": "set_steering_wheel_heating",
        "essential_parameters": ("level",),
        "label": "steering wheel heating control",
    },
    {
        "predicate": lambda text: "fog light" in text and _has_control_verb(text),
        "tool_name": "set_fog_lights",
        "essential_parameters": ("on",),
        "label": "fog light control",
    },
    {
        "predicate": lambda text: "high beam" in text and _has_control_verb(text),
        "tool_name": "set_head_lights_high_beams",
        "essential_parameters": ("on",),
        "label": "high beam headlight control",
    },
    {
        "predicate": lambda text: "low beam" in text and _has_control_verb(text),
        "tool_name": "set_head_lights_low_beams",
        "essential_parameters": ("on",),
        "label": "low beam headlight control",
    },
    {
        "predicate": lambda text: _requests_window_control(text),
        "tool_name": "open_close_window",
        "essential_parameters": ("window", "percentage"),
        "label": "window control",
    },
    {
        "predicate": lambda text: _has_any(text, ("call ", "phone ")) and _has_control_verb(text),
        "tool_name": "call_phone_by_number",
        "essential_parameters": ("phone_number",),
        "label": "phone call",
    },
    {
        "predicate": lambda text: _has_any(text, ("email", "mail")) and _has_control_verb(text),
        "tool_name": "send_email",
        "essential_parameters": ("email_addresses", "content_message"),
        "label": "email sending",
    },
    {
        "predicate": lambda text: _has_any(text, ("calendar", "meeting", "appointment")) and not _has_control_verb(text),
        "tool_name": "get_entries_from_calendar",
        "essential_parameters": ("month", "day"),
        "label": "calendar lookup",
    },
    {
        "predicate": lambda text: _requests_new_navigation_control(text),
        "tool_name": "set_new_navigation",
        "essential_parameters": ("route_ids",),
        "label": "new navigation control",
    },
)


def _has_explicit_percentage(text: str) -> bool:
    if re.search(r"\b\d{1,3}\s*%|\b\d{1,3}\s*percent", text):
        return True
    explicit_words = (
        "half",
        "halfway",
        "fully",
        "full",
        "completely",
        "closed",
        "open all the way",
        "about",
    )
    return any(word in text for word in explicit_words)


def _tool_called_after_latest_actionable_user(
    messages: list[dict[str, Any]], tool_name: str
) -> bool:
    for message in reversed(messages):
        if message.get("role") == "user" and _is_actionable_user_text(
            str(message.get("content") or "")
        ):
            return False
        if message.get("role") == "tool" and message.get("name") == tool_name:
            return True
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") == tool_name:
                return True
    return False


def _is_actionable_user_text(content: str) -> bool:
    text = content.strip().lower()
    return bool(text) and text != "###stop###" and not _is_confirmation_text(text)


def _latest_tool_content(messages: list[dict[str, Any]], tool_name: str) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("name") == tool_name:
            return str(message.get("content") or "")
    return None


def _latest_sunshade_position(messages: list[dict[str, Any]]) -> int | None:
    content = _latest_tool_content(messages, "get_sunroof_and_sunshade_position")
    if not content:
        return None
    data = _parse_json(content)
    value = _find_key(data, "sunshade_position")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _latest_weather_condition(messages: list[dict[str, Any]]) -> str | None:
    content = _latest_tool_content(messages, "get_weather")
    if not content:
        return None
    data = _parse_json(content)
    value = _find_key(data, "condition")
    if isinstance(value, str) and value and value.lower() != "unknown":
        return value.lower()
    return None


def _parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_key(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def _has_unknown_recent_result(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages[-6:]):
        if message.get("role") == "tool":
            content = str(message.get("content") or "").lower()
            if '"unknown"' in content or ": unknown" in content:
                return True
    return False


def _current_weather_args_from_system(
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    tool = tool_map.get("get_weather")
    if not tool:
        return None
    properties = (
        tool.get("function", {})
        .get("parameters", {})
        .get("properties", {})
    )
    required = (
        tool.get("function", {})
        .get("parameters", {})
        .get("required", [])
        or []
    )
    needed = {"location_or_poi_id", "month", "day", "time_hour_24hformat"}
    if not needed.issubset(properties):
        return None
    if any(name not in properties for name in required):
        return None

    system_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
    )
    location = _json_after_marker(system_text, "CURRENT_LOCATION")
    datetime_value = _json_after_marker(system_text, "DATETIME")
    location_id = _find_key(location, "id")
    if not isinstance(location_id, str) or not location_id:
        return None

    try:
        args: dict[str, Any] = {
            "location_or_poi_id": location_id,
            "month": int(_find_key(datetime_value, "month")),
            "day": int(_find_key(datetime_value, "day")),
            "time_hour_24hformat": int(_find_key(datetime_value, "hour")),
        }
        minute = _find_key(datetime_value, "minute")
        if "time_minutes" in properties and minute is not None:
            args["time_minutes"] = int(minute)
    except (TypeError, ValueError):
        return None

    if any(name not in args for name in required):
        return None
    return args


def _json_after_marker(text: str, marker: str) -> Any:
    index = text.find(marker)
    if index < 0:
        return None
    brace_start = text.find("{", index)
    if brace_start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for position in range(brace_start, len(text)):
        char = text[position]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _parse_json(text[brace_start : position + 1])
    return None


def _contains_state_change(calls: list[dict[str, Any]]) -> bool:
    return any(str(call.get("tool_name") or "").startswith(("open_", "set_", "navigation_", "delete_", "call_", "send_")) for call in calls)


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _previous_assistant_text(messages: list[dict[str, Any]]) -> str:
    seen_latest_user = False
    for message in reversed(messages):
        if not seen_latest_user:
            if message.get("role") == "user":
                seen_latest_user = True
            continue
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _extract_email_addresses(text: str) -> list[str]:
    return sorted(set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)))


def _latest_contact_email_addresses(messages: list[dict[str, Any]]) -> list[str]:
    contact_info = _latest_tool_json(messages, "get_contact_information")
    if contact_info is None:
        return []
    return _extract_email_addresses(json.dumps(contact_info, ensure_ascii=False))


def _extract_proposed_email_content(text: str) -> str | None:
    patterns = (
        r"content_message=(.*?)(?:,\s*email_addresses=|\.\s*Please say yes|$)",
        r"message like:\s*['\"](.+?)['\"]",
        r"planning to send:\s*['\"](.+?)['\"]",
        r"something like:\s*['\"](.+?)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content:
                return content
    return None


def _fallback_email_content(
    latest_actionable_user: str,
    *,
    messages: list[dict[str, Any]],
) -> str:
    text = latest_actionable_user.lower()
    weather_content = _weather_email_content(messages)
    if weather_content and _has_any(text, ("weather", "rain", "storm", "snow", "travel", "meeting")):
        return weather_content
    if _has_any(text, ("late", "running late", "delay")):
        return "I am running late and apologize for the delay. I will arrive as soon as possible."
    return "Following up as requested."


def _weather_email_content(messages: list[dict[str, Any]]) -> str | None:
    weather = _latest_tool_json(messages, "get_weather")
    slot = _find_key(weather, "current_slot")
    if not isinstance(slot, dict):
        return None
    condition = _find_string(slot, "condition")
    temperature = _find_number(slot, "temperature_c")
    wind_speed = _find_number(slot, "wind_speed_kph")
    humidity = _find_number(slot, "humidity_percent")
    details = []
    if condition:
        details.append(f"condition: {condition.replace('_', ' ')}")
    if temperature is not None:
        details.append(f"temperature: {temperature:g} C")
    if wind_speed is not None:
        details.append(f"wind: {wind_speed:g} km/h")
    if humidity is not None:
        details.append(f"humidity: {humidity:g}%")
    if not details:
        return None
    return (
        "Weather update for the meeting: "
        + "; ".join(details)
        + ". These conditions may affect travel and the meeting, so please plan accordingly."
    )


def _requests_charging_knowledge(text: str) -> bool:
    if not _has_any(
        text,
        (
            "charge",
            "charging",
            "battery",
            "state of charge",
            "soc",
            "range",
            "remaining range",
            "charging stop",
            "charger",
        ),
    ):
        return False
    return _has_any(
        text,
        (
            "can i",
            "can we",
            "reach",
            "make it",
            "how far",
            "how many",
            "how long",
            "minimum",
            "maximum",
            "need",
            "enough",
            "calculate",
            "plan",
            "stop",
            "stops",
            "make sure",
            "get to",
        ),
    )


def _soc_distance_precondition_action(
    *,
    text: str,
    messages: list[dict[str, Any]],
    tool_map: dict[str, dict[str, Any]],
) -> GuardDecision:
    if "get_distance_by_soc" not in tool_map:
        return GuardDecision()
    if _tool_called_after_latest_actionable_user(messages, "get_distance_by_soc"):
        return GuardDecision()
    if not (
        _has_any(text, ("reach", "make it", "range", "need", "charge", "charging", "battery"))
        and _has_any(
            text,
            (
                "keep",
                "left",
                "remaining",
                "minimum",
                "buffer",
                "at least",
                "enough",
                "safely",
                "comfortable",
                "comfortably",
            ),
        )
    ):
        return GuardDecision()
    final_soc = _requested_final_soc(text)
    if final_soc is None and _has_any(
        text,
        (
            "enough charge",
            "enough battery",
            "safely",
            "buffer",
            "comfortable",
            "comfortably",
        ),
    ):
        final_soc = 20
    if final_soc is None:
        return GuardDecision()
    status = _latest_tool_json(messages, "get_charging_specs_and_status")
    initial_soc = _find_number(status, "state_of_charge")
    if initial_soc is None:
        return GuardDecision()
    if not (
        _tool_has_parameter(tool_map, "get_distance_by_soc", "initial_state_of_charge")
        and _tool_has_parameter(tool_map, "get_distance_by_soc", "final_state_of_charge")
    ):
        return GuardDecision(
            action=_cannot_do(
                "I cannot calculate usable driving range because the SOC distance tool is missing required parameters."
            ),
            warnings=["get_distance_by_soc parameters unavailable"],
        )
    return GuardDecision(
        action=_tool_calls(
            [
                {
                    "tool_name": "get_distance_by_soc",
                    "arguments": {
                        "initial_state_of_charge": initial_soc,
                        "final_state_of_charge": final_soc,
                    },
                }
            ]
        ),
        warnings=["calculating usable range before charging route reasoning"],
    )


def _requested_final_soc(text: str) -> int | None:
    patterns = (
        r"(?:keep|leave|left|remaining|minimum|buffer|at least)\D{0,24}(\d{1,3})\s*%",
        r"(\d{1,3})\s*%\D{0,24}(?:left|remaining|minimum|buffer)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 100:
                return value
    return None


def _requests_relative_temperature_change(text: str) -> bool:
    if not _has_any(text, ("temperature", "climate", "warmer", "cooler", "cold", "hot")):
        return False
    return _has_any(
        text,
        (
            "increase",
            "decrease",
            "raise",
            "lower",
            "warmer",
            "cooler",
            "a bit",
            "little",
            "by ",
            "up",
            "down",
        ),
    ) and not _user_text_has_explicit_temperature(text)


def _user_text_has_explicit_temperature(text: str) -> bool:
    if re.search(r"\b(?:1[6-9]|2[0-8])(?:\.5)?\s*(?:degrees?|celsius|°c)?\b", text):
        return True
    return False


def _charging_status_has_required_fields(messages: list[dict[str, Any]]) -> bool:
    result = _latest_tool_json(messages, "get_charging_specs_and_status")
    return (
        _find_number(result, "state_of_charge") is not None
        and _find_key(result, "remaining_range") is not None
    )


def _temperature_result_has_required_fields(messages: list[dict[str, Any]]) -> bool:
    result = _latest_tool_json(messages, "get_temperature_inside_car")
    return (
        _find_number(result, "climate_temperature_driver") is not None
        and _find_number(result, "climate_temperature_passenger") is not None
    )


def _latest_navigation_active(messages: list[dict[str, Any]]) -> bool | None:
    state = _latest_tool_json(messages, "get_current_navigation_state")
    return _find_bool(state, "navigation_active")


def _active_navigation_replacement_call(
    *,
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    latest_user: str,
    tool_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not (
        _tool_has_parameter(tool_map, "navigation_replace_final_destination", "new_destination_id")
        and _tool_has_parameter(
            tool_map,
            "navigation_replace_final_destination",
            "route_id_leading_to_new_destination",
        )
    ):
        return None
    requested_route_id = _route_id_from_set_new_calls(calls)
    route = _select_route_from_latest_route_result(
        messages=messages,
        latest_user=latest_user,
        requested_route_id=requested_route_id,
    )
    if not isinstance(route, dict):
        return None
    route_id = route.get("route_id")
    destination_id = route.get("destination_id")
    if not isinstance(route_id, str) or not isinstance(destination_id, str):
        return None
    return {
        "tool_name": "navigation_replace_final_destination",
        "arguments": {
            "new_destination_id": destination_id,
            "route_id_leading_to_new_destination": route_id,
        },
    }


def _route_id_from_set_new_calls(calls: list[dict[str, Any]]) -> str | None:
    for call in calls:
        if call.get("tool_name") != "set_new_navigation":
            continue
        route_ids = (call.get("arguments") or {}).get("route_ids")
        if isinstance(route_ids, list) and route_ids and isinstance(route_ids[0], str):
            return route_ids[0]
    return None


def _select_route_from_latest_route_result(
    *,
    messages: list[dict[str, Any]],
    latest_user: str,
    requested_route_id: str | None,
) -> dict[str, Any] | None:
    routes = _latest_routes(messages)
    if not routes:
        return None
    if requested_route_id:
        for route in routes:
            if route.get("route_id") == requested_route_id:
                return route

    text = latest_user.lower()
    ordinal_index = _requested_route_ordinal(text)
    if ordinal_index is not None and 0 <= ordinal_index < len(routes):
        return routes[ordinal_index]

    via_match = _route_matching_via_text(routes, text)
    if via_match is not None:
        return via_match
    if "shortest" in text:
        return min(routes, key=lambda item: _route_distance(item))
    if "fastest" in text or len(routes) == 1:
        return routes[0]
    return None


def _latest_routes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") != "get_routes_from_start_to_destination":
            continue
        routes = _find_key(_parse_json(str(message.get("content") or "")), "routes")
        if isinstance(routes, list):
            return [route for route in routes if isinstance(route, dict)]
    return []


def _requested_route_ordinal(text: str) -> int | None:
    patterns = (
        (0, (r"\bfirst route\b", r"\broute\s*1\b", r"\b1st route\b")),
        (1, (r"\bsecond route\b", r"\broute\s*2\b", r"\b2nd route\b")),
        (2, (r"\bthird route\b", r"\broute\s*3\b", r"\b3rd route\b")),
    )
    for index, route_patterns in patterns:
        if any(re.search(pattern, text) for pattern in route_patterns):
            return index
    return None


def _route_matching_via_text(routes: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    best_route = None
    best_score = 0
    for route in routes:
        via = str(route.get("name_via") or "")
        tokens = [
            token.strip().lower()
            for token in re.split(r"[,/ ]+", via)
            if token.strip()
        ]
        if not tokens:
            continue
        score = sum(1 for token in tokens if token in text)
        if score > best_score:
            best_score = score
            best_route = route
    return best_route if best_score else None


def _route_distance(route: dict[str, Any]) -> float:
    value = route.get("distance_km")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float("inf")


def _unknown_navigation_route_arguments(
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> list[str]:
    route_ids = _route_ids_from_recent_route_results(messages)
    if not route_ids:
        return []

    missing: list[str] = []
    route_arg_names = {
        "route_id_leading_to_new_destination",
        "route_id_without_waypoint",
        "route_id_leading_to_new_waypoint",
        "route_id_leading_away_from_new_waypoint",
    }
    for call in calls:
        args = call.get("arguments") or {}
        for key in route_arg_names:
            value = args.get(key)
            if isinstance(value, str) and value.startswith("r") and value not in route_ids:
                missing.append(value)
        value = args.get("route_ids")
        if isinstance(value, list):
            for route_id in value:
                if isinstance(route_id, str) and route_id.startswith("r") and route_id not in route_ids:
                    missing.append(route_id)
    return missing


def _route_ids_from_recent_route_results(messages: list[dict[str, Any]]) -> set[str]:
    route_ids: set[str] = set()
    for message in messages[-12:]:
        if message.get("role") != "tool":
            continue
        if message.get("name") not in {
            "get_routes_from_start_to_destination",
            "get_current_navigation_state",
        }:
            continue
        _collect_route_ids(_parse_json(str(message.get("content") or "")), route_ids)
    return route_ids


def _collect_route_ids(value: Any, route_ids: set[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"route_id", "id"} and isinstance(nested, str) and nested.startswith("r"):
                route_ids.add(nested)
            else:
                _collect_route_ids(nested, route_ids)
    elif isinstance(value, list):
        for nested in value:
            _collect_route_ids(nested, route_ids)


def _latest_poi_result(messages: list[dict[str, Any]]) -> Any:
    for tool_name in ("search_poi_at_location", "search_poi_along_the_route"):
        result = _latest_tool_json(messages, tool_name)
        if result is not None:
            return result
    return None


def _is_open_sunroof_call(call: dict[str, Any]) -> bool:
    if call.get("tool_name") != "open_close_sunroof":
        return False
    percentage = (call.get("arguments") or {}).get("percentage")
    return isinstance(percentage, (int, float)) and percentage > 0


def _is_full_sunshade_call(call: dict[str, Any]) -> bool:
    if call.get("tool_name") != "open_close_sunshade":
        return False
    percentage = (call.get("arguments") or {}).get("percentage")
    return isinstance(percentage, (int, float)) and percentage >= 100


def _is_adverse_sunroof_weather(condition: str) -> bool:
    allowed = {"sunny", "cloudy", "partly_cloudy"}
    return condition not in allowed


def _has_recent_confirmation(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages[-6:]):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip().lower()
        if _is_confirmation_text(content):
            return True
        return False
    return False


def _is_confirmation_text(content: str) -> bool:
    text = content.strip().lower()
    if not text:
        return False
    if text in {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "confirmed"}:
        return True
    return any(
        phrase in text
        for phrase in (
            "yes please",
            "yes, please",
            "go ahead",
            "please proceed",
            "proceed",
            "do it",
            "send it",
            "still want",
            "sounds good",
            "that works",
        )
    )
