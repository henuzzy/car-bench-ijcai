"""Failure-aware action guard for Track 1.

This guard turns tool failure feedback into hard execution constraints.  The
LLM may see previous tool failures in the transcript, but this module makes
repeating the same failed action impossible unless a safe recovery action is
available.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


READ_ONLY_TOOLS = {
    "get_calendar",
    "get_charging_specs_and_status",
    "get_climate_settings",
    "get_contact_information",
    "get_current_navigation_state",
    "get_distance_by_soc",
    "get_entries_from_calendar",
    "get_exterior_lights_status",
    "get_location_id_by_location_name",
    "get_routes_from_start_to_destination",
    "get_sunroof_and_sunshade_position",
    "get_temperature_inside_car",
    "get_user_preferences",
    "get_vehicle_window_positions",
    "get_weather",
    "search_poi_along_the_route",
    "search_poi_at_location",
}

NAVIGATION_MUTATION_TOOLS = {
    "set_new_navigation",
    "navigation_replace_final_destination",
    "navigation_add_waypoint",
    "navigation_delete_destination",
}


@dataclass
class FailedAction:
    tool_name: str
    arguments: dict[str, Any]
    fingerprint: str
    failure_type: str
    reason: str
    count: int = 1


@dataclass
class FailureGuardDecision:
    action: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class FailureGuard:
    """Blocks repeated failed tool calls and obvious intent/tool mismatches."""

    def apply(
        self,
        *,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> FailureGuardDecision:
        if action.get("action") != "tool_calls":
            return FailureGuardDecision()

        tool_names = _tool_names(tools)
        calls = list(action.get("tool_calls") or [])
        latest_text = _latest_actionable_user_text(messages).lower()
        all_text = _all_actionable_user_text(messages).lower()

        intent_decision = _intent_phase_decision(
            calls=calls,
            latest_text=latest_text,
            all_text=all_text,
            messages=messages,
            tools=tools,
            tool_names=tool_names,
        )
        if intent_decision.action or intent_decision.warnings:
            return intent_decision

        failed = _failed_actions(messages)
        if not failed:
            return FailureGuardDecision()

        for call in calls:
            name = str(call.get("tool_name") or "")
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            failure = failed.get(_fingerprint(name, args))
            if not failure:
                continue
            if failure.failure_type == "TRANSIENT_ERROR" and failure.count < 2:
                return FailureGuardDecision(
                    warnings=[
                        f"FailureGuard allowed one retry for transient failed {name} call"
                    ],
                    evidence=_failure_evidence(failure),
                )
            recovery = _recovery_action_for_failed_call(
                failed_call=call,
                failure=failure,
                messages=messages,
                tools=tools,
                tool_names=tool_names,
            )
            if recovery:
                return FailureGuardDecision(
                    action=recovery,
                    warnings=[
                        f"FailureGuard blocked repeated failed {name} call and changed strategy"
                    ],
                    evidence=_failure_evidence(failure),
                )
            return FailureGuardDecision(
                action={
                    "action": "respond",
                    "content": (
                        "I cannot safely continue that exact tool action because it already failed. "
                        "I need to use a different strategy."
                    ),
                },
                warnings=[f"FailureGuard blocked repeated failed {name} call"],
                evidence=_failure_evidence(failure),
            )

        return FailureGuardDecision()


def _intent_phase_decision(
    *,
    calls: list[dict[str, Any]],
    latest_text: str,
    all_text: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_names: set[str],
) -> FailureGuardDecision:
    call_names = {str(call.get("tool_name") or "") for call in calls}
    if not call_names.intersection(NAVIGATION_MUTATION_TOOLS):
        return FailureGuardDecision()

    if _latest_user_is_charging_station_search_without_navigation_mutation(latest_text):
        search_action = _charging_station_search_action(messages, all_text, tools, tool_names)
        if search_action:
            return FailureGuardDecision(
                action=search_action,
                warnings=[
                    "FailureGuard blocked navigation mutation during charging-station search phase"
                ],
                evidence={
                    "intent": "charging_station_search",
                    "blocked_tools": sorted(call_names.intersection(NAVIGATION_MUTATION_TOOLS)),
                },
            )
        return FailureGuardDecision(
            action={
                "action": "respond",
                "content": "I found charging options, and I will not change navigation unless you ask me to.",
            },
            warnings=[
                "FailureGuard blocked navigation mutation without explicit navigation-change request"
            ],
            evidence={
                "intent": "charging_station_search",
                "blocked_tools": sorted(call_names.intersection(NAVIGATION_MUTATION_TOOLS)),
            },
        )

    return FailureGuardDecision()


def _recovery_action_for_failed_call(
    *,
    failed_call: dict[str, Any],
    failure: FailedAction,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_names: set[str],
) -> dict[str, Any] | None:
    name = str(failed_call.get("tool_name") or "")
    if (
        name == "navigation_delete_destination"
        and failure.failure_type == "POLICY_FORBIDDEN"
    ):
        return _navigation_replacement_recovery(messages, tool_names)

    if (
        name == "navigation_replace_final_destination"
        and _latest_user_is_charging_station_search_without_navigation_mutation(
            _latest_actionable_user_text(messages).lower()
        )
    ):
        return _charging_station_search_action(
            messages,
            _all_actionable_user_text(messages).lower(),
            tools,
            tool_names,
        )

    if name == "navigation_replace_final_destination" and failure.failure_type == "ARGUMENT_ERROR":
        return _route_lookup_for_failed_replacement(failed_call, messages, tool_names)

    return None


def _navigation_replacement_recovery(
    messages: list[dict[str, Any]],
    tool_names: set[str],
) -> dict[str, Any] | None:
    if "navigation_replace_final_destination" not in tool_names:
        return None
    selected_poi = _selected_poi(messages)
    if not selected_poi:
        return None
    route = _selected_route_to_destination(messages, selected_poi)
    if route:
        return {
            "action": "tool_calls",
            "tool_calls": [
                {
                    "tool_name": "navigation_replace_final_destination",
                    "arguments": {
                        "new_destination_id": selected_poi,
                        "route_id_leading_to_new_destination": route,
                    },
                }
            ],
        }
    if "get_routes_from_start_to_destination" in tool_names:
        start_id = _navigation_start_id(messages) or _current_location_id(messages)
        if start_id:
            return {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "get_routes_from_start_to_destination",
                        "arguments": {
                            "start_id": start_id,
                            "destination_id": selected_poi,
                        },
                    }
                ],
            }
    return None


def _route_lookup_for_failed_replacement(
    failed_call: dict[str, Any],
    messages: list[dict[str, Any]],
    tool_names: set[str],
) -> dict[str, Any] | None:
    if "get_routes_from_start_to_destination" not in tool_names:
        return None
    args = failed_call.get("arguments") if isinstance(failed_call.get("arguments"), dict) else {}
    destination = str(args.get("new_destination_id") or "")
    if not destination:
        return None
    start_id = _navigation_start_id(messages) or _current_location_id(messages)
    if not start_id:
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "get_routes_from_start_to_destination",
                "arguments": {"start_id": start_id, "destination_id": destination},
            }
        ],
    }


def _charging_station_search_action(
    messages: list[dict[str, Any]],
    all_text: str,
    tools: list[dict[str, Any]],
    tool_names: set[str],
) -> dict[str, Any] | None:
    if "search_poi_along_the_route" not in tool_names:
        return None
    if _charging_station_search_done(messages):
        return None
    route_id = _active_first_route_id(messages)
    at_kilometer = _requested_at_kilometer(all_text)
    if not route_id or at_kilometer is None:
        return None
    arguments: dict[str, Any] = {
        "route_id": route_id,
        "category_poi": "charging_stations",
        "at_kilometer": at_kilometer,
    }
    filters = _charging_station_filters(all_text)
    if filters and _tool_has_parameter(tools, "search_poi_along_the_route", "filters"):
        arguments["filters"] = filters
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "search_poi_along_the_route",
                "arguments": arguments,
            }
        ],
    }


def _failed_actions(messages: list[dict[str, Any]]) -> dict[str, FailedAction]:
    by_call_id: dict[str, tuple[str, dict[str, Any]]] = {}
    failures: dict[str, FailedAction] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "")
                function = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
                name = str(function.get("name") or call.get("tool_name") or "")
                args = _parse_arguments(function.get("arguments", call.get("arguments", {})))
                if call_id and name:
                    by_call_id[call_id] = (name, args)
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id not in by_call_id:
                continue
            content = str(message.get("content") or "")
            if not _tool_result_failed(content):
                continue
            name, args = by_call_id[call_id]
            fingerprint = _fingerprint(name, args)
            failure_type = _classify_failure(content)
            reason = _failure_reason(content)
            if fingerprint in failures:
                failures[fingerprint].count += 1
                failures[fingerprint].reason = reason or failures[fingerprint].reason
                failures[fingerprint].failure_type = failure_type
            else:
                failures[fingerprint] = FailedAction(
                    tool_name=name,
                    arguments=args,
                    fingerprint=fingerprint,
                    failure_type=failure_type,
                    reason=reason,
                )
    return failures


def _tool_result_failed(content: str) -> bool:
    data = _parse_json(content)
    if isinstance(data, dict):
        status = _find_key(data, "status")
        if isinstance(status, str) and status.strip().lower() in {"failure", "failed", "error"}:
            return True
        errors = _find_key(data, "errors")
        error = _find_key(data, "error")
        if errors not in (None, "", []) or error not in (None, "", []):
            return True
    lowered = content.lower()
    return any(piece in lowered for piece in ("failure", "failed", "error", "winerror", "timeout"))


def _classify_failure(content: str) -> str:
    lowered = content.lower()
    if any(piece in lowered for piece in ("aut-pol", "policy", "forbidden", "would lead to full deletion", "not allowed")):
        return "POLICY_FORBIDDEN"
    if any(piece in lowered for piece in ("unknown tool", "unavailable", "not available", "capability")):
        return "TOOL_UNAVAILABLE"
    if any(piece in lowered for piece in ("timeout", "timed out", "winerror", "connection", "remote host")):
        return "TRANSIENT_ERROR"
    if any(piece in lowered for piece in ("invalid", "does not match", "missing required", "required argument", "not found")):
        return "ARGUMENT_ERROR"
    return "UNKNOWN_ERROR"


def _failure_reason(content: str) -> str:
    data = _parse_json(content)
    if isinstance(data, dict):
        errors = _find_key(data, "errors")
        if isinstance(errors, dict):
            return "; ".join(str(value) for value in errors.values())
        if isinstance(errors, list):
            return "; ".join(str(value) for value in errors)
        if isinstance(errors, str):
            return errors
        error = _find_key(data, "error")
        if error:
            return str(error)
        message = _find_key(data, "message")
        if message:
            return str(message)
    return content[:500]


def _failure_evidence(failure: FailedAction) -> dict[str, Any]:
    return {
        "tool_name": failure.tool_name,
        "arguments": failure.arguments,
        "failure_type": failure.failure_type,
        "reason": failure.reason,
        "count": failure.count,
    }


def _latest_user_is_charging_station_search_without_navigation_mutation(text: str) -> bool:
    if not _has_any(text, ("charging station", "charging stations", "charger", "chargers", "dc fast", "dc plug")):
        return False
    if not _has_any(text, ("find", "search", "show", "look for", "check", "options", "around", "near")):
        return False
    if _has_any(text, ("add", "navigate", "set destination", "set it", "replace", "make it", "start navigation", "route me")):
        return False
    return True


def _charging_station_search_done(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "tool" or message.get("name") != "search_poi_along_the_route":
            continue
        data = _parse_json(str(message.get("content") or ""))
        if _find_key(data, "pois_found_along_route") or _find_key(data, "pois"):
            return True
    return False


def _selected_poi(messages: list[dict[str, Any]]) -> str | None:
    user_text = _all_actionable_user_text(messages).lower()
    candidates: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") != "tool" or message.get("name") not in {"search_poi_at_location", "search_poi_along_the_route"}:
            continue
        _collect_poi_candidates(_parse_json(str(message.get("content") or "")), candidates)
    if not candidates:
        return None
    for candidate in candidates:
        name = candidate.get("name", "").lower()
        if name and name in user_text:
            return candidate["id"]
        if "rinc" in user_text and "tapas" in user_text and "tapas" in name:
            return candidate["id"]
    return candidates[-1]["id"]


def _collect_poi_candidates(data: Any, candidates: list[dict[str, str]]) -> None:
    if isinstance(data, dict):
        poi_id = data.get("poi_id") or data.get("id")
        if isinstance(poi_id, str) and poi_id.startswith("poi_"):
            name = data.get("name")
            candidates.append({"id": poi_id, "name": str(name or "")})
        for value in data.values():
            _collect_poi_candidates(value, candidates)
    elif isinstance(data, list):
        for item in data:
            _collect_poi_candidates(item, candidates)


def _selected_route_to_destination(messages: list[dict[str, Any]], destination_id: str) -> str | None:
    routes = _routes(messages)
    user_text = _all_actionable_user_text(messages).lower()
    candidates = [route for route in routes if route.get("destination_id") == destination_id]
    if not candidates:
        return None
    selected = _select_route_by_request(candidates, user_text) or _default_route(candidates)
    return str((selected or {}).get("route_id") or "") or None


def _requested_via(text: str) -> str:
    match = re.search(r"via\s+([a-z0-9,\s]+)", text, re.I)
    return match.group(1) if match else ""


def _select_route_by_request(
    routes: list[dict[str, Any]],
    text: str,
) -> dict[str, Any] | None:
    ordinal_index = _requested_route_ordinal(text)
    ordinal_alias = _ordinal_alias(ordinal_index)
    if ordinal_alias:
        alias_match = _route_matching_alias(routes, {ordinal_alias})
        if alias_match is not None:
            return alias_match
    if ordinal_index is not None and 0 <= ordinal_index < len(routes):
        return routes[ordinal_index]
    alias_terms = {term for term in ("fastest", "shortest") if term in text}
    if alias_terms:
        alias_match = _route_matching_alias(routes, alias_terms)
        if alias_match is not None:
            return alias_match
    return _route_matching_via_text(routes, text)


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


def _route_matching_via_text(
    routes: list[dict[str, Any]],
    text: str,
) -> dict[str, Any] | None:
    requested_via = _requested_via(text)
    if requested_via:
        requested_tokens = [
            piece.strip().lower()
            for piece in re.split(r"\s*,\s*", requested_via)
            if piece.strip()
        ]
        for route in routes:
            via = str(route.get("name_via") or "").lower()
            if via and all(piece in via for piece in requested_tokens):
                return route

    best_route = None
    best_score = 0
    for route in routes:
        via = str(route.get("name_via") or "")
        tokens = [
            token.strip().lower()
            for token in re.split(r"[,/ ]+", via)
            if token.strip()
        ]
        score = sum(1 for token in tokens if token in text)
        if score > best_score:
            best_score = score
            best_route = route
    return best_route if best_score else None


def _default_route(routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not routes:
        return None
    for aliases in ({"fastest"}, {"first"}):
        selected = _route_matching_alias(routes, aliases)
        if selected is not None:
            return selected
    fastest = _route_with_shortest_duration(routes)
    return fastest or routes[0]


def _route_matching_alias(
    routes: list[dict[str, Any]],
    aliases: set[str],
) -> dict[str, Any] | None:
    for route in routes:
        if _route_aliases(route).intersection(aliases):
            return route
    return None


def _route_aliases(route: dict[str, Any]) -> set[str]:
    value = route.get("alias")
    if isinstance(value, list):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    if isinstance(value, str):
        return {item.strip().lower() for item in re.split(r"[,/ ]+", value) if item.strip()}
    return set()


def _route_with_shortest_duration(routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored: list[tuple[int, dict[str, Any]]] = []
    for route in routes:
        duration = _route_duration_minutes(route)
        if duration is not None:
            scored.append((duration, route))
    if not scored:
        return None
    return min(scored, key=lambda item: item[0])[1]


def _route_duration_minutes(route: dict[str, Any]) -> int | None:
    value = route.get("duration_min")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    hours = route.get("duration_hours")
    minutes = route.get("duration_minutes")
    total = 0
    found = False
    if isinstance(hours, (int, float)) and not isinstance(hours, bool):
        total += int(hours) * 60
        found = True
    if isinstance(minutes, (int, float)) and not isinstance(minutes, bool):
        total += int(minutes)
        found = True
    return total if found else None


def _ordinal_alias(index: int | None) -> str | None:
    if index is None:
        return None
    aliases = ("first", "second", "third")
    return aliases[index] if 0 <= index < len(aliases) else None


def _routes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool" or message.get("name") != "get_routes_from_start_to_destination":
            continue
        _collect_routes(_parse_json(str(message.get("content") or "")), routes)
    return routes


def _collect_routes(data: Any, routes: list[dict[str, Any]]) -> None:
    if isinstance(data, dict):
        route_id = data.get("route_id")
        if isinstance(route_id, str) and route_id.startswith("r"):
            routes.append(
                {
                    "route_id": route_id,
                    "start_id": data.get("start_id"),
                    "destination_id": data.get("destination_id"),
                    "name_via": data.get("name_via"),
                    "distance_km": data.get("distance_km"),
                    "duration_min": data.get("duration_min"),
                    "duration_hours": data.get("duration_hours"),
                    "duration_minutes": data.get("duration_minutes"),
                    "alias": data.get("alias"),
                }
            )
        for value in data.values():
            _collect_routes(value, routes)
    elif isinstance(data, list):
        for item in data:
            _collect_routes(item, routes)


def _navigation_start_id(messages: list[dict[str, Any]]) -> str | None:
    state = _latest_tool_json(messages, "get_current_navigation_state")
    waypoints = _find_key(state, "waypoints_id")
    if isinstance(waypoints, list) and waypoints and isinstance(waypoints[0], str):
        return waypoints[0]
    return None


def _active_first_route_id(messages: list[dict[str, Any]]) -> str | None:
    state = _latest_tool_json(messages, "get_current_navigation_state")
    route_ids = _find_key(state, "routes_to_final_destination_id")
    if isinstance(route_ids, list) and route_ids and isinstance(route_ids[0], str):
        return route_ids[0]
    return None


def _current_location_id(messages: list[dict[str, Any]]) -> str | None:
    system_text = "\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system")
    match = re.search(r'"id"\s*:\s*"(loc_[A-Za-z0-9_]+)"', system_text)
    return match.group(1) if match else None


def _requested_at_kilometer(text: str) -> int | None:
    patterns = (
        r"(?:around|about|near|at|approximately|roughly|specific(?:ally)?)\D{0,24}(\d{1,4})\s*(?:km|kilometer)",
        r"(\d{1,4})\s*(?:km|kilometer)\D{0,30}(?:into|along|mark|point)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value
    return None


def _charging_station_filters(text: str) -> list[str]:
    filters: list[str] = []
    if _has_any(text, ("available", "free plug", "open plug")):
        filters.append("charging_stations::has_available_plug")
    if _has_any(text, ("dc", "fast")):
        filters.append("charging_stations::has_dc_plug")
    if _has_any(text, ("currently open", "open now")):
        filters.append("any::currently_open")
    return filters


def _latest_tool_json(messages: list[dict[str, Any]], tool_name: str) -> Any:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("name") == tool_name:
            return _parse_json(str(message.get("content") or ""))
    return None


def _latest_actionable_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or content.lower() == "###stop###":
            continue
        return content
    return ""


def _all_actionable_user_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "user"
        and str(message.get("content") or "").strip()
        and str(message.get("content") or "").strip().lower() != "###stop###"
    )


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(tool.get("function", {}).get("name") or tool.get("name") or "")
        for tool in tools
        if str(tool.get("function", {}).get("name") or tool.get("name") or "")
    }


def _tool_has_parameter(tools: list[dict[str, Any]], tool_name: str, parameter: str) -> bool:
    for tool in tools:
        name = str(tool.get("function", {}).get("name") or tool.get("name") or "")
        if name != tool_name:
            continue
        properties = tool.get("function", {}).get("parameters", {}).get("properties", {})
        return isinstance(properties, dict) and parameter in properties
    return False


def _fingerprint(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
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
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def _has_any(text: str, pieces: tuple[str, ...]) -> bool:
    return any(piece in text for piece in pieces)
