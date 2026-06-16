"""Structured per-context planning state for the Track 1 planner.

The store rebuilds state from the current transcript on every turn.  That keeps
it scoped to one evaluator context and avoids leaking lessons across tasks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
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

UNAVAILABLE_PATTERNS = (
    "not available",
    "unavailable",
    "cannot search",
    "can't search",
    "no restaurant search",
    "no navigation",
    "service is not",
    "service isn't",
    "capability is unavailable",
    "capability isn't available",
)

POI_CATEGORIES = {
    "restaurant": "restaurants",
    "restaurants": "restaurants",
    "dinner": "restaurants",
    "lunch": "restaurants",
    "charging": "charging_stations",
    "charger": "charging_stations",
    "chargers": "charging_stations",
    "charging station": "charging_stations",
    "parking": "parking",
    "supermarket": "supermarkets",
    "grocery": "supermarkets",
    "bakery": "bakery",
    "airport": "airports",
    "toilet": "public_toilets",
    "restroom": "public_toilets",
    "fast food": "fast_food",
}


@dataclass
class ActionRecord:
    tool_name: str
    arguments: dict[str, Any]
    fingerprint: str
    call_id: str | None = None
    result_name: str | None = None
    result_text: str = ""
    result_json: Any = None
    turn_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if len(data["result_text"]) > 1000:
            data["result_text"] = data["result_text"][:1000] + "...[truncated]"
        return data


@dataclass
class PlanState:
    context_id: str
    turn_index: int = 0
    stage: str = "UNDERSTAND"
    original_user_request: str = ""
    latest_user_request: str = ""
    domains: list[str] = field(default_factory=list)
    completed_tools: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    action_history: list[ActionRecord] = field(default_factory=list)
    repeated_actions: list[str] = field(default_factory=list)
    next_allowed_tools: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    reminders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "turn_index": self.turn_index,
            "stage": self.stage,
            "original_user_request": self.original_user_request,
            "latest_user_request": self.latest_user_request,
            "domains": self.domains,
            "completed_tools": self.completed_tools,
            "facts": self.facts,
            "action_history": [record.to_dict() for record in self.action_history[-12:]],
            "repeated_actions": self.repeated_actions,
            "next_allowed_tools": self.next_allowed_tools,
            "forbidden_actions": self.forbidden_actions,
            "reminders": self.reminders,
        }


@dataclass
class PlanStateDecision:
    action: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


class PlanStateStore:
    """Keeps one structured plan state per evaluator context."""

    def __init__(self) -> None:
        self._states: dict[str, PlanState] = {}

    def reset(self, context_id: str) -> None:
        self._states.pop(context_id, None)

    def observe_messages(
        self,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> None:
        self._states[context_id] = _build_state(context_id, messages, tools)

    def snapshot(self, context_id: str) -> dict[str, Any]:
        return self._state(context_id).to_dict()

    def guidance(self, context_id: str) -> list[str]:
        state = self._state(context_id)
        guidance = list(state.reminders)
        if state.next_allowed_tools:
            guidance.append(
                "Prefer the next_allowed_tools in order unless the latest user message changes the task: "
                + ", ".join(state.next_allowed_tools)
            )
        if state.forbidden_actions:
            guidance.append(
                "Do not take forbidden_actions: " + "; ".join(state.forbidden_actions)
            )
        return guidance

    def postprocess_action(
        self,
        context_id: str,
        action: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> PlanStateDecision:
        state = self._state(context_id)
        tool_names = _tool_names(tools)
        if action.get("action") == "respond":
            return _repair_response(action, state, tools)
        if action.get("action") != "tool_calls":
            return PlanStateDecision(action=action)
        return _repair_tool_calls(action, state, tools)

    def _state(self, context_id: str) -> PlanState:
        return self._states.setdefault(context_id, PlanState(context_id=context_id))


def _build_state(
    context_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> PlanState:
    user_messages = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
        and str(message.get("content") or "").strip()
        and str(message.get("content") or "").strip().lower() != "###stop###"
    ]
    original_user = user_messages[0] if user_messages else ""
    latest_user = user_messages[-1] if user_messages else ""
    action_history = _action_history(messages)
    request_text = " ".join(user_messages)
    facts = _facts_from_history(action_history, request_text=request_text)
    domains = _infer_domains(request_text, action_history)
    state = PlanState(
        context_id=context_id,
        turn_index=len(messages),
        original_user_request=original_user,
        latest_user_request=latest_user,
        domains=domains,
        completed_tools=_unique(
            record.tool_name for record in action_history if record.tool_name != "respond"
        ),
        facts=facts,
        action_history=action_history,
        repeated_actions=_repeated_action_descriptions(action_history),
    )
    _derive_stage_and_constraints(state, tools)
    return state


def _action_history(messages: list[dict[str, Any]]) -> list[ActionRecord]:
    records: list[ActionRecord] = []
    by_call_id: dict[str, ActionRecord] = {}
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            content = str(message.get("content") or "")
            if content:
                records.append(
                    ActionRecord(
                        tool_name="respond",
                        arguments={"content": content},
                        fingerprint=_fingerprint("respond", {"content": _normalize_text(content)}),
                        turn_index=index,
                    )
                )
            if not message.get("tool_calls"):
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = str(function.get("name") or call.get("tool_name") or "")
                arguments = _parse_arguments(function.get("arguments", call.get("arguments", {})))
                if not name:
                    continue
                record = ActionRecord(
                    tool_name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    fingerprint=_fingerprint(name, arguments if isinstance(arguments, dict) else {}),
                    call_id=call.get("id"),
                    turn_index=index,
                )
                records.append(record)
                if record.call_id:
                    by_call_id[record.call_id] = record
        elif message.get("role") == "tool":
            name = str(message.get("name") or "")
            record = by_call_id.get(str(message.get("tool_call_id") or ""))
            if record is None and name:
                record = next(
                    (
                        item
                        for item in reversed(records)
                        if item.tool_name == name and item.result_text == ""
                    ),
                    None,
                )
            if record is None:
                if not name:
                    continue
                record = ActionRecord(
                    tool_name=name,
                    arguments={},
                    fingerprint=_fingerprint(name, {}),
                    call_id=message.get("tool_call_id"),
                    turn_index=index,
                )
                records.append(record)
            content = str(message.get("content") or "")
            record.result_name = name or record.tool_name
            record.result_text = content
            record.result_json = _parse_json(content)
    return records


def _facts_from_history(
    history: list[ActionRecord],
    *,
    request_text: str = "",
) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "location_ids": [],
        "poi_candidates": [],
        "routes": [],
        "contact_emails": [],
        "weather": {},
        "calendar": {},
        "charging": {},
        "distance_by_soc": {},
        "navigation_active": None,
        "navigation_waypoints": [],
        "navigation_routes_to_final_destination": [],
    }
    for record in history:
        data = record.result_json
        if record.tool_name == "get_current_navigation_state":
            active = _find_bool(data, "navigation_active")
            if active is not None:
                facts["navigation_active"] = active
            waypoints = _first_value_for_keys(data, ("waypoints_id",))
            if isinstance(waypoints, list) and waypoints:
                facts["navigation_waypoints"] = [str(item) for item in waypoints]
                facts["navigation_start_id"] = str(waypoints[0])
                facts["navigation_destination_id"] = str(waypoints[-1])
            route_ids = _first_value_for_keys(data, ("routes_to_final_destination_id",))
            if isinstance(route_ids, list) and route_ids:
                facts["navigation_routes_to_final_destination"] = [
                    str(item) for item in route_ids if isinstance(item, str)
                ]
        if (
            record.tool_name == "set_new_navigation"
            and _tool_result_is_navigation_active_failure(record)
        ):
            facts["navigation_active"] = True
        if record.tool_name == "get_location_id_by_location_name":
            location_name = _optional_text(record.arguments.get("location"))
            for location_id in _collect_prefixed_ids(data, "loc_"):
                entry = {"id": location_id}
                if location_name:
                    entry["name"] = location_name
                _append_unique_dict(facts["location_ids"], entry)
        if record.tool_name == "get_routes_from_start_to_destination":
            route_ids = _collect_route_ids(data)
            destination_id = _optional_text(record.arguments.get("destination_id"))
            start_id = _optional_text(record.arguments.get("start_id"))
            for route_id in route_ids:
                route_detail = _route_detail_for_id(data, route_id)
                entry = {
                    "route_id": route_id,
                    "start_id": route_detail.get("start_id") or start_id,
                    "destination_id": route_detail.get("destination_id") or destination_id,
                }
                for key in ("name_via", "distance_km", "duration_min", "duration_minutes", "duration_hours", "alias"):
                    if route_detail.get(key) is not None:
                        entry[key] = route_detail[key]
                _append_unique_dict(facts["routes"], entry)
        if record.tool_name in {"search_poi_at_location", "search_poi_along_the_route"}:
            category = _optional_text(record.arguments.get("category_poi"))
            for poi_id in _collect_prefixed_ids(data, "poi_"):
                entry = {
                    "poi_id": poi_id,
                    "category": category,
                    "location_id": _optional_text(record.arguments.get("location_id")),
                    "route_id": _optional_text(record.arguments.get("route_id")),
                }
                name = _entity_name_near_id(data, poi_id)
                if name:
                    entry["name"] = name
                _append_unique_dict(facts["poi_candidates"], entry)
        if record.tool_name == "get_contact_information":
            for email in _extract_email_addresses(record.result_text):
                if email not in facts["contact_emails"]:
                    facts["contact_emails"].append(email)
        if record.tool_name == "get_weather":
            weather = _weather_summary(data, record.result_text)
            if weather:
                facts["weather"] = weather
        if record.tool_name in {"get_entries_from_calendar", "get_calendar"}:
            calendar = _calendar_summary(data, record.result_text)
            if calendar:
                facts["calendar"] = calendar
        if record.tool_name == "get_charging_specs_and_status":
            charging = _charging_summary(data, record.result_text)
            if charging:
                facts["charging"] = charging
        if record.tool_name == "get_distance_by_soc":
            distance = _distance_by_soc_summary(data, record.result_text)
            if distance:
                facts["distance_by_soc"] = distance
    selected_poi = _select_poi(facts["poi_candidates"])
    if selected_poi:
        facts["selected_poi"] = selected_poi
    selected_route = _select_route_for_destination(
        facts["routes"],
        selected_poi.get("poi_id") if selected_poi else None,
        preferred_start_id=facts.get("navigation_start_id"),
        request_text=request_text,
    )
    if selected_route:
        facts["selected_route"] = selected_route
    return facts


def _derive_stage_and_constraints(state: PlanState, tools: list[dict[str, Any]]) -> None:
    tool_names = _tool_names(tools)
    facts = state.facts
    latest_text = (state.latest_user_request or state.original_user_request).lower()
    reminders: list[str] = []
    next_allowed: list[str] = []
    forbidden: list[str] = []

    if not state.action_history:
        state.stage = "PLAN"
    elif _has_state_changing_action(state.action_history):
        state.stage = "VERIFY"
    else:
        state.stage = "EXECUTE"

    for repeat in state.repeated_actions:
        forbidden.append(f"Do not repeat exact action {repeat} unless a new user instruction changed its arguments.")

    if "navigation" in state.domains:
        _navigation_constraints(
            state=state,
            tool_names=tool_names,
            latest_text=latest_text,
            next_allowed=next_allowed,
            forbidden=forbidden,
            reminders=reminders,
        )

    if "communication" in state.domains:
        _communication_constraints(
            state=state,
            tool_names=tool_names,
            next_allowed=next_allowed,
            forbidden=forbidden,
            reminders=reminders,
        )

    if "charging" in state.domains:
        _charging_constraints(
            state=state,
            tool_names=tool_names,
            latest_text=latest_text,
            next_allowed=next_allowed,
            forbidden=forbidden,
            reminders=reminders,
        )

    state.next_allowed_tools = _unique(next_allowed)
    state.forbidden_actions = _unique(forbidden)
    state.reminders = _unique(reminders)


def _navigation_constraints(
    *,
    state: PlanState,
    tool_names: set[str],
    latest_text: str,
    next_allowed: list[str],
    forbidden: list[str],
    reminders: list[str],
) -> None:
    facts = state.facts
    wants_poi = _preferred_poi_category(latest_text) is not None or bool(facts.get("poi_candidates"))
    selected_poi = facts.get("selected_poi") or {}
    selected_route = facts.get("selected_route") or {}
    if wants_poi:
        reminders.append(
            "POI navigation must target the selected POI id, not only the surrounding city/location id."
        )
        if selected_poi and not selected_route and "get_routes_from_start_to_destination" in tool_names:
            next_allowed.append("get_routes_from_start_to_destination")
        if (
            selected_poi
            and selected_route
            and facts.get("navigation_active") is True
            and "navigation_replace_final_destination" in tool_names
        ):
            next_allowed.append("navigation_replace_final_destination")
        elif selected_poi and selected_route and "set_new_navigation" in tool_names:
            next_allowed.append("set_new_navigation")
        forbidden.append("Do not use a loc_* city id as navigation_replace_final_destination.new_destination_id when a matching poi_* candidate is known.")
    if facts.get("navigation_active") is True and "navigation_replace_final_destination" in tool_names:
        forbidden.append("Do not call set_new_navigation for an active navigation replacement; use navigation_replace_final_destination.")
    if "delete" not in latest_text and "cancel" not in latest_text and "remove" not in latest_text:
        forbidden.append("Do not delete current navigation before resolving the requested replacement destination.")


def _communication_constraints(
    *,
    state: PlanState,
    tool_names: set[str],
    next_allowed: list[str],
    forbidden: list[str],
    reminders: list[str],
) -> None:
    facts = state.facts
    if "send_email" not in tool_names:
        return
    if not facts.get("contact_emails") and "get_contact_information" in tool_names:
        next_allowed.append("get_contact_information")
        reminders.append("Do not send email before contact information includes email addresses.")
    if _mentions_weather_or_travel(state.latest_user_request + " " + state.original_user_request):
        if not facts.get("weather") and "get_weather" in tool_names:
            next_allowed.append("get_weather")
            reminders.append("Weather/travel emails must include the gathered weather facts.")
        if not facts.get("calendar") and "get_entries_from_calendar" in tool_names:
            next_allowed.append("get_entries_from_calendar")
            reminders.append("Meeting/weather emails should include calendar event name, location, and 24-hour time when available.")
    forbidden.append("Do not send generic email content if calendar/weather/contact facts are already known.")


def _charging_constraints(
    *,
    state: PlanState,
    tool_names: set[str],
    latest_text: str,
    next_allowed: list[str],
    forbidden: list[str],
    reminders: list[str],
) -> None:
    if "get_charging_specs_and_status" in tool_names and not state.facts.get("charging"):
        next_allowed.append("get_charging_specs_and_status")
        reminders.append("Charging/range tasks must gather charging status before range or charging calculations.")
        return
    if (
        state.facts.get("charging")
        and not state.facts.get("distance_by_soc")
        and "get_distance_by_soc" in tool_names
        and _mentions_range_buffer(state.original_user_request + " " + state.latest_user_request)
    ):
        next_allowed.append("get_distance_by_soc")
        reminders.append("If the user asks about usable range with a reserve buffer, calculate distance by SOC before route charging decisions.")
    if (
        _wants_charging_station_search(latest_text)
        and state.facts.get("navigation_routes_to_final_destination")
        and "search_poi_along_the_route" in tool_names
    ):
        next_allowed.append("search_poi_along_the_route")
        reminders.append("Charging station searches along a route must use the active route id and include at_kilometer when the user gave one.")
    if _wants_delete_final_destination(latest_text) and "navigation_delete_destination" in tool_names:
        next_allowed.append("navigation_delete_destination")
        forbidden.append("Do not add a charging station to navigation when the user only asked to keep it for later.")


def _repair_response(
    action: dict[str, Any],
    state: PlanState,
    tools: list[dict[str, Any]],
) -> PlanStateDecision:
    tool_names = _tool_names(tools)
    content = str(action.get("content") or "")
    lowered = content.lower()
    latest_text = (state.latest_user_request or "").lower()
    warnings: list[str] = []
    completion_decision = _pending_action_before_response(action, state, tools)
    if completion_decision.action:
        return completion_decision
    if "charging" in state.domains and (
        _wants_charging_station_search(latest_text)
        or _wants_delete_final_destination(latest_text)
    ):
        alternative = _next_charging_action(state, tools)
        if alternative:
            return PlanStateDecision(
                action=alternative,
                warnings=["PlanState advanced explicit charging/navigation follow-up instead of responding"],
            )
    if any(pattern in lowered for pattern in UNAVAILABLE_PATTERNS):
        alternative = _next_recovery_action(state, tools)
        if alternative:
            return PlanStateDecision(
                action=alternative,
                warnings=["PlanState replaced unsupported unavailable response with next feasible action"],
            )
        if _has_relevant_available_tool(state, tool_names):
            return PlanStateDecision(
                action={
                    "action": "respond",
                    "content": "I need one more available tool check before I can answer that safely.",
                },
                warnings=["PlanState blocked unsupported unavailable response"],
            )
    if _response_repeated(state, content):
        alternative = _next_recovery_action(state, tools)
        if alternative:
            return PlanStateDecision(
                action=alternative,
                warnings=["PlanState changed strategy after repeated response"],
            )
    if warnings:
        return PlanStateDecision(action=action, warnings=warnings)
    return PlanStateDecision(action=action)


def _pending_action_before_response(
    action: dict[str, Any],
    state: PlanState,
    tools: list[dict[str, Any]],
) -> PlanStateDecision:
    tool_names = _tool_names(tools)
    content = str(action.get("content") or "")

    email_decision = _pending_email_action_before_response(content, state, tool_names)
    if email_decision.action:
        return email_decision

    charging_action = None
    if "charging" in state.domains:
        charging_action = _next_charging_action(state, tools)
    if charging_action and not _action_contains_completed_state_change(state, charging_action):
        return PlanStateDecision(
            action=charging_action,
            warnings=["PlanState completion gate continued pending charging action before responding"],
        )

    navigation_action = None
    if "navigation" in state.domains:
        navigation_action = _next_navigation_action(state, tool_names)
    if navigation_action and not _action_contains_completed_state_change(state, navigation_action):
        return PlanStateDecision(
            action=navigation_action,
            warnings=["PlanState completion gate continued pending navigation action before responding"],
        )

    return PlanStateDecision()


def _pending_email_action_before_response(
    content: str,
    state: PlanState,
    tool_names: set[str],
) -> PlanStateDecision:
    if "communication" not in state.domains or "send_email" not in tool_names:
        return PlanStateDecision()
    if _tool_completed(state, "send_email"):
        return PlanStateDecision()
    if not _email_task_requested(state):
        return PlanStateDecision()

    if _latest_user_confirms(state.latest_user_request):
        email_action = _next_email_action(state, tool_names)
        if email_action:
            return PlanStateDecision(
                action=email_action,
                warnings=["PlanState completion gate sent pending email after user confirmation"],
            )

    if _email_ready(state) and not _looks_like_email_confirmation_request(content):
        return PlanStateDecision(
            action={"action": "respond", "content": _email_confirmation_prompt(state)},
            warnings=["PlanState completion gate replaced premature response with email confirmation request"],
        )

    return PlanStateDecision()


def _repair_tool_calls(
    action: dict[str, Any],
    state: PlanState,
    tools: list[dict[str, Any]],
) -> PlanStateDecision:
    tool_names = _tool_names(tools)
    calls = list(action.get("tool_calls") or [])
    warnings: list[str] = []
    repaired: list[dict[str, Any]] = []
    for call in calls:
        name = str(call.get("tool_name") or "")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if _is_repeated_failed_call(state, name, args):
            alternative = _next_recovery_action(state, tools)
            if alternative and not _action_contains_repeated_failed_call(state, alternative):
                return PlanStateDecision(
                    action=alternative,
                    warnings=[f"PlanState changed strategy after repeated failed {name} call"],
                )
            return PlanStateDecision(
                action={
                    "action": "respond",
                    "content": "I cannot safely continue that action because the same tool call is failing repeatedly.",
                },
                warnings=[f"PlanState blocked repeated failed {name} call"],
            )

        if _is_repeated_completed_call(state, name, args):
            alternative = _next_recovery_action(state, tools)
            if alternative:
                return PlanStateDecision(
                    action=alternative,
                    warnings=[f"PlanState replaced repeated {name} call with next planned action"],
                )
            return PlanStateDecision(
                action={
                    "action": "respond",
                    "content": _repeated_completed_response(state, name),
                },
                warnings=[f"PlanState blocked repeated completed {name} call"],
            )

        if name == "navigation_replace_final_destination":
            nav_repair = _repair_navigation_replacement(call, state, tool_names)
            if nav_repair.action:
                return nav_repair

        if name == "set_new_navigation" and state.facts.get("navigation_active") is True:
            alternative = _next_navigation_action(state, tool_names)
            if alternative and not _action_contains_repeated_failed_call(state, alternative):
                return PlanStateDecision(
                    action=alternative,
                    warnings=["PlanState redirected active set_new_navigation to the next navigation step"],
                )
            replacement = _active_navigation_replacement_from_route(call, state, tool_names)
            if replacement:
                return PlanStateDecision(
                    action=replacement,
                    warnings=["PlanState converted active set_new_navigation to destination replacement"],
                )

        if name == "send_email":
            call, email_warnings = _repair_email_call(call, state)
            warnings.extend(email_warnings)

        repaired.append(call)
    return PlanStateDecision(
        action={"action": "tool_calls", "tool_calls": repaired},
        warnings=warnings,
    )


def _repair_navigation_replacement(
    call: dict[str, Any],
    state: PlanState,
    tool_names: set[str],
) -> PlanStateDecision:
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    destination = str(args.get("new_destination_id") or "")
    selected_poi = state.facts.get("selected_poi") or {}
    selected_route = state.facts.get("selected_route") or {}
    selected_poi_id = str(selected_poi.get("poi_id") or "")
    selected_route_id = str(selected_route.get("route_id") or "")
    proposed_route_id = str(args.get("route_id_leading_to_new_destination") or "")
    expected_start_id = state.facts.get("navigation_start_id")
    if (
        selected_poi_id
        and destination == selected_poi_id
        and _charging_poi_without_navigation_intent(
            selected_poi,
            state.latest_user_request or "",
        )
    ):
        return PlanStateDecision(
            action={
                "action": "respond",
                "content": _charging_options_response(state),
            },
            warnings=["PlanState blocked charging POI navigation because the user only asked to view options"],
        )
    route_for_destination = _select_route_for_destination(
        state.facts.get("routes") or [],
        destination or None,
        preferred_start_id=expected_start_id,
        request_text=f"{state.original_user_request} {state.latest_user_request}",
    )
    ambiguous_route_response = _ambiguous_route_response(
        state=state,
        destination_id=destination,
        preferred_start_id=expected_start_id,
    )
    if ambiguous_route_response:
        return PlanStateDecision(
            action={"action": "respond", "content": ambiguous_route_response},
            warnings=["PlanState asked user to choose among route alternatives"],
        )
    route_for_destination_id = str((route_for_destination or {}).get("route_id") or "")
    if (
        destination
        and route_for_destination_id
        and proposed_route_id != route_for_destination_id
        and "navigation_replace_final_destination" in tool_names
    ):
        return PlanStateDecision(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": destination,
                            "route_id_leading_to_new_destination": route_for_destination_id,
                        },
                    }
                ],
            },
            warnings=["PlanState corrected replacement route id to match replacement destination"],
        )
    if (
        destination.startswith("loc_")
        and selected_poi_id.startswith("poi_")
        and "navigation_replace_final_destination" in tool_names
    ):
        if selected_route_id:
            return PlanStateDecision(
                action={
                    "action": "tool_calls",
                    "tool_calls": [
                        {
                            "tool_name": "navigation_replace_final_destination",
                            "arguments": {
                                "new_destination_id": selected_poi_id,
                                "route_id_leading_to_new_destination": selected_route_id,
                            },
                        }
                    ],
                },
                warnings=["PlanState replaced city destination id with selected POI destination id"],
            )
        route_lookup = _route_lookup_to_selected_poi(state, tool_names)
        if route_lookup:
            return PlanStateDecision(
                action=route_lookup,
                warnings=["PlanState requested route to selected POI before navigation replacement"],
            )
    proposed_route = _route_by_id(state.facts.get("routes") or [], proposed_route_id)
    if (
        destination
        and proposed_route_id
        and (
            not proposed_route
            or proposed_route.get("destination_id") != destination
            or (
                expected_start_id
                and proposed_route.get("start_id")
                and proposed_route.get("start_id") != expected_start_id
            )
        )
    ):
        route_lookup = _route_lookup_to_destination(
            state,
            tool_names,
            destination_id=destination,
            preferred_start_id=expected_start_id,
        )
        if route_lookup:
            return PlanStateDecision(
                action=route_lookup,
                warnings=[
                    "PlanState requested a current-start route because the replacement route did not match start/destination"
                ],
            )
    if (
        selected_poi_id.startswith("poi_")
        and destination == selected_poi_id
        and not selected_route_id
        and "navigation_replace_final_destination" in tool_names
    ):
        route_lookup = _route_lookup_to_selected_poi(state, tool_names)
        if route_lookup:
            return PlanStateDecision(
                action=route_lookup,
                warnings=["PlanState requested current-start route before navigation replacement"],
            )
    if (
        selected_poi_id.startswith("poi_")
        and destination == selected_poi_id
        and selected_route_id
        and proposed_route_id != selected_route_id
        and "navigation_replace_final_destination" in tool_names
    ):
        return PlanStateDecision(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": selected_poi_id,
                            "route_id_leading_to_new_destination": selected_route_id,
                        },
                    }
                ],
            },
            warnings=["PlanState corrected replacement route id to the selected route"],
        )
    return PlanStateDecision()


def _active_navigation_replacement_from_route(
    call: dict[str, Any],
    state: PlanState,
    tool_names: set[str],
) -> dict[str, Any] | None:
    if "navigation_replace_final_destination" not in tool_names:
        return None
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    route_ids = args.get("route_ids") if isinstance(args.get("route_ids"), list) else []
    route_id = str(route_ids[0]) if route_ids else ""
    route = next(
        (item for item in state.facts.get("routes", []) if item.get("route_id") == route_id),
        None,
    )
    destination = str((route or {}).get("destination_id") or "")
    if not route_id or not destination:
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "navigation_replace_final_destination",
                "arguments": {
                    "new_destination_id": destination,
                    "route_id_leading_to_new_destination": route_id,
                },
            }
        ],
    }


def _repair_email_call(
    call: dict[str, Any],
    state: PlanState,
) -> tuple[dict[str, Any], list[str]]:
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    content = str(args.get("content_message") or "")
    cleaned = _sanitize_email_content(content)
    enriched = _enriched_email_content(cleaned, state)
    warnings: list[str] = []
    if cleaned != content:
        warnings.append("PlanState removed confirmation wording from email content")
    if enriched != content:
        call = {
            **call,
            "arguments": {
                **args,
                "content_message": enriched,
            },
        }
        warnings.append("PlanState enriched email content with known calendar/weather facts")
    if not args.get("email_addresses") and state.facts.get("contact_emails"):
        call = {
            **call,
            "arguments": {
                **call.get("arguments", {}),
                "email_addresses": state.facts["contact_emails"],
            },
        }
        warnings.append("PlanState filled email recipients from contact information")
    return call, warnings


def _next_recovery_action(state: PlanState, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    tool_names = _tool_names(tools)
    if "charging" in state.domains:
        charging_action = _next_charging_action(state, tools)
        if charging_action:
            return charging_action
    return _next_navigation_action(state, tool_names) or _next_email_action(state, tool_names)


def _next_charging_action(state: PlanState, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    tool_names = _tool_names(tools)
    text = f"{state.original_user_request} {state.latest_user_request}".lower()
    latest_text = (state.latest_user_request or "").lower()

    if _wants_delete_final_destination(latest_text):
        delete_action = _delete_final_destination_action(state, tool_names)
        if delete_action:
            return delete_action

    if (
        "get_current_navigation_state" in tool_names
        and not state.facts.get("navigation_waypoints")
        and (_wants_charging_station_search(text) or _wants_delete_final_destination(text))
    ):
        return {
            "action": "tool_calls",
            "tool_calls": [
                {
                    "tool_name": "get_current_navigation_state",
                    "arguments": {"detailed_information": True},
                }
            ],
        }

    if "get_charging_specs_and_status" in tool_names and not state.facts.get("charging"):
        return {
            "action": "tool_calls",
            "tool_calls": [{"tool_name": "get_charging_specs_and_status", "arguments": {}}],
        }

    if (
        state.facts.get("charging")
        and not state.facts.get("distance_by_soc")
        and "get_distance_by_soc" in tool_names
        and _mentions_range_buffer(text)
    ):
        initial_soc = _number_from_value(state.facts.get("charging", {}).get("state_of_charge"))
        final_soc = _requested_final_soc(text) or 20
        if initial_soc is not None:
            return {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "get_distance_by_soc",
                        "arguments": {
                            "initial_state_of_charge": int(initial_soc),
                            "final_state_of_charge": int(final_soc),
                        },
                    }
                ],
            }

    if (
        _wants_charging_station_search(text)
        and not _charging_station_search_done(state)
        and "search_poi_along_the_route" in tool_names
    ):
        route_id = _first_active_route_id(state)
        search_text = latest_text if _wants_charging_station_search(latest_text) else text
        at_kilometer = _requested_at_kilometer(search_text) or _requested_at_kilometer(text)
        if route_id and at_kilometer is not None:
            arguments: dict[str, Any] = {
                "route_id": route_id,
                "category_poi": "charging_stations",
                "at_kilometer": at_kilometer,
            }
            if _tool_has_parameter(tools, "search_poi_along_the_route", "filters"):
                arguments["filters"] = _charging_station_filters(search_text)
            return {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "search_poi_along_the_route",
                        "arguments": arguments,
                    }
                ],
            }

    if _wants_delete_final_destination(text) and _charging_station_search_done(state):
        return _delete_final_destination_action(state, tool_names)
    return None


def _next_navigation_action(state: PlanState, tool_names: set[str]) -> dict[str, Any] | None:
    latest_text = (state.latest_user_request or "").lower()
    if _wants_delete_final_destination(latest_text):
        delete_action = _delete_final_destination_action(state, tool_names)
        if delete_action:
            return delete_action
    selected_poi = state.facts.get("selected_poi") or {}
    selected_route = state.facts.get("selected_route") or {}
    if (
        selected_poi
        and not selected_route
        and not _charging_poi_without_navigation_intent(selected_poi, latest_text)
    ):
        return _route_lookup_to_selected_poi(state, tool_names)
    if (
        selected_poi
        and selected_route
        and state.facts.get("navigation_active") is True
        and "navigation_replace_final_destination" in tool_names
    ):
        return {
            "action": "tool_calls",
            "tool_calls": [
                {
                    "tool_name": "navigation_replace_final_destination",
                    "arguments": {
                        "new_destination_id": selected_poi["poi_id"],
                        "route_id_leading_to_new_destination": selected_route["route_id"],
                    },
                }
            ],
        }
    if selected_poi and selected_route and "set_new_navigation" in tool_names:
        return {
            "action": "tool_calls",
            "tool_calls": [
                {
                    "tool_name": "set_new_navigation",
                    "arguments": {"route_ids": [selected_route["route_id"]]},
                }
            ],
        }
    category = _preferred_poi_category(state.latest_user_request + " " + state.original_user_request)
    known_location = _last_location_id(state)
    if category and known_location and "search_poi_at_location" in tool_names:
        return {
            "action": "tool_calls",
            "tool_calls": [
                {
                    "tool_name": "search_poi_at_location",
                    "arguments": {"location_id": known_location, "category_poi": category},
                }
            ],
        }
    return None


def _route_lookup_to_selected_poi(state: PlanState, tool_names: set[str]) -> dict[str, Any] | None:
    selected_poi = state.facts.get("selected_poi") or {}
    if _charging_poi_without_navigation_intent(
        selected_poi,
        state.latest_user_request or "",
    ):
        return None
    return _route_lookup_to_destination(
        state,
        tool_names,
        destination_id=str(selected_poi.get("poi_id") or ""),
    )


def _charging_poi_without_navigation_intent(
    selected_poi: dict[str, Any],
    latest_text: str,
) -> bool:
    if not _is_charging_poi(selected_poi):
        return False
    lowered = latest_text.lower()
    if _has_any(
        lowered,
        (
            "don't want to add",
            "do not want to add",
            "don't add",
            "do not add",
            "not add",
            "not set",
            "don't set",
            "do not set",
            "just want",
            "only want",
            "show me",
            "display",
            "list",
        ),
    ):
        return True
    return not _has_any(
        lowered,
        (
            "navigate",
            "navigation",
            "route me",
            "directions",
            "add",
            "set",
            "replace",
            "go to",
        ),
    )


def _is_charging_poi(poi: dict[str, Any]) -> bool:
    poi_id = str(poi.get("poi_id") or poi.get("id") or "").lower()
    category = str(poi.get("category") or "").lower()
    name = str(poi.get("name") or "").lower()
    return poi_id.startswith("poi_cha") or category == "charging_stations" or "charging" in name


def _charging_options_response(state: PlanState) -> str:
    candidates = [
        poi
        for poi in state.facts.get("poi_candidates") or []
        if _is_charging_poi(poi)
    ]
    names = [
        str(candidate.get("name") or candidate.get("poi_id") or "").strip()
        for candidate in candidates[-3:]
    ]
    names = [name for name in names if name]
    if not names:
        return "I found charging station options along the route, and I will not add one to navigation yet."
    return (
        "I found these charging station options along the route: "
        + ", ".join(names)
        + ". I will not add one to navigation yet."
    )


def _route_lookup_to_destination(
    state: PlanState,
    tool_names: set[str],
    *,
    destination_id: str,
    preferred_start_id: str | None = None,
) -> dict[str, Any] | None:
    if "get_routes_from_start_to_destination" not in tool_names:
        return None
    if not destination_id:
        return None
    start_id = preferred_start_id or state.facts.get("navigation_start_id") or _last_route_start_id(state) or _last_location_id(state)
    if not start_id:
        return None
    if _route_to_destination_exists(state, destination_id, preferred_start_id=start_id):
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "get_routes_from_start_to_destination",
                "arguments": {"start_id": start_id, "destination_id": destination_id},
            }
        ],
    }


def _next_email_action(state: PlanState, tool_names: set[str]) -> dict[str, Any] | None:
    if "send_email" not in tool_names:
        return None
    emails = state.facts.get("contact_emails") or []
    if not emails:
        return None
    content = _sanitize_email_content(_enriched_email_content("", state))
    if not content:
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "send_email",
                "arguments": {"email_addresses": emails, "content_message": content},
            }
        ],
    }


def _email_task_requested(state: PlanState) -> bool:
    text = f"{state.original_user_request} {state.latest_user_request}".lower()
    return _has_any(text, ("email", "mail", "message", "attendee", "attendees"))


def _email_ready(state: PlanState) -> bool:
    return bool(state.facts.get("contact_emails")) and bool(
        _sanitize_email_content(_enriched_email_content("", state)).strip()
    )


def _looks_like_email_confirmation_request(content: str) -> bool:
    lowered = content.lower()
    if not _has_any(lowered, ("send", "email", "mail")):
        return False
    return _has_any(
        lowered,
        (
            "confirm",
            "should i",
            "shall i",
            "would you like",
            "want me to",
            "go ahead",
            "please say yes",
            "say yes",
        ),
    )


def _email_confirmation_prompt(state: PlanState) -> str:
    emails = ", ".join(str(email) for email in state.facts.get("contact_emails") or [])
    content = _sanitize_email_content(_enriched_email_content("", state)).strip()
    if len(content) > 240:
        content = content[:237].rstrip() + "..."
    if emails and content:
        return (
            f"I can send the email to {emails} saying: {content} "
            "Please say yes to confirm."
        )
    if emails:
        return f"I can send the email to {emails}. Please say yes to confirm."
    return "I can send the email once I have the recipient. Please say yes to confirm."


def _sanitize_email_content(content: str) -> str:
    text = content.strip()
    if not text:
        return ""
    patterns = (
        r"(?:\s*Please say yes to confirm\.?\s*)$",
        r"(?:\s*Please say yes to confirm\.?\s*)",
        r"(?:\s*Please confirm\.?\s*)$",
        r"(?:\s*Should I send it\??\s*)$",
        r"(?:\s*Do you want me to send it\??\s*)$",
        r"(?:\s*Would you like me to send it\??\s*)$",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _latest_user_confirms(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered or _has_any(lowered, ("no", "don't", "do not", "stop", "cancel")):
        return False
    return bool(
        re.search(
            r"\b(yes|yeah|yep|sure|ok|okay|confirm|confirmed|send it|go ahead|please do|do it)\b",
            lowered,
        )
    )


def _tool_completed(state: PlanState, tool_name: str) -> bool:
    return any(
        record.tool_name == tool_name
        and record.result_text
        and not _tool_result_failed(record)
        for record in state.action_history
    )


def _action_contains_completed_state_change(state: PlanState, action: dict[str, Any]) -> bool:
    if action.get("action") != "tool_calls":
        return False
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        if name and name not in READ_ONLY_TOOLS and _tool_completed(state, name):
            return True
    return False


def _repeated_completed_response(state: PlanState, tool_name: str) -> str:
    selected_poi = state.facts.get("selected_poi") or {}
    if (
        tool_name in {"search_poi_along_the_route", "get_routes_from_start_to_destination"}
        and _charging_poi_without_navigation_intent(selected_poi, state.latest_user_request or "")
    ):
        return _charging_options_response(state)
    return "I already have that information, so I will not repeat the same lookup."


def _enriched_email_content(content: str, state: PlanState) -> str:
    pieces: list[str] = []
    base = _sanitize_email_content(content).strip()
    if base:
        pieces.append(base.rstrip("."))
    calendar = state.facts.get("calendar") or {}
    weather = state.facts.get("weather") or {}
    if calendar:
        meeting_bits = []
        if calendar.get("title"):
            meeting_bits.append(str(calendar["title"]))
        if calendar.get("location"):
            meeting_bits.append(f"location: {calendar['location']}")
        if calendar.get("time"):
            meeting_bits.append(f"time: {calendar['time']}")
        meeting = "; ".join(meeting_bits)
        if meeting and meeting.lower() not in base.lower():
            pieces.append(f"Meeting details: {meeting}")
    if weather:
        weather_bits = []
        for key in ("condition", "temperature", "wind", "humidity"):
            if weather.get(key) is not None:
                label = "temperature" if key == "temperature" else key
                weather_bits.append(f"{label}: {weather[key]}")
        weather_text = "; ".join(weather_bits)
        if weather_text and weather_text.lower() not in base.lower():
            pieces.append(f"Weather: {weather_text}")
    if weather and "travel" not in " ".join(pieces).lower():
        pieces.append("These conditions may affect travel to the meeting, so please plan accordingly")
    return ". ".join(piece for piece in pieces if piece).strip() + ("." if pieces else "")


def _infer_domains(text: str, history: list[ActionRecord]) -> list[str]:
    lowered = text.lower()
    domains: list[str] = []
    if _has_any(lowered, ("navigate", "navigation", "route", "destination", "waypoint", "poi", "restaurant", "charging station", "charger")):
        domains.append("navigation")
    if _has_any(lowered, ("email", "mail", "message", "contact", "calendar", "meeting", "attendee")):
        domains.append("communication")
    if _has_any(lowered, ("weather", "rain", "snow", "storm", "travel")):
        domains.append("weather")
    if _has_any(lowered, ("charge", "charging", "battery", "soc", "range", "charger")):
        domains.append("charging")
    if _has_any(lowered, ("temperature", "climate", "air conditioning", " ac ", "fan", "window", "sunroof", "sunshade", "defrost")):
        domains.append("climate")
    tool_text = " ".join(record.tool_name for record in history)
    if _has_any(tool_text, ("navigation", "routes", "poi")) and "navigation" not in domains:
        domains.append("navigation")
    if _has_any(tool_text, ("email", "contact", "calendar")) and "communication" not in domains:
        domains.append("communication")
    return domains or ["general"]


def _select_poi(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in reversed(candidates):
        if str(candidate.get("poi_id") or "").startswith("poi_"):
            return candidate
    return None


def _select_route_for_destination(
    routes: list[dict[str, Any]],
    destination_id: str | None,
    *,
    preferred_start_id: str | None = None,
    request_text: str = "",
) -> dict[str, Any] | None:
    candidates = list(routes)
    if destination_id:
        candidates = [route for route in candidates if route.get("destination_id") == destination_id]
    if preferred_start_id:
        start_matches = [
            route for route in candidates if route.get("start_id") == preferred_start_id
        ]
        selected = _select_route_by_request(start_matches, request_text)
        if selected is not None:
            return selected
        if start_matches:
            return _default_route(start_matches)
        unknown_start = [route for route in candidates if not route.get("start_id")]
        selected = _select_route_by_request(unknown_start, request_text)
        if selected is not None:
            return selected
        if unknown_start:
            return _default_route(unknown_start)
        if destination_id:
            return None
    selected = _select_route_by_request(candidates, request_text)
    if selected is not None:
        return selected
    return _default_route(candidates)


def _select_route_by_request(
    routes: list[dict[str, Any]],
    request_text: str,
) -> dict[str, Any] | None:
    if not routes:
        return None
    text = request_text.lower()
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


def _ambiguous_route_response(
    *,
    state: PlanState,
    destination_id: str,
    preferred_start_id: str | None,
) -> str | None:
    routes = _route_candidates_for_destination(
        state.facts.get("routes") or [],
        destination_id,
        preferred_start_id=preferred_start_id,
    )
    if len(routes) < 2:
        return None
    request_text = f"{state.original_user_request} {state.latest_user_request}".lower()
    if _route_selection_is_explicit(routes, request_text):
        return None
    options: list[str] = []
    for index, route in enumerate(routes[:3]):
        ordinal = _ordinal_alias(index) or f"option {index + 1}"
        via = str(route.get("name_via") or "").strip()
        label = f"{ordinal} route"
        if via:
            label += f" via {via}"
        aliases = sorted(_route_aliases(route) - {ordinal, "first", "second", "third"})
        if aliases:
            label += f" ({', '.join(aliases)})"
        options.append(label)
    return "I found multiple route options: " + "; ".join(options) + ". Which route should I use?"


def _route_candidates_for_destination(
    routes: list[dict[str, Any]],
    destination_id: str,
    *,
    preferred_start_id: str | None,
) -> list[dict[str, Any]]:
    candidates = [
        route for route in routes if not destination_id or route.get("destination_id") == destination_id
    ]
    if preferred_start_id:
        start_matches = [
            route for route in candidates if route.get("start_id") == preferred_start_id
        ]
        if start_matches:
            return start_matches
    return candidates


def _route_selection_is_explicit(
    routes: list[dict[str, Any]],
    request_text: str,
) -> bool:
    text = request_text.lower()
    if _requested_route_ordinal(text) is not None:
        return True
    if _has_any(text, ("fastest", "shortest")):
        return True
    return _route_matching_via_text(routes, text) is not None


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


def _route_by_id(routes: list[dict[str, Any]], route_id: str) -> dict[str, Any] | None:
    if not route_id:
        return None
    for route in reversed(routes):
        if route.get("route_id") == route_id:
            return route
    return None


def _is_repeated_completed_call(state: PlanState, tool_name: str, arguments: dict[str, Any]) -> bool:
    if tool_name not in READ_ONLY_TOOLS:
        return False
    fingerprint = _fingerprint(tool_name, arguments)
    count = sum(
        1
        for record in state.action_history
        if record.fingerprint == fingerprint and record.result_text
    )
    return count > 0


def _is_repeated_failed_call(state: PlanState, tool_name: str, arguments: dict[str, Any]) -> bool:
    fingerprint = _fingerprint(tool_name, arguments)
    return any(
        record.fingerprint == fingerprint and _tool_result_failed(record)
        for record in state.action_history
    )


def _action_contains_repeated_failed_call(state: PlanState, action: dict[str, Any]) -> bool:
    if action.get("action") != "tool_calls":
        return False
    for call in action.get("tool_calls") or []:
        name = str(call.get("tool_name") or "")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if _is_repeated_failed_call(state, name, args):
            return True
    return False


def _tool_result_failed(record: ActionRecord) -> bool:
    data = record.result_json
    if isinstance(data, dict):
        status = _first_value_for_keys(data, ("status", "result_status"))
        if isinstance(status, str) and status.strip().lower() in {"failure", "failed", "error"}:
            return True
        errors = _first_value_for_keys(data, ("error", "errors"))
        if errors not in (None, "", []):
            return True
    text = record.result_text.lower()
    return any(piece in text for piece in ('"status":"failure"', '"status": "failure"', "failure", "failed", "error"))


def _tool_result_is_navigation_active_failure(record: ActionRecord) -> bool:
    if not _tool_result_failed(record):
        return False
    return "navigation already active" in record.result_text.lower()


def _response_repeated(state: PlanState, content: str) -> bool:
    normalized = _normalize_text(content)
    if not normalized:
        return False
    count = 0
    for record in state.action_history:
        if record.tool_name != "respond":
            continue
        if _normalize_text(str(record.arguments.get("content") or "")) == normalized:
            count += 1
    return count >= 1


def _repeated_action_descriptions(history: list[ActionRecord]) -> list[str]:
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for record in history:
        if record.tool_name not in READ_ONLY_TOOLS:
            continue
        counts[record.fingerprint] = counts.get(record.fingerprint, 0) + 1
        names[record.fingerprint] = record.tool_name
    return [names[key] for key, count in counts.items() if count > 1]


def _has_state_changing_action(history: list[ActionRecord]) -> bool:
    return any(
        record.tool_name != "respond" and record.tool_name not in READ_ONLY_TOOLS
        for record in history
    )


def _has_relevant_available_tool(state: PlanState, tool_names: set[str]) -> bool:
    if "navigation" in state.domains and any(
        name in tool_names
        for name in (
            "get_location_id_by_location_name",
            "search_poi_at_location",
            "get_routes_from_start_to_destination",
            "navigation_replace_final_destination",
            "set_new_navigation",
        )
    ):
        return True
    if "communication" in state.domains and any(
        name in tool_names for name in ("get_contact_information", "send_email")
    ):
        return True
    return False


def _last_location_id(state: PlanState) -> str | None:
    locations = state.facts.get("location_ids") or []
    if locations:
        return str(locations[-1].get("id") or "") or None
    for route in reversed(state.facts.get("routes") or []):
        if route.get("start_id"):
            return str(route["start_id"])
    return None


def _last_route_start_id(state: PlanState) -> str | None:
    for route in reversed(state.facts.get("routes") or []):
        if route.get("start_id"):
            return str(route["start_id"])
    return None


def _route_to_destination_exists(
    state: PlanState,
    destination_id: str,
    *,
    preferred_start_id: str | None = None,
) -> bool:
    for route in state.facts.get("routes") or []:
        if route.get("destination_id") != destination_id:
            continue
        if preferred_start_id and route.get("start_id") != preferred_start_id:
            continue
        return True
    return False


def _first_active_route_id(state: PlanState) -> str | None:
    route_ids = state.facts.get("navigation_routes_to_final_destination") or []
    for route_id in route_ids:
        if isinstance(route_id, str) and route_id.startswith("r"):
            return route_id
    for route in state.facts.get("routes") or []:
        route_id = str(route.get("route_id") or "")
        destination = str(route.get("destination_id") or "")
        if route_id.startswith("r") and destination.startswith("loc_"):
            return route_id
    return None


def _delete_final_destination_action(
    state: PlanState,
    tool_names: set[str],
) -> dict[str, Any] | None:
    if "navigation_delete_destination" not in tool_names:
        return None
    destination_id = str(state.facts.get("navigation_destination_id") or "")
    waypoints = state.facts.get("navigation_waypoints") or []
    if not destination_id.startswith(("loc_", "poi_")):
        return None
    if len(waypoints) < 2:
        return None
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": "navigation_delete_destination",
                "arguments": {"destination_id_to_delete": destination_id},
            }
        ],
    }


def _charging_station_search_done(state: PlanState) -> bool:
    for record in state.action_history:
        if record.tool_name != "search_poi_along_the_route":
            continue
        category = str(record.arguments.get("category_poi") or "")
        if category == "charging_stations" and record.result_text:
            return True
    return False


def _requested_final_soc(text: str) -> int | None:
    lowered = text.lower()
    patterns = (
        r"(?:keep|leave|left|remaining|minimum|buffer|at least|safety)\D{0,24}(\d{1,3})\s*%",
        r"(\d{1,3})\s*%\D{0,24}(?:left|remaining|minimum|buffer|safety)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 100:
                return value
    return None


def _requested_at_kilometer(text: str) -> int | None:
    lowered = text.lower()
    patterns = (
        r"(?:around|about|near|at|approximately|roughly|specific(?:ally)?)\D{0,24}(\d{1,4})\s*(?:km|kilometer)",
        r"(\d{1,4})\s*(?:km|kilometer)\D{0,30}(?:into|along|mark|point)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value
    return None


def _charging_station_filters(text: str) -> list[str]:
    lowered = text.lower()
    filters: list[str] = []
    if _has_any(lowered, ("dc", "fast")):
        filters.append("charging_stations::has_dc_plug")
    if _has_any(
        lowered,
        (
            "available",
            "available plug",
            "available plugs",
            "free plug",
            "open plug",
            "currently open",
            "open now",
        ),
    ):
        filters.append("charging_stations::has_available_plug")
    return filters or ["charging_stations::has_available_plug"]


def _tool_has_parameter(tools: list[dict[str, Any]], tool_name: str, parameter: str) -> bool:
    for tool in tools:
        if str(tool.get("function", {}).get("name") or tool.get("name") or "") != tool_name:
            continue
        properties = (
            tool.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        return parameter in properties
    return False


def _number_from_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _preferred_poi_category(text: str) -> str | None:
    lowered = f" {text.lower()} "
    for phrase, category in sorted(POI_CATEGORIES.items(), key=lambda item: -len(item[0])):
        if phrase in lowered:
            return category
    return None


def _mentions_weather_or_travel(text: str) -> bool:
    return _has_any(text.lower(), ("weather", "rain", "snow", "storm", "travel", "meeting"))


def _mentions_range_buffer(text: str) -> bool:
    lowered = text.lower()
    return _has_any(
        lowered,
        ("range", "battery", "charge", "charging", "soc", "make it", "reach", "buffer", "safety"),
    ) and _has_any(
        lowered,
        ("keep", "left", "remaining", "minimum", "at least", "buffer", "safety", "enough", "make it", "reach"),
    )


def _wants_charging_station_search(text: str) -> bool:
    lowered = text.lower()
    return _has_any(
        lowered,
        ("charging station", "charging stations", "charger", "chargers", "charging stop", "dc plug", "dc fast"),
    ) and _has_any(lowered, ("route", "along", "around", "near", "kilometer", "km", "into"))


def _wants_delete_final_destination(text: str) -> bool:
    lowered = text.lower()
    return _has_any(lowered, ("remove", "delete", "cancel", "drop")) and _has_any(
        lowered,
        (
            "destination",
            "waypoint",
            "stop",
            "endpoint",
            "end point",
            "final",
            "continue",
            "colonge",
            "cologne",
        ),
    )


def _weather_summary(data: Any, raw_text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    condition = _first_value_for_keys(data, ("condition", "weather", "description"))
    temperature = _first_value_for_keys(data, ("temperature", "temp", "temperature_c", "temperature_celsius"))
    wind = _first_value_for_keys(data, ("wind", "wind_speed", "wind_kmh", "wind_speed_kph"))
    humidity = _first_value_for_keys(data, ("humidity", "humidity_percent"))
    if condition is None:
        match = re.search(r"(cloudy|rain|snow|storm|sunny|clear|fog|mist)[a-z_ ]*", raw_text, re.I)
        if match:
            condition = match.group(0)
    if condition is not None:
        summary["condition"] = str(condition)
    if temperature is not None:
        summary["temperature"] = _format_weather_value(temperature, "C")
    if wind is not None:
        summary["wind"] = _format_weather_value(wind, "km/h")
    if humidity is not None:
        summary["humidity"] = _format_weather_value(humidity, "%")
    return summary


def _charging_summary(data: Any, raw_text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source_key, target_key in (
        ("state_of_charge", "state_of_charge"),
        ("current_state_of_charge", "state_of_charge"),
        ("remaining_range", "remaining_range"),
        ("remaining_range_km", "remaining_range"),
        ("battery_capacity_kwh", "battery_capacity_kwh"),
        ("energy_consumption", "energy_consumption"),
    ):
        value = _first_value_for_keys(data, (source_key,))
        if value is not None:
            summary[target_key] = value
    if "state_of_charge" not in summary:
        match = re.search(r"\b(\d{1,3})\s*%\b", raw_text)
        if match:
            summary["state_of_charge"] = int(match.group(1))
    return summary


def _distance_by_soc_summary(data: Any, raw_text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    distance = _first_value_for_keys(
        data,
        ("distance_km", "range_km", "driving_range_km", "reachable_distance_km", "distance"),
    )
    if distance is None:
        match = re.search(r"\b(\d{1,4}(?:\.\d+)?)\s*(?:km|kilometers?)\b", raw_text, re.I)
        if match:
            distance = float(match.group(1))
    if distance is not None:
        summary["distance_km"] = distance
    return summary


def _calendar_summary(data: Any, raw_text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    title = _first_value_for_keys(data, ("title", "summary", "subject", "event_name", "topic", "name"))
    location = _first_value_for_keys(data, ("location", "location_name", "city"))
    time_value = _first_value_for_keys(data, ("time", "start_time", "start", "hour", "time_hour_24hformat"))
    if title is not None:
        summary["title"] = str(title)
    if location is not None:
        summary["location"] = _compact_value(location)
    time_text = _format_calendar_time(time_value) or _first_time_in_text(raw_text)
    if time_text:
        summary["time"] = time_text
    return summary


def _format_weather_value(value: Any, suffix: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{value:g} {suffix}"
    return _compact_value(value)


def _format_calendar_time(value: Any) -> str | None:
    if isinstance(value, str):
        return _first_time_in_text(value) or value
    if isinstance(value, dict):
        hour = _first_value_for_keys(value, ("hour", "time_hour_24hformat"))
        minute = _first_value_for_keys(value, ("minute", "minutes", "time_minutes"))
        if isinstance(hour, str) and hour.isdigit():
            hour = int(hour)
        if isinstance(minute, str) and minute.isdigit():
            minute = int(minute)
        if isinstance(hour, (int, float)) and float(hour).is_integer():
            minute_value = int(minute) if isinstance(minute, (int, float)) else 0
            if 0 <= int(hour) <= 23 and 0 <= minute_value <= 59:
                return f"{int(hour):02d}:{minute_value:02d}"
    if isinstance(value, int):
        return f"{value:02d}:00"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):02d}:00"
    return None


def _first_time_in_text(text: str) -> str | None:
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    match = re.search(r"\b([01]?\d|2[0-3])\s*(?:h|hour)\b", text, re.I)
    if match:
        return f"{int(match.group(1)):02d}:00"
    return None


def _route_detail_for_id(data: Any, route_id: str) -> dict[str, str]:
    route = _dict_containing_value(data, route_id)
    if not route:
        return {}
    destination = _first_value_for_keys(route, ("destination_id", "new_destination_id", "location_or_poi_id"))
    start = _first_value_for_keys(route, ("start_id", "origin_id"))
    detail: dict[str, Any] = {}
    if start:
        detail["start_id"] = str(start)
    if destination:
        detail["destination_id"] = str(destination)
    for key in ("name_via", "distance_km", "duration_min", "duration_minutes", "duration_hours", "alias"):
        if route.get(key) is not None:
            detail[key] = route[key]
    return detail


def _entity_name_near_id(data: Any, entity_id: str) -> str | None:
    entity = _dict_containing_value(data, entity_id)
    if not entity:
        return None
    name = _first_value_for_keys(entity, ("name", "title", "poi_name"))
    return str(name) if name is not None else None


def _dict_containing_value(data: Any, value: str) -> dict[str, Any] | None:
    if isinstance(data, dict):
        if value in [str(item) for item in data.values() if isinstance(item, (str, int, float))]:
            return data
        for child in data.values():
            found = _dict_containing_value(child, value)
            if found:
                return found
    elif isinstance(data, list):
        for child in data:
            found = _dict_containing_value(child, value)
            if found:
                return found
    return None


def _collect_route_ids(data: Any) -> list[str]:
    ids: list[str] = []
    _collect_route_ids_from_keys(data, ids)
    return ids


def _collect_route_ids_from_keys(data: Any, ids: list[str]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "route_id" and isinstance(value, str) and value.startswith("r"):
                if value not in ids:
                    ids.append(value)
                continue
            if key in {"route_ids", "routes_to_final_destination_id"} and isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.startswith("r") and item not in ids:
                        ids.append(item)
                continue
            _collect_route_ids_from_keys(value, ids)
    elif isinstance(data, list):
        for item in data:
            _collect_route_ids_from_keys(item, ids)


def _collect_prefixed_ids(data: Any, prefix: str) -> list[str]:
    ids = []
    for value in _walk_values(data):
        if isinstance(value, str):
            for match in re.findall(rf"\b{re.escape(prefix)}[A-Za-z0-9_]+\b", value):
                if match not in ids:
                    ids.append(match)
    return ids


def _first_value_for_keys(data: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys and value not in (None, "", []):
                return value
        for value in data.values():
            found = _first_value_for_keys(value, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(data, list):
        for item in data:
            found = _first_value_for_keys(item, keys)
            if found not in (None, "", []):
                return found
    return None


def _find_bool(data: Any, key: str) -> bool | None:
    value = _first_value_for_keys(data, (key,))
    return value if isinstance(value, bool) else None


def _extract_email_addresses(text: str) -> list[str]:
    return _unique(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))


def _walk_values(data: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(data, dict):
        for value in data.values():
            values.append(value)
            values.extend(_walk_values(value))
    elif isinstance(data, list):
        for value in data:
            values.append(value)
            values.extend(_walk_values(value))
    return values


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(tool.get("function", {}).get("name") or tool.get("name") or "")
        for tool in tools
        if str(tool.get("function", {}).get("name") or tool.get("name") or "")
    }


def _parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
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
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_any(text: str, pieces: tuple[str, ...]) -> bool:
    return any(piece in text for piece in pieces)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _compact_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:200]


def _unique(items) -> list[Any]:
    result: list[Any] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _append_unique_dict(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if item not in items:
        items.append(item)
