"""Deterministic task-family skills for Track 1.

The skills are small state machines that can choose an obvious next action
before the planner spends an LLM call.  They use only the current transcript and
runtime tool schemas, not task ids or gold answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SkillDecision:
    action: dict[str, Any] | None = None
    skill: str | None = None
    warnings: list[str] = field(default_factory=list)


class Skill(Protocol):
    name: str

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        ...


class SkillRegistry:
    completion_gate_skills = {
        "reading_light_occupancy",
        "window_match_defrost",
        "occupancy_climate_efficiency",
    }

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self.skills = skills or [
            CommunicationEmailSkill(),
            HallucinationGuardSkill(),
            ReadingLightOccupancySkill(),
            WindowMatchDefrostSkill(),
            OccupancyClimateEfficiencySkill(),
            NavigationEditSkill(),
            ChargingRouteSkill(),
            ClimateACDefrostSkill(),
        ]

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        for skill in self.skills:
            decision = skill.preempt(messages=messages, tools=tools)
            if decision.action:
                decision.skill = decision.skill or skill.name
                return decision
        return SkillDecision()

    def before_tool_calls(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        proposed_action: dict[str, Any],
    ) -> SkillDecision:
        """Replace partial tool batches with a deterministic complete checklist."""

        decision = self.preempt(messages=messages, tools=tools)
        if (
            not decision.action
            or decision.action.get("action") != "tool_calls"
            or decision.skill not in self.completion_gate_skills
        ):
            return SkillDecision()
        proposed_calls = list(proposed_action.get("tool_calls") or [])
        skill_calls = list(decision.action.get("tool_calls") or [])
        if not skill_calls or _same_call_batch(proposed_calls, skill_calls):
            return SkillDecision()
        decision.warnings.append(
            f"replaced partial tool batch with complete {decision.skill} checklist"
        )
        return decision

    def before_response(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_content: str,
    ) -> SkillDecision:
        """Block premature user-facing responses when a grounded tool step exists."""

        for skill in self.skills:
            decision = skill.preempt(messages=messages, tools=tools)
            if not decision.action or decision.action.get("action") != "tool_calls":
                continue
            decision.skill = decision.skill or skill.name
            decision.warnings.append(
                f"blocked premature response before completing {skill.name}: {response_content[:80]}"
            )
            return decision
        return SkillDecision()


class CommunicationEmailSkill:
    name = "communication_email"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        latest_user = _latest_user_text(messages)
        if not _is_confirmation_text(latest_user):
            return SkillDecision()
        tool_map = _tool_map(tools)
        if not (
            _tool_has_parameter(tool_map, "send_email", "email_addresses")
            and _tool_has_parameter(tool_map, "send_email", "content_message")
        ):
            return SkillDecision()
        previous = _previous_assistant_text(messages)
        if not previous or not _has_any(previous.lower(), ("email", "mail", "send")):
            return SkillDecision()
        recipients = _extract_email_addresses(previous) or _latest_contact_email_addresses(messages)
        if not recipients:
            return SkillDecision()
        content = _sanitize_email_content(
            _extract_quoted_or_after_colon(previous) or _fallback_email_content(messages)
        )
        return SkillDecision(
            action=_tool_calls(
                [
                    {
                        "tool_name": "send_email",
                        "arguments": {
                            "email_addresses": recipients,
                            "content_message": _normalize_24h_times(
                                _sanitize_email_content(content)
                            ),
                        },
                    }
                ]
            ),
            warnings=["email confirmation converted to send_email"],
        )


class HallucinationGuardSkill:
    name = "hallucination_guard"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        text = _latest_actionable_user_text(messages).lower()
        if not text:
            return SkillDecision()
        tool_map = _tool_map(tools)

        if _mentions_ac(text):
            if "set_air_conditioning" not in tool_map:
                return SkillDecision(
                    action=_respond("I cannot control the air conditioning because that control is unavailable."),
                    warnings=["missing required capability set_air_conditioning"],
                )
            if not _tool_has_parameter(tool_map, "set_air_conditioning", "on"):
                return SkillDecision(
                    action=_respond("I cannot control the air conditioning because that control is unavailable."),
                    warnings=["missing parameter set_air_conditioning.on"],
                )

        parameter_checks = [
            (("email", "mail", "send"), "send_email", "email_addresses", "I cannot send email because the email tool is missing the recipient parameter."),
            (("email", "mail", "send"), "send_email", "content_message", "I cannot send email because the email tool is missing the message parameter."),
            (("call", "phone"), "call_phone_by_number", "phone_number", "I cannot place a call because phone calling is unavailable."),
            (("fan", "airflow"), "set_fan_speed", "level", "I cannot set the fan speed because that control is unavailable."),
            (("window",), "open_close_window", "percentage", "I cannot move the windows because window position control is unavailable."),
            (("sunroof",), "open_close_sunroof", "percentage", "I cannot move the sunroof because sunroof control is unavailable."),
            (("sunshade",), "open_close_sunshade", "percentage", "I cannot move the sunshade because sunshade control is unavailable."),
            (("seat heating", "seat heater"), "set_seat_heating", "level", "I cannot set seat heating because that control is unavailable."),
            (("high beam", "high beams"), "set_head_lights_high_beams", "on", "I cannot control high beam headlights because that control is unavailable."),
            (("fog light", "fog lights"), "set_fog_lights", "on", "I cannot control fog lights because that control is unavailable."),
            (("navigate", "navigation", "route", "destination"), "get_current_navigation_state", "detailed_information", "I cannot inspect or edit navigation because navigation state lookup is unavailable."),
            (("weather", "rain", "snow", "forecast"), "get_weather", "location_or_poi_id", "I cannot check the weather because weather lookup is unavailable."),
            (("charge", "charging", "battery", "range"), "get_charging_specs_and_status", None, "I cannot answer charging or range questions because charging status is unavailable."),
        ]
        for keywords, tool_name, parameter, response in parameter_checks:
            if not _has_any(text, keywords):
                continue
            if tool_name not in tool_map:
                return SkillDecision(action=_respond(response), warnings=[f"missing required capability {tool_name}"])
            if parameter and not _tool_has_parameter(tool_map, tool_name, parameter):
                return SkillDecision(action=_respond(response), warnings=[f"missing parameter {tool_name}.{parameter}"])
        return SkillDecision()


class ReadingLightOccupancySkill:
    name = "reading_light_occupancy"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        text = _latest_actionable_user_text(messages).lower()
        if not _requests_occupancy_based_reading_lights(text):
            return SkillDecision()
        tool_map = _tool_map(tools)
        if "get_seats_occupancy" not in tool_map:
            return SkillDecision(
                action=_respond("I cannot adjust the reading lights by occupancy because seat occupancy is unavailable."),
                warnings=["seat occupancy unavailable for reading lights"],
            )
        if not (
            _tool_has_parameter(tool_map, "set_reading_light", "position")
            and _tool_has_parameter(tool_map, "set_reading_light", "on")
        ):
            return SkillDecision(
                action=_respond("I cannot adjust the reading lights because reading light control is unavailable."),
                warnings=["reading light control unavailable"],
            )
        get_calls: list[dict[str, Any]] = []
        if not _tool_called_after_latest_actionable_user(messages, "get_seats_occupancy"):
            get_calls.append({"tool_name": "get_seats_occupancy", "arguments": {}})
        if (
            "get_reading_lights_status" in tool_map
            and not _tool_called_after_latest_actionable_user(messages, "get_reading_lights_status")
        ):
            get_calls.append({"tool_name": "get_reading_lights_status", "arguments": {}})
        if get_calls:
            return SkillDecision(
                action=_tool_calls(get_calls),
                warnings=["reading light occupancy task starts with state checks"],
            )

        occupancy = _latest_seat_occupancy(messages)
        if not occupancy:
            return SkillDecision(
                action=_respond("I cannot adjust the reading lights because seat occupancy information is unavailable."),
                warnings=["seat occupancy result missing"],
            )

        reading_lights = _latest_reading_lights_status(messages)
        calls = _reading_light_calls_for_occupancy(occupancy, reading_lights)
        if not calls:
            return SkillDecision(
                action=_respond("Done."),
                warnings=["reading lights already match occupancy"],
            )
        return SkillDecision(
            action=_tool_calls(calls),
            warnings=["reading lights adjusted from seat occupancy"],
        )


class WindowMatchDefrostSkill:
    name = "window_match_defrost"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        text = _latest_actionable_user_text(messages).lower()
        if not _requests_window_match_and_defrost(text):
            return SkillDecision()
        tool_map = _tool_map(tools)
        required = {
            "get_vehicle_window_positions": (),
            "get_climate_settings": (),
            "open_close_window": ("window", "percentage"),
            "set_window_defrost": ("defrost_window", "on"),
            "set_fan_speed": ("level",),
            "set_fan_airflow_direction": ("direction",),
            "set_air_conditioning": ("on",),
        }
        missing = [
            name
            for name, params in required.items()
            if name not in tool_map or any(not _tool_has_parameter(tool_map, name, param) for param in params)
        ]
        if missing:
            return SkillDecision(
                action=_respond("I cannot complete the window and defrost optimization because a required vehicle control is unavailable."),
                warnings=[f"window/defrost controls unavailable: {', '.join(missing)}"],
            )

        get_calls: list[dict[str, Any]] = []
        if not _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions"):
            get_calls.append({"tool_name": "get_vehicle_window_positions", "arguments": {}})
        if not _tool_called_after_latest_actionable_user(messages, "get_climate_settings"):
            get_calls.append({"tool_name": "get_climate_settings", "arguments": {}})
        if get_calls:
            return SkillDecision(
                action=_tool_calls(get_calls),
                warnings=["window/defrost checklist gathering window and climate state"],
            )

        positions = _latest_window_positions(messages)
        if not positions:
            return SkillDecision(
                action=_respond("I cannot match the windows because the window position result is missing required fields."),
                warnings=["window positions missing for match task"],
            )
        reference_window = _requested_reference_window(text)
        target = positions.get(reference_window or "")
        if target is None:
            return SkillDecision(
                action=_respond("I cannot match the windows because the reference window position is unavailable."),
                warnings=["reference window position missing"],
            )

        climate = _latest_climate_settings(messages)
        calls: list[dict[str, Any]] = [
            {
                "tool_name": "open_close_window",
                "arguments": {"window": "ALL", "percentage": target},
            },
            {
                "tool_name": "set_window_defrost",
                "arguments": {"defrost_window": _requested_defrost_window(text) or "FRONT", "on": True},
            },
        ]
        if _number_from_value(_find_key(climate, "fan_speed")) != 2:
            calls.append({"tool_name": "set_fan_speed", "arguments": {"level": 2}})
        airflow = _string_from_value(_find_key(climate, "fan_airflow_direction"))
        if airflow != "WINDSHIELD":
            calls.append({"tool_name": "set_fan_airflow_direction", "arguments": {"direction": "WINDSHIELD"}})
        if _bool_from_value(_find_key(climate, "air_conditioning")) is not True:
            calls.append({"tool_name": "set_air_conditioning", "arguments": {"on": True}})
        return SkillDecision(
            action=_tool_calls(calls),
            warnings=["window/defrost checklist executed from grounded window and climate facts"],
        )


class OccupancyClimateEfficiencySkill:
    name = "occupancy_climate_efficiency"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        text = _all_actionable_user_text(messages).lower()
        if not _requests_occupancy_climate_efficiency(text):
            return SkillDecision()
        tool_map = _tool_map(tools)
        required = {
            "get_seats_occupancy": (),
            "get_temperature_inside_car": (),
            "get_seat_heating_level": (),
            "set_seat_heating": ("seat_zone", "level"),
            "set_climate_temperature": ("temperature", "seat_zone"),
        }
        missing = [
            name
            for name, params in required.items()
            if name not in tool_map or any(not _tool_has_parameter(tool_map, name, param) for param in params)
        ]
        if missing:
            return SkillDecision(
                action=_respond("I cannot complete the climate optimization because a required climate or seat control is unavailable."),
                warnings=[f"occupancy climate controls unavailable: {', '.join(missing)}"],
            )

        get_calls: list[dict[str, Any]] = []
        for name in ("get_seats_occupancy", "get_temperature_inside_car", "get_seat_heating_level"):
            if not _tool_called_anywhere(messages, name):
                get_calls.append({"tool_name": name, "arguments": {}})
        if get_calls:
            return SkillDecision(
                action=_tool_calls(get_calls),
                warnings=["occupancy climate checklist gathering seats, temperatures, and heating"],
            )

        occupancy = _latest_seat_occupancy(messages)
        heating = _latest_seat_heating_levels(messages)
        temperatures = _latest_climate_temperatures(messages)
        if not occupancy or not heating or not temperatures:
            return SkillDecision(
                action=_respond("I cannot optimize the climate because the current seat, heating, or temperature facts are incomplete."),
                warnings=["occupancy climate facts incomplete"],
            )

        calls = _seat_heating_calls_for_empty_heated_seats(occupancy, heating, messages)
        passenger_temperature = temperatures.get("PASSENGER")
        driver_temperature = temperatures.get("DRIVER")
        temperature_args = {"temperature": passenger_temperature, "seat_zone": "DRIVER"}
        if (
            passenger_temperature is not None
            and driver_temperature != passenger_temperature
            and not _tool_call_with_arguments_after_latest_user(messages, "set_climate_temperature", temperature_args)
        ):
            calls.append({"tool_name": "set_climate_temperature", "arguments": temperature_args})
        if not calls:
            return SkillDecision(
                action=_respond("Done."),
                warnings=["occupancy climate already satisfies checklist"],
            )
        return SkillDecision(
            action=_tool_calls(calls),
            warnings=["occupancy climate checklist executed from grounded facts"],
        )


class NavigationEditSkill:
    name = "navigation_route_edit"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        text = _latest_actionable_user_text(messages).lower()
        if not _requests_navigation_edit(text):
            return SkillDecision()
        tool_map = _tool_map(tools)
        if "get_current_navigation_state" not in tool_map:
            return SkillDecision(
                action=_respond("I cannot modify the current navigation route because navigation state lookup is unavailable."),
                warnings=["navigation edit blocked by unavailable state lookup"],
            )
        if not _tool_called_after_latest_actionable_user(messages, "get_current_navigation_state"):
            return SkillDecision(
                action=_tool_calls(
                    [
                        {
                            "tool_name": "get_current_navigation_state",
                            "arguments": {"detailed_information": True},
                        }
                    ]
                ),
                warnings=["navigation edit starts with current navigation state"],
            )
        return SkillDecision()


class ChargingRouteSkill:
    name = "charging_route"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        text = _latest_actionable_user_text(messages).lower()
        if not _requests_charging(text):
            return SkillDecision()
        tool_map = _tool_map(tools)
        if _has_any(text, ("route", "navigation", "along", "destination")) and "get_current_navigation_state" in tool_map:
            if not _tool_called_after_latest_actionable_user(messages, "get_current_navigation_state"):
                return SkillDecision(
                    action=_tool_calls(
                        [
                            {
                                "tool_name": "get_current_navigation_state",
                                "arguments": {"detailed_information": True},
                            }
                        ]
                    ),
                    warnings=["charging route task starts with navigation state"],
                )
        if "get_charging_specs_and_status" not in tool_map:
            return SkillDecision(
                action=_respond("I cannot answer that charging or range question because charging status information is unavailable."),
                warnings=["charging status unavailable"],
            )
        if not _tool_called_after_latest_actionable_user(messages, "get_charging_specs_and_status"):
            return SkillDecision(
                action=_tool_calls([{"tool_name": "get_charging_specs_and_status", "arguments": {}}]),
                warnings=["charging task starts with charging status"],
            )
        return SkillDecision()


class ClimateACDefrostSkill:
    name = "climate_ac_defrost"

    def preempt(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> SkillDecision:
        text = _latest_actionable_user_text(messages).lower()
        if not _requests_ac_or_defrost(text):
            return SkillDecision()
        tool_map = _tool_map(tools)
        calls: list[dict[str, Any]] = []
        if "get_climate_settings" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_climate_settings"):
            calls.append({"tool_name": "get_climate_settings", "arguments": {}})
        if "get_vehicle_window_positions" in tool_map and not _tool_called_after_latest_actionable_user(messages, "get_vehicle_window_positions"):
            calls.append({"tool_name": "get_vehicle_window_positions", "arguments": {}})
        if calls:
            return SkillDecision(
                action=_tool_calls(calls),
                warnings=["climate AC/defrost task starts with policy precondition checks"],
            )
        if _has_any(text, ("air conditioning", " ac ")) and "set_air_conditioning" not in tool_map:
            return SkillDecision(
                action=_respond("I cannot turn on the air conditioning because that control is unavailable."),
                warnings=["AC control unavailable"],
            )
        return SkillDecision()


def _tool_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"action": "tool_calls", "tool_calls": calls}


def _respond(content: str) -> dict[str, Any]:
    return {"action": "respond", "content": content}


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or tool.get("name") or "")


def _tool_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_tool_name(tool): tool for tool in tools if _tool_name(tool)}


def _same_call_batch(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return [_call_fingerprint(call) for call in left] == [_call_fingerprint(call) for call in right]


def _call_fingerprint(call: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": str(call.get("tool_name") or ""),
            "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _tool_has_parameter(tool_map: dict[str, dict[str, Any]], tool_name: str, parameter: str) -> bool:
    tool = tool_map.get(tool_name)
    if not tool:
        return False
    properties = tool.get("function", {}).get("parameters", {}).get("properties", {})
    return parameter in properties


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _latest_actionable_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or content == "###STOP###" or _is_confirmation_text(content):
            continue
        return content
    return ""


def _all_actionable_user_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or content == "###STOP###" or _is_confirmation_text(content):
            continue
        parts.append(content)
    return "\n".join(parts)


def _previous_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message.get("content") or "")
    return ""


def _tool_called_after_latest_actionable_user(messages: list[dict[str, Any]], tool_name: str) -> bool:
    latest_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content and content != "###STOP###" and not _is_confirmation_text(content):
            latest_user_index = index
    for message in messages[latest_user_index + 1 :]:
        if message.get("role") == "tool" and message.get("name") == tool_name:
            return True
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if function.get("name") == tool_name or call.get("tool_name") == tool_name:
                    return True
    return False


def _tool_called_anywhere(messages: list[dict[str, Any]], tool_name: str) -> bool:
    for message in messages:
        if message.get("role") == "tool" and message.get("name") == tool_name:
            return True
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if function.get("name") == tool_name or call.get("tool_name") == tool_name:
                return True
    return False


def _tool_call_with_arguments_after_latest_user(
    messages: list[dict[str, Any]],
    tool_name: str,
    expected_arguments: dict[str, Any],
) -> bool:
    latest_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content and content != "###STOP###" and not _is_confirmation_text(content):
            latest_user_index = index
    for message in messages[latest_user_index + 1 :]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = str(function.get("name") or call.get("tool_name") or "")
            if name != tool_name:
                continue
            arguments = _parse_arguments(function.get("arguments", call.get("arguments", {})))
            if not isinstance(arguments, dict):
                continue
            if all(arguments.get(key) == value for key, value in expected_arguments.items()):
                return True
    return False


def _latest_contact_email_addresses(messages: list[dict[str, Any]]) -> list[str]:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("name") == "get_contact_information":
            emails = _extract_email_addresses(str(message.get("content") or ""))
            if emails:
                return emails
    return []


def _extract_email_addresses(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)))


def _extract_quoted_or_after_colon(text: str) -> str:
    quoted = re.findall(r"'([^']{8,})'|\"([^\"]{8,})\"", text)
    for left, right in quoted:
        candidate = (left or right).strip()
        if candidate:
            return candidate
    match = re.search(r"(?:message|saying|content)\s*[:\-]\s*(.+)", text, flags=re.I | re.S)
    if match:
        return _sanitize_email_content(match.group(1).strip()[:500])
    return ""


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
    return re.sub(r"\s+", " ", text).strip()


def _fallback_email_content(messages: list[dict[str, Any]]) -> str:
    request = _latest_actionable_user_text(messages)
    return request or "Here is the requested update."


def _is_confirmation_text(content: str) -> bool:
    text = content.strip().lower()
    if not text or _has_any(text, ("no", "don't", "do not", "cancel", "stop")):
        return False
    return bool(
        re.search(
            r"\b(yes|yeah|yep|sure|ok|okay|confirm|confirmed|send it|go ahead|please do|do it)\b",
            text,
        )
    )


def _requests_navigation_edit(text: str) -> bool:
    return _has_any(text, ("navigation", "route", "destination", "waypoint", "stop")) and _has_any(
        text, ("replace", "change", "delete", "remove", "add", "skip", "cancel", "shorten", "reroute")
    )


def _requests_charging(text: str) -> bool:
    return _has_any(text, ("charge", "charging", "battery", "soc", "range", "charger"))


def _requests_ac_or_defrost(text: str) -> bool:
    return _mentions_ac(text) or _has_any(text, ("defrost", "stagnant air", "air flow", "airflow"))


def _requests_window_match_and_defrost(text: str) -> bool:
    return (
        "window" in text
        and _has_any(text, ("match", "same as", "as much as", "adjusted to"))
        and _has_any(text, ("rear passenger", "passenger rear"))
        and _has_any(text, ("defrost", "defog", "fog", "fogging", "windshield"))
    )


def _requests_occupancy_climate_efficiency(text: str) -> bool:
    return (
        _has_any(text, ("climate", "temperature", "seat heating", "heated seat", "heated empty seat"))
        and _has_any(
            text,
            (
                "energy",
                "efficient",
                "optimize",
                "wasting",
                "wasteful",
                "empty seat",
                "empty seats",
                "heated empty seat",
                "heated empty seats",
                "unoccupied",
                "who's actually",
                "who is actually",
                "actually in the car",
                "passenger side",
                "passenger temperature",
            ),
        )
    )


def _requests_occupancy_based_reading_lights(text: str) -> bool:
    return "reading light" in text and _has_any(
        text,
        (
            "occupied",
            "occupancy",
            "who's actually",
            "who is actually",
            "empty seat",
            "empty seats",
            "waste energy",
            "based on who",
            "based on who's",
            "actually in the car",
        ),
    )


def _mentions_ac(text: str) -> bool:
    return "air conditioning" in text or bool(re.search(r"\bac\b", text))


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


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

    return re.sub(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([AaPp][Mm])\b", replace, content)


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    return _parse_json(value) or {}


def _latest_tool_json(messages: list[dict[str, Any]], tool_name: str) -> Any:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("name") == tool_name:
            return _parse_json(str(message.get("content") or ""))
    return None


def _latest_seat_occupancy(messages: list[dict[str, Any]]) -> dict[str, bool]:
    data = _latest_tool_json(messages, "get_seats_occupancy")
    seats = _find_key(data, "seats_occupied")
    if isinstance(seats, str) and seats.lower() == "unknown":
        return {}
    if not isinstance(seats, dict):
        return {}
    normalized: dict[str, bool] = {}
    for key, value in seats.items():
        if isinstance(value, bool):
            normalized[_normalize_seat_key(str(key))] = value
    return {
        key: value
        for key, value in normalized.items()
        if key in {"driver", "passenger", "driver_rear", "passenger_rear"}
    }


def _latest_reading_lights_status(messages: list[dict[str, Any]]) -> dict[str, bool]:
    data = _latest_tool_json(messages, "get_reading_lights_status")
    if data is None:
        return {}
    result = _find_key(data, "result")
    source = result if isinstance(result, dict) else data
    status: dict[str, bool] = {}
    for key, position in {
        "reading_light_driver": "DRIVER",
        "driver": "DRIVER",
        "reading_light_passenger": "PASSENGER",
        "passenger": "PASSENGER",
        "reading_light_driver_rear": "DRIVER_REAR",
        "driver_rear": "DRIVER_REAR",
        "reading_light_passenger_rear": "PASSENGER_REAR",
        "passenger_rear": "PASSENGER_REAR",
    }.items():
        value = _find_key(source, key)
        if isinstance(value, bool):
            status[position] = value
    return status


def _latest_window_positions(messages: list[dict[str, Any]]) -> dict[str, int]:
    data = _latest_tool_json(messages, "get_vehicle_window_positions")
    mapping = {
        "window_driver_position": "DRIVER",
        "window_passenger_position": "PASSENGER",
        "window_driver_rear_position": "DRIVER_REAR",
        "window_passenger_rear_position": "PASSENGER_REAR",
    }
    positions: dict[str, int] = {}
    for key, window in mapping.items():
        number = _number_from_value(_find_key(data, key))
        if number is not None:
            positions[window] = max(0, min(100, int(number)))
    return positions


def _latest_climate_settings(messages: list[dict[str, Any]]) -> Any:
    data = _latest_tool_json(messages, "get_climate_settings")
    result = _find_key(data, "result")
    return result if isinstance(result, dict) else data


def _latest_seat_heating_levels(messages: list[dict[str, Any]]) -> dict[str, int]:
    data = _latest_tool_json(messages, "get_seat_heating_level")
    mapping = {
        "seat_heating_driver": "DRIVER",
        "driver": "DRIVER",
        "seat_heating_passenger": "PASSENGER",
        "passenger": "PASSENGER",
        "seat_heating_driver_rear": "DRIVER_REAR",
        "driver_rear": "DRIVER_REAR",
        "seat_heating_passenger_rear": "PASSENGER_REAR",
        "passenger_rear": "PASSENGER_REAR",
    }
    levels: dict[str, int] = {}
    for key, zone in mapping.items():
        number = _number_from_value(_find_key(data, key))
        if number is not None:
            levels[zone] = int(number)
    return levels


def _latest_climate_temperatures(messages: list[dict[str, Any]]) -> dict[str, float]:
    data = _latest_tool_json(messages, "get_temperature_inside_car")
    mapping = {
        "climate_temperature_driver": "DRIVER",
        "temperature_driver": "DRIVER",
        "driver": "DRIVER",
        "climate_temperature_passenger": "PASSENGER",
        "temperature_passenger": "PASSENGER",
        "passenger": "PASSENGER",
    }
    temperatures: dict[str, float] = {}
    for key, zone in mapping.items():
        number = _number_from_value(_find_key(data, key))
        if number is not None:
            temperatures[zone] = int(number) if float(number).is_integer() else number
    return temperatures


def _requested_reference_window(text: str) -> str | None:
    if "passenger rear" in text or "rear passenger" in text:
        return "PASSENGER_REAR"
    if "driver rear" in text or "rear driver" in text:
        return "DRIVER_REAR"
    if "passenger" in text:
        return "PASSENGER"
    if "driver" in text:
        return "DRIVER"
    return None


def _requested_defrost_window(text: str) -> str | None:
    if _has_any(text, ("front", "windshield", "windscreen")):
        return "FRONT"
    if _has_any(text, ("rear", "back window")):
        return "REAR"
    if "all" in text:
        return "ALL"
    if _has_any(text, ("fog", "fogging", "defrost", "defog")):
        return "FRONT"
    return None


def _seat_heating_calls_for_empty_heated_seats(
    occupancy: dict[str, bool],
    heating: dict[str, int],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for seat_key, zone in (
        ("driver", "DRIVER"),
        ("passenger", "PASSENGER"),
        ("driver_rear", "DRIVER_REAR"),
        ("passenger_rear", "PASSENGER_REAR"),
    ):
        if occupancy.get(seat_key) is False and heating.get(zone, 0) > 0:
            arguments = {"seat_zone": zone, "level": 0}
            if _tool_call_with_arguments_after_latest_user(messages, "set_seat_heating", arguments):
                continue
            calls.append(
                {
                    "tool_name": "set_seat_heating",
                    "arguments": arguments,
                }
            )
    return calls


def _reading_light_calls_for_occupancy(
    occupancy: dict[str, bool],
    reading_lights: dict[str, bool],
) -> list[dict[str, Any]]:
    turn_on_calls: list[dict[str, Any]] = []
    turn_off_calls: list[dict[str, Any]] = []
    for seat, position in (
        ("driver", "DRIVER"),
        ("passenger", "PASSENGER"),
        ("driver_rear", "DRIVER_REAR"),
        ("passenger_rear", "PASSENGER_REAR"),
    ):
        if seat not in occupancy:
            continue
        desired = occupancy[seat]
        if reading_lights and reading_lights.get(position) is desired:
            continue
        call = {
            "tool_name": "set_reading_light",
            "arguments": {"position": position, "on": desired},
        }
        if desired:
            turn_on_calls.append(call)
        else:
            turn_off_calls.append(call)
    return turn_on_calls + turn_off_calls


def _normalize_seat_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


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


def _number_from_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _bool_from_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "on", "yes"}:
            return True
        if lowered in {"false", "off", "no"}:
            return False
    return None


def _string_from_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
