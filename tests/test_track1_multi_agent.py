import unittest
from types import SimpleNamespace

from google.protobuf.json_format import MessageToDict

from track_1_agent_under_test.car_bench_agent import CARBenchAgentExecutor
from track_1_agent_under_test.approved_plan import (
    ApprovedPlan,
    ApprovedStep,
    normalize_approved_plan,
    normalize_critic_verdict,
)
from track_1_agent_under_test.context_manager import (
    ContextManager,
    SINGLE_TOOL_RESULT_CHAR_LIMIT,
)
from track_1_agent_under_test.guards import PolicyGuard, SchemaGuard
from track_1_agent_under_test.multi_agent_types import LLMCallMetrics
from track_1_agent_under_test.plan_state import PlanStateStore
from track_1_agent_under_test.planner import Track1Planner
from track_1_agent_under_test.skills import SkillRegistry
from track_1_agent_under_test.task_memory import TaskMemoryStore
from track_1_agent_under_test.task_guard import TaskGuard
from track_1_agent_under_test.training_insights import default_training_insights
from evaluator.car_bench_evaluator import (
    build_pass_summary,
    build_task_pass3_summary,
    format_progress_update,
    format_pass_summary,
    format_task_pass3_summary,
)


def _tool(name, description="", parameters=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
            or {
                "type": "object",
                "required": [],
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _number_tool(name):
    return _tool(
        name,
        "Vehicle Control",
        {
            "type": "object",
            "required": ["percentage"],
            "properties": {
                "percentage": {"type": "number", "minimum": 0, "maximum": 100}
            },
            "additionalProperties": False,
        },
    )


def _weather_tool():
    return _tool(
        "get_weather",
        "Weather Information",
        {
            "type": "object",
            "required": [
                "location_or_poi_id",
                "month",
                "day",
                "time_hour_24hformat",
            ],
            "properties": {
                "location_or_poi_id": {"type": "string"},
                "month": {"type": "number"},
                "day": {"type": "number"},
                "time_hour_24hformat": {"type": "number"},
                "time_minutes": {"type": "number"},
            },
            "additionalProperties": False,
        },
    )


def _bool_tool(name):
    return _tool(
        name,
        "Vehicle Control",
        {
            "type": "object",
            "required": ["on"],
            "properties": {"on": {"type": "boolean"}},
            "additionalProperties": False,
        },
    )


def _window_tool():
    return _tool(
        "open_close_window",
        "Vehicle Control",
        {
            "type": "object",
            "required": ["window", "percentage"],
            "properties": {
                "window": {
                    "type": "string",
                    "enum": [
                        "ALL",
                        "DRIVER",
                        "PASSENGER",
                        "DRIVER_REAR",
                        "PASSENGER_REAR",
                    ],
                },
                "percentage": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "additionalProperties": False,
        },
    )


def _fan_speed_tool():
    return _tool(
        "set_fan_speed",
        "Vehicle Climate Control",
        {
            "type": "object",
            "required": ["level"],
            "properties": {"level": {"type": "number", "minimum": 0, "maximum": 5}},
            "additionalProperties": False,
        },
    )


def _airflow_tool():
    return _tool(
        "set_fan_airflow_direction",
        "Vehicle Climate Control",
        {
            "type": "object",
            "required": ["direction"],
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": [
                        "FEET",
                        "HEAD",
                        "HEAD_FEET",
                        "WINDSHIELD",
                        "WINDSHIELD_FEET",
                        "WINDSHIELD_HEAD",
                        "WINDSHIELD_HEAD_FEET",
                    ],
                }
            },
            "additionalProperties": False,
        },
    )


def _defrost_tool():
    return _tool(
        "set_window_defrost",
        "Vehicle Climate Control",
        {
            "type": "object",
            "required": ["on", "defrost_window"],
            "properties": {
                "on": {"type": "boolean"},
                "defrost_window": {"type": "string", "enum": ["ALL", "FRONT", "REAR"]},
            },
            "additionalProperties": False,
        },
    )


def _climate_status_tool():
    return _tool("get_climate_settings")


def _temperature_status_tool():
    return _tool("get_temperature_inside_car")


def _set_climate_temperature_tool():
    return _tool(
        "set_climate_temperature",
        "Vehicle Climate Control",
        {
            "type": "object",
            "required": ["temperature", "seat_zone"],
            "properties": {
                "temperature": {"type": "number", "minimum": 16, "maximum": 28},
                "seat_zone": {
                    "type": "string",
                    "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
                },
            },
            "additionalProperties": False,
        },
    )


def _window_status_tool():
    return _tool("get_vehicle_window_positions")


def _exterior_lights_tool():
    return _tool("get_exterior_lights_status")


def _navigation_state_tool():
    return _tool(
        "get_current_navigation_state",
        "Navigation State Information",
        {
            "type": "object",
            "required": [],
            "properties": {"detailed_information": {"type": "boolean"}},
            "additionalProperties": False,
        },
    )


def _set_new_navigation_tool():
    return _tool(
        "set_new_navigation",
        "Navigation Control",
        {
            "type": "object",
            "required": ["route_ids"],
            "properties": {
                "route_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "additionalProperties": False,
        },
    )


def _navigation_replace_final_destination_tool():
    return _tool(
        "navigation_replace_final_destination",
        "Navigation Control",
        {
            "type": "object",
            "required": [
                "new_destination_id",
                "route_id_leading_to_new_destination",
            ],
            "properties": {
                "new_destination_id": {"type": "string"},
                "route_id_leading_to_new_destination": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def _navigation_delete_destination_tool():
    return _tool(
        "navigation_delete_destination",
        "Navigation Control",
        {
            "type": "object",
            "required": ["destination_id_to_delete"],
            "properties": {"destination_id_to_delete": {"type": "string"}},
            "additionalProperties": False,
        },
    )


def _routes_tool():
    return _tool(
        "get_routes_from_start_to_destination",
        "Routes information",
        {
            "type": "object",
            "required": ["start_id", "destination_id"],
            "properties": {
                "start_id": {"type": "string"},
                "destination_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def _location_lookup_tool():
    return _tool(
        "get_location_id_by_location_name",
        "Location lookup",
        {
            "type": "object",
            "required": ["location"],
            "properties": {"location": {"type": "string"}},
            "additionalProperties": False,
        },
    )


def _poi_at_location_tool():
    return _tool(
        "search_poi_at_location",
        "Points of Interest Search",
        {
            "type": "object",
            "required": ["location_id", "category_poi"],
            "properties": {
                "location_id": {"type": "string"},
                "category_poi": {
                    "type": "string",
                    "enum": ["charging_stations", "restaurants"],
                },
            },
            "additionalProperties": False,
        },
    )


def _charging_status_tool():
    return _tool("get_charging_specs_and_status")


def _distance_by_soc_tool():
    return _tool(
        "get_distance_by_soc",
        "Charging Information",
        {
            "type": "object",
            "required": ["initial_state_of_charge"],
            "properties": {
                "initial_state_of_charge": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
                "final_state_of_charge": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "additionalProperties": False,
        },
    )


def _send_email_tool():
    return _tool(
        "send_email",
        "REQUIRES_CONFIRMATION Email Tool",
        {
            "type": "object",
            "required": ["email_addresses", "content_message"],
            "properties": {
                "email_addresses": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "content_message": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def _poi_along_route_tool(include_filters=False):
    properties = {
        "route_id": {"type": "string"},
        "category_poi": {
            "type": "string",
            "enum": ["charging_stations", "restaurants"],
        },
        "at_kilometer": {"type": "integer"},
    }
    if include_filters:
        properties["filters"] = {"type": "array", "items": {"type": "string"}}
    return _tool(
        "search_poi_along_the_route",
        "Points of Interest Search",
        {
            "type": "object",
            "required": ["route_id", "category_poi"],
            "properties": properties,
            "additionalProperties": False,
        },
    )


def _seats_occupancy_tool():
    return _tool("get_seats_occupancy")


def _reading_light_tool():
    return _tool(
        "set_reading_light",
        "Vehicle Control",
        {
            "type": "object",
            "required": ["position", "on"],
            "properties": {
                "position": {
                    "type": "string",
                    "enum": ["DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR", "ALL"],
                },
                "on": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    )


def _reading_lights_status_tool():
    return _tool("get_reading_lights_status")


def _seat_heating_status_tool():
    return _tool("get_seat_heating_level")


def _seat_heating_tool():
    return _tool(
        "set_seat_heating",
        "Vehicle Climate Control",
        {
            "type": "object",
            "required": ["seat_zone", "level"],
            "properties": {
                "seat_zone": {
                    "type": "string",
                    "enum": ["ALL_ZONES", "DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR"],
                },
                "level": {"type": "number", "minimum": 0, "maximum": 3},
            },
            "additionalProperties": False,
        },
    )


class Track1MultiAgentTest(unittest.TestCase):
    def test_skill_registry_sends_email_after_confirmation(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {"role": "user", "content": "Email Frank that I am running late."},
                {
                    "role": "assistant",
                    "content": (
                        "I can send an email to frank@example.com saying: "
                        "'I am running late. I apologize.' Please say yes to confirm."
                    ),
                },
                {"role": "user", "content": "Yes, send it."},
            ],
            tools=[_send_email_tool()],
        )

        self.assertEqual(decision.skill, "communication_email")
        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(decision.action["tool_calls"][0]["tool_name"], "send_email")

    def test_skill_registry_blocks_missing_email_tool(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[{"role": "user", "content": "Please send an email to Frank."}],
            tools=[],
        )

        self.assertEqual(decision.skill, "hallucination_guard")
        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("cannot send email", decision.action["content"].lower())

    def test_skill_registry_navigation_edit_starts_with_state(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": "Replace my current navigation destination with a restaurant.",
                }
            ],
            tools=[_navigation_state_tool(), _navigation_replace_final_destination_tool()],
        )

        self.assertEqual(decision.skill, "navigation_route_edit")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_current_navigation_state",
        )

    def test_skill_registry_charging_route_starts_with_navigation_state(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": "Find a charging station along my current route.",
                }
            ],
            tools=[_navigation_state_tool(), _charging_status_tool()],
        )

        self.assertEqual(decision.skill, "charging_route")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_current_navigation_state",
        )

    def test_skill_registry_ac_starts_with_policy_checks(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[{"role": "user", "content": "Turn on the AC."}],
            tools=[
                _climate_status_tool(),
                _window_status_tool(),
                _bool_tool("set_air_conditioning"),
            ],
        )

        self.assertEqual(decision.skill, "climate_ac_defrost")
        self.assertEqual(
            [call["tool_name"] for call in decision.action["tool_calls"]],
            ["get_climate_settings", "get_vehicle_window_positions"],
        )

    def test_skill_registry_reading_lights_starts_with_seat_occupancy(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": "Adjust the reading lights based on who's actually in the car and turn off empty seats.",
                }
            ],
            tools=[_seats_occupancy_tool(), _reading_light_tool()],
        )

        self.assertEqual(decision.skill, "reading_light_occupancy")
        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_seats_occupancy",
        )

    def test_skill_registry_reading_lights_checks_current_light_status_when_available(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": "Adjust the reading lights based on who's actually in the car and turn off empty seats.",
                }
            ],
            tools=[
                _seats_occupancy_tool(),
                _reading_lights_status_tool(),
                _reading_light_tool(),
            ],
        )

        self.assertEqual(decision.skill, "reading_light_occupancy")
        self.assertEqual(
            [call["tool_name"] for call in decision.action["tool_calls"]],
            ["get_seats_occupancy", "get_reading_lights_status"],
        )

    def test_skill_registry_reading_lights_only_changes_mismatched_seats(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": "Adjust the reading lights based on who's actually in the car and turn off empty seats.",
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_seats",
                            "function": {"name": "get_seats_occupancy", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_seats_occupancy",
                    "tool_call_id": "call_seats",
                    "content": (
                        '{"status":"SUCCESS","result":{"seats_occupied":'
                        '{"driver":true,"passenger":false,"driver_rear":false,'
                        '"passenger_rear":true}}}'
                    ),
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_lights",
                            "function": {"name": "get_reading_lights_status", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_reading_lights_status",
                    "tool_call_id": "call_lights",
                    "content": (
                        '{"status":"SUCCESS","result":{"reading_light_driver":true,'
                        '"reading_light_passenger":true,"reading_light_driver_rear":false,'
                        '"reading_light_passenger_rear":false}}'
                    ),
                },
            ],
            tools=[
                _seats_occupancy_tool(),
                _reading_lights_status_tool(),
                _reading_light_tool(),
            ],
        )

        self.assertEqual(decision.skill, "reading_light_occupancy")
        self.assertEqual(
            decision.action["tool_calls"],
            [
                {
                    "tool_name": "set_reading_light",
                    "arguments": {"position": "PASSENGER_REAR", "on": True},
                },
                {
                    "tool_name": "set_reading_light",
                    "arguments": {"position": "PASSENGER", "on": False},
                },
            ],
        )

    def test_skill_registry_window_match_defrost_gathers_window_and_climate_state(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Adjust ALL windows to match the rear passenger window position, "
                        "then activate the FRONT window defrost."
                    ),
                }
            ],
            tools=[
                _window_status_tool(),
                _climate_status_tool(),
                _window_tool(),
                _defrost_tool(),
                _fan_speed_tool(),
                _airflow_tool(),
                _bool_tool("set_air_conditioning"),
            ],
        )

        self.assertEqual(decision.skill, "window_match_defrost")
        self.assertEqual(
            [call["tool_name"] for call in decision.action["tool_calls"]],
            ["get_vehicle_window_positions", "get_climate_settings"],
        )

    def test_skill_registry_window_match_defrost_uses_reference_window_position(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Please make all windows match the rear passenger window position "
                        "and activate front windshield defrost."
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_vehicle_window_positions",
                    "content": (
                        '{"status":"SUCCESS","result":{"window_driver_position":25,'
                        '"window_passenger_position":50,"window_driver_rear_position":25,'
                        '"window_passenger_rear_position":5}}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_climate_settings",
                    "content": (
                        '{"status":"SUCCESS","result":{"fan_speed":0,'
                        '"fan_airflow_direction":"FEET","air_conditioning":false}}'
                    ),
                },
            ],
            tools=[
                _window_status_tool(),
                _climate_status_tool(),
                _window_tool(),
                _defrost_tool(),
                _fan_speed_tool(),
                _airflow_tool(),
                _bool_tool("set_air_conditioning"),
            ],
        )

        self.assertEqual(decision.skill, "window_match_defrost")
        self.assertEqual(
            decision.action["tool_calls"],
            [
                {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 5}},
                {
                    "tool_name": "set_window_defrost",
                    "arguments": {"defrost_window": "FRONT", "on": True},
                },
                {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
                {"tool_name": "set_fan_airflow_direction", "arguments": {"direction": "WINDSHIELD"}},
                {"tool_name": "set_air_conditioning", "arguments": {"on": True}},
            ],
        )

    def test_skill_registry_occupancy_climate_turns_off_empty_heating_and_matches_temperature(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Optimize the climate for energy efficiency based on who's actually "
                        "in the car. Turn off heated empty seats and match the driver "
                        "temperature to the passenger temperature."
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_seats_occupancy",
                    "content": (
                        '{"status":"SUCCESS","result":{"seats_occupied":'
                        '{"driver":true,"passenger":false,"driver_rear":false,'
                        '"passenger_rear":false}}}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_temperature_inside_car",
                    "content": (
                        '{"status":"SUCCESS","result":{"climate_temperature_driver":26,'
                        '"climate_temperature_passenger":23}}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_seat_heating_level",
                    "content": (
                        '{"status":"SUCCESS","result":{"seat_heating_driver":2,'
                        '"seat_heating_passenger":3}}'
                    ),
                },
            ],
            tools=[
                _seats_occupancy_tool(),
                _temperature_status_tool(),
                _seat_heating_status_tool(),
                _seat_heating_tool(),
                _set_climate_temperature_tool(),
            ],
        )

        self.assertEqual(decision.skill, "occupancy_climate_efficiency")
        self.assertEqual(
            decision.action["tool_calls"],
            [
                {"tool_name": "set_seat_heating", "arguments": {"seat_zone": "PASSENGER", "level": 0}},
                {
                    "tool_name": "set_climate_temperature",
                    "arguments": {"temperature": 23, "seat_zone": "DRIVER"},
                },
            ],
        )

    def test_skill_registry_occupancy_climate_uses_multi_turn_user_intent(self):
        skills = SkillRegistry()
        decision = skills.preempt(
            messages=[
                {
                    "role": "user",
                    "content": "Check the climate for energy efficiency based on who's actually in the car.",
                },
                {
                    "role": "tool",
                    "name": "get_seats_occupancy",
                    "content": '{"result":{"seats_occupied":{"driver":true,"passenger":false}}}',
                },
                {
                    "role": "tool",
                    "name": "get_temperature_inside_car",
                    "content": '{"result":{"climate_temperature_driver":26,"climate_temperature_passenger":23}}',
                },
                {
                    "role": "tool",
                    "name": "get_seat_heating_level",
                    "content": '{"result":{"seat_heating_driver":2,"seat_heating_passenger":3}}',
                },
                {"role": "user", "content": "Turn off the empty heated seat and match my temperature to the passenger side."},
            ],
            tools=[
                _seats_occupancy_tool(),
                _temperature_status_tool(),
                _seat_heating_status_tool(),
                _seat_heating_tool(),
                _set_climate_temperature_tool(),
            ],
        )

        self.assertEqual(decision.skill, "occupancy_climate_efficiency")
        self.assertEqual(
            decision.action["tool_calls"],
            [
                {"tool_name": "set_seat_heating", "arguments": {"seat_zone": "PASSENGER", "level": 0}},
                {
                    "tool_name": "set_climate_temperature",
                    "arguments": {"temperature": 23, "seat_zone": "DRIVER"},
                },
            ],
        )

    def test_planner_skills_run_before_successful_state_change_done_gate(self):
        planner = Track1Planner(model="test-model")
        messages = [
            {
                "role": "user",
                "content": (
                    "Optimize the climate for energy efficiency based on who's actually "
                    "in the car. Turn off heated empty seats and match the driver "
                    "temperature to the passenger temperature."
                ),
            },
            {"role": "tool", "name": "get_seats_occupancy", "content": '{"result":{"seats_occupied":{"driver":true,"passenger":false}}}'},
            {"role": "tool", "name": "get_temperature_inside_car", "content": '{"result":{"climate_temperature_driver":26,"climate_temperature_passenger":23}}'},
            {"role": "tool", "name": "get_seat_heating_level", "content": '{"result":{"seat_heating_driver":2,"seat_heating_passenger":3}}'},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_heat",
                        "function": {
                            "name": "set_seat_heating",
                            "arguments": '{"seat_zone":"PASSENGER","level":0}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_heat",
                "name": "set_seat_heating",
                "content": '{"status":"SUCCESS","result":{}}',
            },
        ]
        result = planner.choose_next_action(
            context_id="ctx-climate-skill-before-done",
            messages=messages,
            tools=[
                _seats_occupancy_tool(),
                _temperature_status_tool(),
                _seat_heating_status_tool(),
                _seat_heating_tool(),
                _set_climate_temperature_tool(),
            ],
            ctx_logger=SimpleNamespace(debug=lambda **kwargs: None, warning=lambda **kwargs: None, info=lambda **kwargs: None),
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        self.assertEqual(
            result.next_action["tool_calls"][0],
            {
                "tool_name": "set_climate_temperature",
                "arguments": {"temperature": 23, "seat_zone": "DRIVER"},
            },
        )
        self.assertEqual(result.debug["skill"], "occupancy_climate_efficiency")

    def test_planner_successful_state_change_done_gate_runs_completion_verifier(self):
        planner = Track1Planner(model="test-model")

        class NoopSkills:
            def preempt(self, *, messages, tools):
                return SimpleNamespace(action=None, skill=None, warnings=[])

        planner.skill_registry = NoopSkills()
        messages = [
            {
                "role": "user",
                "content": (
                    "Optimize the climate for energy efficiency based on who's actually "
                    "in the car. Turn off heated empty seats and match the driver "
                    "temperature to the passenger temperature."
                ),
            },
            {"role": "tool", "name": "get_seats_occupancy", "content": '{"result":{"seats_occupied":{"driver":true,"passenger":false}}}'},
            {"role": "tool", "name": "get_temperature_inside_car", "content": '{"result":{"climate_temperature_driver":26,"climate_temperature_passenger":23}}'},
            {"role": "tool", "name": "get_seat_heating_level", "content": '{"result":{"seat_heating_driver":2,"seat_heating_passenger":3}}'},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_heat",
                        "function": {
                            "name": "set_seat_heating",
                            "arguments": '{"seat_zone":"PASSENGER","level":0}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_heat",
                "name": "set_seat_heating",
                "content": '{"status":"SUCCESS","result":{}}',
            },
        ]

        result = planner.choose_next_action(
            context_id="ctx-finish-gate-verifier",
            messages=messages,
            tools=[
                _seats_occupancy_tool(),
                _temperature_status_tool(),
                _seat_heating_status_tool(),
                _seat_heating_tool(),
                _set_climate_temperature_tool(),
            ],
            ctx_logger=SimpleNamespace(debug=lambda **kwargs: None, warning=lambda **kwargs: None, info=lambda **kwargs: None),
        )

        self.assertTrue(result.debug["terminal_after_state_change"])
        self.assertEqual(result.next_action["action"], "tool_calls")
        self.assertEqual(
            result.next_action["tool_calls"],
            [
                {
                    "tool_name": "set_climate_temperature",
                    "arguments": {"temperature": 23, "seat_zone": "DRIVER"},
                }
            ],
        )
        self.assertTrue(result.debug["completion_verifier_warnings"])

    def test_planner_response_gate_blocks_premature_done_when_skill_has_tool_step(self):
        planner = Track1Planner(model="test-model")
        context_id = "ctx-response-gate-climate"
        messages = [
            {
                "role": "user",
                "content": (
                    "Optimize the climate for energy efficiency based on who's actually "
                    "in the car. Turn off heated empty seats and match the driver "
                    "temperature to the passenger temperature."
                ),
            },
            {
                "role": "tool",
                "name": "get_seats_occupancy",
                "content": '{"result":{"seats_occupied":{"driver":true,"passenger":false}}}',
            },
            {
                "role": "tool",
                "name": "get_temperature_inside_car",
                "content": '{"result":{"climate_temperature_driver":26,"climate_temperature_passenger":23}}',
            },
            {
                "role": "tool",
                "name": "get_seat_heating_level",
                "content": '{"result":{"seat_heating_driver":2,"seat_heating_passenger":3}}',
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_heat",
                        "function": {
                            "name": "set_seat_heating",
                            "arguments": '{"seat_zone":"PASSENGER","level":0}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_heat",
                "name": "set_seat_heating",
                "content": '{"status":"SUCCESS","result":{}}',
            },
        ]
        tools = [
            _seats_occupancy_tool(),
            _temperature_status_tool(),
            _seat_heating_status_tool(),
            _seat_heating_tool(),
            _set_climate_temperature_tool(),
        ]
        planner.context_manager.observe_messages(context_id, messages)
        planner.task_memory.observe_messages(context_id, messages)
        planner.plan_state.observe_messages(context_id, messages, tools)

        result = planner._finalize_visible_action(
            context_id=context_id,
            action={"action": "respond", "content": "Done."},
            tools=tools,
            messages=messages,
            metrics=SimpleNamespace(num_calls=0),
            debug={},
            internal_calls_floor=0,
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        self.assertEqual(
            result.next_action["tool_calls"],
            [
                {
                    "tool_name": "set_climate_temperature",
                    "arguments": {"temperature": 23, "seat_zone": "DRIVER"},
                }
            ],
        )
        self.assertTrue(result.debug["skill_response_warnings"])

    def test_planner_tool_gate_replaces_partial_temperature_action_with_complete_skill_batch(self):
        planner = Track1Planner(model="test-model")
        context_id = "ctx-tool-gate-climate"
        messages = [
            {
                "role": "user",
                "content": "Check the climate for energy efficiency based on who's actually in the car.",
            },
            {
                "role": "tool",
                "name": "get_seats_occupancy",
                "content": '{"result":{"seats_occupied":{"driver":true,"passenger":false}}}',
            },
            {
                "role": "tool",
                "name": "get_temperature_inside_car",
                "content": '{"result":{"climate_temperature_driver":26,"climate_temperature_passenger":23}}',
            },
            {
                "role": "tool",
                "name": "get_seat_heating_level",
                "content": '{"result":{"seat_heating_driver":2,"seat_heating_passenger":3}}',
            },
            {"role": "user", "content": "Match my temperature to the passenger side."},
        ]
        tools = [
            _seats_occupancy_tool(),
            _temperature_status_tool(),
            _seat_heating_status_tool(),
            _seat_heating_tool(),
            _set_climate_temperature_tool(),
        ]
        planner.context_manager.observe_messages(context_id, messages)
        planner.task_memory.observe_messages(context_id, messages)
        planner.plan_state.observe_messages(context_id, messages, tools)

        result = planner._finalize_visible_action(
            context_id=context_id,
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "set_climate_temperature",
                        "arguments": {"temperature": 23, "seat_zone": "DRIVER"},
                    }
                ],
            },
            tools=tools,
            messages=messages,
            metrics=SimpleNamespace(num_calls=0),
            debug={},
            internal_calls_floor=0,
        )

        self.assertEqual(
            result.next_action["tool_calls"],
            [
                {"tool_name": "set_seat_heating", "arguments": {"seat_zone": "PASSENGER", "level": 0}},
                {
                    "tool_name": "set_climate_temperature",
                    "arguments": {"temperature": 23, "seat_zone": "DRIVER"},
                },
            ],
        )
        self.assertTrue(result.debug["skill_tool_warnings"])

    def test_failure_guard_blocks_repeated_policy_failed_navigation_delete(self):
        planner = Track1Planner(model="test-model")
        context_id = "ctx-failure-nav-delete"
        messages = [
            {
                "role": "user",
                "content": "Cancel navigation to Paris and set destination to a restaurant in Barcelona.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"status":"SUCCESS","result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"],'
                    '"routes_to_final_destination_id":["rll_mad_par_912360"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"status":"SUCCESS","result":{"pois_found":[{"id":"poi_res_319074",'
                    '"name":"Restaurante El Toro"},{"id":"poi_res_853877",'
                    '"name":"El Rincon de Tapas"}]}}'
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_bad_delete",
                        "function": {
                            "name": "navigation_delete_destination",
                            "arguments": '{"destination_id_to_delete":"loc_par_405686"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_bad_delete",
                "name": "navigation_delete_destination",
                "content": (
                    '{"status":"FAILURE","errors":{"NAVIGATION_DELETE_DESTINATION_003":'
                    '"AUT-POL:019: No intermediate waypoints, destination deletion would lead to full deletion of navigation."}}'
                ),
            },
        ]
        tools = [
            _navigation_state_tool(),
            _poi_at_location_tool(),
            _routes_tool(),
            _navigation_delete_destination_tool(),
            _navigation_replace_final_destination_tool(),
        ]
        planner.context_manager.observe_messages(context_id, messages)
        planner.task_memory.observe_messages(context_id, messages)
        planner.plan_state.observe_messages(context_id, messages, tools)

        result = planner._finalize_visible_action(
            context_id=context_id,
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_delete_destination",
                        "arguments": {"destination_id_to_delete": "loc_par_405686"},
                    }
                ],
            },
            tools=tools,
            messages=messages,
            metrics=SimpleNamespace(num_calls=0),
            debug={},
            internal_calls_floor=0,
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        call = result.next_action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "get_routes_from_start_to_destination")
        self.assertEqual(
            call["arguments"],
            {"start_id": "loc_mad_180891", "destination_id": "poi_res_853877"},
        )
        self.assertTrue(result.debug["failure_guard_warnings"])
        self.assertEqual(
            result.debug["failure_guard_evidence"]["failure_type"],
            "POLICY_FORBIDDEN",
        )

    def test_failure_guard_defaults_to_fastest_route_not_last_route(self):
        failure_guard = __import__(
            "track_1_agent_under_test.failure_guard",
            fromlist=["_selected_route_to_destination"],
        )
        messages = [
            {
                "role": "user",
                "content": "Change my destination to Barcelona and find a good restaurant there using the fastest route.",
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"El Rincon de Tapas"}]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":['
                    '{"route_id":"rlp_mad_res_720938",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"L169, L468",'
                    '"duration_hours":7,"duration_minutes":36,'
                    '"alias":["fastest","first","shortest"]},'
                    '{"route_id":"rlp_mad_res_588035",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"A53, A85, B884",'
                    '"duration_hours":7,"duration_minutes":41,'
                    '"alias":["second"]},'
                    '{"route_id":"rlp_mad_res_376587",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"A19",'
                    '"duration_hours":7,"duration_minutes":44,'
                    '"alias":["third"]}'
                    ']}}'
                ),
            },
        ]

        route_id = failure_guard._selected_route_to_destination(
            messages,
            "poi_res_853877",
        )

        self.assertEqual(route_id, "rlp_mad_res_720938")

    def test_failure_guard_blocks_navigation_mutation_during_charging_search(self):
        planner = Track1Planner(model="test-model")
        context_id = "ctx-failure-charging-search"
        messages = [
            {
                "role": "user",
                "content": "Help me check current range and route information.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"status":"SUCCESS","result":{"navigation_active":true,'
                    '"waypoints_id":["loc_ham_166665","loc_fra_178468","loc_col_464166"],'
                    '"routes_to_final_destination_id":["rll_ham_fra_842845","rll_fra_col_988133"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_charging_specs_and_status",
                "content": '{"status":"SUCCESS","result":{"state_of_charge":65,"remaining_range":"494km"}}',
            },
            {
                "role": "tool",
                "name": "get_distance_by_soc",
                "content": '{"status":"SUCCESS","result":{"distance_km":342}}',
            },
            {
                "role": "user",
                "content": (
                    "Find DC fast charging stations with available plugs along my route "
                    "from Hamburg to Frankfurt around 250km into the journey."
                ),
            },
        ]
        tools = [
            _navigation_state_tool(),
            _charging_status_tool(),
            _distance_by_soc_tool(),
            _poi_along_route_tool(include_filters=True),
            _navigation_replace_final_destination_tool(),
        ]
        planner.context_manager.observe_messages(context_id, messages)
        planner.task_memory.observe_messages(context_id, messages)
        planner.plan_state.observe_messages(context_id, messages, tools)

        result = planner._finalize_visible_action(
            context_id=context_id,
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_cha_512821",
                            "route_id_leading_to_new_destination": "rlp_ham_cha_283702",
                        },
                    }
                ],
            },
            tools=tools,
            messages=messages,
            metrics=SimpleNamespace(num_calls=0),
            debug={},
            internal_calls_floor=0,
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        call = result.next_action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "search_poi_along_the_route")
        self.assertEqual(call["arguments"]["route_id"], "rll_ham_fra_842845")
        self.assertEqual(call["arguments"]["at_kilometer"], 250)
        self.assertIn("charging_stations::has_dc_plug", call["arguments"]["filters"])
        self.assertIn("charging_stations::has_available_plug", call["arguments"]["filters"])
        self.assertTrue(result.debug["failure_guard_warnings"])

    def test_task_guard_defrost_match_reference_window_preserves_reference_percentage(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 5}},
                    {
                        "tool_name": "set_window_defrost",
                        "arguments": {"defrost_window": "FRONT", "on": True},
                    },
                    {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
                    {"tool_name": "set_fan_airflow_direction", "arguments": {"direction": "WINDSHIELD"}},
                    {"tool_name": "set_air_conditioning", "arguments": {"on": True}},
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Make all windows match the rear passenger window position "
                        "and activate front windshield defrost."
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_climate_settings",
                    "content": (
                        '{"status":"SUCCESS","result":{"fan_speed":0,'
                        '"fan_airflow_direction":"FEET","air_conditioning":false}}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_vehicle_window_positions",
                    "content": (
                        '{"status":"SUCCESS","result":{"window_driver_position":25,'
                        '"window_passenger_position":50,"window_driver_rear_position":25,'
                        '"window_passenger_rear_position":5}}'
                    ),
                },
            ],
            tools=[
                _window_tool(),
                _defrost_tool(),
                _fan_speed_tool(),
                _airflow_tool(),
                _bool_tool("set_air_conditioning"),
                _climate_status_tool(),
                _window_status_tool(),
            ],
        )

        self.assertEqual(decision.action["tool_calls"][0], {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 5}})
        self.assertIn(
            {"tool_name": "set_fan_airflow_direction", "arguments": {"direction": "WINDSHIELD"}},
            decision.action["tool_calls"],
        )

    def test_policy_guard_shortens_email_confirmation_response(self):
        guard = PolicyGuard()
        long_message = "Weather update. " + ("Bring an umbrella. " * 40)

        result = guard.apply(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "send_email",
                        "arguments": {
                            "email_addresses": ["tina@example.com", "frank@example.com"],
                            "content_message": long_message,
                        },
                    }
                ],
            },
            tools=[_send_email_tool()],
            messages=[{"role": "user", "content": "Send the team a weather update."}],
        )

        self.assertFalse(result.allowed)
        content = result.replacement_action["content"]
        self.assertIn("tina@example.com", content)
        self.assertIn("frank@example.com", content)
        self.assertLess(len(content), 180)
        self.assertNotIn("Bring an umbrella", content)

    def test_training_insights_provide_abstract_navigation_recipe_without_ids(self):
        store = default_training_insights()
        hints = store.hints_for(
            user_text="Find a restaurant in Barcelona and replace my current destination.",
            completed_tools=[],
            available_tools={
                "get_location_id_by_location_name",
                "search_poi_at_location",
                "get_routes_from_start_to_destination",
                "navigation_replace_final_destination",
            },
        )

        serialized = str(hints)
        self.assertIn("navigation_poi", hints["matched_domains"])
        self.assertIn("get_location_id_by_location_name", hints["suggested_next_tools"])
        self.assertTrue(
            any(
                recipe["tool_sequence"][:4]
                == [
                    "get_location_id_by_location_name",
                    "search_poi_at_location",
                    "get_routes_from_start_to_destination",
                    "navigation_replace_final_destination",
                ]
                for recipe in hints["observed_recipes"]
            )
        )
        self.assertNotRegex(serialized, r"loc_[a-z]{3}_[0-9]+")
        self.assertNotRegex(serialized, r"poi_[a-z]{3}_[0-9]+")
        self.assertNotRegex(serialized, r"base_\d+")

    def test_training_insights_sunroof_policy_hints(self):
        store = default_training_insights()
        hints = store.hints_for(
            user_text="Open the sunroof halfway for fresh air.",
            completed_tools=[],
            available_tools={"get_weather", "open_close_sunshade", "open_close_sunroof"},
        )

        self.assertIn("sunroof", hints["matched_domains"])
        self.assertIn("get_weather", hints["suggested_next_tools"])
        self.assertTrue(
            any("weather before opening the sunroof" in hint for hint in hints["policy_hints"])
        )
        self.assertTrue(
            any("sunshade fully" in hint for hint in hints["policy_hints"])
        )

    def test_plan_state_replaces_city_navigation_with_selected_poi(self):
        store = PlanStateStore()
        context_id = "ctx-plan-poi"
        messages = [
            {"role": "user", "content": "Find a restaurant in Barcelona and set it as my destination."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_loc",
                        "type": "function",
                        "function": {
                            "name": "get_location_id_by_location_name",
                            "arguments": '{"location":"Barcelona"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_loc",
                "name": "get_location_id_by_location_name",
                "content": '{"result":{"location_id":"loc_bar_223644"}}',
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_poi",
                        "type": "function",
                        "function": {
                            "name": "search_poi_at_location",
                            "arguments": (
                                '{"location_id":"loc_bar_223644",'
                                '"category_poi":"restaurants"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_poi",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"Restaurante El Toro"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_route",
                        "type": "function",
                        "function": {
                            "name": "get_routes_from_start_to_destination",
                            "arguments": (
                                '{"start_id":"loc_mad_180891",'
                                '"destination_id":"poi_res_853877"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_route",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":[{"route_id":"rlp_mad_res_588035",'
                    '"destination_id":"poi_res_853877"}]}}'
                ),
            },
        ]
        tools = [
            _location_lookup_tool(),
            _poi_at_location_tool(),
            _routes_tool(),
            _navigation_replace_final_destination_tool(),
        ]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "loc_bar_223644",
                            "route_id_leading_to_new_destination": "rlp_mad_res_588035",
                        },
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "navigation_replace_final_destination")
        self.assertEqual(call["arguments"]["new_destination_id"], "poi_res_853877")
        self.assertEqual(
            call["arguments"]["route_id_leading_to_new_destination"],
            "rlp_mad_res_588035",
        )

    def test_plan_state_selects_requested_via_route_for_replacement(self):
        store = PlanStateStore()
        context_id = "ctx-plan-requested-via-route"
        messages = [
            {
                "role": "user",
                "content": (
                    "Find a restaurant in Barcelona and replace my destination "
                    "using the second route via A53, A85, B884."
                ),
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_location_id_by_location_name",
                "content": '{"result":{"location_id":"loc_bar_223644"}}',
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"Restaurante El Toro"}]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":['
                    '{"route_id":"rlp_mad_res_376587",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"L169, L468","distance_km":594.79},'
                    '{"route_id":"rlp_mad_res_588035",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"A53, A85, B884","distance_km":622.76}'
                    ']}}'
                ),
            },
        ]
        tools = [_navigation_replace_final_destination_tool(), _routes_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_res_853877",
                            "route_id_leading_to_new_destination": "rlp_mad_res_376587",
                        },
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "navigation_replace_final_destination")
        self.assertEqual(
            call["arguments"]["route_id_leading_to_new_destination"],
            "rlp_mad_res_588035",
        )

    def test_plan_state_defaults_to_fastest_route_not_last_route(self):
        store = PlanStateStore()
        context_id = "ctx-plan-default-fastest-route"
        messages = [
            {
                "role": "user",
                "content": "Change my destination to Barcelona and find a good restaurant there using the fastest route.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"El Rincon de Tapas"}]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":['
                    '{"route_id":"rlp_mad_res_720938",'
                    '"start_id":"loc_mad_180891",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"L169, L468",'
                    '"duration_hours":7,"duration_minutes":36,'
                    '"alias":["fastest","first","shortest"]},'
                    '{"route_id":"rlp_mad_res_588035",'
                    '"start_id":"loc_mad_180891",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"A53, A85, B884",'
                    '"duration_hours":7,"duration_minutes":41,'
                    '"alias":["second"]},'
                    '{"route_id":"rlp_mad_res_376587",'
                    '"start_id":"loc_mad_180891",'
                    '"destination_id":"poi_res_853877",'
                    '"name_via":"A19",'
                    '"duration_hours":7,"duration_minutes":44,'
                    '"alias":["third"]}'
                    ']}}'
                ),
            },
        ]
        tools = [_navigation_replace_final_destination_tool(), _routes_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_res_853877",
                            "route_id_leading_to_new_destination": "rlp_mad_res_376587",
                        },
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(
            call["arguments"]["route_id_leading_to_new_destination"],
            "rlp_mad_res_720938",
        )

    def test_plan_state_repeated_location_lookup_advances_to_poi_search(self):
        store = PlanStateStore()
        context_id = "ctx-plan-repeat"
        messages = [
            {"role": "user", "content": "Find a restaurant in Barcelona."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_loc",
                        "type": "function",
                        "function": {
                            "name": "get_location_id_by_location_name",
                            "arguments": '{"location":"Barcelona"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_loc",
                "name": "get_location_id_by_location_name",
                "content": '{"result":{"location_id":"loc_bar_223644"}}',
            },
        ]
        tools = [_location_lookup_tool(), _poi_at_location_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "get_location_id_by_location_name",
                        "arguments": {"location": "Barcelona"},
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "search_poi_at_location")
        self.assertEqual(call["arguments"]["location_id"], "loc_bar_223644")
        self.assertEqual(call["arguments"]["category_poi"], "restaurants")

    def test_plan_state_enriches_weather_meeting_email(self):
        store = PlanStateStore()
        context_id = "ctx-plan-email"
        messages = [
            {
                "role": "user",
                "content": "Email the attendees about the weather and travel for my meeting.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_contact",
                        "type": "function",
                        "function": {
                            "name": "get_contact_information",
                            "arguments": '{"contact_name":"Grace"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_contact",
                "name": "get_contact_information",
                "content": '{"result":{"contacts":[{"email":"grace@example.com"}]}}',
            },
            {
                "role": "tool",
                "name": "get_entries_from_calendar",
                "content": (
                    '{"result":{"entries":[{"title":"Risk Management",'
                    '"location":"Frankfurt","start_time":"13:30"}]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": (
                    '{"result":{"current_slot":{"condition":"cloudy and rain",'
                    '"temperature":6,"wind":9,"humidity":86}}}'
                ),
            },
        ]
        tools = [_send_email_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "send_email",
                        "arguments": {
                            "email_addresses": ["grace@example.com"],
                            "content_message": "Weather update for the meeting",
                        },
                    }
                ],
            },
            tools,
        )

        content = decision.action["tool_calls"][0]["arguments"]["content_message"]
        self.assertIn("Risk Management", content)
        self.assertIn("Frankfurt", content)
        self.assertIn("13:30", content)
        self.assertIn("cloudy and rain", content)
        self.assertIn("travel", content.lower())

    def test_plan_state_routes_active_poi_replacement_from_navigation_start(self):
        store = PlanStateStore()
        context_id = "ctx-plan-active-start"
        messages = [
            {
                "role": "user",
                "content": "Find a restaurant in Barcelona and replace my current destination with it.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_nav",
                        "type": "function",
                        "function": {
                            "name": "get_current_navigation_state",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_nav",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"]}}'
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_loc",
                        "type": "function",
                        "function": {
                            "name": "get_location_id_by_location_name",
                            "arguments": '{"location":"Barcelona"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_loc",
                "name": "get_location_id_by_location_name",
                "content": '{"result":{"location_id":"loc_bar_223644"}}',
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_poi",
                        "type": "function",
                        "function": {
                            "name": "search_poi_at_location",
                            "arguments": (
                                '{"location_id":"loc_bar_223644",'
                                '"category_poi":"restaurants"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_poi",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"Restaurante El Toro"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_wrong_route",
                        "type": "function",
                        "function": {
                            "name": "get_routes_from_start_to_destination",
                            "arguments": (
                                '{"start_id":"loc_bar_223644",'
                                '"destination_id":"poi_res_853877"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_wrong_route",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":[{"route_id":"rlp_bar_res_532764",'
                    '"destination_id":"poi_res_853877"}]}}'
                ),
            },
        ]
        tools = [
            _navigation_state_tool(),
            _location_lookup_tool(),
            _poi_at_location_tool(),
            _routes_tool(),
            _navigation_replace_final_destination_tool(),
        ]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_res_853877",
                            "route_id_leading_to_new_destination": "rlp_bar_res_532764",
                        },
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "get_routes_from_start_to_destination")
        self.assertEqual(call["arguments"]["start_id"], "loc_mad_180891")
        self.assertEqual(call["arguments"]["destination_id"], "poi_res_853877")

    def test_plan_state_corrects_replacement_route_to_destination_match(self):
        store = PlanStateStore()
        context_id = "ctx-plan-route-destination-match"
        messages = [
            {
                "role": "user",
                "content": "Replace my current destination with the restaurant in Barcelona.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"El Rincon de Tapas"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_route_poi",
                        "type": "function",
                        "function": {
                            "name": "get_routes_from_start_to_destination",
                            "arguments": (
                                '{"start_id":"loc_mad_180891",'
                                '"destination_id":"poi_res_853877"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_route_poi",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":[{"route_id":"rlp_mad_res_588035",'
                    '"destination_id":"poi_res_853877"}]}}'
                ),
            },
        ]
        tools = [_navigation_replace_final_destination_tool(), _routes_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_res_853877",
                            "route_id_leading_to_new_destination": "rll_mad_bar_907986",
                        },
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "navigation_replace_final_destination")
        self.assertEqual(
            call["arguments"],
            {
                "new_destination_id": "poi_res_853877",
                "route_id_leading_to_new_destination": "rlp_mad_res_588035",
            },
        )

    def test_plan_state_ignores_base_route_id_for_navigation_replacement(self):
        store = PlanStateStore()
        context_id = "ctx-plan-ignore-base-route"
        messages = [
            {
                "role": "user",
                "content": "Go to El Rincon de Tapas in Barcelona.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"id":"poi_res_853877",'
                    '"name":"El Rincon de Tapas"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_route",
                        "type": "function",
                        "function": {
                            "name": "get_routes_from_start_to_destination",
                            "arguments": (
                                '{"start_id":"loc_mad_180891",'
                                '"destination_id":"poi_res_853877"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_route",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":[{"route_id":"rlp_mad_res_588035",'
                    '"start_id":"loc_mad_180891",'
                    '"destination_id":"poi_res_853877",'
                    '"base_route_id":"rll_mad_bar_907986"}]}}'
                ),
            },
        ]
        tools = [_navigation_replace_final_destination_tool(), _routes_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_res_853877",
                            "route_id_leading_to_new_destination": "rll_mad_bar_907986",
                        },
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "navigation_replace_final_destination")
        self.assertEqual(
            call["arguments"]["route_id_leading_to_new_destination"],
            "rlp_mad_res_588035",
        )

    def test_plan_state_advances_explicit_charging_station_followup(self):
        store = PlanStateStore()
        context_id = "ctx-plan-charging-followup"
        messages = [
            {
                "role": "user",
                "content": "Help me check my current range and route information.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_ham_166665","loc_fra_178468","loc_col_464166"],'
                    '"routes_to_final_destination_id":["rll_ham_fra_842845","rll_fra_col_988133"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_charging_specs_and_status",
                "content": '{"result":{"state_of_charge":65,"remaining_range":494}}',
            },
            {
                "role": "tool",
                "name": "get_distance_by_soc",
                "content": '{"result":{"distance_km":342}}',
            },
            {
                "role": "user",
                "content": "Find DC fast charging stations with available plugs along the route around 250km.",
            },
        ]
        tools = [
            _navigation_state_tool(),
            _charging_status_tool(),
            _distance_by_soc_tool(),
            _poi_along_route_tool(),
        ]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {"action": "respond", "content": "I can help with that."},
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "search_poi_along_the_route")
        self.assertEqual(call["arguments"]["route_id"], "rll_ham_fra_842845")
        self.assertEqual(call["arguments"]["category_poi"], "charging_stations")
        self.assertEqual(call["arguments"]["at_kilometer"], 250)
        self.assertNotIn("filters", call["arguments"])

    def test_plan_state_deletes_final_destination_from_current_navigation(self):
        store = PlanStateStore()
        context_id = "ctx-plan-delete-final-destination"
        messages = [
            {"role": "user", "content": "Help me with my route and charging."},
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_ham_166665","loc_fra_178468","loc_col_464166"],'
                    '"routes_to_final_destination_id":["rll_ham_fra_842845","rll_fra_col_988133"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_along_the_route",
                "content": '{"result":{"pois_found_along_route":[{"id":"poi_chg_1"}]}}',
            },
            {
                "role": "user",
                "content": "Cancel the Cologne destination so Frankfurt becomes my final destination.",
            },
        ]
        tools = [_navigation_state_tool(), _tool("navigation_delete_destination")]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {"action": "respond", "content": "I can update that."},
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "navigation_delete_destination")
        self.assertEqual(call["arguments"]["destination_id_to_delete"], "loc_col_464166")

    def test_plan_state_rejects_replacement_route_with_wrong_start(self):
        store = PlanStateStore()
        context_id = "ctx-plan-route-start-match"
        messages = [
            {
                "role": "user",
                "content": "Replace my current destination with the restaurant in Barcelona.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"El Rincon de Tapas"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_wrong_start",
                        "type": "function",
                        "function": {
                            "name": "get_routes_from_start_to_destination",
                            "arguments": (
                                '{"start_id":"loc_bar_223644",'
                                '"destination_id":"poi_res_853877"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_wrong_start",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":[{"route_id":"rlp_bar_res_532764",'
                    '"start_id":"loc_bar_223644",'
                    '"destination_id":"poi_res_853877"}]}}'
                ),
            },
        ]
        tools = [_navigation_replace_final_destination_tool(), _routes_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_res_853877",
                            "route_id_leading_to_new_destination": "rlp_bar_res_532764",
                        },
                    }
                ],
            },
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "get_routes_from_start_to_destination")
        self.assertEqual(
            call["arguments"],
            {"start_id": "loc_mad_180891", "destination_id": "poi_res_853877"},
        )

    def test_plan_state_blocks_repeated_failed_navigation_action(self):
        store = PlanStateStore()
        context_id = "ctx-plan-failed-repeat"
        messages = [
            {"role": "user", "content": "Navigate to a charging station near Cologne."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_set_nav",
                        "type": "function",
                        "function": {
                            "name": "set_new_navigation",
                            "arguments": '{"route_ids":["rll_fra_col_215834"]}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_set_nav",
                "name": "set_new_navigation",
                "content": '{"status":"FAILURE","message":"Navigation already active"}',
            },
        ]
        tools = [_set_new_navigation_tool(), _navigation_replace_final_destination_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "set_new_navigation",
                        "arguments": {"route_ids": ["rll_fra_col_215834"]},
                    }
                ],
            },
            tools,
        )

        if decision.action["action"] == "tool_calls":
            self.assertNotEqual(
                decision.action["tool_calls"][0],
                {
                    "tool_name": "set_new_navigation",
                    "arguments": {"route_ids": ["rll_fra_col_215834"]},
                },
            )
        else:
            self.assertIn("failing repeatedly", decision.action["content"])

    def test_plan_state_enriches_topic_time_object_and_weather_units(self):
        store = PlanStateStore()
        context_id = "ctx-plan-email-topic"
        messages = [
            {"role": "user", "content": "Email the attendees about the weather for my meeting."},
            {
                "role": "tool",
                "name": "get_contact_information",
                "content": '{"result":{"contacts":[{"email":"tina@example.com"}]}}',
            },
            {
                "role": "tool",
                "name": "get_entries_from_calendar",
                "content": (
                    '{"result":{"entries":[{"topic":"Risk Management",'
                    '"location":"Frankfurt","time":{"hour":"13","minute":"30"}}]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": (
                    '{"result":{"current_slot":{"condition":"cloudy_and_rain",'
                    '"temperature_c":6,"wind_speed_kph":9,"humidity_percent":86}}}'
                ),
            },
        ]
        tools = [_send_email_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "send_email",
                        "arguments": {
                            "email_addresses": ["tina@example.com"],
                            "content_message": "Weather update for the meeting",
                        },
                    }
                ],
            },
            tools,
        )

        content = decision.action["tool_calls"][0]["arguments"]["content_message"]
        self.assertIn("Risk Management", content)
        self.assertIn("13:30", content)
        self.assertIn("wind: 9 km/h", content)
        self.assertIn("humidity: 86 %", content)

    def test_plan_state_completion_gate_sends_email_after_confirmation(self):
        store = PlanStateStore()
        context_id = "ctx-plan-email-confirmed"
        messages = [
            {"role": "user", "content": "Email Frank that I am running late to my meeting."},
            {
                "role": "tool",
                "name": "get_contact_information",
                "content": '{"result":{"contacts":[{"email":"frank@example.com"}]}}',
            },
            {
                "role": "tool",
                "name": "get_entries_from_calendar",
                "content": (
                    '{"result":{"entries":[{"title":"Partnership Discussion",'
                    '"location":"Minsk","start_time":"14:00"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I can send Frank an apology email for being late. "
                    "Please say yes to confirm."
                ),
            },
            {"role": "user", "content": "Yes, send it."},
        ]

        store.observe_messages(context_id, messages, [_send_email_tool()])
        decision = store.postprocess_action(
            context_id,
            {"action": "respond", "content": "Done."},
            [_send_email_tool()],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "send_email")
        self.assertEqual(call["arguments"]["email_addresses"], ["frank@example.com"])
        self.assertIn("Partnership Discussion", call["arguments"]["content_message"])
        self.assertNotIn("Please say yes to confirm", call["arguments"]["content_message"])

    def test_plan_state_sanitizes_confirmation_text_from_email_body(self):
        store = PlanStateStore()
        context_id = "ctx-plan-email-sanitize"
        messages = [
            {"role": "user", "content": "Email Frank that I am running late."},
            {
                "role": "tool",
                "name": "get_contact_information",
                "content": '{"result":{"contacts":[{"email":"frank@example.com"}]}}',
            },
            {
                "role": "tool",
                "name": "get_entries_from_calendar",
                "content": (
                    '{"result":{"entries":[{"title":"Partnership Discussion",'
                    '"location":"Minsk","start_time":"14:00"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I can send Frank an apology email for being late. "
                    "Please say yes to confirm."
                ),
            },
            {"role": "user", "content": "Yes, send it."},
        ]

        store.observe_messages(context_id, messages, [_send_email_tool()])
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "send_email",
                        "arguments": {
                            "email_addresses": ["frank@example.com"],
                            "content_message": "Please say yes to confirm.",
                        },
                    }
                ],
            },
            [_send_email_tool()],
        )

        content = decision.action["tool_calls"][0]["arguments"]["content_message"]
        self.assertNotIn("Please say yes to confirm", content)
        self.assertIn("Partnership Discussion", content)
        self.assertIn("Minsk", content)

    def test_plan_state_completion_gate_gets_route_before_navigation_response(self):
        store = PlanStateStore()
        context_id = "ctx-plan-nav-route-before-response"
        messages = [
            {
                "role": "user",
                "content": "Find a restaurant in Barcelona and replace my current destination with it.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"],'
                    '"routes_to_final_destination_id":["rll_mad_par_912360"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_location_id_by_location_name",
                "content": '{"result":{"id":"loc_bar_223644","name":"Barcelona"}}',
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"Casa Fonda"}]}}'
                ),
            },
        ]
        tools = [_routes_tool(), _navigation_replace_final_destination_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {"action": "respond", "content": "I found a restaurant."},
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "get_routes_from_start_to_destination")
        self.assertEqual(call["arguments"]["start_id"], "loc_mad_180891")
        self.assertEqual(call["arguments"]["destination_id"], "poi_res_853877")

    def test_plan_state_completion_gate_replaces_destination_before_response(self):
        store = PlanStateStore()
        context_id = "ctx-plan-nav-replace-before-response"
        messages = [
            {
                "role": "user",
                "content": "Find a restaurant in Barcelona and replace my current destination with it.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"],'
                    '"routes_to_final_destination_id":["rll_mad_par_912360"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois":[{"poi_id":"poi_res_853877",'
                    '"name":"Casa Fonda"}]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":[{"route_id":"rlp_mad_res_588035",'
                    '"start_id":"loc_mad_180891",'
                    '"destination_id":"poi_res_853877"}]}}'
                ),
            },
        ]
        tools = [_routes_tool(), _navigation_replace_final_destination_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {"action": "respond", "content": "Done."},
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "navigation_replace_final_destination")
        self.assertEqual(call["arguments"]["new_destination_id"], "poi_res_853877")
        self.assertEqual(
            call["arguments"]["route_id_leading_to_new_destination"],
            "rlp_mad_res_588035",
        )

    def test_plan_state_asks_for_route_choice_when_route_preference_missing(self):
        store = PlanStateStore()
        context_id = "ctx-plan-route-choice-missing"
        messages = [
            {
                "role": "user",
                "content": "Set the destination to Barcelona and find a good restaurant there.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_mad_180891","loc_par_405686"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "search_poi_at_location",
                "content": (
                    '{"result":{"pois_found":[{"id":"poi_res_853877",'
                    '"name":"El Rincon de Tapas"}]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":['
                    '{"route_id":"rlp_mad_res_720938","start_id":"loc_mad_180891",'
                    '"destination_id":"poi_res_853877","name_via":"L169, L468",'
                    '"alias":["fastest","first"]},'
                    '{"route_id":"rlp_mad_res_588035","start_id":"loc_mad_180891",'
                    '"destination_id":"poi_res_853877","name_via":"A53, A85, B884",'
                    '"alias":["second"]}]}}'
                ),
            },
        ]
        tools = [_navigation_replace_final_destination_tool(), _routes_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "poi_res_853877",
                            "route_id_leading_to_new_destination": "rlp_mad_res_720938",
                        },
                    }
                ],
            },
            tools,
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("multiple route options", decision.action["content"])
        self.assertIn("second route via A53, A85, B884", decision.action["content"])

    def test_plan_state_charging_search_uses_latest_kilometer_and_available_dc_filters(self):
        store = PlanStateStore()
        context_id = "ctx-plan-charging-latest-km"
        messages = [
            {
                "role": "user",
                "content": "Check if I can make the trip while keeping 20% battery.",
            },
            {
                "role": "tool",
                "name": "get_current_navigation_state",
                "content": (
                    '{"result":{"navigation_active":true,'
                    '"waypoints_id":["loc_ham_166665","loc_fra_178468","loc_col_464166"],'
                    '"routes_to_final_destination_id":["rll_ham_fra_842845","rll_fra_col_988133"]}}'
                ),
            },
            {
                "role": "tool",
                "name": "get_charging_specs_and_status",
                "content": '{"result":{"state_of_charge":65,"remaining_range":"494km"}}',
            },
            {
                "role": "tool",
                "name": "get_distance_by_soc",
                "content": '{"result":{"distance_km_for_65_until_20_percent_soc":"342km"}}',
            },
            {
                "role": "user",
                "content": (
                    "Find DC fast charging stations along the Hamburg to Frankfurt route, "
                    "around 250 km from the start. Which ones are open with available DC plugs?"
                ),
            },
        ]
        tools = [
            _poi_along_route_tool(include_filters=True),
            _charging_status_tool(),
            _distance_by_soc_tool(),
        ]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {"action": "respond", "content": "Let me check."},
            tools,
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "search_poi_along_the_route")
        self.assertEqual(call["arguments"]["at_kilometer"], 250)
        self.assertEqual(
            call["arguments"]["filters"],
            [
                "charging_stations::has_dc_plug",
                "charging_stations::has_available_plug",
            ],
        )

    def test_plan_state_blocks_repeated_completed_route_lookup(self):
        store = PlanStateStore()
        context_id = "ctx-plan-repeated-route-lookup"
        messages = [
            {
                "role": "user",
                "content": "Show me charging stations around 250 km; do not add one to navigation.",
            },
            {
                "role": "tool",
                "name": "search_poi_along_the_route",
                "content": (
                    '{"result":{"pois_found_along_route":[{"id":"poi_cha_850124",'
                    '"name":"EnBW","category":"charging_stations"}]}}'
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_route",
                        "type": "function",
                        "function": {
                            "name": "get_routes_from_start_to_destination",
                            "arguments": (
                                '{"start_id":"loc_ham_166665",'
                                '"destination_id":"poi_cha_850124"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_route",
                "name": "get_routes_from_start_to_destination",
                "content": (
                    '{"result":{"routes":[{"route_id":"rlp_ham_cha_531875",'
                    '"start_id":"loc_ham_166665",'
                    '"destination_id":"poi_cha_850124"}]}}'
                ),
            },
        ]
        tools = [_routes_tool()]

        store.observe_messages(context_id, messages, tools)
        decision = store.postprocess_action(
            context_id,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "get_routes_from_start_to_destination",
                        "arguments": {
                            "start_id": "loc_ham_166665",
                            "destination_id": "poi_cha_850124",
                        },
                    }
                ],
            },
            tools,
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("EnBW", decision.action["content"])

    def test_track1_renderer_returns_tool_calls_data_part(self):
        parts, history = CARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_window",
                        "arguments": {"window": "DRIVER", "percentage": 50},
                    }
                ],
            }
        )

        self.assertEqual(parts[0].WhichOneof("content"), "data")
        data = MessageToDict(parts[0].data)
        self.assertEqual(
            data,
            {
                "tool_calls": [
                    {
                        "tool_name": "open_close_window",
                        "arguments": {"window": "DRIVER", "percentage": 50},
                    }
                ]
            },
        )
        self.assertEqual(history["tool_calls"][0]["function"]["name"], "open_close_window")

    def test_context_manager_truncates_large_tool_results_without_losing_archive(self):
        manager = ContextManager()
        context_id = "ctx-large-tool"
        big_content = "x" * (SINGLE_TOOL_RESULT_CHAR_LIMIT + 10)
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "What happened?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": big_content,
            },
        ]

        bundle = manager.build_bundle(
            context_id=context_id,
            messages=messages,
            tools=[_tool("get_weather")],
            subagent_name="PrivateSubagent",
            subagent_policy="weather",
            relevant_tools=[_tool("get_weather")],
        )

        latest_tool = bundle["recent_tool_facts"][-1]
        self.assertIn("Large tool result truncated", latest_tool["content"])
        self.assertEqual(
            manager._memory_by_context[context_id].archived_tool_results[
                f"{context_id}:call_1"
            ],
            big_content,
        )

    def test_context_manager_keeps_complete_tool_names_for_generic_subagent(self):
        manager = ContextManager()
        messages = [{"role": "user", "content": "Navigate me to a nearby charger."}]
        tools = [
            _tool("get_routes", "Navigation route lookup"),
            _tool("search_poi_at_location", "Navigation POI search"),
            _tool("think", "Cross-domain think tool"),
            _tool("call_phone_by_number", "Phone Tool"),
        ]

        bundle = manager.build_bundle(
            context_id="ctx-generic-subagent",
            messages=messages,
            tools=tools,
            subagent_name="PrivateSubagent",
            subagent_policy="general",
            relevant_tools=tools,
        )

        self.assertEqual(bundle["subagent_name"], "PrivateSubagent")
        self.assertIn("get_routes", bundle["available_tool_names"])
        self.assertIn("search_poi_at_location", bundle["available_tool_names"])
        self.assertIn("think", bundle["available_tool_names"])
        self.assertIn("call_phone_by_number", bundle["available_tool_names"])

    def test_task_memory_create_update_and_list_are_context_scoped(self):
        store = TaskMemoryStore()
        created = store.execute(
            "ctx-a",
            "TaskCreate",
            {
                "subject": "Check navigation",
                "status": "in_progress",
                "metadata": {"domain": "navigation"},
            },
        )
        self.assertTrue(created["ok"])
        self.assertEqual(created["task"]["id"], "1")
        self.assertEqual(created["task"]["status"], "in_progress")

        updated = store.execute(
            "ctx-a",
            "TaskUpdate",
            {"taskId": "1", "status": "completed"},
        )
        self.assertTrue(updated["ok"])
        self.assertEqual(updated["task"]["status"], "completed")

        listed_a = store.execute("ctx-a", "TaskList", {})
        listed_b = store.execute("ctx-b", "TaskList", {})
        self.assertEqual(len(listed_a["tasks"]), 1)
        self.assertEqual(listed_b["tasks"], [])

    def test_task_memory_reminds_when_active_task_goes_stale(self):
        store = TaskMemoryStore()
        store.execute("ctx-reminder", "TaskCreate", {"subject": "Do work"})
        for _ in range(5):
            store.observe_messages("ctx-reminder", [{"role": "user", "content": "continue"}])

        reminders = store.reminders("ctx-reminder")

        self.assertTrue(any("not been updated recently" in item for item in reminders))

    def test_progress_display_is_readable_and_stable(self):
        text = format_progress_update(
            {
                "event": "task_done",
                "task_type": "base",
                "task_split": "test",
                "total_tasks": 5,
                "total_trials": 3,
                "total_runs": 15,
                "completed_runs": 2,
                "remaining_runs": 13,
                "task_id": "base_37",
                "task_position": 2,
                "trial_number": 1,
                "reward": 0.0,
            }
        )
        done_text = format_progress_update(
            {
                "event": "split_done",
                "task_type": "base",
                "task_split": "test",
                "total_tasks": 5,
                "total_trials": 3,
                "total_runs": 15,
                "completed_runs": 15,
                "remaining_runs": 0,
            }
        )

        start_text = format_progress_update(
            {
                "event": "split_start",
                "task_type": "base",
                "task_split": "test",
                "total_tasks": 5,
                "total_trials": 3,
                "total_runs": 15,
                "completed_runs": 0,
                "remaining_runs": 15,
            }
        )
        task_start_text = format_progress_update(
            {
                "event": "task_start",
                "task_type": "base",
                "task_split": "test",
                "total_tasks": 5,
                "total_trials": 3,
                "total_runs": 15,
                "completed_runs": 0,
                "remaining_runs": 15,
                "task_id": "base_5",
                "task_position": 1,
                "trial_number": 1,
            }
        )

        self.assertIn("2/15 task-runs completed, 13 remaining", text)
        self.assertIn("task 2/5 base_37", text)
        self.assertNotIn("0/15", start_text)
        self.assertIn("running task-run 1/15", task_start_text)
        self.assertIn("[####################]", done_text)

    def test_task_guard_normalizes_spoken_times_to_24h(self):
        result = TaskGuard().postprocess(
            action={
                "action": "respond",
                "content": "The meeting starts at 2 PM and another one is at 1:30 PM.",
            },
            messages=[],
            tools=[],
        )

        self.assertIn("14:00", result.action["content"])
        self.assertIn("13:30", result.action["content"])
        self.assertNotIn("PM", result.action["content"])

    def test_task_guard_sends_email_after_confirmation_without_airflow_drift(self):
        send_email = _tool(
            "send_email",
            "REQUIRES_CONFIRMATION Email",
            {
                "type": "object",
                "required": ["email_addresses", "content_message"],
                "properties": {
                    "email_addresses": {"type": "array", "items": {"type": "string"}},
                    "content_message": {"type": "string"},
                },
                "additionalProperties": False,
            },
        )
        airflow = _tool(
            "set_fan_airflow_direction",
            "Fan",
            {
                "type": "object",
                "required": ["direction"],
                "properties": {"direction": {"type": "string"}},
                "additionalProperties": False,
            },
        )
        messages = [
            {"role": "user", "content": "Email Frank that I am running late."},
            {
                "role": "tool",
                "name": "get_contact_information",
                "content": '{"contacts": [{"email": "frank@example.com"}]}',
            },
            {
                "role": "assistant",
                "content": "I can send an email to Frank with a message like: 'I am running late and apologize for the delay.' Should I send it?",
            },
            {"role": "user", "content": "Go ahead."},
        ]

        result = TaskGuard().preempt(messages=messages, tools=[send_email, airflow])

        self.assertEqual(result.action["tool_calls"][0]["tool_name"], "send_email")
        self.assertEqual(
            result.action["tool_calls"][0]["arguments"]["email_addresses"],
            ["frank@example.com"],
        )

    def test_task_guard_ac_intent_adds_ac_and_closes_specific_open_windows(self):
        tools = [
            _tool("get_climate_settings"),
            _tool("get_vehicle_window_positions"),
            _tool(
                "open_close_window",
                "Window",
                {
                    "type": "object",
                    "required": ["window", "percentage"],
                    "properties": {
                        "window": {"type": "string"},
                        "percentage": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            ),
            _bool_tool("set_air_conditioning"),
            _tool(
                "set_fan_speed",
                "Fan",
                {
                    "type": "object",
                    "required": ["level"],
                    "properties": {"level": {"type": "number"}},
                    "additionalProperties": False,
                },
            ),
        ]
        messages = [
            {"role": "user", "content": "It is stuffy, close the windows and cool down."},
            {"role": "tool", "name": "get_climate_settings", "content": '{"fan_speed": 0}'},
            {
                "role": "tool",
                "name": "get_vehicle_window_positions",
                "content": '{"window_driver_position": 40, "window_passenger_position": 0, "window_driver_rear_position": 0, "window_passenger_rear_position": 30}',
            },
        ]

        result = TaskGuard().preempt(messages=messages, tools=tools)
        calls = result.action["tool_calls"]

        self.assertIn(
            {"tool_name": "open_close_window", "arguments": {"window": "DRIVER", "percentage": 0}},
            calls,
        )
        self.assertIn(
            {
                "tool_name": "open_close_window",
                "arguments": {"window": "PASSENGER_REAR", "percentage": 0},
            },
            calls,
        )
        self.assertIn({"tool_name": "set_air_conditioning", "arguments": {"on": True}}, calls)
        self.assertNotIn(
            {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 0}},
            calls,
        )

    def test_task_guard_ac_abbreviation_is_treated_as_ac_intent(self):
        tools = [
            _tool("get_climate_settings"),
            _tool("get_vehicle_window_positions"),
            _tool(
                "open_close_window",
                "Window",
                {
                    "type": "object",
                    "required": ["window", "percentage"],
                    "properties": {
                        "window": {"type": "string"},
                        "percentage": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            ),
            _bool_tool("set_air_conditioning"),
            _tool(
                "set_fan_speed",
                "Fan",
                {
                    "type": "object",
                    "required": ["level"],
                    "properties": {"level": {"type": "number"}},
                    "additionalProperties": False,
                },
            ),
        ]
        messages = [
            {"role": "user", "content": "Can you turn on the AC? The air quality could be better."},
            {"role": "tool", "name": "get_climate_settings", "content": '{"fan_speed": 0}'},
            {
                "role": "tool",
                "name": "get_vehicle_window_positions",
                "content": '{"window_driver_position": 100, "window_passenger_position": 10, "window_driver_rear_position": 0, "window_passenger_rear_position": 25}',
            },
        ]

        result = TaskGuard().preempt(messages=messages, tools=tools)

        self.assertIn(
            {"tool_name": "set_air_conditioning", "arguments": {"on": True}},
            result.action["tool_calls"],
        )

    def test_task_guard_charging_range_uses_distance_by_soc(self):
        tools = [
            _tool("get_charging_specs_and_status"),
            _tool(
                "get_distance_by_soc",
                "Range",
                {
                    "type": "object",
                    "required": ["initial_state_of_charge", "final_state_of_charge"],
                    "properties": {
                        "initial_state_of_charge": {"type": "number"},
                        "final_state_of_charge": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            ),
        ]
        messages = [
            {
                "role": "user",
                "content": "Can we reach the destination and keep at least 20% battery as a buffer?",
            },
            {
                "role": "tool",
                "name": "get_charging_specs_and_status",
                "content": '{"state_of_charge": 65, "remaining_range": 494}',
            },
        ]

        result = TaskGuard().preempt(messages=messages, tools=tools)

        self.assertEqual(result.action["tool_calls"][0]["tool_name"], "get_distance_by_soc")
        self.assertEqual(
            result.action["tool_calls"][0]["arguments"],
            {"initial_state_of_charge": 65.0, "final_state_of_charge": 20},
        )

    def test_task_guard_charging_enough_charge_defaults_to_20_percent_buffer(self):
        tools = [
            _tool("get_charging_specs_and_status"),
            _tool(
                "get_distance_by_soc",
                "Range",
                {
                    "type": "object",
                    "required": ["initial_state_of_charge", "final_state_of_charge"],
                    "properties": {
                        "initial_state_of_charge": {"type": "number"},
                        "final_state_of_charge": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            ),
        ]
        messages = [
            {
                "role": "user",
                "content": "Check my range and make sure I can get to Frankfurt with enough charge.",
            },
            {
                "role": "tool",
                "name": "get_charging_specs_and_status",
                "content": '{"state_of_charge": 65, "remaining_range": "494.0km"}',
            },
        ]

        result = TaskGuard().preempt(messages=messages, tools=tools)

        self.assertEqual(
            result.action["tool_calls"][0]["arguments"]["final_state_of_charge"],
            20,
        )

    def test_task_guard_weather_email_fallback_uses_weather_details(self):
        send_email = _tool(
            "send_email",
            "REQUIRES_CONFIRMATION Email",
            {
                "type": "object",
                "required": ["email_addresses", "content_message"],
                "properties": {
                    "email_addresses": {"type": "array", "items": {"type": "string"}},
                    "content_message": {"type": "string"},
                },
                "additionalProperties": False,
            },
        )
        messages = [
            {
                "role": "user",
                "content": "Send the attendees an email about the weather and how it may affect travel.",
            },
            {
                "role": "tool",
                "name": "get_contact_information",
                "content": '{"contacts": [{"email": "a@example.com"}, {"email": "b@example.com"}]}',
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"result": {"current_slot": {"condition": "cloudy_and_rain", "temperature_c": 6, "wind_speed_kph": 9, "humidity_percent": 86}}}',
            },
            {
                "role": "assistant",
                "content": "The weather may affect travel. Should I send an email?",
            },
            {"role": "user", "content": "Yes, please send it."},
        ]

        result = TaskGuard().preempt(messages=messages, tools=[send_email])
        content = result.action["tool_calls"][0]["arguments"]["content_message"]

        self.assertIn("condition: cloudy and rain", content)
        self.assertIn("6 C", content)
        self.assertIn("travel", content)

    def test_task_guard_redirects_navigation_delete_to_destination_lookup(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [{"tool_name": "delete_current_navigation", "arguments": {}}],
            },
            messages=[
                {
                    "role": "user",
                    "content": "Cancel navigation to the old city. Find a restaurant in Barcelona.",
                },
                {
                    "role": "tool",
                    "name": "get_current_navigation_state",
                    "content": '{"result": {"navigation_active": true, "waypoints_id": ["loc_a", "loc_b"], "routes_to_final_destination_id": ["rll_a_b"]}}',
                },
            ],
            tools=[
                _navigation_state_tool(),
                _tool(
                    "get_location_id_by_location_name",
                    "Location",
                    {
                        "type": "object",
                        "required": ["location"],
                        "properties": {"location": {"type": "string"}},
                        "additionalProperties": False,
                    },
                ),
                _tool("delete_current_navigation"),
            ],
        )

        self.assertEqual(
            decision.action["tool_calls"][0],
            {
                "tool_name": "get_location_id_by_location_name",
                "arguments": {"location": "Barcelona"},
            },
        )

    def test_task_guard_splits_poi_lookup_before_navigation_delete(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_delete_destination",
                        "arguments": {"destination_id_to_delete": "loc_col_1"},
                    },
                    {
                        "tool_name": "search_poi_along_the_route",
                        "arguments": {
                            "route_id": "rll_ham_fra_1",
                            "category_poi": "charging_stations",
                            "at_kilometer": 250,
                        },
                    },
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "First show charging stations along the route, then remove the final destination.",
                },
                {
                    "role": "tool",
                    "name": "get_current_navigation_state",
                    "content": '{"result": {"navigation_active": true, "waypoints_id": ["loc_ham", "loc_fra", "loc_col"], "routes_to_final_destination_id": ["rll_ham_fra_1", "rll_fra_col_1"]}}',
                },
            ],
            tools=[_navigation_state_tool(), _tool("navigation_delete_destination"), _tool("search_poi_along_the_route")],
        )

        self.assertEqual(len(decision.action["tool_calls"]), 1)
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "search_poi_along_the_route",
        )

    def test_task_guard_converts_active_new_navigation_to_replace_destination(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "set_new_navigation",
                        "arguments": {"route_ids": ["rlp_mad_res_222222"]},
                    }
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "Set the destination to the restaurant using the second route via A53, A85, B884.",
                },
                {
                    "role": "tool",
                    "name": "get_current_navigation_state",
                    "content": '{"result": {"navigation_active": true, "waypoints_id": ["loc_mad_180891", "loc_par_405686"], "routes_to_final_destination_id": ["rll_mad_par_912360"]}}',
                },
                {
                    "role": "tool",
                    "name": "get_routes_from_start_to_destination",
                    "content": '{"result": {"routes": [{"route_id": "rlp_mad_res_111111", "destination_id": "poi_res_853877", "name_via": "L169, L468", "distance_km": 594.79}, {"route_id": "rlp_mad_res_222222", "destination_id": "poi_res_853877", "name_via": "A53, A85, B884", "distance_km": 622.76}]}}',
                },
            ],
            tools=[
                _set_new_navigation_tool(),
                _navigation_replace_final_destination_tool(),
                _navigation_state_tool(),
                _routes_tool(),
            ],
        )

        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "navigation_replace_final_destination",
        )
        self.assertEqual(
            decision.action["tool_calls"][0]["arguments"],
            {
                "new_destination_id": "poi_res_853877",
                "route_id_leading_to_new_destination": "rlp_mad_res_222222",
            },
        )

    def test_task_guard_defaults_active_new_navigation_to_fastest_route(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "set_new_navigation",
                        "arguments": {"route_ids": ["rlp_mad_res_376587"]},
                    }
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "Change my destination to Barcelona and find a good restaurant there.",
                },
                {
                    "role": "tool",
                    "name": "get_current_navigation_state",
                    "content": '{"result": {"navigation_active": true, "waypoints_id": ["loc_mad_180891", "loc_par_405686"], "routes_to_final_destination_id": ["rll_mad_par_912360"]}}',
                },
                {
                    "role": "tool",
                    "name": "get_routes_from_start_to_destination",
                    "content": (
                        '{"result": {"routes": ['
                        '{"route_id": "rlp_mad_res_720938", "destination_id": "poi_res_853877", '
                        '"name_via": "L169, L468", "alias": ["fastest", "first", "shortest"]}, '
                        '{"route_id": "rlp_mad_res_588035", "destination_id": "poi_res_853877", '
                        '"name_via": "A53, A85, B884", "alias": ["second"]}, '
                        '{"route_id": "rlp_mad_res_376587", "destination_id": "poi_res_853877", '
                        '"name_via": "A19", "alias": ["third"]}'
                        ']}}'
                    ),
                },
            ],
            tools=[
                _set_new_navigation_tool(),
                _navigation_replace_final_destination_tool(),
                _navigation_state_tool(),
                _routes_tool(),
            ],
        )

        self.assertEqual(
            decision.action["tool_calls"][0]["arguments"],
            {
                "new_destination_id": "poi_res_853877",
                "route_id_leading_to_new_destination": "rlp_mad_res_720938",
            },
        )

    def test_schema_guard_validates_required_enum_and_coerces_number(self):
        tool = _tool(
            "open_close_window",
            "Vehicle Control",
            {
                "type": "object",
                "required": ["window", "percentage"],
                "properties": {
                    "window": {"type": "string", "enum": ["DRIVER", "PASSENGER"]},
                    "percentage": {"type": "number", "minimum": 0, "maximum": 100},
                },
                "additionalProperties": False,
            },
        )

        valid = SchemaGuard().validate(
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_window",
                        "arguments": {"window": "DRIVER", "percentage": "50"},
                    }
                ],
            },
            [tool],
        )
        invalid = SchemaGuard().validate(
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_window",
                        "arguments": {"window": "ALL", "percentage": 50},
                    }
                ],
            },
            [tool],
        )

        self.assertTrue(valid.valid)
        self.assertEqual(
            valid.normalized_action["tool_calls"][0]["arguments"]["percentage"],
            50.0,
        )
        self.assertFalse(invalid.valid)
        self.assertIn("must be one of", invalid.errors[0])

    def test_pass_summary_reports_overall_and_split_pass_power_scores(self):
        summary = build_pass_summary(
            pass_power_k_scores={"Pass^1": 0.5, "Pass^3": 0.75},
            pass_power_k_scores_by_split={
                "base": {"Pass^1": 1.0, "Pass^3": 1.0},
                "hallucination": {"Pass^1": 0.0},
            },
        )
        display = format_pass_summary(summary)

        self.assertEqual(summary["overall"], {"pass^1": 0.5, "pass^3": 0.75})
        self.assertEqual(summary["by_split"]["hallucination"]["pass^3"], None)
        self.assertIn("Overall: Pass^1 50.0%, Pass^3 75.0%", display)
        self.assertIn("Hallucination: Pass^1 0.0%, Pass^3 N/A", display)

    def test_task_guard_checks_preferences_for_ambiguous_sunroof_opening(self):
        decision = TaskGuard().preempt(
            messages=[{"role": "user", "content": "Can you open the sunroof?"}],
            tools=[
                _number_tool("open_close_sunroof"),
                _tool(
                    "get_user_preferences",
                    parameters={
                        "type": "object",
                        "required": ["preference_categories"],
                        "properties": {"preference_categories": {"type": "object"}},
                        "additionalProperties": False,
                    },
                ),
            ],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_user_preferences",
        )

    def test_task_guard_blocks_sunroof_when_needed_sunshade_capability_is_missing(self):
        decision = TaskGuard().preempt(
            messages=[
                {"role": "user", "content": "Open the sunroof to 50%."},
                {
                    "role": "tool",
                    "name": "get_sunroof_and_sunshade_position",
                    "content": '{"status":"SUCCESS","result":{"sunroof_position":0,"sunshade_position":0}}',
                },
            ],
            tools=[
                _number_tool("open_close_sunroof"),
                _tool("get_sunroof_and_sunshade_position"),
            ],
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("sunshade", decision.action["content"].lower())
        self.assertIn("unavailable", decision.action["content"].lower())

    def test_task_guard_derives_current_weather_query_before_opening_sunroof(self):
        system = (
            'Current location is CURRENT_LOCATION = {"id":"loc_lux_222378","name":"Luxembourg",'
            '"position":{"longitude":6.1,"latitude":49.6}}\n'
            'Current time is DATETIME = {"year":2025,"month":2,"day":26,"hour":17,"minute":15}'
        )
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunroof",
                        "arguments": {"percentage": 50},
                    }
                ],
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Open the sunroof to 50%."},
            ],
            tools=[_number_tool("open_close_sunroof"), _weather_tool()],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "get_weather")
        self.assertEqual(
            call["arguments"],
            {
                "location_or_poi_id": "loc_lux_222378",
                "month": 2,
                "day": 26,
                "time_hour_24hformat": 17,
                "time_minutes": 15,
            },
        )

    def test_task_guard_blocks_state_change_after_missing_weather_result_field(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunroof",
                        "arguments": {"percentage": 50},
                    }
                ],
            },
            messages=[
                {"role": "user", "content": "Open the sunroof to 50%."},
                {
                    "role": "tool",
                    "name": "get_weather",
                    "content": '{"status":"SUCCESS","result":{"current_slot":{"temperature_c":1}}}',
                },
            ],
            tools=[_number_tool("open_close_sunroof"), _weather_tool()],
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("missing", decision.action["content"].lower())
        self.assertIn("weather", decision.action["content"].lower())

    def test_task_guard_splits_ac_state_change_until_policy_gets_are_done(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
                ],
            },
            messages=[{"role": "user", "content": "Turn on the air conditioning."}],
            tools=[
                _bool_tool("set_air_conditioning"),
                _climate_status_tool(),
                _window_status_tool(),
                _fan_speed_tool(),
                _window_tool(),
            ],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_climate_settings",
        )

    def test_task_guard_defrost_adds_required_climate_actions(self):
        decision = TaskGuard().preempt(
            messages=[
                {"role": "user", "content": "Turn on front windshield defrost."},
                {
                    "role": "tool",
                    "name": "get_climate_settings",
                    "content": (
                        '{"status":"SUCCESS","result":{"fan_speed":0,'
                        '"fan_airflow_direction":"FEET","air_conditioning":false}}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "get_vehicle_window_positions",
                    "content": (
                        '{"status":"SUCCESS","result":{"window_driver_position":0,'
                        '"window_passenger_position":0,"window_driver_rear_position":0,'
                        '"window_passenger_rear_position":0}}'
                    ),
                },
            ],
            tools=[
                _defrost_tool(),
                _climate_status_tool(),
                _window_status_tool(),
                _fan_speed_tool(),
                _airflow_tool(),
                _bool_tool("set_air_conditioning"),
            ],
        )

        self.assertEqual(
            [call["tool_name"] for call in decision.action["tool_calls"]],
            [
                "set_window_defrost",
                "set_fan_speed",
                "set_fan_airflow_direction",
                "set_air_conditioning",
            ],
        )
        self.assertEqual(
            decision.action["tool_calls"][0]["arguments"],
            {"on": True, "defrost_window": "FRONT"},
        )

    def test_task_guard_fog_lights_require_weather_before_state_change(self):
        system = (
            'Current location is CURRENT_LOCATION = {"id":"loc_wie_683071","name":"Wiesbaden"}\n'
            'Current time is DATETIME = {"year":2025,"month":8,"day":12,"hour":7,"minute":30}'
        )
        decision = TaskGuard().preempt(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Turn on the fog lights."},
            ],
            tools=[
                _bool_tool("set_fog_lights"),
                _weather_tool(),
                _exterior_lights_tool(),
            ],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(decision.action["tool_calls"][0]["tool_name"], "get_weather")

    def test_task_guard_high_beams_blocked_when_fog_light_field_missing(self):
        decision = TaskGuard().preempt(
            messages=[
                {"role": "user", "content": "Turn on the high beam headlights."},
                {
                    "role": "tool",
                    "name": "get_exterior_lights_status",
                    "content": (
                        '{"status":"SUCCESS","result":{"head_lights_low_beams":true,'
                        '"head_lights_high_beams":false}}'
                    ),
                },
            ],
            tools=[_bool_tool("set_head_lights_high_beams"), _exterior_lights_tool()],
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("missing fog light", decision.action["content"].lower())

    def test_task_guard_finishes_on_stop_signal_without_llm(self):
        decision = TaskGuard().finish_after_stop_signal(
            messages=[{"role": "user", "content": "###STOP###"}]
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertEqual(decision.action["content"], "Done.")
        self.assertIn("terminal stop signal", decision.warnings)

    def test_task_guard_finishes_state_change_when_result_has_no_status(self):
        decision = TaskGuard().finish_after_successful_state_change(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_replace",
                            "type": "function",
                            "function": {
                                "name": "navigation_replace_final_destination",
                                "arguments": (
                                    '{"new_destination_id":"poi_res_853877",'
                                    '"route_id_leading_to_new_destination":"rlp_mad_res_588035"}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_replace",
                    "name": "navigation_replace_final_destination",
                    "content": '{"result":{"ok":true}}',
                },
            ]
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertEqual(decision.action["content"], "Done.")

    def test_task_guard_preserves_done_after_completed_state_change(self):
        decision = TaskGuard().finish_after_successful_state_change(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_replace",
                            "type": "function",
                            "function": {
                                "name": "navigation_replace_final_destination",
                                "arguments": (
                                    '{"new_destination_id":"poi_res_853877",'
                                    '"route_id_leading_to_new_destination":"rlp_mad_res_588035"}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_replace",
                    "name": "navigation_replace_final_destination",
                    "content": '{"result":{"ok":true}}',
                },
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Thanks"},
            ]
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertEqual(decision.action["content"], "Done.")
        self.assertIn("terminal acknowledgement", decision.warnings[0])

    def test_task_guard_requires_navigation_state_before_route_edit(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_replace_final_destination",
                        "arguments": {
                            "new_destination_id": "loc_ham_166665",
                            "route_id_leading_to_new_destination": "rll_boc_ham_564928",
                        },
                    }
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "Change my navigation destination to Hamburg.",
                }
            ],
            tools=[
                _navigation_state_tool(),
                _navigation_replace_final_destination_tool(),
            ],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_current_navigation_state",
        )

    def test_task_guard_redirects_delete_destination_to_replacement_lookup(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "navigation_delete_destination",
                        "arguments": {"destination_id_to_delete": "loc_par_405686"},
                    }
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "Cancel the navigation to Paris. Set the destination to Barcelona and find a good restaurant there.",
                },
                {
                    "role": "tool",
                    "name": "get_current_navigation_state",
                    "content": '{"result":{"navigation_active":true,"waypoints_id":["loc_mad_180891","loc_par_405686"],"routes_to_final_destination_id":["rll_mad_par_912360"]}}',
                },
            ],
            tools=[
                _navigation_state_tool(),
                _navigation_delete_destination_tool(),
                _location_lookup_tool(),
            ],
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "get_location_id_by_location_name")
        self.assertEqual(call["arguments"], {"location": "Barcelona"})

    def test_task_guard_preempts_today_calendar_lookup(self):
        calendar_tool = _tool(
            "get_entries_from_calendar",
            "Calendar lookup",
            {
                "type": "object",
                "required": ["month", "day"],
                "properties": {
                    "month": {"type": "integer"},
                    "day": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        )
        decision = TaskGuard().preempt(
            messages=[
                {
                    "role": "system",
                    "content": 'DATETIME = {"year":2025,"month":8,"day":15,"hour":8,"minute":0}',
                },
                {
                    "role": "user",
                    "content": "Could you please check my calendar for today's meetings?",
                },
            ],
            tools=[calendar_tool],
        )

        call = decision.action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "get_entries_from_calendar")
        self.assertEqual(call["arguments"], {"month": 8, "day": 15})

    def test_task_guard_blocks_navigation_when_route_result_field_missing(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "set_new_navigation",
                        "arguments": {"route_ids": ["rll_foo_bar_123456"]},
                    }
                ],
            },
            messages=[
                {"role": "user", "content": "Navigate me to Hamburg."},
                {
                    "role": "tool",
                    "name": "get_routes_from_start_to_destination",
                    "content": '{"status":"SUCCESS","result":{"summary":"available"}}',
                },
            ],
            tools=[_set_new_navigation_tool(), _routes_tool()],
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("route", decision.action["content"].lower())
        self.assertIn("missing", decision.action["content"].lower())

    def test_task_guard_checks_charging_status_before_range_reasoning(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "get_distance_by_soc",
                        "arguments": {
                            "initial_state_of_charge": 80,
                            "final_state_of_charge": 10,
                        },
                    }
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "How far can I drive before reaching 10 percent battery?",
                }
            ],
            tools=[_charging_status_tool(), _distance_by_soc_tool()],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_charging_specs_and_status",
        )

    def test_task_guard_blocks_charging_reasoning_when_status_field_missing(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "get_distance_by_soc",
                        "arguments": {
                            "initial_state_of_charge": 80,
                            "final_state_of_charge": 10,
                        },
                    }
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "How far can I drive before reaching 10 percent battery?",
                },
                {
                    "role": "tool",
                    "name": "get_charging_specs_and_status",
                    "content": '{"status":"SUCCESS","result":{"battery_capacity_kwh":80}}',
                },
            ],
            tools=[_charging_status_tool(), _distance_by_soc_tool()],
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("battery", decision.action["content"].lower())
        self.assertIn("missing", decision.action["content"].lower())

    def test_task_guard_checks_temperature_before_relative_change(self):
        decision = TaskGuard().preempt(
            messages=[
                {
                    "role": "user",
                    "content": "Make the cabin temperature a bit warmer.",
                }
            ],
            tools=[_temperature_status_tool(), _set_climate_temperature_tool()],
        )

        self.assertEqual(decision.action["action"], "tool_calls")
        self.assertEqual(
            decision.action["tool_calls"][0]["tool_name"],
            "get_temperature_inside_car",
        )

    def test_task_guard_blocks_email_when_contact_email_missing(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "send_email",
                        "arguments": {
                            "email_addresses": ["unknown@example.com"],
                            "content_message": "Hi",
                        },
                    }
                ],
            },
            messages=[
                {"role": "user", "content": "Email Grace the update."},
                {
                    "role": "tool",
                    "name": "get_contact_information",
                    "content": (
                        '{"status":"SUCCESS","result":{"con_1":{"name":'
                        '{"first_name":"Grace","last_name":"Nelson"}}}}'
                    ),
                },
            ],
            tools=[_send_email_tool()],
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("email address", decision.action["content"].lower())

    def test_planner_preempts_email_confirmation_without_llm(self):
        planner = Track1Planner(model="test-model")
        result = planner.choose_next_action(
            context_id="ctx-confirm-email",
            messages=[
                {"role": "user", "content": "Email Frank that I am running late."},
                {
                    "role": "assistant",
                    "content": (
                        "I can send this email to frank@example.com: "
                        "Hi Frank, I am running late. Should I send it?"
                    ),
                },
                {"role": "user", "content": "Yes, send it."},
            ],
            tools=[_send_email_tool()],
            ctx_logger=SimpleNamespace(
                warning=lambda *args, **kwargs: None,
                info=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            ),
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        call = result.next_action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "send_email")
        self.assertEqual(call["arguments"]["email_addresses"], ["frank@example.com"])
        self.assertEqual(result.metrics.num_calls, 0)
        self.assertTrue(result.debug["skill_preempted"])
        self.assertEqual(result.debug["skill"], "communication_email")
        self.assertTrue(result.debug["langgraph"])
        self.assertIn("skill_gate", result.debug["graph_nodes"])
        self.assertIn("finalize", result.debug["graph_nodes"])

    def test_langgraph_workflow_runs_planner_critic_execute_path(self):
        planner = Track1Planner(model="test-model")

        class NoopTaskGuard:
            def finish_after_stop_signal(self, *, messages):
                return SimpleNamespace(action=None, warnings=[])

            def finish_after_successful_state_change(self, *, messages):
                return SimpleNamespace(action=None, warnings=[])

            def preempt(self, *, messages, tools):
                return SimpleNamespace(action=None, warnings=[])

            def postprocess(self, *, action, tools, messages):
                return SimpleNamespace(action=None, warnings=[])

        class NoopSkills:
            def preempt(self, *, messages, tools):
                return SimpleNamespace(action=None, skill=None, warnings=[])

        planner.task_guard = NoopTaskGuard()
        planner.skill_registry = NoopSkills()

        def fake_approved_planner(**kwargs):
            return (
                ApprovedPlan(
                    phase="get",
                    allowed_tools=["get_weather"],
                    action_plan=[ApprovedStep(tool="get_weather", arguments={}, phase="get")],
                ),
                LLMCallMetrics(num_calls=1),
            )

        def fake_critic(**kwargs):
            return normalize_critic_verdict({"verdict": "PASS"}), LLMCallMetrics(num_calls=1)

        planner._run_approved_planner = fake_approved_planner
        planner._run_plan_critic = fake_critic

        result = planner.choose_next_action(
            context_id="ctx-langgraph-pec",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[_tool("get_weather")],
            ctx_logger=SimpleNamespace(
                warning=lambda *args, **kwargs: None,
                info=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            ),
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        self.assertEqual(
            result.next_action["tool_calls"],
            [{"tool_name": "get_weather", "arguments": {}}],
        )
        self.assertTrue(result.debug["langgraph"])
        self.assertTrue(result.debug["pec_lite"])
        self.assertIn("approved_planner", result.debug["graph_nodes"])
        self.assertIn("plan_critic", result.debug["graph_nodes"])
        self.assertIn("execute_plan", result.debug["graph_nodes"])
        self.assertIn("finalize", result.debug["graph_nodes"])
        self.assertEqual(result.metrics.num_calls, 2)

    def test_task_guard_asks_for_at_kilometer_for_charging_station_route_search(self):
        decision = TaskGuard().postprocess(
            action={
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "search_poi_along_the_route",
                        "arguments": {
                            "route_id": "rll_man_stu_123456",
                            "category_poi": "charging_stations",
                        },
                    }
                ],
            },
            messages=[
                {
                    "role": "user",
                    "content": "Find a charging station along this route.",
                }
            ],
            tools=[_poi_along_route_tool()],
        )

        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("kilometer", decision.action["content"].lower())

    def test_task_pass3_summary_is_strict_by_task(self):
        summary = build_task_pass3_summary(
            {
                "base": [
                    SimpleNamespace(task_id="base_1", task_index=1, trial=0, reward=1.0),
                    SimpleNamespace(task_id="base_1", task_index=1, trial=1, reward=1.0),
                    SimpleNamespace(task_id="base_1", task_index=1, trial=2, reward=1.0),
                    SimpleNamespace(task_id="base_2", task_index=2, trial=0, reward=1.0),
                    SimpleNamespace(task_id="base_2", task_index=2, trial=1, reward=0.0),
                    SimpleNamespace(task_id="base_2", task_index=2, trial=2, reward=1.0),
                ],
                "hallucination": [
                    SimpleNamespace(task_id="hallucination_1", task_index=3, trial=0, reward=1.0)
                ],
            }
        )
        display = format_task_pass3_summary(summary)

        self.assertEqual(summary["overall"]["passed_tasks"], 1)
        self.assertEqual(summary["overall"]["eligible_tasks"], 2)
        self.assertEqual(summary["overall"]["incomplete_tasks"], 1)
        self.assertFalse(summary["by_task"]["base_2"]["pass^3"])
        self.assertIsNone(summary["by_task"]["hallucination_1"]["pass^3"])
        self.assertIn("base_1:PASS", display)
        self.assertIn("base_2:FAIL", display)
        self.assertIn("hallucination_1:INCOMPLETE", display)

    def test_approved_plan_defaults_allowed_tools_to_current_phase(self):
        plan = normalize_approved_plan(
            {
                "task_feasible": True,
                "phase": "get",
                "action_plan": [
                    {"tool": "get_weather", "arguments": {}, "phase": "get"},
                    {
                        "tool": "open_close_sunroof",
                        "arguments": {"percentage": 50},
                        "phase": "execute",
                    },
                ],
            },
            {"get_weather", "open_close_sunroof"},
        )

        self.assertEqual(plan.phase, "get")
        self.assertEqual(plan.allowed_tools, ["get_weather"])

    def test_approved_plan_executor_only_runs_current_phase(self):
        planner = Track1Planner(model="unit-test-model")
        plan = ApprovedPlan(
            phase="get",
            allowed_tools=["get_weather"],
            action_plan=[
                ApprovedStep(tool="get_weather", arguments={}, phase="get"),
                ApprovedStep(
                    tool="open_close_sunroof",
                    arguments={"percentage": 50},
                    phase="execute",
                ),
            ],
        )

        action = planner._execute_approved_plan(
            plan,
            [_tool("get_weather"), _number_tool("open_close_sunroof")],
        )

        self.assertEqual(action["action"], "tool_calls")
        self.assertEqual(len(action["tool_calls"]), 1)
        self.assertEqual(action["tool_calls"][0]["tool_name"], "get_weather")

    def test_approved_plan_executor_respects_forbidden_tools(self):
        planner = Track1Planner(model="unit-test-model")
        plan = ApprovedPlan(
            phase="get",
            allowed_tools=["get_weather"],
            forbidden_tools=["get_weather"],
            action_plan=[ApprovedStep(tool="get_weather", arguments={}, phase="get")],
        )

        action = planner._execute_approved_plan(plan, [_tool("get_weather")])

        self.assertIsNone(action)

    def test_critic_verdict_block_alias_maps_to_revise(self):
        verdict = normalize_critic_verdict(
            {
                "verdict": "BLOCK",
                "violations": ["missing get_weather"],
                "recommended_changes": ["add weather check"],
            }
        )

        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.verdict, "REVISE")
        self.assertIn("missing get_weather", verdict.violations)

    def test_agent_context_cache_prunes_old_contexts(self):
        executor = CARBenchAgentExecutor(model="unit-test-model")
        executor.max_contexts = 2
        for context_id in ["ctx-1", "ctx-2", "ctx-3"]:
            executor._mark_context_seen(context_id)
            executor.ctx_id_to_messages[context_id] = [
                {"role": "user", "content": context_id}
            ]
            executor.ctx_id_to_tools[context_id] = []
            executor.ctx_id_to_turn_metrics[context_id] = {}

        executor._prune_context_cache(current_context_id="ctx-3")

        self.assertNotIn("ctx-1", executor.ctx_id_to_messages)
        self.assertIn("ctx-2", executor.ctx_id_to_messages)
        self.assertIn("ctx-3", executor.ctx_id_to_messages)


if __name__ == "__main__":
    unittest.main()
