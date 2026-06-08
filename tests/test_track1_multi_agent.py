import unittest
from types import SimpleNamespace

from google.protobuf.json_format import MessageToDict

from track_1_agent_under_test.car_bench_agent import CARBenchAgentExecutor
from track_1_agent_under_test.context_manager import (
    ContextManager,
    SINGLE_TOOL_RESULT_CHAR_LIMIT,
)
from track_1_agent_under_test.guards import SchemaGuard
from track_1_agent_under_test.task_memory import TaskMemoryStore
from track_1_agent_under_test.task_guard import TaskGuard
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


def _poi_along_route_tool():
    return _tool(
        "search_poi_along_the_route",
        "Points of Interest Search",
        {
            "type": "object",
            "required": ["route_id", "category_poi"],
            "properties": {
                "route_id": {"type": "string"},
                "category_poi": {
                    "type": "string",
                    "enum": ["charging_stations", "restaurants"],
                },
                "at_kilometer": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    )


class Track1MultiAgentTest(unittest.TestCase):
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
