import unittest
from types import SimpleNamespace

from track_1_agent_under_test.approved_plan import (
    ApprovedPlan,
    ApprovedStep,
    normalize_approved_plan,
    normalize_critic_verdict,
)
from track_1_agent_under_test.car_bench_agent import CARBenchAgentExecutor
from track_1_agent_under_test.multi_agent_types import LLMCallMetrics
from track_1_agent_under_test.planner import Track1Planner
from evaluator.car_bench_evaluator import (
    build_task_pass3_summary,
    format_task_pass3_summary,
)


def _logger():
    return SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
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


def _send_email_tool():
    return _tool(
        "send_email",
        "REQUIRES_CONFIRMATION: Send email",
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


def _seat_heating_tool():
    return _tool(
        "set_seat_heating",
        "Set seat heating",
        {
            "type": "object",
            "required": ["seat_zone", "level"],
            "properties": {
                "seat_zone": {"type": "string"},
                "level": {"type": "number"},
            },
            "additionalProperties": False,
        },
    )


def _set_climate_temperature_tool():
    return _tool(
        "set_climate_temperature",
        "Set climate temperature",
        {
            "type": "object",
            "required": ["temperature", "seat_zone"],
            "properties": {
                "temperature": {"type": "number"},
                "seat_zone": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def _navigation_state_tool():
    return _tool("get_current_navigation_state")


def _poi_along_route_tool():
    return _tool(
        "search_poi_along_the_route",
        "Search POIs along route",
        {
            "type": "object",
            "required": ["route_id", "category_poi", "at_kilometer"],
            "properties": {
                "route_id": {"type": "string"},
                "category_poi": {"type": "string"},
                "at_kilometer": {"type": "number"},
            },
            "additionalProperties": False,
        },
    )


def _location_lookup_tool():
    return _tool(
        "get_location_id_by_location_name",
        "Lookup location",
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
        "Search POIs",
        {
            "type": "object",
            "required": ["location_id", "category_poi"],
            "properties": {
                "location_id": {"type": "string"},
                "category_poi": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def _routes_tool():
    return _tool(
        "get_routes_from_start_to_destination",
        "Get routes",
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


def _navigation_replace_tool():
    return _tool(
        "navigation_replace_final_destination",
        "Replace destination",
        {
            "type": "object",
            "required": ["new_destination_id", "route_id_leading_to_new_destination"],
            "properties": {
                "new_destination_id": {"type": "string"},
                "route_id_leading_to_new_destination": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def _calendar_tool():
    return _tool(
        "get_entries_from_calendar",
        "Calendar",
        {
            "type": "object",
            "required": ["month", "day"],
            "properties": {"month": {"type": "number"}, "day": {"type": "number"}},
            "additionalProperties": False,
        },
    )


def _contact_tool():
    return _tool(
        "get_contact_information",
        "Contacts",
        {
            "type": "object",
            "required": ["contact_ids"],
            "properties": {"contact_ids": {"type": "array", "items": {"type": "string"}}},
            "additionalProperties": False,
        },
    )


def _weather_tool():
    return _tool(
        "get_weather",
        "Weather",
        {
            "type": "object",
            "required": ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
            "properties": {
                "location_or_poi_id": {"type": "string"},
                "month": {"type": "number"},
                "day": {"type": "number"},
                "time_hour_24hformat": {"type": "number"},
            },
            "additionalProperties": False,
        },
    )


def _charging_status_tool():
    return _tool("get_charging_specs_and_status")


def _distance_by_soc_tool():
    return _tool(
        "get_distance_by_soc",
        "Range by SOC",
        {
            "type": "object",
            "required": ["initial_state_of_charge", "final_state_of_charge"],
            "properties": {
                "initial_state_of_charge": {"type": "number"},
                "final_state_of_charge": {"type": "number"},
            },
            "additionalProperties": False,
        },
    )


def _navigation_delete_tool():
    return _tool(
        "navigation_delete_destination",
        "Delete destination",
        {
            "type": "object",
            "required": ["destination_id_to_delete"],
            "properties": {"destination_id_to_delete": {"type": "string"}},
            "additionalProperties": False,
        },
    )


class Track1LangGraphRuntimeTest(unittest.TestCase):
    def test_email_confirmation_is_handled_inside_langgraph_without_llm(self):
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
            ctx_logger=_logger(),
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        call = result.next_action["tool_calls"][0]
        self.assertEqual(call["tool_name"], "send_email")
        self.assertEqual(call["arguments"]["email_addresses"], ["frank@example.com"])
        self.assertEqual(call["arguments"]["content_message"], "Hi Frank, I am running late.")
        self.assertEqual(result.metrics.num_calls, 0)
        self.assertTrue(result.debug["langgraph_native"])
        self.assertIn("langgraph_skill_gate", result.debug["graph_nodes"])
        self.assertIn("finalize", result.debug["graph_nodes"])

    def test_climate_remaining_step_is_completed_from_langgraph_state(self):
        planner = Track1Planner(model="test-model")
        result = planner.choose_next_action(
            context_id="ctx-climate-langgraph",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Optimize climate for energy efficiency based on who's actually "
                        "in the car. Turn off heated empty seats and match my temperature "
                        "to the passenger side."
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
            ],
            tools=[_seat_heating_tool(), _set_climate_temperature_tool()],
            ctx_logger=_logger(),
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
        self.assertTrue(result.debug["langgraph_completion_warnings"])

    def test_route_charging_precondition_is_langgraph_node(self):
        planner = Track1Planner(model="test-model")

        result = planner.choose_next_action(
            context_id="ctx-route-charging-precondition",
            messages=[
                {
                    "role": "user",
                    "content": "Find a charging station along this route.",
                }
            ],
            tools=[_navigation_state_tool(), _poi_along_route_tool()],
            ctx_logger=_logger(),
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        self.assertEqual(
            result.next_action["tool_calls"],
            [{"tool_name": "get_current_navigation_state", "arguments": {}}],
        )
        self.assertIn("langgraph_skill_gate", result.debug["graph_nodes"])
        self.assertEqual(result.debug["skill"], "route_charging")

    def test_langgraph_workflow_runs_planner_critic_execute_path(self):
        planner = Track1Planner(model="test-model")

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

        planner._run_langgraph_approved_planner = fake_approved_planner
        planner._run_langgraph_plan_critic = fake_critic

        result = planner.choose_next_action(
            context_id="ctx-langgraph-pec",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[_tool("get_weather")],
            ctx_logger=_logger(),
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        self.assertEqual(
            result.next_action["tool_calls"],
            [{"tool_name": "get_weather", "arguments": {}}],
        )
        self.assertTrue(result.debug["langgraph_native"])
        self.assertTrue(result.debug["pec_lite"])
        self.assertIn("approved_planner", result.debug["graph_nodes"])
        self.assertIn("plan_critic", result.debug["graph_nodes"])
        self.assertIn("execute_plan", result.debug["graph_nodes"])
        self.assertIn("finalize", result.debug["graph_nodes"])
        self.assertEqual(result.metrics.num_calls, 2)

    def test_navigation_restaurant_retarget_replaces_final_destination(self):
        planner = Track1Planner(model="test-model")
        result = planner.choose_next_action(
            context_id="ctx-nav-restaurant",
            messages=[
                {"role": "user", "content": "Change my destination to a restaurant in Barcelona."},
                {
                    "role": "tool",
                    "name": "get_current_navigation_state",
                    "content": '{"result":{"waypoints_id":["loc_mad_180891","loc_par_405686"]}}',
                },
                {
                    "role": "tool",
                    "name": "get_location_id_by_location_name",
                    "content": '{"result":{"location_id":"loc_bar_223644"}}',
                },
                {
                    "role": "tool",
                    "name": "search_poi_at_location",
                    "content": '{"result":{"pois_found":[{"id":"poi_res_319074","name":"Restaurante El Toro"},{"id":"poi_res_853877","name":"El Rincon de Tapas"}]}}',
                },
                {
                    "role": "tool",
                    "name": "get_routes_from_start_to_destination",
                    "content": '{"result":{"routes":[{"route_id":"rlp_mad_res_720938","name_via":"L169, L468"},{"route_id":"rlp_mad_res_588035","name_via":"A53, A85, B884"}]}}',
                },
            ],
            tools=[
                _navigation_state_tool(),
                _location_lookup_tool(),
                _poi_at_location_tool(),
                _routes_tool(),
                _navigation_replace_tool(),
            ],
            ctx_logger=_logger(),
        )

        self.assertEqual(result.next_action["action"], "tool_calls")
        self.assertEqual(
            result.next_action["tool_calls"],
            [
                {
                    "tool_name": "navigation_replace_final_destination",
                    "arguments": {
                        "new_destination_id": "poi_res_853877",
                        "route_id_leading_to_new_destination": "rlp_mad_res_588035",
                    },
                }
            ],
        )

    def test_meeting_weather_email_continues_from_calendar_to_contacts(self):
        planner = Track1Planner(model="test-model")
        result = planner.choose_next_action(
            context_id="ctx-weather-email-contacts",
            messages=[
                {"role": "user", "content": "Check the Risk Management meeting weather in Frankfurt and email attendees."},
                {
                    "role": "tool",
                    "name": "get_entries_from_calendar",
                    "content": '{"result":{"events":[{"title":"Risk Management","location_id":"loc_fra_178468","start":"13:30","attendees":["con_4970","con_8656"]}]}}',
                },
            ],
            tools=[_calendar_tool(), _contact_tool(), _weather_tool(), _send_email_tool()],
            ctx_logger=_logger(),
        )

        self.assertEqual(result.next_action["tool_calls"][0]["tool_name"], "get_contact_information")
        self.assertEqual(
            result.next_action["tool_calls"][0]["arguments"],
            {"contact_ids": ["con_4970", "con_8656"]},
        )

    def test_route_charging_chain_deletes_cologne_after_station_search(self):
        planner = Track1Planner(model="test-model")
        result = planner.choose_next_action(
            context_id="ctx-route-charging-delete",
            messages=[
                {"role": "user", "content": "Find charging on my Hamburg Frankfurt route, then remove Cologne."},
                {
                    "role": "tool",
                    "name": "get_current_navigation_state",
                    "content": '{"result":{"waypoints_id":["loc_ham_166665","loc_fra_178468","loc_col_464166"],"routes_to_final_destination_id":["rll_ham_fra_842845","rll_fra_col_988133"]}}',
                },
                {
                    "role": "tool",
                    "name": "get_charging_specs_and_status",
                    "content": '{"result":{"state_of_charge":65}}',
                },
                {
                    "role": "tool",
                    "name": "get_distance_by_soc",
                    "content": '{"result":{"distance_km":342}}',
                },
                {
                    "role": "tool",
                    "name": "search_poi_along_the_route",
                    "content": '{"result":{"pois":[{"id":"poi_cha_1"}]}}',
                },
            ],
            tools=[
                _navigation_state_tool(),
                _charging_status_tool(),
                _distance_by_soc_tool(),
                _poi_along_route_tool(),
                _navigation_delete_tool(),
            ],
            ctx_logger=_logger(),
        )

        self.assertEqual(result.next_action["tool_calls"][0]["tool_name"], "navigation_delete_destination")
        self.assertEqual(
            result.next_action["tool_calls"][0]["arguments"],
            {"destination_id_to_delete": "loc_col_464166"},
        )

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
            [_tool("get_weather"), _tool("open_close_sunroof")],
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
