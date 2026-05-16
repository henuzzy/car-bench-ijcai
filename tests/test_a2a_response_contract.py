import unittest

from google.protobuf.json_format import MessageToDict

from a2a.helpers.proto_helpers import new_text_part
from agentbeats.sync_client import (
    build_send_message_jsonrpc_request,
    create_message_with_parts,
)
from track_2_agent_under_test_codex.car_bench_agent import CARBenchAgentExecutor
from track_2_agent_under_test_codex.codex_client import CodexTokenUsage, add_token_usage
from track_2_agent_under_test_codex_planner.planner_agent import (
    PlannerExecutorCARBenchAgentExecutor,
)
from turn_metrics import (
    AVG_LLM_CALL_TIME_MS,
    COMPLETION_TOKENS,
    COST,
    MODEL,
    NUM_LLM_CALLS,
    NUM_PASSES,
    PROMPT_TOKENS,
    THINKING_TOKENS,
)


class A2AResponseContractTest(unittest.TestCase):
    def test_sync_client_serializes_a2a_1_json_field_names(self) -> None:
        message = create_message_with_parts(
            parts=[new_text_part("hello")],
            context_id="ctx-1",
        )

        payload = build_send_message_jsonrpc_request(message)
        serialized_message = payload["params"]["message"]

        self.assertEqual(payload["method"], "SendMessage")
        self.assertIn("messageId", serialized_message)
        self.assertIn("contextId", serialized_message)
        self.assertNotIn("message_id", serialized_message)
        self.assertNotIn("context_id", serialized_message)
        self.assertEqual(serialized_message["parts"], [{"text": "hello"}])

    def test_codex_token_usage_parses_app_server_notification_payload(self) -> None:
        usage = CodexTokenUsage.from_app_server(
            {
                "last": {
                    "inputTokens": 100,
                    "cachedInputTokens": 30,
                    "outputTokens": 12,
                    "reasoningOutputTokens": 4,
                    "totalTokens": 116,
                },
                "total": {
                    "inputTokens": 999,
                    "cachedInputTokens": 333,
                    "outputTokens": 222,
                    "reasoningOutputTokens": 111,
                    "totalTokens": 1665,
                },
            }
        )

        self.assertIsNotNone(usage)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.cached_input_tokens, 30)
        self.assertEqual(usage.output_tokens, 12)
        self.assertEqual(usage.reasoning_output_tokens, 4)
        self.assertEqual(usage.total_tokens, 116)

    def test_codex_token_usage_can_be_aggregated_across_internal_calls(self) -> None:
        usage = add_token_usage(
            CodexTokenUsage(input_tokens=10, output_tokens=3, total_tokens=13),
            CodexTokenUsage(
                input_tokens=20,
                output_tokens=4,
                reasoning_output_tokens=2,
                total_tokens=26,
            ),
        )

        self.assertEqual(usage.input_tokens, 30)
        self.assertEqual(usage.output_tokens, 7)
        self.assertEqual(usage.reasoning_output_tokens, 2)
        self.assertEqual(usage.total_tokens, 39)

    def test_codex_respond_action_returns_text_part(self) -> None:
        parts, history_message = CARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "respond",
                "content": "Done.",
                "tool_calls": [],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "text")
        self.assertEqual(parts[0].text, "Done.")
        self.assertEqual(history_message, {"role": "assistant", "content": "Done."})

    def test_codex_tool_action_returns_tool_calls_data_part(self) -> None:
        parts, history_message = CARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "tool_calls",
                "content": "",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "data")
        data = MessageToDict(parts[0].data)
        self.assertEqual(
            data,
            {
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ]
            },
        )
        self.assertIsNone(history_message["content"])
        self.assertEqual(
            history_message["tool_calls"][0]["function"]["name"],
            "open_close_sunshade",
        )

    def test_codex_turn_metrics_are_public_metadata_shape(self) -> None:
        executor = CARBenchAgentExecutor(model="gpt-5.3-codex-spark")

        executor._record_turn_metrics(
            "ctx",
            100.0,
            token_usage=CodexTokenUsage(
                input_tokens=1200,
                cached_input_tokens=400,
                output_tokens=80,
                reasoning_output_tokens=25,
                total_tokens=1305,
            ),
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[PROMPT_TOKENS], 1200)
        self.assertEqual(metrics[COMPLETION_TOKENS], 80)
        self.assertEqual(metrics[THINKING_TOKENS], 25)
        self.assertEqual(metrics[COST], 0.0)
        self.assertEqual(metrics[MODEL], "gpt-5.3-codex-spark")
        self.assertEqual(metrics[NUM_LLM_CALLS], 1)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 100.0)
        self.assertEqual(metrics[NUM_PASSES], 1)
        self.assertNotIn("_total_llm_time_ms", metrics)

    def test_planner_executor_metrics_report_internal_passes(self) -> None:
        executor = PlannerExecutorCARBenchAgentExecutor(
            planner_model="gpt-5.5",
            executor_model="gpt-5.3-codex-spark",
        )
        executor._last_internal_call_count = 2

        executor._record_turn_metrics(
            "ctx",
            300.0,
            token_usage=CodexTokenUsage(
                input_tokens=3000,
                output_tokens=200,
                reasoning_output_tokens=75,
                total_tokens=3275,
            ),
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[MODEL], "gpt-5.5->gpt-5.3-codex-spark")
        self.assertEqual(metrics[PROMPT_TOKENS], 3000)
        self.assertEqual(metrics[COMPLETION_TOKENS], 200)
        self.assertEqual(metrics[THINKING_TOKENS], 75)
        self.assertEqual(metrics[NUM_LLM_CALLS], 2)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 150.0)
        self.assertEqual(metrics[NUM_PASSES], 2)


if __name__ == "__main__":
    unittest.main()
