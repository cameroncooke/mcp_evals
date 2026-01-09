import json
import pathlib
import tempfile
import unittest

from evals.eval_reporting import (
    aggregate,
    compute_cold_equivalent_cost,
    compute_cost,
    is_cold_row,
    parse_usage,
    recompute_rows,
    write_csv,
    write_markdown_summary,
)


class TestEvalReporting(unittest.TestCase):
    def test_parse_usage_openai_responses_cached_input(self) -> None:
        usage = {"usage": {"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 40}}
        out = parse_usage(usage)
        self.assertEqual(out["output_tokens"], 50)
        self.assertEqual(out["cached_read_tokens"], 40)
        self.assertEqual(out["uncached_input_tokens"], 60)

    def test_parse_usage_openai_prompt_tokens_details(self) -> None:
        usage = {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 20},
            }
        }
        out = parse_usage(usage)
        self.assertEqual(out["output_tokens"], 80)
        self.assertEqual(out["cached_read_tokens"], 20)
        self.assertEqual(out["uncached_input_tokens"], 100)

    def test_parse_usage_anthropic_cache_fields(self) -> None:
        usage = {
            "usage": {
                "input_tokens": 90,
                "output_tokens": 10,
                "cache_read_input_tokens": 30,
                "cache_write_tokens": 5,
            }
        }
        out = parse_usage(usage)
        self.assertEqual(out["uncached_input_tokens"], 90)
        self.assertEqual(out["output_tokens"], 10)
        self.assertEqual(out["cached_read_tokens"], 30)
        self.assertEqual(out["cache_write_tokens"], 5)

    def test_compute_cost_with_cache_multipliers(self) -> None:
        unified = {
            "uncached_input_tokens": 100,
            "cached_read_tokens": 50,
            "cache_write_tokens": 10,
            "cache_write_ttl": "1h",
            "output_tokens": 20,
        }
        pricing = {
            "input_per_token": 0.01,
            "output_per_token": 0.02,
            "cache_read_multiplier": 0.1,
            "cache_write_multiplier_5m": 1.25,
            "cache_write_multiplier_1h": 2.0,
        }
        cost = compute_cost(unified, pricing)
        self.assertAlmostEqual(cost or 0.0, 1.65, places=6)

    def test_compute_cost_prefers_billed_cost(self) -> None:
        unified = {"billed_cost_usd": 9.99}
        cost = compute_cost(unified, {"input_per_token": 0.01, "output_per_token": 0.02})
        self.assertEqual(cost, 9.99)

    def test_compute_cold_equivalent_cost(self) -> None:
        unified = {
            "uncached_input_tokens": 100,
            "cached_read_tokens": 50,
            "output_tokens": 20,
        }
        pricing = {"input_per_token": 0.01, "output_per_token": 0.02}
        cold_cost, cache_rate = compute_cold_equivalent_cost(unified, pricing)
        # total_input = 100 + 50 + 0 = 150 (no cache_write_tokens)
        # cold_cost = 150 * 0.01 + 20 * 0.02 = 1.5 + 0.4 = 1.9
        # cache_rate = 50 / 150 = 1/3
        self.assertAlmostEqual(cold_cost or 0.0, 1.9, places=6)
        self.assertAlmostEqual(cache_rate or 0.0, 1.0 / 3.0, places=6)

    def test_compute_cold_equivalent_cost_with_cache_write(self) -> None:
        """Test that cache_write_tokens are included in total_input calculation."""
        unified = {
            "uncached_input_tokens": 100,
            "cached_read_tokens": 30,
            "cache_write_tokens": 20,
            "output_tokens": 10,
        }
        pricing = {"input_per_token": 0.01, "output_per_token": 0.02}
        cold_cost, cache_rate = compute_cold_equivalent_cost(unified, pricing)
        # total_input = 100 + 30 + 20 = 150
        # cold_cost = 150 * 0.01 + 10 * 0.02 = 1.5 + 0.2 = 1.7
        # cache_rate = 30 / 150 = 0.2
        self.assertAlmostEqual(cold_cost or 0.0, 1.7, places=6)
        self.assertAlmostEqual(cache_rate or 0.0, 0.2, places=6)

    def test_aggregate_basic(self) -> None:
        rows = [
            {
                "agent_id": "a",
                "scenario": "s",
                "task_id": "t",
                "task_kind": "capability",
                "baseline_run": True,
                "success": True,
                "wall_time_sec": 1,
                "billed_cost_usd": 0.1,
                "cold_equivalent_cost_usd": 0.2,
                "cache_savings_usd": 0.1,
                "cache_read_rate": 0.5,
                "xcodebuild_calls": 0,
                "xcrun_calls": 0,
                "simctl_calls": 0,
                "mcp_tool_calls": 0,
                "tool_error_total": 0,
                "tool_error_mcp": 0,
                "tool_error_non_mcp": 0,
            },
            {
                "agent_id": "a",
                "scenario": "s",
                "task_id": "t",
                "task_kind": "capability",
                "baseline_run": False,
                "success": True,
                "wall_time_sec": 10,
                "billed_cost_usd": 1.0,
                "cold_equivalent_cost_usd": 1.5,
                "cache_savings_usd": 0.5,
                "cache_read_rate": 0.25,
                "xcodebuild_calls": 2,
                "xcrun_calls": 3,
                "simctl_calls": 1,
                "mcp_tool_calls": 4,
                "tool_error_total": 0,
                "tool_error_mcp": 0,
                "tool_error_non_mcp": 0,
            },
            {
                "agent_id": "a",
                "scenario": "s",
                "task_id": "t",
                "task_kind": "capability",
                "baseline_run": False,
                "success": False,
                "wall_time_sec": 30,
                "billed_cost_usd": 3.0,
                "cold_equivalent_cost_usd": 4.0,
                "cache_savings_usd": 1.0,
                "cache_read_rate": 0.5,
                "xcodebuild_calls": 4,
                "xcrun_calls": 1,
                "simctl_calls": 2,
                "mcp_tool_calls": 2,
                "tool_error_total": 1,
                "tool_error_mcp": 1,
                "tool_error_non_mcp": 0,
            },
        ]
        summary = aggregate(rows)
        self.assertEqual(len(summary), 1)
        cell = summary[0]
        self.assertEqual(cell["runs"], 2)
        self.assertAlmostEqual(cell["success_rate"], 0.5, places=6)
        self.assertAlmostEqual(cell["pass_at_3"], 0.875, places=6)
        self.assertAlmostEqual(cell["pass_pow_3"], 0.125, places=6)
        self.assertAlmostEqual(cell["time_median"], 20.0, places=6)
        self.assertAlmostEqual(cell["time_p90"], 28.0, places=6)
        self.assertAlmostEqual(cell["cost_median"], 2.0, places=6)
        self.assertAlmostEqual(cell["cold_cost_median"], 2.75, places=6)
        self.assertAlmostEqual(cell["cache_savings_mean"], 0.75, places=6)
        self.assertAlmostEqual(cell["cache_read_rate_mean"], 0.375, places=6)
        self.assertAlmostEqual(cell["cost_per_success_mean"], 4.0, places=6)
        self.assertAlmostEqual(cell["time_per_success_mean"], 40.0, places=6)
        self.assertAlmostEqual(cell["xcodebuild_calls_mean"], 3.0, places=6)
        self.assertAlmostEqual(cell["mcp_tool_calls_mean"], 3.0, places=6)
        self.assertAlmostEqual(cell["tool_error_mean"], 0.5, places=6)
        self.assertAlmostEqual(cell["tool_error_mcp_mean"], 0.5, places=6)

    def test_aggregate_adjusts_mcp_session_defaults_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            error_log_path = pathlib.Path(tmpdir) / "tool_errors.jsonl"
            errors = [
                {"tool_kind": "mcp", "payload": "Error: Missing required session defaults"},
                {"tool_kind": "mcp", "payload": "Some other MCP error"},
            ]
            with open(error_log_path, "w") as f:
                for err in errors:
                    f.write(json.dumps(err) + "\n")

            rows = [
                {
                    "agent_id": "a",
                    "scenario": "mcp_unprimed",
                    "task_id": "t",
                    "task_kind": "capability",
                    "baseline_run": False,
                    "success": True,
                    "wall_time_sec": 10,
                    "billed_cost_usd": 1.0,
                    "cold_equivalent_cost_usd": 1.0,
                    "cache_savings_usd": 0.0,
                    "cache_read_rate": 0.0,
                    "xcodebuild_calls": 0,
                    "xcrun_calls": 0,
                    "simctl_calls": 0,
                    "mcp_tool_calls": 1,
                    "tool_error_total": 2,
                    "tool_error_mcp": 2,
                    "tool_error_non_mcp": 0,
                    "tool_error_log_path": str(error_log_path),
                }
            ]

            summary = aggregate(rows)
            self.assertEqual(len(summary), 1)
            cell = summary[0]
            self.assertAlmostEqual(cell["tool_error_mean"], 1.0, places=6)
            self.assertAlmostEqual(cell["tool_error_mcp_mean"], 1.0, places=6)

    def test_recompute_rows_updates_cached_fields_and_baselines(self) -> None:
        rows = [
            {
                "agent_id": "a",
                "scenario": "s",
                "baseline_run": True,
                "billed_cost_usd": 1.0,
                "provider_usage": {"usage": {"input_tokens": 10, "output_tokens": 5}},
            },
            {
                "agent_id": "a",
                "scenario": "s",
                "baseline_run": False,
                "billed_cost_usd": 2.5,
                "provider_usage": {
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "cached_input_tokens": 4,
                    }
                },
            },
        ]
        pricing = {"a": {"input_per_token": 0.01, "output_per_token": 0.02}}
        recompute_rows(rows, pricing)
        self.assertEqual(rows[1]["cached_read_tokens"], 4)
        self.assertEqual(rows[1]["uncached_input_tokens"], 16)
        self.assertEqual(rows[1]["baseline_cost_usd"], 1.0)
        self.assertAlmostEqual(rows[1]["marginal_cost_usd"], 1.5, places=6)

    def test_write_csv_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "out.csv"
            write_csv(path, [])
            self.assertFalse(path.exists())

    def test_write_markdown_summary_formatting(self) -> None:
        rows = [
            {
                "agent_id": "a",
                "scenario": "s",
                "task_id": "t",
                "task_kind": "capability",
                "runs": 2,
                "success_rate": 0.5,
                "time_median": 20.0,
                "time_p90": 28.0,
                "time_cv": 0.7,
                "cost_median": 2.0,
                "cost_p90": 2.8,
                "cost_cv": 0.7,
                "cold_cost_median": 2.75,
                "cold_cost_p90": 3.75,
                "cold_cost_cv": 0.64,
                "cache_savings_mean": 0.75,
                "cache_read_rate_mean": 0.375,
                "pass_at_3": 0.875,
                "pass_pow_3": 0.125,
                "cost_per_success_mean": 4.0,
                "xcodebuild_calls_mean": 3.0,
                "xcrun_calls_mean": 2.0,
                "simctl_calls_mean": 1.5,
                "mcp_tool_calls_mean": 3.0,
                "tool_error_mean": 0.5,
                "tool_error_mcp_mean": 0.5,
                "tool_error_non_mcp_mean": 0.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "summary.md"
            write_markdown_summary(path, rows)
            text = path.read_text()
            self.assertIn("| agent_id | scenario | task_id |", text)
            self.assertIn("50.0%", text)
            self.assertIn("20.0", text)
            self.assertIn("2.000000", text)

    def test_is_cold_row(self) -> None:
        self.assertTrue(is_cold_row({"cached_read_tokens": 0}))
        self.assertTrue(is_cold_row({"cached_read_tokens": None}))
        self.assertFalse(is_cold_row({"cached_read_tokens": 1}))


if __name__ == "__main__":
    unittest.main()
