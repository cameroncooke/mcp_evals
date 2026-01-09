"""
Integration tests for eval reporting using real run data fixtures.

Tests verify:
1. Manifest generation (failure analysis, tool error reports)
2. Aggregation statistics (pass@k, percentiles, cost metrics)
3. Metrics computation from command logs
4. Agent stubbing for non-deterministic report generation
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

from evals.eval_reporting import aggregate, recompute_rows
from evals import (
    build_failure_analysis_manifest,
    build_tool_error_report_manifest,
    load_jsonl,
    write_jsonl,
    SuiteConfig,
    AgentConfig,
)
from evals.metrics import (
    count_mcp_tool_usage,
    compute_time_to_first_mcp_build_sec,
    count_simctl_invocations,
    read_command_log_entries,
)


FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> List[Dict[str, Any]]:
    """Load a JSONL fixture file."""
    path = FIXTURES_DIR / name
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json_fixture(name: str) -> Dict[str, Any]:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / name
    with open(path) as f:
        return json.load(f)


class TestManifestGeneration(unittest.TestCase):
    """Test manifest building without agent execution."""

    def setUp(self) -> None:
        self.rows = load_fixture("runs_sample.jsonl")
        self.expected = load_json_fixture("expected_summary.json")

    def test_build_failure_analysis_manifest(self) -> None:
        """Verify failure analysis manifest includes only failed non-baseline runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            manifest_path = build_failure_analysis_manifest(self.rows, out_dir)

            self.assertTrue(manifest_path.exists())
            with open(manifest_path) as f:
                manifest = json.load(f)

            expected_test = self.expected["manifest_tests"]["failure_analysis"]
            self.assertEqual(len(manifest["entries"]), expected_test["expected_entries"])

            actual_run_ids = [e["run_id"] for e in manifest["entries"]]
            self.assertEqual(sorted(actual_run_ids), sorted(expected_test["expected_run_ids"]))

            # Verify structure
            self.assertIn("generated_at", manifest)
            self.assertIn("run_dir", manifest)
            self.assertIn("total_failures", manifest)
            self.assertEqual(manifest["total_failures"], expected_test["expected_entries"])

    def test_build_tool_error_report_manifest(self) -> None:
        """Verify tool error manifest includes only runs with tool_error_total > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            manifest_path = build_tool_error_report_manifest(self.rows, out_dir)

            self.assertTrue(manifest_path.exists())
            with open(manifest_path) as f:
                manifest = json.load(f)

            expected_test = self.expected["manifest_tests"]["tool_error_report"]
            self.assertEqual(len(manifest["entries"]), expected_test["expected_entries"])

            actual_run_ids = [e["run_id"] for e in manifest["entries"]]
            self.assertEqual(sorted(actual_run_ids), sorted(expected_test["expected_run_ids"]))

    def test_tool_error_manifest_includes_summary_stats(self) -> None:
        """Verify tool error manifest includes total run counts and error breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            manifest_path = build_tool_error_report_manifest(self.rows, out_dir)

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Verify summary stats are present
            self.assertIn("total_runs", manifest)
            self.assertIn("runs_with_errors", manifest)
            self.assertIn("runs_without_errors", manifest)
            self.assertIn("error_rate_percent", manifest)
            self.assertIn("total_runs_by_agent", manifest)
            self.assertIn("runs_with_errors_by_agent", manifest)
            self.assertIn("errors_by_scenario", manifest)

            # Verify consistency: runs_with_errors + runs_without_errors = total_runs
            self.assertEqual(
                manifest["runs_with_errors"] + manifest["runs_without_errors"],
                manifest["total_runs"]
            )

            # Verify errors_by_scenario has MCP/non-MCP breakdown
            for scenario, stats in manifest["errors_by_scenario"].items():
                self.assertIn("total", stats)
                self.assertIn("mcp", stats)
                self.assertIn("non_mcp", stats)
                # total should equal mcp + non_mcp
                self.assertEqual(stats["total"], stats["mcp"] + stats["non_mcp"])

    def test_discovery_errors_counts_all_session_defaults(self) -> None:
        """Verify ALL session-defaults errors are counted and excluded from real errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)

            # Create a tool error log with multiple session-defaults errors
            error_log_path = out_dir / "tool_errors.jsonl"
            errors = [
                {"tool_kind": "mcp", "payload": "Error: Missing required session defaults"},
                {"tool_kind": "mcp", "payload": "Error: Missing required session defaults"},
                {"tool_kind": "mcp", "payload": "Error: Missing required session defaults"},
                {"tool_kind": "mcp", "payload": "Some other MCP error"},
            ]
            with open(error_log_path, "w") as f:
                for err in errors:
                    f.write(json.dumps(err) + "\n")

            # Create a row referencing the error log (MCP scenario)
            rows = [{
                "run_id": "test-discovery",
                "agent_id": "test-agent",
                "scenario": "mcp_unprimed",
                "task_id": "test-task",
                "baseline_run": False,
                "tool_error_total": 4,
                "tool_error_mcp": 4,
                "tool_error_non_mcp": 0,
                "tool_error_log_path": str(error_log_path),
            }]

            manifest_path = build_tool_error_report_manifest(rows, out_dir)
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Should count ALL 3 session-defaults errors (expected MCP workflow)
            self.assertEqual(manifest["mcp_discovery_errors"], 3)
            # Total MCP errors should be 4
            self.assertEqual(manifest["total_mcp_errors"], 4)
            # Adjusted (real errors) should be 4 - 3 = 1
            self.assertEqual(manifest["adjusted_mcp_errors"], 1)
            # Per-scenario breakdown should also have session_defaults = 3
            self.assertEqual(
                manifest["errors_by_scenario"]["mcp_unprimed"]["session_defaults"], 3
            )

    def test_manifest_excludes_baseline_runs(self) -> None:
        """Verify baseline runs are never included in failure analysis manifest."""
        # Add a failing baseline run to test exclusion
        rows_with_failing_baseline = self.rows + [{
            "run_id": "test-baseline-failure",
            "baseline_run": True,
            "success": False,
            "failure_reason": "tests_failed",
            "tool_error_total": 10,
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            manifest_path = build_failure_analysis_manifest(rows_with_failing_baseline, out_dir)

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Should still only have the original failed run
            self.assertEqual(manifest["total_failures"], 1)
            self.assertNotIn("test-baseline-failure", [e["run_id"] for e in manifest["entries"]])


class TestAggregation(unittest.TestCase):
    """Test summary statistics computation."""

    def setUp(self) -> None:
        self.rows = load_fixture("runs_sample.jsonl")
        self.expected = load_json_fixture("expected_summary.json")

    def test_aggregate_excludes_baseline(self) -> None:
        """Verify baseline runs are excluded from aggregation."""
        summary = aggregate(self.rows)
        self.assertEqual(len(summary), self.expected["expected_groups"])

        # Verify no baseline task appears
        task_ids = [r["task_id"] for r in summary]
        self.assertNotIn("baseline", task_ids)

    def test_aggregate_success_rate(self) -> None:
        """Verify pass@1, pass@3, pass^3 formulas."""
        summary = aggregate(self.rows)

        # Find the failed run group
        failed_group = next(
            r for r in summary
            if r["agent_id"] == "codex"
            and r["scenario"] == "shell_unprimed"
            and r["task_id"] == "hn_api_cache_ttl"
        )

        # For a single failed run: p=0
        self.assertEqual(failed_group["success_rate"], 0.0)
        self.assertEqual(failed_group["pass_at_1"], 0.0)
        self.assertEqual(failed_group["pass_at_3"], 0.0)  # 1 - (1-0)^3 = 0
        self.assertEqual(failed_group["pass_pow_3"], 0.0)  # 0^3 = 0

        # Find a successful run group
        success_group = next(
            r for r in summary
            if r["agent_id"] == "codex"
            and r["scenario"] == "mcp_unprimed"
            and r["task_id"] == "smoke_build_install_launch"
        )

        # For a single successful run: p=1
        self.assertEqual(success_group["success_rate"], 1.0)
        self.assertEqual(success_group["pass_at_1"], 1.0)
        self.assertEqual(success_group["pass_at_3"], 1.0)  # 1 - (1-1)^3 = 1
        self.assertEqual(success_group["pass_pow_3"], 1.0)  # 1^3 = 1

    def test_aggregate_percentiles_single_run(self) -> None:
        """Verify percentiles with single-run groups return the single value."""
        summary = aggregate(self.rows)

        # Find the MCP smoke test group
        expected_group = next(
            g for g in self.expected["groups"]
            if g["task_id"] == "smoke_build_install_launch"
        )
        actual_group = next(
            r for r in summary
            if r["task_id"] == "smoke_build_install_launch"
        )

        # With single run, p10/median/p90/mean should all equal the single value
        self.assertAlmostEqual(
            actual_group["time_median"],
            expected_group["time_median"],
            places=5
        )
        self.assertAlmostEqual(
            actual_group["cost_median"],
            expected_group["cost_median"],
            places=8
        )

    def test_aggregate_cost_per_success(self) -> None:
        """Verify cost_per_success is null when no successes."""
        summary = aggregate(self.rows)

        # Failed run should have null cost_per_success
        failed_group = next(
            r for r in summary
            if r["task_id"] == "hn_api_cache_ttl" and r["scenario"] == "shell_unprimed"
        )
        self.assertIsNone(failed_group["cost_per_success_mean"])

        # Successful run should have cost_per_success = cost / 1
        success_group = next(
            r for r in summary
            if r["task_id"] == "smoke_build_install_launch"
        )
        self.assertIsNotNone(success_group["cost_per_success_mean"])

    def test_aggregate_tool_errors(self) -> None:
        """Verify tool error aggregation."""
        summary = aggregate(self.rows)

        for expected_group in self.expected["groups"]:
            actual_group = next(
                r for r in summary
                if r["agent_id"] == expected_group["agent_id"]
                and r["scenario"] == expected_group["scenario"]
                and r["task_id"] == expected_group["task_id"]
            )
            self.assertAlmostEqual(
                actual_group["tool_error_mean"],
                expected_group["tool_error_mean"],
                places=5
            )

    def test_aggregate_mcp_tool_calls(self) -> None:
        """Verify MCP tool call aggregation."""
        summary = aggregate(self.rows)

        for expected_group in self.expected["groups"]:
            actual_group = next(
                r for r in summary
                if r["agent_id"] == expected_group["agent_id"]
                and r["scenario"] == expected_group["scenario"]
                and r["task_id"] == expected_group["task_id"]
            )
            if expected_group["mcp_tool_calls_mean"] is not None:
                self.assertAlmostEqual(
                    actual_group["mcp_tool_calls_mean"] or 0,
                    expected_group["mcp_tool_calls_mean"],
                    places=5
                )


class TestMetricsComputation(unittest.TestCase):
    """Test metrics extracted from command logs."""

    def test_count_simctl_invocations(self) -> None:
        """Verify simctl invocation counting from cmd_log."""
        cmd_log_path = FIXTURES_DIR / "cmd_log_sample.jsonl"
        count = count_simctl_invocations(cmd_log_path, source="agent")
        # From fixture: simctl list, simctl install, simctl launch = 3
        self.assertEqual(count, 3)

    def test_read_command_log_entries(self) -> None:
        """Verify command log parsing."""
        cmd_log_path = FIXTURES_DIR / "cmd_log_sample.jsonl"
        entries = read_command_log_entries(cmd_log_path, source="agent")
        self.assertEqual(len(entries), 5)

        # Check first entry is xcrun
        self.assertEqual(entries[0]["cmd"], "xcrun")
        self.assertEqual(entries[0]["argv"][0], "simctl")

    def test_count_mcp_tool_usage(self) -> None:
        """Verify MCP tool usage counting."""
        mcp_log_path = FIXTURES_DIR / "mcp_tool_calls_sample.jsonl"
        counts = count_mcp_tool_usage(mcp_log_path)

        self.assertEqual(counts["total"], 10)
        # build_sim is an xcodebuild tool
        self.assertGreaterEqual(counts["xcodebuild"], 1)
        # install_app_sim, launch_app_sim, list_sims are simctl tools
        self.assertGreaterEqual(counts["simctl"], 3)

    def test_time_to_first_mcp_build(self) -> None:
        """Verify time-to-first-mcp-build calculation."""
        mcp_log_path = FIXTURES_DIR / "mcp_tool_calls_sample.jsonl"
        ts_start = "2026-01-21T23:02:23.781895+00:00"

        result = compute_time_to_first_mcp_build_sec(ts_start, mcp_log_path)

        # build_sim is at 2026-01-21T23:03:23.357703Z
        # Start is 2026-01-21T23:02:23.781895Z
        # Delta should be about 59.5 seconds
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result, 59.575808, delta=0.1)

    def test_time_to_first_mcp_build_no_build_tools(self) -> None:
        """Verify None returned when no build tools in log."""
        # Create a temp file with only non-build tools
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"ts": "2026-01-21T23:02:28Z", "tool": "list_schemes"}\n')
            f.write('{"ts": "2026-01-21T23:02:30Z", "tool": "discover_projs"}\n')
            log_path = pathlib.Path(f.name)

        try:
            result = compute_time_to_first_mcp_build_sec(
                "2026-01-21T23:02:00+00:00",
                log_path
            )
            self.assertIsNone(result)
        finally:
            log_path.unlink()


class TestJSONLRoundtrip(unittest.TestCase):
    """Test JSONL read/write consistency."""

    def test_load_write_roundtrip(self) -> None:
        """Verify JSONL data survives write/load cycle."""
        original_rows = load_fixture("runs_sample.jsonl")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pathlib.Path(tmpdir) / "roundtrip.jsonl"
            write_jsonl(out_path, original_rows)
            reloaded_rows = load_jsonl(out_path)

        self.assertEqual(len(reloaded_rows), len(original_rows))
        for orig, reloaded in zip(original_rows, reloaded_rows):
            self.assertEqual(orig["run_id"], reloaded["run_id"])
            self.assertEqual(orig["success"], reloaded["success"])
            self.assertAlmostEqual(
                orig["wall_time_sec"],
                reloaded["wall_time_sec"],
                places=10
            )

    def test_load_nonexistent_returns_empty(self) -> None:
        """Verify loading nonexistent file returns empty list."""
        result = load_jsonl(pathlib.Path("/nonexistent/path.jsonl"))
        self.assertEqual(result, [])


class TestRecomputeRows(unittest.TestCase):
    """Test row recomputation with pricing."""

    def test_recompute_rows_cache_metrics(self) -> None:
        """Verify cache metrics are computed correctly."""
        rows = load_fixture("runs_sample.jsonl")

        # Use a simple pricing model
        pricing_by_agent = {
            "codex": {
                "input_per_token": 0.000001,
                "output_per_token": 0.000002,
                "cache_read_multiplier": 0.1,
            }
        }

        recomputed = recompute_rows(rows, pricing_by_agent)

        for row in recomputed:
            if row.get("cached_read_tokens", 0) > 0:
                # Should have cache_read_rate set
                self.assertIsNotNone(row.get("cache_read_rate"))
                self.assertGreater(row["cache_read_rate"], 0)

    def test_recompute_rows_baseline_cost(self) -> None:
        """Verify baseline cost propagates to non-baseline runs."""
        rows = load_fixture("runs_sample.jsonl")
        pricing_by_agent = {"codex": {}}

        recomputed = recompute_rows(rows, pricing_by_agent)

        # Find non-baseline runs and verify they have baseline_cost_usd set
        for row in recomputed:
            if not row.get("baseline_run"):
                # shell_unprimed scenario has baseline, so should have baseline_cost
                if row["scenario"] == "shell_unprimed":
                    self.assertIsNotNone(row.get("baseline_cost_usd"))


class TestAgentStubbing(unittest.TestCase):
    """Test report generation with stubbed agent."""

    def _make_suite_config(self, **kwargs) -> SuiteConfig:
        """Create a SuiteConfig with default test values."""
        defaults = {
            "output_root": "/tmp/test_output",
            "timeout_sec": 300,
            "trials_per_cell": 1,
            "random_seed": 42,
            "run_baselines": False,
            "keep_workdirs": False,
            "post_run_report": True,
        }
        defaults.update(kwargs)
        return SuiteConfig(**defaults)

    def _make_agent_config(self, **kwargs) -> AgentConfig:
        """Create an AgentConfig with default test values."""
        defaults = {
            "id": "test-agent",
            "kind": "claude_cli",
            "command": ["echo", "test"],
            "env": {},
            "pricing": {},
        }
        defaults.update(kwargs)
        return AgentConfig(**defaults)

    def test_failure_analysis_report_with_stubbed_agent(self) -> None:
        """Verify failure analysis runs with mocked agent."""
        from evals.reporting import run_failure_analysis_report

        rows = load_fixture("runs_sample.jsonl")
        suite = self._make_suite_config(
            post_run_report=True,
            post_run_report_agent="test-agent",
        )
        agents = [self._make_agent_config()]

        # Mock the agent adapter
        mock_adapter = mock.MagicMock()
        mock_adapter.run.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)

            with mock.patch("evals.reporting.make_agent", return_value=mock_adapter):
                result = run_failure_analysis_report(suite, agents, out_dir, rows)

            # Should have called make_agent and adapter.run
            mock_adapter.run.assert_called_once()

            # Manifest should be created
            manifest_path = out_dir / "failure_analysis_manifest.json"
            self.assertTrue(manifest_path.exists())

    def test_tool_error_report_with_stubbed_agent(self) -> None:
        """Verify tool error report runs with mocked agent."""
        from evals.reporting import run_post_run_report

        rows = load_fixture("runs_sample.jsonl")
        suite = self._make_suite_config(
            post_run_report=True,
            post_run_report_agent="test-agent",
        )
        agents = [self._make_agent_config()]

        mock_adapter = mock.MagicMock()
        mock_adapter.run.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)

            with mock.patch("evals.reporting.make_agent", return_value=mock_adapter):
                result = run_post_run_report(suite, agents, out_dir, rows)

            # Should have called adapter.run
            mock_adapter.run.assert_called_once()

            # Manifest should be created
            manifest_path = out_dir / "tool_error_report_manifest.json"
            self.assertTrue(manifest_path.exists())

    def test_report_skipped_when_disabled(self) -> None:
        """Verify reports are skipped when post_run_report=False."""
        from evals.reporting import run_failure_analysis_report, run_post_run_report

        rows = load_fixture("runs_sample.jsonl")
        suite = self._make_suite_config(post_run_report=False)
        agents = [self._make_agent_config()]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir)
            result1 = run_failure_analysis_report(suite, agents, out_dir, rows)
            result2 = run_post_run_report(suite, agents, out_dir, rows)

        self.assertIsNone(result1)
        self.assertIsNone(result2)


class TestResultsToRows(unittest.TestCase):
    """Test TrialResult to row conversion - critical for accurate output."""

    def test_results_to_rows_basic_conversion(self) -> None:
        """Verify results_to_rows converts TrialResult correctly."""
        from evals.trial import TrialResult
        from evals.reporting import results_to_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create command log file
            cmd_log_path = pathlib.Path(tmpdir) / "cmd_log.jsonl"
            cmd_log_path.write_text(
                '{"ts": "2026-01-20T10:00:05Z", "cmd": "xcodebuild", "argv": ["-scheme", "App", "-destination", "id=ABC", "build"], "source": "agent"}\n'
                '{"ts": "2026-01-20T10:00:10Z", "cmd": "xcrun", "argv": ["simctl", "install", "ABC", "app.app"], "source": "agent"}\n'
                '{"ts": "2026-01-20T10:00:15Z", "cmd": "xcrun", "argv": ["simctl", "launch", "ABC", "com.app"], "source": "agent"}\n'
            )

            trial = TrialResult(
                run_id="test-run-123",
                ts_start="2026-01-20T10:00:00+00:00",
                ts_end="2026-01-20T10:01:00+00:00",
                agent_id="test-agent",
                agent_kind="claude_cli",
                scenario="shell_unprimed",
                task_id="test_task",
                task_kind="capability",
                baseline_run=False,
                success=True,
                failure_reason=None,
                grader_results=[{"type": "ios_test_pass", "ok": True}],
                exit_code=0,
                wall_time_sec=60.0,
                model="claude-3",
                provider_usage=None,
                uncached_input_tokens=1000,
                output_tokens=500,
                cached_read_tokens=2000,
                cache_write_tokens=0,
                cache_write_ttl=None,
                billed_cost_usd=0.05,
                cost_source="api",
                cold_equivalent_cost_usd=0.08,
                cache_savings_usd=0.03,
                cache_read_rate=0.67,
                baseline_cost_usd=0.01,
                marginal_cost_usd=0.04,
                command_invocations={"xcodebuild": 1, "xcrun": 2},
                mcp_tool_invocations=None,
                tool_error_total=0,
                tool_error_mcp=0,
                tool_error_non_mcp=0,
                workdir=tmpdir,
                transcript_path=str(pathlib.Path(tmpdir) / "transcript.txt"),
                agent_output_json_path=None,
                command_log_path=str(cmd_log_path),
                mcp_tool_log_path=None,
                tool_error_log_path=None,
                tool_error_context_log_path=None,
            )

            rows = results_to_rows([trial])

            self.assertEqual(len(rows), 1)
            row = rows[0]

            # Verify basic fields
            self.assertEqual(row["run_id"], "test-run-123")
            self.assertEqual(row["agent_id"], "test-agent")
            self.assertEqual(row["scenario"], "shell_unprimed")
            self.assertEqual(row["success"], True)
            self.assertEqual(row["wall_time_sec"], 60.0)

            # Verify computed metrics
            self.assertEqual(row["xcodebuild_calls"], 1)
            self.assertEqual(row["xcrun_calls"], 2)
            self.assertEqual(row["simctl_calls"], 2)  # simctl install + launch
            self.assertEqual(row["destination_count"], 1)  # id=ABC
            self.assertEqual(row["destination_churn"], 0)  # 1 destination = 0 churn

            # Verify time_to_first_xcodebuild (5 seconds from start)
            self.assertAlmostEqual(row["time_to_first_xcodebuild_sec"], 5.0, places=1)

    def test_results_to_rows_destination_churn(self) -> None:
        """Verify destination churn is computed correctly."""
        from evals.trial import TrialResult
        from evals.reporting import results_to_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create command log with multiple different destinations
            cmd_log_path = pathlib.Path(tmpdir) / "cmd_log.jsonl"
            cmd_log_path.write_text(
                '{"ts": "2026-01-20T10:00:05Z", "cmd": "xcodebuild", "argv": ["-destination", "id=AAA", "build"], "source": "agent"}\n'
                '{"ts": "2026-01-20T10:00:10Z", "cmd": "xcodebuild", "argv": ["-destination", "id=BBB", "build"], "source": "agent"}\n'
                '{"ts": "2026-01-20T10:00:15Z", "cmd": "xcodebuild", "argv": ["-destination", "id=CCC", "build"], "source": "agent"}\n'
            )

            trial = TrialResult(
                run_id="churn-test",
                ts_start="2026-01-20T10:00:00+00:00",
                ts_end="2026-01-20T10:01:00+00:00",
                agent_id="test-agent",
                agent_kind="claude_cli",
                scenario="shell_unprimed",
                task_id="test_task",
                task_kind="capability",
                baseline_run=False,
                success=True,
                failure_reason=None,
                grader_results=None,
                exit_code=0,
                wall_time_sec=60.0,
                model=None,
                provider_usage=None,
                uncached_input_tokens=None,
                output_tokens=None,
                cached_read_tokens=None,
                cache_write_tokens=None,
                cache_write_ttl=None,
                billed_cost_usd=None,
                cost_source=None,
                cold_equivalent_cost_usd=None,
                cache_savings_usd=None,
                cache_read_rate=None,
                baseline_cost_usd=None,
                marginal_cost_usd=None,
                command_invocations={"xcodebuild": 3},
                mcp_tool_invocations=None,
                tool_error_total=0,
                tool_error_mcp=0,
                tool_error_non_mcp=0,
                workdir=tmpdir,
                transcript_path=str(pathlib.Path(tmpdir) / "transcript.txt"),
                agent_output_json_path=None,
                command_log_path=str(cmd_log_path),
                mcp_tool_log_path=None,
                tool_error_log_path=None,
                tool_error_context_log_path=None,
            )

            rows = results_to_rows([trial])
            row = rows[0]

            # 3 distinct destinations = churn of 2
            self.assertEqual(row["destination_count"], 3)
            self.assertEqual(row["destination_churn"], 2)

    def test_results_to_rows_mcp_metrics(self) -> None:
        """Verify MCP tool metrics are computed correctly."""
        from evals.trial import TrialResult
        from evals.reporting import results_to_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty command log
            cmd_log_path = pathlib.Path(tmpdir) / "cmd_log.jsonl"
            cmd_log_path.write_text("")

            # Create MCP tool log - use correct tool names from SIMCTL_MCP_TOOLS
            mcp_log_path = pathlib.Path(tmpdir) / "mcp_tool_calls.jsonl"
            mcp_log_path.write_text(
                '{"ts": "2026-01-20T10:00:30Z", "tool": "build_sim", "arguments": {}}\n'
                '{"ts": "2026-01-20T10:00:45Z", "tool": "install_sim", "arguments": {}}\n'
                '{"ts": "2026-01-20T10:00:50Z", "tool": "launch_app_sim", "arguments": {}}\n'
            )

            trial = TrialResult(
                run_id="mcp-test",
                ts_start="2026-01-20T10:00:00+00:00",
                ts_end="2026-01-20T10:01:00+00:00",
                agent_id="test-agent",
                agent_kind="claude_cli",
                scenario="mcp_unprimed",
                task_id="test_task",
                task_kind="capability",
                baseline_run=False,
                success=True,
                failure_reason=None,
                grader_results=None,
                exit_code=0,
                wall_time_sec=60.0,
                model=None,
                provider_usage=None,
                uncached_input_tokens=None,
                output_tokens=None,
                cached_read_tokens=None,
                cache_write_tokens=None,
                cache_write_ttl=None,
                billed_cost_usd=None,
                cost_source=None,
                cold_equivalent_cost_usd=None,
                cache_savings_usd=None,
                cache_read_rate=None,
                baseline_cost_usd=None,
                marginal_cost_usd=None,
                command_invocations={},
                mcp_tool_invocations=None,  # Should be computed from log
                tool_error_total=0,
                tool_error_mcp=0,
                tool_error_non_mcp=0,
                workdir=tmpdir,
                transcript_path=str(pathlib.Path(tmpdir) / "transcript.txt"),
                agent_output_json_path=None,
                command_log_path=str(cmd_log_path),
                mcp_tool_log_path=str(mcp_log_path),
                tool_error_log_path=None,
                tool_error_context_log_path=None,
            )

            rows = results_to_rows([trial])
            row = rows[0]

            # Should count MCP tools
            self.assertEqual(row["mcp_tool_calls"], 3)
            # build_sim is xcodebuild-like
            self.assertGreaterEqual(row["mcp_xcodebuild_calls"], 1)
            # install_app_sim, launch_app_sim are simctl-like
            self.assertGreaterEqual(row["mcp_simctl_calls"], 2)
            # time_to_first_mcp_build should be 30 seconds
            self.assertAlmostEqual(row["time_to_first_mcp_build_sec"], 30.0, places=1)


class TestCountInvocations(unittest.TestCase):
    """Test command invocation counting."""

    def test_count_invocations_basic(self) -> None:
        """Verify basic command counting."""
        from evals.metrics import count_invocations

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"cmd": "xcodebuild", "argv": ["-list"], "source": "agent"}\n')
            f.write('{"cmd": "xcodebuild", "argv": ["build"], "source": "agent"}\n')
            f.write('{"cmd": "xcrun", "argv": ["simctl", "list"], "source": "agent"}\n')
            log_path = pathlib.Path(f.name)

        try:
            counts = count_invocations(log_path)
            self.assertEqual(counts["xcodebuild"], 2)
            self.assertEqual(counts["xcrun"], 1)
        finally:
            log_path.unlink()

    def test_count_invocations_source_filter(self) -> None:
        """Verify source filtering works."""
        from evals.metrics import count_invocations

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"cmd": "xcodebuild", "argv": [], "source": "agent"}\n')
            f.write('{"cmd": "xcodebuild", "argv": [], "source": "harness"}\n')
            f.write('{"cmd": "xcodebuild", "argv": [], "source": "agent"}\n')
            log_path = pathlib.Path(f.name)

        try:
            # Count only agent commands
            counts = count_invocations(log_path, source="agent")
            self.assertEqual(counts["xcodebuild"], 2)

            # Count all
            counts_all = count_invocations(log_path)
            self.assertEqual(counts_all["xcodebuild"], 3)
        finally:
            log_path.unlink()

    def test_count_invocations_nonexistent_file(self) -> None:
        """Verify empty dict returned for missing file."""
        from evals.metrics import count_invocations

        counts = count_invocations(pathlib.Path("/nonexistent/path.jsonl"))
        self.assertEqual(counts, {})

    def test_count_invocations_malformed_json(self) -> None:
        """Verify malformed lines are skipped."""
        from evals.metrics import count_invocations

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"cmd": "xcodebuild", "argv": []}\n')
            f.write('not valid json\n')
            f.write('{"cmd": "xcrun", "argv": []}\n')
            log_path = pathlib.Path(f.name)

        try:
            counts = count_invocations(log_path)
            self.assertEqual(counts["xcodebuild"], 1)
            self.assertEqual(counts["xcrun"], 1)
        finally:
            log_path.unlink()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in metrics computation."""

    def test_aggregate_with_mixed_success_rates(self) -> None:
        """Verify aggregation with partial success rates."""
        rows = [
            {"agent_id": "a", "scenario": "s", "task_id": "t", "task_kind": "cap",
             "baseline_run": False, "success": True, "wall_time_sec": 100.0,
             "billed_cost_usd": 0.1, "mcp_tool_calls": 5, "tool_error_total": 0,
             "xcodebuild_calls": 1, "xcrun_calls": 2, "simctl_calls": 2},
            {"agent_id": "a", "scenario": "s", "task_id": "t", "task_kind": "cap",
             "baseline_run": False, "success": False, "wall_time_sec": 200.0,
             "billed_cost_usd": 0.2, "mcp_tool_calls": 3, "tool_error_total": 5,
             "xcodebuild_calls": 2, "xcrun_calls": 1, "simctl_calls": 1},
            {"agent_id": "a", "scenario": "s", "task_id": "t", "task_kind": "cap",
             "baseline_run": False, "success": True, "wall_time_sec": 150.0,
             "billed_cost_usd": 0.15, "mcp_tool_calls": 4, "tool_error_total": 1,
             "xcodebuild_calls": 1, "xcrun_calls": 2, "simctl_calls": 2},
        ]

        summary = aggregate(rows)
        self.assertEqual(len(summary), 1)

        row = summary[0]
        # 2 out of 3 succeeded
        self.assertAlmostEqual(row["success_rate"], 2/3, places=5)
        self.assertAlmostEqual(row["pass_at_1"], 2/3, places=5)
        # pass@3 = 1 - (1 - 2/3)^3 = 1 - (1/3)^3 = 1 - 1/27 = 26/27
        self.assertAlmostEqual(row["pass_at_3"], 26/27, places=5)

        # cost_per_success = total_cost / successes = 0.45 / 2 = 0.225
        self.assertAlmostEqual(row["cost_per_success_mean"], 0.225, places=5)

        # tool_error_mean = (0 + 5 + 1) / 3 = 2
        self.assertAlmostEqual(row["tool_error_mean"], 2.0, places=5)

    def test_percentile_computation(self) -> None:
        """Verify percentile math with known values."""
        from evals.eval_reporting import percentile

        # Simple case
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(percentile(values, 50), 30.0)  # Median
        self.assertEqual(percentile(values, 0), 10.0)   # Min
        self.assertEqual(percentile(values, 100), 50.0) # Max

        # Empty list
        self.assertIsNone(percentile([], 50))

        # Single value
        self.assertEqual(percentile([42.0], 10), 42.0)
        self.assertEqual(percentile([42.0], 90), 42.0)

    def test_cv_computation(self) -> None:
        """Verify coefficient of variation computation."""
        from evals.eval_reporting import cv, mean, stdev

        values = [100.0, 100.0, 100.0]
        # CV of identical values should be 0
        self.assertEqual(cv(values), 0.0)

        # CV = stdev / mean
        values2 = [10.0, 20.0, 30.0]
        m = mean(values2)
        s = stdev(values2)
        assert m is not None
        assert s is not None
        cv_val = cv(values2)
        assert cv_val is not None
        self.assertAlmostEqual(cv_val, s / m, places=10)


if __name__ == "__main__":
    unittest.main()
