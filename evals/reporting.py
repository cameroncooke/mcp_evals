"""
Reporting utilities for the eval suite.

Provides functions for JSONL I/O, results aggregation, and post-run reports.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from evals.infrastructure import run_cmd, safe_mkdir, now_ts
from evals.metrics import (
    read_command_log_entries,
    extract_xcodebuild_destination,
    compute_time_to_first_xcodebuild_sec,
    compute_xcodebuild_repeat_count,
    count_simctl_invocations,
    count_mcp_tool_usage,
    compute_time_to_first_mcp_build_sec,
)
from evals.agents import make_agent

if TYPE_CHECKING:
    from evals.config import AgentConfig, SuiteConfig
    from evals.trial import TrialResult


def load_prompt(name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Args:
        name: Name of the prompt file (without extension)

    Returns:
        Content of the prompt file
    """
    prompt_dir = pathlib.Path(__file__).parent / "prompts"
    return (prompt_dir / f"{name}.md").read_text()


def results_to_rows(
    results: List["TrialResult"], include_provider_usage: bool = False
) -> List[Dict[str, Any]]:
    """Convert TrialResult objects to row dictionaries for output."""
    rows: List[Dict[str, Any]] = []
    for r in results:
        entries = read_command_log_entries(
            pathlib.Path(r.command_log_path), source="agent"
        )
        destinations: List[str] = []
        for entry in entries:
            if entry.get("cmd") != "xcodebuild":
                continue
            argv = [str(x) for x in (entry.get("argv") or [])]
            dest = extract_xcodebuild_destination(argv)
            if dest:
                destinations.append(dest)
        distinct_destinations = len(set(destinations))
        row = {
            "run_id": r.run_id,
            "ts_start": r.ts_start,
            "ts_end": r.ts_end,
            "agent_id": r.agent_id,
            "agent_kind": r.agent_kind,
            "scenario": r.scenario,
            "task_id": r.task_id,
            "task_kind": r.task_kind,
            "baseline_run": r.baseline_run,
            "success": r.success,
            "failure_reason": r.failure_reason,
            "grader_results": r.grader_results,
            "exit_code": r.exit_code,
            "wall_time_sec": r.wall_time_sec,
            "model": r.model,
            "uncached_input_tokens": r.uncached_input_tokens,
            "cached_read_tokens": r.cached_read_tokens,
            "cache_write_tokens": r.cache_write_tokens,
            "cache_write_ttl": r.cache_write_ttl,
            "output_tokens": r.output_tokens,
            "billed_cost_usd": r.billed_cost_usd,
            "cost_source": r.cost_source,
            "cold_equivalent_cost_usd": r.cold_equivalent_cost_usd,
            "cache_savings_usd": r.cache_savings_usd,
            "cache_read_rate": r.cache_read_rate,
            "baseline_cost_usd": r.baseline_cost_usd,
            "marginal_cost_usd": r.marginal_cost_usd,
            "xcodebuild_calls": r.command_invocations.get("xcodebuild", 0),
            "xcrun_calls": r.command_invocations.get("xcrun", 0),
            "simctl_calls": count_simctl_invocations(
                pathlib.Path(r.command_log_path), source="agent"
            ),
            "time_to_first_xcodebuild_sec": compute_time_to_first_xcodebuild_sec(
                r.ts_start, entries
            ),
            "xcodebuild_repeat_count": compute_xcodebuild_repeat_count(entries),
            "destination_count": distinct_destinations,
            "destination_churn": max(0, distinct_destinations - 1),
            "mcp_tool_calls": r.mcp_tool_invocations,
            "tool_error_total": r.tool_error_total,
            "tool_error_mcp": r.tool_error_mcp,
            "tool_error_non_mcp": r.tool_error_non_mcp,
            "transcript_path": r.transcript_path,
            "agent_output_json_path": r.agent_output_json_path,
            "command_log_path": r.command_log_path,
            "mcp_tool_log_path": r.mcp_tool_log_path,
            "tool_error_log_path": r.tool_error_log_path,
            "tool_error_context_log_path": r.tool_error_context_log_path,
        }
        mcp_counts = count_mcp_tool_usage(
            pathlib.Path(r.mcp_tool_log_path) if r.mcp_tool_log_path else None
        )
        row["mcp_xcodebuild_calls"] = mcp_counts["xcodebuild"]
        row["mcp_simctl_calls"] = mcp_counts["simctl"]
        row["mcp_tool_calls"] = mcp_counts["total"] if row["mcp_tool_calls"] is None else row["mcp_tool_calls"]
        mcp_log_path = pathlib.Path(r.mcp_tool_log_path) if r.mcp_tool_log_path else None
        row["time_to_first_mcp_build_sec"] = compute_time_to_first_mcp_build_sec(
            r.ts_start, mcp_log_path
        )
        if include_provider_usage:
            row["provider_usage"] = r.provider_usage
        rows.append(row)
    return rows


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: pathlib.Path, row: Dict[str, Any]) -> None:
    """Append a single row to a JSONL file (creates if doesn't exist)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load all rows from a JSONL file."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_completed_trial_keys(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], int]:
    """
    Extract keys for completed trials from existing runs.

    Returns dict mapping (agent_id, scenario, task_id) to trial count.
    """
    from collections import defaultdict
    trial_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in rows:
        key = (r["agent_id"], r["scenario"], r["task_id"])
        trial_counts[key] += 1
    return dict(trial_counts)


def select_post_run_agent(
    suite: "SuiteConfig", agents_cfgs: List["AgentConfig"]
) -> Optional["AgentConfig"]:
    """Select an agent to use for post-run reports."""
    if not agents_cfgs:
        return None
    if suite.post_run_report_agent:
        for a in agents_cfgs:
            if a.id == suite.post_run_report_agent:
                return a
        return None
    for a in agents_cfgs:
        if a.id == "codex":
            return a
    return agents_cfgs[0]


def prepare_report_env(
    suite: "SuiteConfig", agent_cfg: "AgentConfig", out_dir: pathlib.Path
) -> Dict[str, str]:
    """Prepare environment for running a report agent."""
    env = dict(os.environ)
    env.update(suite.env or {})
    env.update(agent_cfg.env or {})
    env["EVAL_AGENT_EXTRA_ARGS"] = ""
    if suite.clean_agent_env:
        clean_root = out_dir / "report_agent_env"
        safe_mkdir(clean_root)
        if agent_cfg.kind == "codex_cli":
            codex_home = clean_root / "codex_home"
            safe_mkdir(codex_home)
            env["CODEX_HOME"] = str(codex_home)
            auth_src = pathlib.Path.home() / ".codex" / "auth.json"
            auth_dst = codex_home / "auth.json"
            if auth_src.exists():
                try:
                    shutil.copy2(auth_src, auth_dst)
                except Exception:
                    pass
    return env


SESSION_DEFAULTS_DISCOVERY_PATTERNS = [
    "Missing required session defaults",
]

SIBLING_CASCADE_PATTERN = "Sibling tool call errored"

# Patterns indicating the tool worked correctly but reported build/test failure
# These are compile errors or test failures during iterative development - not tool bugs
BUILD_TEST_FAILURE_INDICATORS = [
    "Testing failed:",
    "tests failed",
    "Build failed",
    # Swift/clang compile errors (file path + line number pattern)
    ".swift:",  # e.g., "HNApi.swift:63:34: error:"
]

# Patterns that look like build errors but are actually agent mistakes (wrong params)
# These should NOT be excluded from real errors
BUILD_TEST_EXCLUSIONS = [
    "Unable to find a device",  # Wrong simulator name
    "simulator not found",
    "No such file or directory",  # Wrong path
]


def _parse_transcript_tool_results(transcript_path: Optional[str]) -> Dict[str, str]:
    """Parse transcript to extract tool result content by tool name.

    Returns a dict mapping tool_name to the full result content string.
    For MCP tools with status=failed, extracts result.content[].text.
    For bash tools, extracts the output field.
    """
    results: Dict[str, str] = {}
    if not transcript_path:
        return results
    path = pathlib.Path(transcript_path)
    if not path.exists():
        return results

    try:
        current_tool: Optional[str] = None
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                # Track tool calls to know which tool the result belongs to
                if line.startswith("TOOL_CALL "):
                    parts = line.split(" ", 2)
                    if len(parts) >= 2:
                        current_tool = parts[1]
                # Extract tool results
                elif line.startswith("TOOL_RESULT ") and current_tool:
                    result_str = line[len("TOOL_RESULT "):]
                    try:
                        result = json.loads(result_str)
                        content_text = ""

                        # Handle MCP tool results
                        if isinstance(result.get("result"), dict):
                            mcp_result = result["result"]
                            content = mcp_result.get("content")
                            if isinstance(content, list):
                                texts = [
                                    c.get("text", "")
                                    for c in content
                                    if isinstance(c, dict)
                                ]
                                content_text = "\n".join(t for t in texts if t)

                        # Handle bash tool results
                        elif "output" in result:
                            content_text = str(result.get("output", ""))

                        # Store with tool name as key (may overwrite if same tool called multiple times)
                        # For classification, we just need the content patterns
                        if content_text and current_tool:
                            # Append to existing content for this tool
                            if current_tool in results:
                                results[current_tool] += "\n" + content_text
                            else:
                                results[current_tool] = content_text
                    except json.JSONDecodeError:
                        pass
                    current_tool = None
    except Exception:
        pass
    return results


def _classify_tool_errors(
    tool_error_log_path: Optional[str],
    transcript_path: Optional[str] = None,
) -> Dict[str, int]:
    """Classify tool errors into categories for fair comparison.

    Returns counts for:
    - session_defaults: All session-defaults discovery errors (expected MCP workflow)
    - sibling_cascade: Errors caused by sibling tool call failing (cascade, not agent mistake)
    - build_test_failure: Tool worked correctly but reported build/test failure (expected iteration)
    - real_errors: Actual tool errors that aren't in above categories

    If transcript_path is provided, will look up actual error content from transcript
    when the tool_errors.jsonl payload is missing content (e.g., Codex MCP errors).
    """
    counts = {
        "session_defaults": 0,
        "sibling_cascade": 0,
        "build_test_failure": 0,
        "real_errors": 0,
        "total": 0,
    }
    if not tool_error_log_path:
        return counts
    log_path = pathlib.Path(tool_error_log_path)
    if not log_path.exists():
        return counts

    # Parse transcript for actual content (used when tool_errors.jsonl is missing it)
    transcript_content = _parse_transcript_tool_results(transcript_path)

    try:
        with open(log_path, "r") as f:
            for line in f:
                try:
                    err = json.loads(line)
                    payload = str(err.get("payload", ""))
                    tool_kind = err.get("tool_kind", "")
                    tool_name = err.get("tool_name", "")
                    counts["total"] += 1

                    # Check if payload is missing actual content (Codex bug)
                    # These look like: {"error": null, "status": "failed"}
                    if '{"error":' in payload and '"status":' in payload:
                        try:
                            payload_obj = json.loads(payload)
                            # If it's just error/status without message, try transcript
                            if (
                                payload_obj.get("error") is None
                                and payload_obj.get("status") == "failed"
                                and "message" not in payload_obj
                                and tool_name in transcript_content
                            ):
                                payload = transcript_content[tool_name]
                        except json.JSONDecodeError:
                            pass

                    # Check for session-defaults discovery (MCP only)
                    if tool_kind == "mcp" and any(
                        p in payload for p in SESSION_DEFAULTS_DISCOVERY_PATTERNS
                    ):
                        counts["session_defaults"] += 1
                    # Check for sibling cascade
                    elif SIBLING_CASCADE_PATTERN in payload:
                        counts["sibling_cascade"] += 1
                    # Check for build/test failures with useful content
                    # These are status=failed but the tool worked correctly
                    # Exclude patterns that indicate agent mistakes (wrong params)
                    elif tool_kind == "mcp" and any(
                        indicator in payload for indicator in BUILD_TEST_FAILURE_INDICATORS
                    ) and not any(
                        excl in payload for excl in BUILD_TEST_EXCLUSIONS
                    ):
                        counts["build_test_failure"] += 1
                    else:
                        counts["real_errors"] += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return counts


def _count_discovery_errors(tool_error_log_path: Optional[str]) -> int:
    """Count ALL session-defaults discovery errors in a tool error log.

    These are expected MCP workflow - agent must discover project/workspace
    before using most tools. Not agent mistakes.
    """
    if not tool_error_log_path:
        return 0
    log_path = pathlib.Path(tool_error_log_path)
    if not log_path.exists():
        return 0
    count = 0
    try:
        with open(log_path, "r") as f:
            for line in f:
                try:
                    err = json.loads(line)
                    if err.get("tool_kind") != "mcp":
                        continue
                    payload = err.get("payload", "")
                    if any(p in payload for p in SESSION_DEFAULTS_DISCOVERY_PATTERNS):
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


def build_tool_error_report_manifest(
    rows: List[Dict[str, Any]], out_dir: pathlib.Path
) -> pathlib.Path:
    """Build a manifest of runs with tool errors for analysis."""
    # Filter out baseline runs for stats
    non_baseline_rows = [r for r in rows if not r.get("baseline_run")]

    # Count totals by agent
    total_by_agent: Dict[str, int] = {}
    errors_by_agent: Dict[str, int] = {}
    real_errors_by_agent: Dict[str, int] = {}
    # Count errors by scenario with MCP/non-MCP breakdown
    errors_by_scenario: Dict[str, Dict[str, int]] = {}
    # Track error classifications for adjustment
    total_classification = {
        "session_defaults": 0,
        "sibling_cascade": 0,
        "build_test_failure": 0,
        "real_errors": 0,
    }
    # Track per-agent adjusted errors
    adjusted_by_agent: Dict[str, Dict[str, int]] = {}

    for r in non_baseline_rows:
        agent = r.get("agent_id", "unknown")
        scenario = r.get("scenario", "unknown")
        total_by_agent[agent] = total_by_agent.get(agent, 0) + 1
        try:
            err_total = int(r.get("tool_error_total") or 0)
            err_mcp = int(r.get("tool_error_mcp") or 0)
            err_non_mcp = int(r.get("tool_error_non_mcp") or 0)
        except Exception:
            err_total = err_mcp = err_non_mcp = 0
        if err_total > 0:
            errors_by_agent[agent] = errors_by_agent.get(agent, 0) + 1

        # Classify errors for this trial
        classification = _classify_tool_errors(
            r.get("tool_error_log_path"), r.get("transcript_path")
        )
        for key in ["session_defaults", "sibling_cascade", "build_test_failure", "real_errors"]:
            total_classification[key] += classification[key]

        # Track per-agent real errors
        if agent not in adjusted_by_agent:
            adjusted_by_agent[agent] = {"trials": 0, "real_errors": 0}
        adjusted_by_agent[agent]["trials"] += 1
        adjusted_by_agent[agent]["real_errors"] += classification["real_errors"]
        if classification["real_errors"] > 0:
            real_errors_by_agent[agent] = real_errors_by_agent.get(agent, 0) + 1

        # Accumulate scenario stats
        if scenario not in errors_by_scenario:
            errors_by_scenario[scenario] = {
                "total": 0,
                "mcp": 0,
                "non_mcp": 0,
                "session_defaults": 0,
                "sibling_cascade": 0,
                "build_test_failure": 0,
                "real_errors": 0,
            }
        errors_by_scenario[scenario]["total"] += err_total
        errors_by_scenario[scenario]["mcp"] += err_mcp
        errors_by_scenario[scenario]["non_mcp"] += err_non_mcp
        errors_by_scenario[scenario]["session_defaults"] += classification["session_defaults"]
        errors_by_scenario[scenario]["sibling_cascade"] += classification["sibling_cascade"]
        errors_by_scenario[scenario]["build_test_failure"] += classification["build_test_failure"]
        errors_by_scenario[scenario]["real_errors"] += classification["real_errors"]

    entries: List[Dict[str, Any]] = []
    for r in rows:
        try:
            total = int(r.get("tool_error_total") or 0)
        except Exception:
            total = 0
        if total <= 0:
            continue
        # Add classification to each entry
        classification = _classify_tool_errors(
            r.get("tool_error_log_path"), r.get("transcript_path")
        )
        entries.append(
            {
                "run_id": r.get("run_id"),
                "agent_id": r.get("agent_id"),
                "scenario": r.get("scenario"),
                "task_id": r.get("task_id"),
                "baseline_run": r.get("baseline_run"),
                "tool_error_total": r.get("tool_error_total"),
                "tool_error_mcp": r.get("tool_error_mcp"),
                "tool_error_non_mcp": r.get("tool_error_non_mcp"),
                "session_defaults_errors": classification["session_defaults"],
                "sibling_cascade_errors": classification["sibling_cascade"],
                "build_test_failure_errors": classification["build_test_failure"],
                "real_errors": classification["real_errors"],
                "tool_error_log_path": r.get("tool_error_log_path"),
                "tool_error_context_log_path": r.get("tool_error_context_log_path"),
                "transcript_path": r.get("transcript_path"),
                "command_log_path": r.get("command_log_path"),
                "mcp_tool_log_path": r.get("mcp_tool_log_path"),
            }
        )

    total_runs = len(non_baseline_rows)
    runs_with_errors = len([e for e in entries if not e.get("baseline_run")])
    runs_with_real_errors = len([
        e for e in entries
        if not e.get("baseline_run") and e.get("real_errors", 0) > 0
    ])

    # Compute totals
    total_errors = sum(s.get("total", 0) for s in errors_by_scenario.values())
    total_real_errors = total_classification["real_errors"]
    excluded_errors = (
        total_classification["session_defaults"]
        + total_classification["sibling_cascade"]
        + total_classification["build_test_failure"]
    )

    manifest = {
        "generated_at": now_ts(),
        "run_dir": str(out_dir),
        "runs_csv": str(out_dir / "runs.csv"),
        "summary_csv": str(out_dir / "summary.csv"),
        # Summary stats for context
        "total_runs": total_runs,
        "runs_with_errors": runs_with_errors,
        "runs_with_real_errors": runs_with_real_errors,
        "runs_without_errors": total_runs - runs_with_errors,
        "error_rate_percent": round(runs_with_errors / total_runs * 100, 1) if total_runs > 0 else 0,
        "real_error_rate_percent": round(runs_with_real_errors / total_runs * 100, 1) if total_runs > 0 else 0,
        "total_runs_by_agent": total_by_agent,
        "runs_with_errors_by_agent": errors_by_agent,
        "runs_with_real_errors_by_agent": real_errors_by_agent,
        "errors_by_scenario": errors_by_scenario,
        # Error classification breakdown
        "error_classification": {
            "total_errors": total_errors,
            "session_defaults": total_classification["session_defaults"],
            "sibling_cascade": total_classification["sibling_cascade"],
            "build_test_failure": total_classification["build_test_failure"],
            "real_errors": total_real_errors,
            "excluded_errors": excluded_errors,
            "exclusion_note": (
                "session_defaults: Expected MCP discovery workflow. "
                "sibling_cascade: Caused by sibling tool failure. "
                "build_test_failure: Tool worked correctly, reported build/test error."
            ),
        },
        # Per-agent adjusted stats
        "adjusted_by_agent": adjusted_by_agent,
        # Legacy fields for compatibility
        "mcp_discovery_errors": total_classification["session_defaults"],
        "total_mcp_errors": sum(s.get("mcp", 0) for s in errors_by_scenario.values()),
        "adjusted_mcp_errors": total_real_errors,
        "entries": entries,
    }
    path = out_dir / "tool_error_report_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path


def run_post_run_report(
    suite: "SuiteConfig",
    agents_cfgs: List["AgentConfig"],
    out_dir: pathlib.Path,
    rows: List[Dict[str, Any]],
) -> Optional[pathlib.Path]:
    """Run a post-run tool error report using an agent."""
    if not suite.post_run_report:
        return None
    agent_cfg = select_post_run_agent(suite, agents_cfgs)
    if agent_cfg is None:
        print("Post-run report skipped: no matching agent.", flush=True)
        return None
    manifest_path = build_tool_error_report_manifest(rows, out_dir)
    report_path = out_dir / (suite.post_run_report_path or "tool_error_report.md")
    transcript_path = out_dir / "tool_error_report.transcript.txt"
    out_json = out_dir / "tool_error_report.agent_output.json"

    # Load prompt from external file or use inline
    try:
        prompt_template = load_prompt("tool_error_report")
        prompt = prompt_template.format(
            manifest_path=manifest_path,
            report_path=report_path,
        )
    except Exception:
        # Fallback to inline prompt
        prompt = (
            "Generate a concise post-run report on tool errors.\n\n"
            f"Manifest: {manifest_path}\n"
            f"Write report to: {report_path}\n\n"
            "Requirements:\n"
            "- Summarize tool errors by agent/scenario.\n"
            "- For each error with context, note whether follow-up calls repeat the "
            "same command/tool or change parameters (possible agent correction).\n"
            "- Call out likely agent mistakes vs transient/environmental failures.\n"
            "- Do not run builds/tests; only read files.\n"
            "- Keep it short and actionable.\n"
            "Then exit.\n"
        )

    env = prepare_report_env(suite, agent_cfg, out_dir)
    adapter = make_agent(agent_cfg)
    adapter.run(
        prompt=prompt,
        workdir=out_dir,
        out_json=out_json,
        env=env,
        timeout_sec=min(600, max(120, suite.timeout_sec // 3)),
        stall_timeout_sec=120,
        transcript_path=transcript_path,
        transcript_mode="minimal",
        stream_output=False,
    )
    return report_path


def build_failure_analysis_manifest(
    rows: List[Dict[str, Any]], out_dir: pathlib.Path
) -> pathlib.Path:
    """Build a manifest of failed runs for analysis."""
    entries: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("baseline_run"):
            continue
        if r.get("success"):
            continue
        entries.append(
            {
                "run_id": r.get("run_id"),
                "agent_id": r.get("agent_id"),
                "scenario": r.get("scenario"),
                "task_id": r.get("task_id"),
                "success": r.get("success"),
                "failure_reason": r.get("failure_reason"),
                "exit_code": r.get("exit_code"),
                "wall_time_sec": r.get("wall_time_sec"),
                "grader_results": r.get("grader_results"),
                "tool_error_total": r.get("tool_error_total"),
                "tool_error_mcp": r.get("tool_error_mcp"),
                "tool_error_non_mcp": r.get("tool_error_non_mcp"),
                "transcript_path": r.get("transcript_path"),
                "tool_error_log_path": r.get("tool_error_log_path"),
                "tool_error_context_log_path": r.get("tool_error_context_log_path"),
                "command_log_path": r.get("command_log_path"),
                "mcp_tool_log_path": r.get("mcp_tool_log_path"),
            }
        )
    manifest = {
        "generated_at": now_ts(),
        "run_dir": str(out_dir),
        "total_failures": len(entries),
        "runs_jsonl": str(out_dir / "runs.jsonl"),
        "entries": entries,
    }
    path = out_dir / "failure_analysis_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path


def run_failure_analysis_report(
    suite: "SuiteConfig",
    agents_cfgs: List["AgentConfig"],
    out_dir: pathlib.Path,
    rows: List[Dict[str, Any]],
) -> Optional[pathlib.Path]:
    """Run an agent to analyze failed runs and classify root causes."""
    if not suite.post_run_report:
        return None

    failures = [r for r in rows if not r.get("baseline_run") and not r.get("success")]
    if not failures:
        print("Failure analysis skipped: no failures to analyze.", flush=True)
        return None

    agent_cfg = select_post_run_agent(suite, agents_cfgs)
    if agent_cfg is None:
        print("Failure analysis skipped: no matching agent.", flush=True)
        return None

    from collections import Counter
    print(f"\nFailure analysis: {len(failures)} failed runs", flush=True)
    by_reason = Counter(f.get("failure_reason") for f in failures)
    for reason, count in by_reason.most_common():
        print(f"  {reason}: {count}", flush=True)

    manifest_path = build_failure_analysis_manifest(rows, out_dir)
    report_path = out_dir / "failure_analysis_report.md"
    transcript_path = out_dir / "failure_analysis.transcript.txt"
    out_json = out_dir / "failure_analysis.agent_output.json"

    # Load prompt from external file
    try:
        prompt_template = load_prompt("failure_analysis")
        prompt = prompt_template.format(
            manifest_path=manifest_path,
            report_path=report_path,
        )
    except Exception:
        # Fallback to inline prompt if template not found
        prompt = f"""Analyze failed evaluation runs and produce a detailed failure analysis report.

## Manifest Location
{manifest_path}

## Output Report
Write to: {report_path}

## Instructions
- Read each transcript file listed in the manifest
- Classify each failure as AGENT_MISTAKE, ENVIRONMENTAL, TASK_ISSUE, or UNKNOWN
- Group similar failures into themes
- Include specific examples - code snippets, tool calls, error messages
- Do not run builds/tests; only read files

Then exit.
"""

    env = prepare_report_env(suite, agent_cfg, out_dir)
    adapter = make_agent(agent_cfg)
    print(f"Running failure analysis on {len(failures)} failed runs...", flush=True)
    adapter.run(
        prompt=prompt,
        workdir=out_dir,
        out_json=out_json,
        env=env,
        timeout_sec=min(900, max(180, suite.timeout_sec // 2)),
        stall_timeout_sec=180,
        transcript_path=transcript_path,
        transcript_mode="minimal",
        stream_output=False,
    )
    return report_path
