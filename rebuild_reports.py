#!/usr/bin/env python3
"""
Recompute derived metrics and summaries for an existing run directory.

Usage:
  python3 rebuild_reports.py --run runs/20260119_114416 --config config.yaml --in-place
  python3 rebuild_reports.py --run runs/20260119_114416 --config config.yaml --failure-analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from evals.eval_reporting import (
    aggregate,
    is_cold_row,
    recompute_rows,
    write_csv,
    write_markdown_summary,
)
from evals.agents import extract_mcp_error_payload, format_tool_result

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`"
    ) from e

from evals import (
    SuiteConfig,
    AgentConfig,
    run_failure_analysis_report,
    run_post_run_report,
)


def load_rows(run_dir: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = run_dir / "runs.jsonl"
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def backup(path: pathlib.Path) -> None:
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv_with_header(
    path: pathlib.Path, rows: List[Dict[str, Any]], header: List[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})


def load_pricing_by_agent(config_path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    config = yaml.safe_load(config_path.read_text()) or {}
    agents = config.get("agents") or []
    pricing_by_agent: Dict[str, Dict[str, float]] = {}
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("id")
        if not agent_id:
            continue
        pricing = agent.get("pricing") or {}
        if isinstance(pricing, dict):
            pricing_by_agent[str(agent_id)] = pricing
    return pricing_by_agent


def load_suite_and_agents(config_path: pathlib.Path):
    """Load suite config and agents from config file."""
    config = yaml.safe_load(config_path.read_text()) or {}
    suite = SuiteConfig(**config.get("suite", {}))
    agents = [AgentConfig(**a) for a in config.get("agents", [])]
    return suite, agents


def parse_transcript_mcp_results(
    transcript_path: pathlib.Path,
) -> Dict[str, Dict[str, Any]]:
    """Parse transcript to extract MCP tool results.

    Returns a dict mapping (tool_name, call_index) to the full result dict.
    call_index tracks multiple calls to the same tool.
    """
    results: Dict[str, Dict[str, Any]] = {}
    tool_call_counts: Dict[str, int] = {}

    if not transcript_path.exists():
        return results

    content = transcript_path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    current_tool: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Match TOOL_CALL for MCP tools
        if line.startswith("TOOL_CALL mcp__"):
            match = re.match(r"TOOL_CALL (mcp__\S+)", line)
            if match:
                current_tool = match.group(1)
                # Track call index for this tool
                if current_tool not in tool_call_counts:
                    tool_call_counts[current_tool] = 0
                tool_call_counts[current_tool] += 1

        # Match TOOL_RESULT and associate with previous tool
        elif line.startswith("TOOL_RESULT ") and current_tool:
            result_json = line[len("TOOL_RESULT ") :]
            try:
                result = json.loads(result_json)
                key = f"{current_tool}:{tool_call_counts[current_tool]}"
                results[key] = result
            except json.JSONDecodeError:
                pass
            current_tool = None

        i += 1

    return results


def backfill_tool_errors(run_dir: pathlib.Path) -> Tuple[int, int]:
    """Backfill tool_errors.jsonl files with corrected MCP error payloads.

    Parses transcripts to extract actual error content that was missing
    from the original tool_errors.jsonl files (Codex MCP errors).

    Returns (trials_updated, errors_fixed) counts.
    """
    trials_dir = run_dir / "trials"
    if not trials_dir.exists():
        return 0, 0

    trials_updated = 0
    errors_fixed = 0

    for trial_dir in sorted(trials_dir.iterdir()):
        if not trial_dir.is_dir():
            continue

        tool_errors_path = trial_dir / "tool_errors.jsonl"
        transcript_path = trial_dir / "transcript.txt"

        if not tool_errors_path.exists():
            continue

        # Check if this is a Codex trial (only Codex has the bug)
        if not trial_dir.name.startswith("codex-"):
            continue

        # Parse transcript for MCP results
        mcp_results = parse_transcript_mcp_results(transcript_path)
        if not mcp_results:
            continue

        # Read existing tool errors
        errors: List[Dict[str, Any]] = []
        with tool_errors_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    errors.append(json.loads(line))

        # Track MCP call indices per tool
        mcp_call_indices: Dict[str, int] = {}
        modified = False

        for error in errors:
            tool_name = error.get("tool_name", "")
            tool_kind = error.get("tool_kind", "")

            # Only fix MCP errors from Codex
            if tool_kind != "mcp" or not tool_name.startswith("mcp__"):
                continue

            # Check if payload needs fixing (missing message)
            payload_str = error.get("payload", "")
            if not isinstance(payload_str, str):
                continue

            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            # Only fix if it's the broken pattern: null error, failed status, no message
            if not (
                payload.get("error") is None
                and payload.get("status") == "failed"
                and "message" not in payload
            ):
                continue

            # Find corresponding transcript result
            if tool_name not in mcp_call_indices:
                mcp_call_indices[tool_name] = 0
            mcp_call_indices[tool_name] += 1
            key = f"{tool_name}:{mcp_call_indices[tool_name]}"

            transcript_result = mcp_results.get(key)
            if not transcript_result:
                continue

            # Use the shared function to extract proper payload
            fixed_payload = extract_mcp_error_payload(transcript_result)
            if fixed_payload and fixed_payload.get("message"):
                error["payload"] = format_tool_result(fixed_payload)
                modified = True
                errors_fixed += 1

        if modified:
            # Backup and rewrite
            backup(tool_errors_path)
            with tool_errors_path.open("w", encoding="utf-8") as f:
                for err in errors:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")

            # Update summary
            summary_path = trial_dir / "tool_error_summary.json"
            if summary_path.exists():
                # Summary counts don't change, just backup for consistency
                backup(summary_path)

            trials_updated += 1

    return trials_updated, errors_fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory (e.g., runs/20260119_114416)")
    ap.add_argument(
        "--config", default="config.yaml", help="Config used for the run (for pricing)"
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite runs.jsonl/runs.csv/summary.* in place (backups created)",
    )
    ap.add_argument(
        "--failure-analysis",
        action="store_true",
        help="Run failure analysis report on failed runs",
    )
    ap.add_argument(
        "--post-run-report",
        action="store_true",
        help="Run post-run tool error report",
    )
    ap.add_argument(
        "--agent",
        help="Override agent to use for analysis (default: from config post_run_report_agent)",
    )
    ap.add_argument(
        "--backfill-errors",
        action="store_true",
        help="Backfill tool_errors.jsonl with corrected MCP error payloads from transcripts",
    )
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    # load config to get pricing
    pricing_by_agent = load_pricing_by_agent(pathlib.Path(args.config))

    rows = load_rows(run_dir)
    rows = recompute_rows(rows, pricing_by_agent)

    if not args.in_place:
        print("Dry run complete. Use --in-place to write outputs.")
        return

    # Back up existing outputs
    for name in ["runs.jsonl", "runs.csv", "summary.csv", "summary.md", "summary_cold.csv", "summary_cold.md"]:
        backup(run_dir / name)

    # Write runs.jsonl
    write_jsonl(run_dir / "runs.jsonl", rows)

    # Rebuild runs.csv using existing header if present
    runs_csv = run_dir / "runs.csv"
    header: List[str] = []
    if runs_csv.exists():
        with runs_csv.open() as f:
            reader = csv.reader(f)
            header = next(reader, [])
    if not header:
        # fallback: use keys from first row
        header = list(rows[0].keys()) if rows else []
    write_csv_with_header(runs_csv, rows, header)

    # Summaries
    summary = aggregate(rows)
    write_csv(run_dir / "summary.csv", summary)
    write_markdown_summary(run_dir / "summary.md", summary)

    summary_cold = aggregate(rows, row_filter=is_cold_row)
    write_csv(run_dir / "summary_cold.csv", summary_cold)
    write_markdown_summary(run_dir / "summary_cold.md", summary_cold)

    print(f"Rebuilt reports in {run_dir}")

    # Backfill tool errors if requested
    if args.backfill_errors:
        trials_updated, errors_fixed = backfill_tool_errors(run_dir)
        print(f"Backfilled tool errors: {trials_updated} trials updated, {errors_fixed} errors fixed")

    # Run post-run reports if requested
    if args.failure_analysis or args.post_run_report:
        suite, agents = load_suite_and_agents(pathlib.Path(args.config))
        suite.post_run_report = True
        if args.agent:
            suite.post_run_report_agent = args.agent

        if args.post_run_report:
            report_path = run_post_run_report(suite, agents, run_dir, rows)
            if report_path:
                print(f"Tool error report: {report_path}")

        if args.failure_analysis:
            report_path = run_failure_analysis_report(suite, agents, run_dir, rows)
            if report_path:
                print(f"Failure analysis report: {report_path}")


if __name__ == "__main__":
    main()
