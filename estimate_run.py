#!/usr/bin/env python3
"""
Estimate cost, time, and tokens for a full evaluation run based on a previous run's data.
"""

import csv
import sys
from pathlib import Path


def load_runs(runs_csv_path: Path) -> list[dict]:
    """Load runs from CSV file."""
    with open(runs_csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def estimate_run(previous_run_dir: Path, target_trials: int = 30) -> dict:
    """
    Estimate cost, time, and tokens for a target number of trials
    based on data from a previous run.
    """
    runs_csv = previous_run_dir / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found in {previous_run_dir}")

    runs = load_runs(runs_csv)

    # Separate baselines from task trials
    baselines = [r for r in runs if r.get("baseline_run") == "True"]
    task_trials = [r for r in runs if r.get("baseline_run") == "False"]

    # Count unique cells (agent × scenario × task combinations)
    task_cells = set()
    for r in task_trials:
        task_cells.add((r["agent_id"], r["scenario"], r["task_id"]))

    baseline_cells = set()
    for r in baselines:
        baseline_cells.add((r["agent_id"], r["scenario"]))

    # Trials per cell in previous run
    trials_per_cell_prev = len(task_trials) // len(task_cells) if task_cells else 1

    # Calculate totals from previous run
    prev_stats = {
        "task_trials": len(task_trials),
        "baselines": len(baselines),
        "trials_per_cell": trials_per_cell_prev,
        "task_cells": len(task_cells),
        "baseline_cells": len(baseline_cells),
    }

    # Aggregate by agent
    agent_stats = {}
    for r in task_trials:
        agent = r["agent_id"]
        if agent not in agent_stats:
            agent_stats[agent] = {
                "cost": 0.0,
                "time": 0.0,
                "uncached_input_tokens": 0,
                "cached_read_tokens": 0,
                "output_tokens": 0,
                "trials": 0,
            }
        agent_stats[agent]["cost"] += float(r.get("billed_cost_usd") or 0)
        agent_stats[agent]["time"] += float(r.get("wall_time_sec") or 0)
        agent_stats[agent]["uncached_input_tokens"] += int(r.get("uncached_input_tokens") or 0)
        agent_stats[agent]["cached_read_tokens"] += int(r.get("cached_read_tokens") or 0)
        agent_stats[agent]["output_tokens"] += int(r.get("output_tokens") or 0)
        agent_stats[agent]["trials"] += 1

    # Baseline totals
    baseline_cost = sum(float(r.get("billed_cost_usd") or 0) for r in baselines)
    baseline_time = sum(float(r.get("wall_time_sec") or 0) for r in baselines)

    # Calculate projections
    scale_factor = target_trials / trials_per_cell_prev

    projections = {
        "target_trials": target_trials,
        "task_trials_projected": len(task_cells) * target_trials,
        "baselines_projected": len(baseline_cells),  # baselines don't scale with trials
        "total_runs_projected": len(task_cells) * target_trials + len(baseline_cells),
    }

    # Per-agent projections
    agent_projections = {}
    total_cost = baseline_cost
    total_time = baseline_time
    total_tokens = 0

    for agent, stats in agent_stats.items():
        avg_cost = stats["cost"] / stats["trials"]
        avg_time = stats["time"] / stats["trials"]
        avg_tokens = (
            stats["uncached_input_tokens"]
            + stats["cached_read_tokens"]
            + stats["output_tokens"]
        ) / stats["trials"]

        # Number of trials for this agent in projected run
        agent_cells = sum(1 for c in task_cells if c[0] == agent)
        projected_trials = agent_cells * target_trials

        agent_projections[agent] = {
            "avg_cost_per_trial": avg_cost,
            "avg_time_per_trial": avg_time,
            "avg_tokens_per_trial": avg_tokens,
            "projected_trials": projected_trials,
            "projected_cost": avg_cost * projected_trials,
            "projected_time_sec": avg_time * projected_trials,
            "projected_tokens": avg_tokens * projected_trials,
        }

        total_cost += avg_cost * projected_trials
        total_time += avg_time * projected_trials
        total_tokens += avg_tokens * projected_trials

    # Add overhead estimate (time between runs for setup/teardown)
    # Calculate from actual elapsed time vs sum of wall times
    if task_trials:
        first_start = min(r["ts_start"] for r in runs)
        last_end = max(r["ts_end"] for r in runs)
        # Parse ISO timestamps
        from datetime import datetime

        def parse_ts(ts):
            # Handle timezone suffix
            if "+" in ts:
                ts = ts.split("+")[0]
            return datetime.fromisoformat(ts)

        elapsed = (parse_ts(last_end) - parse_ts(first_start)).total_seconds()
        sum_wall_time = sum(float(r.get("wall_time_sec") or 0) for r in runs)
        overhead_ratio = elapsed / sum_wall_time if sum_wall_time > 0 else 1.0
    else:
        overhead_ratio = 1.0

    # Apply overhead to time estimate
    total_time_with_overhead = total_time * overhead_ratio

    return {
        "previous_run": {
            "path": str(previous_run_dir),
            "task_trials": prev_stats["task_trials"],
            "baselines": prev_stats["baselines"],
            "trials_per_cell": prev_stats["trials_per_cell"],
            "total_cost": sum(float(r.get("billed_cost_usd") or 0) for r in runs),
            "total_time_sec": sum(float(r.get("wall_time_sec") or 0) for r in runs),
        },
        "projections": {
            "target_trials_per_cell": target_trials,
            "total_runs": projections["total_runs_projected"],
            "task_trials": projections["task_trials_projected"],
            "baselines": projections["baselines_projected"],
        },
        "agent_projections": agent_projections,
        "totals": {
            "cost_usd": total_cost,
            "baseline_cost_usd": baseline_cost,
            "time_sec": total_time,
            "time_with_overhead_sec": total_time_with_overhead,
            "time_hours": total_time_with_overhead / 3600,
            "tokens": total_tokens,
            "overhead_ratio": overhead_ratio,
        },
    }


def format_report(estimate: dict) -> str:
    """Format estimate as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RUN ESTIMATE")
    lines.append("=" * 60)

    prev = estimate["previous_run"]
    lines.append(f"\nBased on: {prev['path']}")
    lines.append(f"  - {prev['task_trials']} task trials + {prev['baselines']} baselines")
    lines.append(f"  - {prev['trials_per_cell']} trial(s) per cell")
    lines.append(f"  - Total cost: ${prev['total_cost']:.2f}")
    lines.append(f"  - Total time: {prev['total_time_sec']/60:.1f} minutes")

    proj = estimate["projections"]
    lines.append(f"\nProjected run ({proj['target_trials_per_cell']} trials per cell):")
    lines.append(f"  - {proj['task_trials']} task trials + {proj['baselines']} baselines")
    lines.append(f"  - {proj['total_runs']} total runs")

    lines.append("\n" + "-" * 60)
    lines.append("PER-AGENT BREAKDOWN")
    lines.append("-" * 60)

    for agent, stats in estimate["agent_projections"].items():
        lines.append(f"\n{agent.upper()}:")
        lines.append(f"  Avg cost/trial:    ${stats['avg_cost_per_trial']:.4f}")
        lines.append(f"  Avg time/trial:    {stats['avg_time_per_trial']:.1f} sec")
        lines.append(f"  Avg tokens/trial:  {stats['avg_tokens_per_trial']:,.0f}")
        lines.append(f"  Projected trials:  {stats['projected_trials']}")
        lines.append(f"  Projected cost:    ${stats['projected_cost']:.2f}")
        lines.append(f"  Projected time:    {stats['projected_time_sec']/3600:.1f} hours")
        lines.append(f"  Projected tokens:  {stats['projected_tokens']/1e6:.1f}M")

    totals = estimate["totals"]
    lines.append("\n" + "=" * 60)
    lines.append("TOTALS")
    lines.append("=" * 60)
    lines.append(f"\n  Estimated cost:     ${totals['cost_usd']:.2f}")
    lines.append(f"    (baselines:       ${totals['baseline_cost_usd']:.2f})")
    lines.append(f"  Estimated time:     {totals['time_hours']:.1f} hours")
    lines.append(f"    (overhead factor: {totals['overhead_ratio']:.2f}x)")
    lines.append(f"  Estimated tokens:   {totals['tokens']/1e6:.1f}M")
    lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Estimate cost/time for evaluation run")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to previous run directory (e.g., runs/20260120_202830)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Target trials per cell (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )

    args = parser.parse_args()

    estimate = estimate_run(args.run, args.trials)

    if args.json:
        import json

        print(json.dumps(estimate, indent=2))
    else:
        print(format_report(estimate))


if __name__ == "__main__":
    main()
