#!/usr/bin/env python3

"""
Agent Eval Suite v2

Implements:
- 3 scenarios: shell_unprimed, shell_primed, mcp_unprimed
- Clean isolated trials via git worktrees
- Deterministic graders (unit tests, sim install/launch checks)
- Per-trial JSONL + CSV logging with raw usage + derived unified metrics
- Baseline "do-nothing" trials per {agent, scenario} for overhead
- Aggregation with median/p10/p90/std/CV + pass@k and pass^k

This harness is designed to follow principles from:
- Anthropic: "Demystifying evals for AI agents" (outcome-based grading, deterministic graders, transcript review)
- Your Evaluation Plan PDF (unified cost, baseline overhead, schema overhead, variance metrics)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from evals.eval_reporting import (
    aggregate,
    is_cold_row,
    write_csv,
    write_markdown_summary,
)

from evals import (
    SCENARIOS,
    # Config
    AgentConfig,
    ProjectConfig,
    MCPConfig,
    SuiteConfig,
    TaskSpec,
    load_config,
    load_tasks,
    validate_project_config,
    validate_suite_config,
    # Infrastructure
    run_cmd,
    safe_mkdir,
    now_ts,
    resolve_repo_layout,
    resolve_developer_dir,
    scrub_env,
    # Worktrees
    make_worktree,
    remove_worktree,
    # Graders
    capture_forbidden_baseline,
    run_graders,
    # Agents
    make_agent,
    # Trial
    TrialResult,
    run_one_trial,
    # Reporting
    results_to_rows,
    write_jsonl,
    append_jsonl,
    load_jsonl,
    get_completed_trial_keys,
    run_post_run_report,
    run_failure_analysis_report,
)


def resolve_suite_path(path_str: str) -> pathlib.Path:
    """Resolve a path relative to the suite root."""
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return p
    return pathlib.Path(__file__).resolve().parent / p


def validate_reference_solutions(
    tasks: List[TaskSpec], suite: SuiteConfig, project: ProjectConfig, out_dir: pathlib.Path
) -> None:
    """Validate tasks using reference solutions before running trials."""
    if not suite.validate_tasks:
        return

    repo_root = pathlib.Path(project.repo_root or project.repo_path)
    repo_subdir = project.repo_subdir or ""
    ref_dir = out_dir / "reference_checks"
    safe_mkdir(ref_dir)
    worktrees_root = ref_dir / "worktrees"
    rows: List[Dict[str, Any]] = []

    print("Validating tasks with reference solutions...")
    for task in tasks:
        if not task.reference_patch:
            print(f"- Skipping {task.id} (no reference_patch)")
            continue

        import time
        run_id = f"ref-{task.id}-{int(time.time() * 1000)}"
        wt = make_worktree(
            repo_root, project.base_ref, worktrees_root, run_id, suite.fetch_remote
        )
        workdir = wt / repo_subdir if repo_subdir else wt

        env = dict(os.environ)
        env["EVAL_SUITE_ROOT"] = str(pathlib.Path(__file__).resolve().parent)
        env["EVAL_RUN_ID"] = run_id
        env["EVAL_REPO_ROOT"] = str(wt)
        env["EVAL_REPO_SUBDIR"] = repo_subdir if repo_subdir else "."
        env["EVAL_REPO_WORKDIR"] = str(workdir)

        ok = True
        reason: Optional[str] = None
        grader_results: Optional[List[Dict[str, Any]]] = None
        try:
            for scmd in task.setup_commands:
                rc, out, err = run_cmd(
                    [str(x) for x in scmd], cwd=str(workdir), env=env, timeout=600
                )
                if rc != 0:
                    ok = False
                    reason = "setup_failed"
                    break

            if ok:
                ok, reason = capture_forbidden_baseline(
                    task.graders, workdir, project
                )
                if not ok:
                    reason = reason or "forbidden_baseline_failed"

            if ok:
                patch_path = resolve_suite_path(task.reference_patch)
                if not patch_path.exists():
                    ok = False
                    reason = "reference_patch_missing"
                else:
                    cmd = ["git", "-C", str(wt), "apply"]
                    if repo_subdir:
                        cmd += ["--directory", repo_subdir]
                    cmd.append(str(patch_path))
                    rc, out, err = run_cmd(
                        cmd,
                        cwd=str(workdir),
                        env=env,
                        timeout=600,
                    )
                    if rc != 0:
                        ok = False
                        reason = "reference_patch_apply_failed"

            if ok:
                graders_for_ref: List[Dict[str, Any]] = []
                for g in task.graders:
                    if g.get("type") == "ios_test_pass":
                        gg = dict(g)
                        gg["log_path"] = str(ref_dir / f"{task.id}_ios_test_pass.log")
                        graders_for_ref.append(gg)
                    else:
                        graders_for_ref.append(g)
                ok, reason, grader_results = run_graders(
                    graders_for_ref, workdir, project, timeout_sec=suite.timeout_sec
                )
        finally:
            rows.append(
                {
                    "task_id": task.id,
                    "task_kind": task.kind,
                    "success": bool(ok),
                    "failure_reason": reason,
                    "grader_results": grader_results,
                    "reference_patch": task.reference_patch,
                }
            )
            remove_worktree(repo_root, wt)

        if not ok:
            write_jsonl(ref_dir / "reference_checks.jsonl", rows)
            raise SystemExit(
                f"Reference solution failed for task {task.id}: {reason}"
            )

    write_jsonl(ref_dir / "reference_checks.jsonl", rows)
    print("Reference solutions validated.")


def resolve_spm_cache_dir(suite: SuiteConfig) -> Optional[pathlib.Path]:
    """Resolve SwiftPM cache directory from suite config."""
    if not suite.spm_cache_dir:
        return None
    p = pathlib.Path(str(suite.spm_cache_dir)).expanduser()
    if not p.is_absolute():
        p = pathlib.Path(suite.output_root).resolve() / p
    return p


def prewarm_swiftpm(
    suite: SuiteConfig,
    project: ProjectConfig,
    out_dir: pathlib.Path,
    spm_cache_dir: pathlib.Path,
) -> None:
    """Pre-warm SwiftPM package cache before running trials."""
    primed = project.build_params or {}
    workspace = primed.get("workspace")
    proj = primed.get("project")
    scheme = primed.get("scheme")
    log_path = out_dir / "spm_prewarm.log"

    if not scheme or (not workspace and not proj):
        log_path.write_text(
            "SKIP: missing scheme/workspace/project for SwiftPM prewarm.\n"
        )
        return

    repo_root = pathlib.Path(project.repo_root or project.repo_path)
    repo_subdir = project.repo_subdir or ""
    repo_effective = repo_root / repo_subdir if repo_subdir else repo_root

    cmd = ["xcodebuild", "-resolvePackageDependencies"]
    if workspace:
        cmd += ["-workspace", str(repo_effective / workspace)]
    elif proj:
        cmd += ["-project", str(repo_effective / proj)]
    cmd += ["-scheme", str(scheme)]
    cmd += ["-clonedSourcePackagesDirPath", str(spm_cache_dir)]

    env = dict(os.environ)
    env.update(suite.env or {})
    dev_dir = resolve_developer_dir()
    if dev_dir and "DEVELOPER_DIR" not in env:
        env["DEVELOPER_DIR"] = dev_dir

    rc, out, err = run_cmd(cmd, cwd=str(repo_effective), env=env, timeout=900)
    log_path.write_text(
        "CMD: "
        + " ".join(cmd)
        + "\n\n"
        + f"rc={rc}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}\n"
    )


def collect_host_metadata(
    project: ProjectConfig, agents_cfgs: List[AgentConfig], suite: SuiteConfig
) -> Dict[str, Any]:
    """Collect metadata about the host environment."""
    def cmd_result(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        rc, out, err = run_cmd(cmd, cwd=cwd)
        return {
            "cmd": cmd,
            "rc": rc,
            "stdout": out.strip(),
            "stderr": err.strip(),
        }

    meta: Dict[str, Any] = {
        "ts": now_ts(),
        "project": {
            "repo_path": project.repo_path,
            "base_ref": project.base_ref,
        },
        "suite": {
            "output_root": suite.output_root,
            "trials_per_cell": suite.trials_per_cell,
            "run_baselines": suite.run_baselines,
            "random_seed": suite.random_seed,
            "prewarm_spm": suite.prewarm_spm,
            "spm_cache_dir": suite.spm_cache_dir,
            "env": scrub_env(suite.env),
        },
        "agents": [
            {
                "id": a.id,
                "kind": a.kind,
                "command": a.command,
                "env": scrub_env(a.env),
            }
            for a in agents_cfgs
        ],
        "host": {
            "python_version": sys.version.splitlines()[0],
        },
        "tools": {},
    }

    repo_root, _ = resolve_repo_layout(project.repo_path)
    meta["project"]["repo_root"] = repo_root
    rc, out, _ = run_cmd(["git", "-C", repo_root, "rev-parse", "HEAD"])
    if rc == 0:
        meta["project"]["git_head"] = out.strip()
    rc, out, _ = run_cmd(["git", "-C", repo_root, "rev-parse", project.base_ref])
    if rc == 0:
        meta["project"]["base_ref_sha"] = out.strip()

    meta["host"]["sw_vers"] = cmd_result(["sw_vers"])
    meta["host"]["uname"] = cmd_result(["uname", "-a"])
    meta["tools"]["xcodebuild"] = cmd_result(["xcodebuild", "-version"])
    meta["tools"]["xcode_select"] = cmd_result(["xcode-select", "-p"])
    meta["tools"]["claude_cli"] = cmd_result(["claude", "--version"])
    meta["tools"]["codex_cli"] = cmd_result(["codex", "--version"])

    simctl = cmd_result(["xcrun", "simctl", "list", "runtimes", "-j"])
    if simctl["rc"] == 0:
        try:
            meta["tools"]["simulator_runtimes"] = json.loads(simctl["stdout"])
        except Exception:
            meta["tools"]["simulator_runtimes_raw"] = simctl
    else:
        meta["tools"]["simulator_runtimes_raw"] = simctl

    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--tasks", required=True, help="Path to tasks.yaml")
    ap.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Subset of task ids to run (space-separated)",
    )
    ap.add_argument("--trials", type=int, default=None, help="Override trials_per_cell")
    ap.add_argument(
        "--scenarios", nargs="*", default=SCENARIOS, help="Subset of scenarios to run"
    )
    ap.add_argument(
        "--agents", nargs="*", default=None, help="Subset of agent ids to run"
    )
    ap.add_argument(
        "--validate-tasks",
        action="store_true",
        help="Validate tasks using reference solutions before running trials",
    )
    ap.add_argument(
        "--validate-tasks-only",
        action="store_true",
        help="Validate tasks using reference solutions and exit",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Stream agent stdout/stderr to console (also written to transcript)",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an interrupted run from the given run directory (e.g., runs/20260120_202830)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what trials would be run without actually running them",
    )
    args = ap.parse_args()

    suite, project, agents_cfgs, mcp = load_config(args.config)
    all_agents_cfgs = agents_cfgs  # Keep full list for post-run reports
    validate_suite_config(suite)
    validate_project_config(project)
    if args.trials is not None:
        suite.trials_per_cell = args.trials

    tasks = load_tasks(args.tasks)
    if args.task_ids:
        allowed = set(args.task_ids)
        tasks = [t for t in tasks if t.id in allowed]
        if not tasks:
            raise SystemExit("No matching task ids for --task-ids")
    if args.validate_tasks:
        suite.validate_tasks = True
    if args.validate_tasks_only:
        suite.validate_tasks = True

    # Handle --resume
    resuming = False
    completed_trials: Dict[Tuple[str, str, str], int] = {}
    resume_agents: Optional[set] = None
    resume_scenarios: Optional[set] = None
    resume_tasks: Optional[set] = None
    out_dir: Optional[pathlib.Path] = None

    if args.resume:
        out_dir = pathlib.Path(args.resume).resolve()
        if not out_dir.exists():
            raise SystemExit(f"Resume directory does not exist: {out_dir}")
        if not (out_dir / "runs.jsonl").exists():
            print(f"Warning: No runs.jsonl found in {out_dir}, starting fresh")
        else:
            existing_rows = load_jsonl(out_dir / "runs.jsonl")
            completed_trials = get_completed_trial_keys(existing_rows)
            total_completed = sum(completed_trials.values())

            resume_agents = set(r.get("agent_id") for r in existing_rows if r.get("agent_id"))
            resume_scenarios = set(r.get("scenario") for r in existing_rows if r.get("scenario"))
            resume_tasks = set(r.get("task_id") for r in existing_rows if r.get("task_id") and r.get("task_id") != "baseline")

            print(f"Resuming run from {out_dir}")
            print(f"  Found {total_completed} completed trials across {len(completed_trials)} cells")
            print(f"  Inferred agents: {sorted(resume_agents)}")
            print(f"  Inferred scenarios: {sorted(resume_scenarios)}")
            print(f"  Inferred tasks: {sorted(resume_tasks)}")
            resuming = True

    # Apply agent/scenario filters
    if args.resume and resuming:
        if not args.agents and resume_agents:
            agents_cfgs = [a for a in agents_cfgs if a.id in resume_agents]
            print(f"  Using agents from existing run: {[a.id for a in agents_cfgs]}")
        elif args.agents:
            agents_cfgs = [a for a in agents_cfgs if a.id in set(args.agents)]
            print(f"  Filtering to specified agents: {args.agents}")

        if args.scenarios == SCENARIOS and resume_scenarios:
            scenarios = [s for s in SCENARIOS if s in resume_scenarios]
            print(f"  Using scenarios from existing run: {scenarios}")
        else:
            scenarios = [s for s in args.scenarios if s in SCENARIOS]
            print(f"  Using specified scenarios: {scenarios}")
    else:
        if args.agents:
            agents_cfgs = [a for a in agents_cfgs if a.id in set(args.agents)]
        scenarios = [s for s in args.scenarios if s in SCENARIOS]

    if not scenarios:
        raise SystemExit(f"No valid scenarios. Options: {SCENARIOS}")

    # Create output directory
    if not args.resume:
        out_dir = (
            pathlib.Path(suite.output_root)
            .resolve()
            / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    if out_dir is None:
        raise SystemExit("Failed to resolve output directory")
    safe_mkdir(out_dir)
    spm_cache_dir = resolve_spm_cache_dir(suite)
    if spm_cache_dir:
        safe_mkdir(spm_cache_dir)
        suite.env = dict(suite.env or {})
        suite.env.setdefault("SWIFTPM_PACKAGE_CACHE_PATH", str(spm_cache_dir))
        if suite.prewarm_spm:
            prewarm_swiftpm(suite, project, out_dir, spm_cache_dir)
    meta = collect_host_metadata(project, agents_cfgs, suite)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    validate_reference_solutions(tasks, suite, project, out_dir)
    if args.validate_tasks_only:
        return

    # Build plan
    rng = random.Random(suite.random_seed)
    plan: List[Tuple[AgentConfig, TaskSpec, str, bool]] = []
    baseline_task = TaskSpec(
        id="baseline",
        description="baseline",
        prompt="",
        setup_commands=[],
        graders=[],
        kind="baseline",
        reference_patch=None,
    )

    if suite.plan_mode == "random":
        for a in agents_cfgs:
            for s in scenarios:
                if suite.run_baselines:
                    for _ in range(suite.baseline_trials_per_scenario):
                        plan.append((a, baseline_task, s, True))
        for a in agents_cfgs:
            for t in tasks:
                for s in scenarios:
                    for _ in range(suite.trials_per_cell):
                        plan.append((a, t, s, False))
        rng.shuffle(plan)
    else:
        for s in scenarios:
            for a in agents_cfgs:
                if suite.run_baselines:
                    for _ in range(suite.baseline_trials_per_scenario):
                        plan.append((a, baseline_task, s, True))
                block_trials: List[Tuple[AgentConfig, TaskSpec, str, bool]] = []
                for t in tasks:
                    for _ in range(suite.trials_per_cell):
                        block_trials.append((a, t, s, False))
                if suite.shuffle_within_scenario:
                    rng.shuffle(block_trials)
                plan.extend(block_trials)

    # Run with incremental writes and resume support
    results: List[TrialResult] = []
    session_trial_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    trials_to_skip = dict(completed_trials) if resuming else {}

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - showing what would be executed")
        print("=" * 60)
        to_run_by_cell: Dict[Tuple[str, str, str], int] = defaultdict(int)

    total_plan = len(plan)
    skipped = 0
    trials_to_run = 0
    for idx, (a_cfg, task, scenario, baseline) in enumerate(plan):
        cell_key = (a_cfg.id, scenario, task.id)

        if trials_to_skip.get(cell_key, 0) > 0:
            trials_to_skip[cell_key] -= 1
            skipped += 1
            continue

        if args.dry_run:
            to_run_by_cell[cell_key] += 1
            trials_to_run += 1
            continue

        print(f"[{idx + 1 - skipped}/{total_plan - skipped}] Running {a_cfg.id}/{scenario}/{task.id}...")

        adapter = make_agent(a_cfg)
        tr = run_one_trial(
            suite=suite,
            project=project,
            mcp=mcp,
            agent_cfg=a_cfg,
            agent=adapter,
            task=task,
            scenario=scenario,
            baseline_run=baseline,
            out_dir=out_dir,
            stream_agent_output=args.stream,
        )
        results.append(tr)
        session_trial_counts[cell_key] += 1

        row_jsonl = results_to_rows([tr], include_provider_usage=True)[0]
        append_jsonl(out_dir / "runs.jsonl", row_jsonl)

    if skipped > 0:
        print(f"Skipped {skipped} already-completed trials")

    if args.dry_run:
        print(f"\nTotal trials in plan: {total_plan}")
        print(f"Trials to skip (already completed): {skipped}")
        print(f"Trials to run: {trials_to_run}")

        if trials_to_run > 0:
            print("\nTrials to run by cell:")
            for (agent, scenario, task_id), count in sorted(to_run_by_cell.items()):
                if count > 0:
                    have = completed_trials.get((agent, scenario, task_id), 0)
                    need = have + count
                    print(f"  {agent}/{scenario}/{task_id}: {have} -> {need} (+{count})")
        else:
            print("\nNo trials to run - all cells are complete!")

        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE - no trials were executed")
        print("=" * 60)
        return

    # Load all results for final aggregation
    all_rows_jsonl = load_jsonl(out_dir / "runs.jsonl")
    all_rows = [{k: v for k, v in r.items() if k != "provider_usage"} for r in all_rows_jsonl]
    rows_jsonl_by_id = {r["run_id"]: r for r in all_rows_jsonl}

    write_csv(out_dir / "runs.csv", all_rows)
    rows = all_rows

    # Build baseline map for overhead decomposition
    baseline_cost: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if r["baseline_run"] and r.get("billed_cost_usd") is not None:
            baseline_cost[(r["agent_id"], r["scenario"])] = float(r["billed_cost_usd"])

    for r in rows:
        if r["baseline_run"]:
            continue
        bc = baseline_cost.get((r["agent_id"], r["scenario"]))
        r["baseline_cost_usd"] = bc
        if bc is not None and r.get("billed_cost_usd") is not None:
            r["marginal_cost_usd"] = float(r["billed_cost_usd"]) - float(bc)
        jr = rows_jsonl_by_id.get(r["run_id"])
        if jr is not None:
            jr["baseline_cost_usd"] = r["baseline_cost_usd"]
            jr["marginal_cost_usd"] = r["marginal_cost_usd"]

    write_jsonl(out_dir / "runs.jsonl", all_rows_jsonl)
    write_csv(out_dir / "runs.csv", rows)

    summary = aggregate(rows)
    write_csv(out_dir / "summary.csv", summary)
    write_markdown_summary(out_dir / "summary.md", summary)
    if suite.summary_cache_strata:
        summary_cold = aggregate(rows, row_filter=is_cold_row)
        write_csv(out_dir / "summary_cold.csv", summary_cold)
        write_markdown_summary(out_dir / "summary_cold.md", summary_cold)

    report_path = run_post_run_report(suite, all_agents_cfgs, out_dir, rows)
    failure_report_path = run_failure_analysis_report(suite, all_agents_cfgs, out_dir, rows)

    print(f"Wrote results to: {out_dir}")
    print(f"- runs.jsonl")
    print(f"- runs.csv")
    print(f"- summary.csv")
    print(f"- summary.md")
    if suite.summary_cache_strata:
        print(f"- summary_cold.csv")
        print(f"- summary_cold.md")
    if report_path:
        print(f"- {report_path.name}")
    if failure_report_path:
        print(f"- {failure_report_path.name}")


if __name__ == "__main__":
    main()
