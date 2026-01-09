"""
Trial execution for the eval suite.

Provides the TrialResult dataclass and run_one_trial orchestrator function.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shlex
import shutil
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from evals.eval_reporting import compute_cold_equivalent_cost, compute_cost, parse_usage

from evals.infrastructure import (
    run_cmd,
    safe_mkdir,
    now_ts,
    resolve_simulator_udid_for_project,
    resolve_developer_dir,
    ensure_simulator_booted,
    reset_simulator_app_state,
)
from evals.metrics import count_invocations, count_mcp_tool_invocations
from evals.worktrees import make_worktree, remove_worktree
from evals.graders import capture_forbidden_baseline, run_graders
from evals.agents import (
    AgentAdapter,
    make_command_shims,
    build_prompt,
    read_tool_error_summary,
    ccusage_codex_cost,
    codex_mcp_args_from_json,
)

if TYPE_CHECKING:
    from evals.config import AgentConfig, MCPConfig, ProjectConfig, SuiteConfig, TaskSpec


@dataclasses.dataclass
class TrialResult:
    """Result of a single trial run."""
    run_id: str
    ts_start: str
    ts_end: str
    agent_id: str
    agent_kind: str
    scenario: str
    task_id: str
    task_kind: str
    baseline_run: bool

    # Outcome
    success: bool
    failure_reason: Optional[str]
    grader_results: Optional[List[Dict[str, Any]]]
    exit_code: Optional[int]

    # Time
    wall_time_sec: float

    # Usage + cost (unified)
    model: Optional[str]
    provider_usage: Optional[Dict[str, Any]]
    uncached_input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached_read_tokens: Optional[int]
    cache_write_tokens: Optional[int]
    cache_write_ttl: Optional[str]  # "5m"|"1h"|None
    billed_cost_usd: Optional[float]
    cost_source: Optional[str]
    cold_equivalent_cost_usd: Optional[float]
    cache_savings_usd: Optional[float]
    cache_read_rate: Optional[float]

    # Overhead decomposition (filled in post-process)
    baseline_cost_usd: Optional[float]
    marginal_cost_usd: Optional[float]

    # Instrumentation
    command_invocations: Dict[str, int]  # xcodebuild/xcrun/...
    mcp_tool_invocations: Optional[int]
    tool_error_total: int
    tool_error_mcp: int
    tool_error_non_mcp: int

    # Artifacts
    workdir: Optional[str]
    transcript_path: str
    agent_output_json_path: Optional[str]
    command_log_path: str
    mcp_tool_log_path: Optional[str]
    tool_error_log_path: Optional[str]
    tool_error_context_log_path: Optional[str]


def run_one_trial(
    suite: "SuiteConfig",
    project: "ProjectConfig",
    mcp: "MCPConfig",
    agent_cfg: "AgentConfig",
    agent: AgentAdapter,
    task: "TaskSpec",
    scenario: str,
    baseline_run: bool,
    out_dir: pathlib.Path,
    stream_agent_output: bool,
) -> TrialResult:
    """
    Run a single trial and return the result.

    Handles:
    - Worktree isolation
    - Command shim setup
    - MCP server lifecycle
    - Agent execution
    - Grading
    - Result collection
    """
    run_id = f"{agent_cfg.id}-{scenario}-{task.id}-{'baseline' if baseline_run else 'trial'}-{int(time.time()*1000)}"
    ts_start = now_ts()

    # Prepare output paths
    trial_dir = out_dir / "trials" / run_id
    safe_mkdir(trial_dir)
    transcript_path = trial_dir / "transcript.txt"
    agent_out_json = trial_dir / "agent_output.json"
    cmd_log_path = trial_dir / "cmd_log.jsonl"
    tool_error_log_path = trial_dir / "tool_errors.jsonl"
    tool_error_summary_path = trial_dir / "tool_error_summary.json"
    tool_error_context_log_path = trial_dir / "tool_error_context.jsonl"
    mcp_log_path = None

    # Worktree isolation
    repo_root = pathlib.Path(project.repo_root or project.repo_path)
    worktrees_root = out_dir / "worktrees"
    wt = make_worktree(
        repo_root,
        project.base_ref,
        worktrees_root,
        run_id,
        suite.fetch_remote,
    )

    repo_subdir = project.repo_subdir or ""
    workdir = wt / repo_subdir if repo_subdir else wt
    if not workdir.exists():
        raise RuntimeError(f"Workdir does not exist: {workdir}")

    # Setup shims (logs xcodebuild/xcrun invocations in *all* scenarios)
    shim_dir = trial_dir / "shims"
    shim_env = make_command_shims(
        shim_dir, cmd_log_path, commands=["xcodebuild", "xcrun"]
    )

    # Build environment
    env = dict(os.environ)
    env.update(suite.env or {})
    env.update(agent_cfg.env or {})
    env.update(shim_env)
    env["EVAL_SUITE_ROOT"] = str(pathlib.Path(__file__).resolve().parent.parent)
    env["EVAL_RUN_ID"] = run_id
    env["EVAL_REPO_ROOT"] = str(wt)
    env["EVAL_REPO_SUBDIR"] = repo_subdir if repo_subdir else "."
    env["EVAL_REPO_WORKDIR"] = str(workdir)
    dev_dir = resolve_developer_dir()
    if dev_dir and "DEVELOPER_DIR" not in env:
        env["DEVELOPER_DIR"] = dev_dir

    # Clean agent environment if configured
    if suite.clean_agent_env:
        clean_root = trial_dir / "agent_env"
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

    # MCP setup for mcp_unprimed scenarios
    if mcp.enabled and scenario in ("mcp_unprimed", "mcp_unprimed_v2"):
        env["EVAL_MCP_ENABLED"] = "1"
        env.update(mcp.env or {})
        if env.get("EVAL_MCP_TOOL_LOG"):
            path_val = env["EVAL_MCP_TOOL_LOG"]
            if "{RUN_ID}" in path_val:
                path_val = path_val.replace("{RUN_ID}", run_id)
                env["EVAL_MCP_TOOL_LOG"] = path_val
            mcp_log_path = pathlib.Path(path_val)
        else:
            env["EVAL_MCP_TOOL_LOG"] = str(trial_dir / "mcp_tool_calls.jsonl")
            mcp_log_path = pathlib.Path(env["EVAL_MCP_TOOL_LOG"])
        if mcp_log_path:
            safe_mkdir(mcp_log_path.parent)
        if mcp.start_command:
            run_cmd(mcp.start_command, cwd=str(wt), env=env, timeout=120)
    else:
        env["EVAL_MCP_ENABLED"] = "0"

    # Scenario-specific agent args
    env["EVAL_AGENT_EXTRA_ARGS"] = ""
    suite_root = pathlib.Path(__file__).resolve().parent.parent
    if agent_cfg.kind == "claude_code_cli":
        cfg_root = suite_root / "mcp_configs"
        if scenario == "mcp_unprimed":
            mcp_cfg = cfg_root / "xcodebuildmcp_only.json"
            if mcp_cfg.exists():
                env["EVAL_AGENT_EXTRA_ARGS"] = f"--strict-mcp-config --mcp-config {mcp_cfg}"
        elif scenario == "mcp_unprimed_v2":
            mcp_cfg = cfg_root / "xcodebuildmcp_v2.json"
            if mcp_cfg.exists():
                env["EVAL_AGENT_EXTRA_ARGS"] = f"--strict-mcp-config --mcp-config {mcp_cfg}"
        else:
            empty_cfg = cfg_root / "empty.json"
            env["EVAL_AGENT_EXTRA_ARGS"] = f"--strict-mcp-config --mcp-config {empty_cfg}"
    elif agent_cfg.kind == "codex_cli":
        cfg_root = suite_root / "mcp_configs"
        if scenario == "mcp_unprimed":
            mcp_cfg = cfg_root / "xcodebuildmcp_only.json"
            extra_args = codex_mcp_args_from_json(mcp_cfg)
            if extra_args:
                env["EVAL_AGENT_EXTRA_ARGS"] = shlex.join(extra_args)
        elif scenario == "mcp_unprimed_v2":
            mcp_cfg = cfg_root / "xcodebuildmcp_v2.json"
            extra_args = codex_mcp_args_from_json(mcp_cfg)
            if extra_args:
                env["EVAL_AGENT_EXTRA_ARGS"] = shlex.join(extra_args)
        else:
            env["EVAL_AGENT_EXTRA_ARGS"] = shlex.join(["-c", "mcp_servers={}"])

    # Prepend shims to PATH
    env["PATH"] = f"{shim_dir}:{env.get('PATH','')}"

    # Task setup
    for scmd in task.setup_commands:
        rc, out, err = run_cmd(
            [str(x) for x in scmd], cwd=str(workdir), env=env, timeout=600
        )
        if rc != 0:
            ts_end = now_ts()
            wall = 0.0
            inv = count_invocations(cmd_log_path, source="agent")
            mcp_calls = count_mcp_tool_invocations(mcp_log_path)
            tool_errors = read_tool_error_summary(tool_error_summary_path)
            if mcp.stop_command and env.get("EVAL_MCP_ENABLED") == "1":
                run_cmd(mcp.stop_command, cwd=str(wt), env=env, timeout=120)
            if not suite.keep_workdirs:
                remove_worktree(repo_root, wt)
            return TrialResult(
                run_id=run_id,
                ts_start=ts_start,
                ts_end=ts_end,
                agent_id=agent_cfg.id,
                agent_kind=agent_cfg.kind,
                scenario=scenario,
                task_id=task.id,
                task_kind=task.kind,
                baseline_run=baseline_run,
                success=False,
                failure_reason="setup_failed",
                grader_results=None,
                exit_code=rc,
                wall_time_sec=wall,
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
                command_invocations=inv,
                mcp_tool_invocations=mcp_calls,
                tool_error_total=tool_errors["total"],
                tool_error_mcp=tool_errors["mcp"],
                tool_error_non_mcp=tool_errors["non_mcp"],
                workdir=str(workdir) if suite.keep_workdirs else None,
                transcript_path=str(transcript_path),
                agent_output_json_path=None,
                command_log_path=str(cmd_log_path),
                mcp_tool_log_path=str(mcp_log_path) if mcp_log_path else None,
                tool_error_log_path=str(tool_error_log_path),
                tool_error_context_log_path=str(tool_error_context_log_path),
            )

    # Capture forbidden baseline
    baseline_path = trial_dir / "forbidden_baseline.json"
    baseline_ok, baseline_reason = capture_forbidden_baseline(
        task.graders, workdir, project, baseline_path=baseline_path
    )
    if not baseline_ok:
        ts_end = now_ts()
        wall = 0.0
        inv = count_invocations(cmd_log_path, source="agent")
        mcp_calls = count_mcp_tool_invocations(mcp_log_path)
        tool_errors = read_tool_error_summary(tool_error_summary_path)
        if mcp.stop_command and env.get("EVAL_MCP_ENABLED") == "1":
            run_cmd(mcp.stop_command, cwd=str(wt), env=env, timeout=120)
        if not suite.keep_workdirs:
            remove_worktree(repo_root, wt)
        return TrialResult(
            run_id=run_id,
            ts_start=ts_start,
            ts_end=ts_end,
            agent_id=agent_cfg.id,
            agent_kind=agent_cfg.kind,
            scenario=scenario,
            task_id=task.id,
            task_kind=task.kind,
            baseline_run=baseline_run,
            success=False,
            failure_reason=baseline_reason or "forbidden_baseline_failed",
            grader_results=None,
            exit_code=None,
            wall_time_sec=wall,
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
            command_invocations=inv,
            mcp_tool_invocations=mcp_calls,
            tool_error_total=tool_errors["total"],
            tool_error_mcp=tool_errors["mcp"],
            tool_error_non_mcp=tool_errors["non_mcp"],
            workdir=str(workdir) if suite.keep_workdirs else None,
            transcript_path=str(transcript_path),
            agent_output_json_path=None,
            command_log_path=str(cmd_log_path),
            mcp_tool_log_path=str(mcp_log_path) if mcp_log_path else None,
            tool_error_log_path=str(tool_error_log_path),
            tool_error_context_log_path=str(tool_error_context_log_path),
        )

    # Pre-run simulator cleanup for install/launch graders
    if not baseline_run:
        needs_sim_reset = any(
            g.get("type") in ("ios_install_check", "ios_launch_check")
            for g in task.graders
        )
        if needs_sim_reset:
            bundle_id = project.build_params.get("bundle_id")
            udid = resolve_simulator_udid_for_project(
                project, project.build_params.get("destination")
            )
            if bundle_id and udid:
                ensure_simulator_booted(udid)
                reset_simulator_app_state(udid, bundle_id)

    # Build prompt
    if baseline_run:
        prompt = "Respond with exactly: OK\nThen exit.\n"
    else:
        prompt = build_prompt(
            task.prompt, scenario, project.build_params, project.simulator_name
        )

    print(
        f"[trial {run_id}] transcript={transcript_path} workdir={workdir}",
        flush=True,
    )

    # Resolve timeouts
    scenario_timeout = suite.timeout_sec
    if suite.scenario_timeouts_sec and scenario in suite.scenario_timeouts_sec:
        scenario_timeout = int(suite.scenario_timeouts_sec[scenario])
    stall_timeout = suite.stall_timeout_sec

    # Run agent
    t0 = time.time()
    exit_code, provider_usage, timeout_reason = agent.run(
        prompt,
        workdir,
        agent_out_json,
        env,
        scenario_timeout,
        stall_timeout,
        transcript_path,
        transcript_mode=suite.transcript_mode,
        stream_output=stream_agent_output,
        cmd_log_path=cmd_log_path,
        tool_error_log_path=tool_error_log_path,
        tool_error_summary_path=tool_error_summary_path,
        tool_error_context_log_path=tool_error_context_log_path,
    )
    wall = time.time() - t0

    # Stop MCP after agent completes
    if mcp.enabled and scenario in ("mcp_unprimed", "mcp_unprimed_v2") and mcp.stop_command:
        run_cmd(mcp.stop_command, cwd=str(wt), env=env, timeout=120)

    # Grade (skip for baseline)
    success = True
    failure_reason = None
    grader_results: Optional[List[Dict[str, Any]]] = None
    if not baseline_run:
        graders_for_run: List[Dict[str, Any]] = []
        ios_test_idx = 0
        for g in task.graders:
            if g.get("type") == "git_diff_forbidden":
                gg = dict(g)
                gg["baseline_path"] = str(baseline_path)
                graders_for_run.append(gg)
            elif g.get("type") == "ios_test_pass":
                gg = dict(g)
                ios_test_idx += 1
                gg["log_path"] = str(
                    trial_dir / f"{task.id}_ios_test_pass_{ios_test_idx}.log"
                )
                graders_for_run.append(gg)
            else:
                graders_for_run.append(g)
        success, failure_reason, grader_results = run_graders(
            graders_for_run, workdir, project, timeout_sec=suite.timeout_sec, env=env
        )

    # Parse usage + compute cost
    unified = parse_usage(provider_usage)
    cost_source: Optional[str] = None
    billed: Optional[float] = None
    if agent_cfg.kind == "claude_code_cli":
        if unified.get("billed_cost_usd") is not None:
            billed = float(unified["billed_cost_usd"])
            cost_source = "provider"
        else:
            billed = compute_cost(unified, agent_cfg.pricing)
            cost_source = "computed" if billed is not None else None
    elif agent_cfg.kind == "codex_cli":
        if suite.use_ccusage_for_codex:
            cc_cost = ccusage_codex_cost(env.get("CODEX_HOME"))
            if cc_cost is not None:
                billed = cc_cost
                cost_source = "ccusage"
        if billed is None:
            unified_for_compute = dict(unified)
            unified_for_compute["billed_cost_usd"] = None
            billed = compute_cost(unified_for_compute, agent_cfg.pricing)
            cost_source = "computed" if billed is not None else None
    else:
        billed = compute_cost(unified, agent_cfg.pricing)
        cost_source = "computed" if billed is not None else None

    cold_equiv, cache_read_rate = compute_cold_equivalent_cost(
        unified, agent_cfg.pricing
    )
    cache_savings = None
    if cold_equiv is not None and billed is not None:
        cache_savings = float(cold_equiv) - float(billed)

    inv = count_invocations(cmd_log_path, source="agent")
    mcp_calls = count_mcp_tool_invocations(mcp_log_path)
    tool_errors = read_tool_error_summary(tool_error_summary_path)

    ts_end = now_ts()

    # Cleanup
    if not suite.keep_workdirs:
        remove_worktree(repo_root, wt)

    return TrialResult(
        run_id=run_id,
        ts_start=ts_start,
        ts_end=ts_end,
        agent_id=agent_cfg.id,
        agent_kind=agent_cfg.kind,
        scenario=scenario,
        task_id=task.id,
        task_kind=task.kind,
        baseline_run=baseline_run,
        success=bool(success) and exit_code == 0,
        failure_reason=(
            failure_reason
            if exit_code == 0
            else (failure_reason or timeout_reason or "agent_exit_nonzero")
        ),
        exit_code=exit_code,
        wall_time_sec=float(wall),
        model=unified.get("model"),
        provider_usage=provider_usage,
        uncached_input_tokens=unified.get("uncached_input_tokens"),
        output_tokens=unified.get("output_tokens"),
        cached_read_tokens=unified.get("cached_read_tokens"),
        cache_write_tokens=unified.get("cache_write_tokens"),
        cache_write_ttl=unified.get("cache_write_ttl"),
        billed_cost_usd=billed,
        cost_source=cost_source,
        cold_equivalent_cost_usd=cold_equiv,
        cache_savings_usd=cache_savings,
        cache_read_rate=cache_read_rate,
        baseline_cost_usd=None,
        marginal_cost_usd=None,
        command_invocations=inv,
        mcp_tool_invocations=mcp_calls,
        tool_error_total=tool_errors["total"],
        tool_error_mcp=tool_errors["mcp"],
        tool_error_non_mcp=tool_errors["non_mcp"],
        workdir=str(workdir) if suite.keep_workdirs else None,
        transcript_path=str(transcript_path),
        agent_output_json_path=str(agent_out_json) if agent_out_json.exists() else None,
        command_log_path=str(cmd_log_path),
        mcp_tool_log_path=str(mcp_log_path) if mcp_log_path else None,
        tool_error_log_path=str(tool_error_log_path),
        tool_error_context_log_path=str(tool_error_context_log_path),
        grader_results=grader_results,
    )
