# Eval Suite: How It Works (Step by Step)

This document describes **exactly how the eval harness runs**, **how results are computed**, and **what each metric means**. All scoring is **programmatic** in `run_suite.py` (no LLM judging).

**Related docs:**
- [README.md](README.md) — Evaluation overview and hypotheses
- [SETUP.md](SETUP.md) — Installation and usage

## Eval outcomes (primary vs secondary)

Primary outcomes (what the eval optimizes for):
- **Success rate** (deterministic graders).
- **Agent runtime** (`wall_time_sec`) — agent-only time, not harness overhead.
- **Token cost** (`cost_*` or `cold_cost_*`, depending on analysis).

Secondary/diagnostic metrics:
- Tool call counts, tool errors, cache read rates, and efficiency/thrash signals
  (e.g., repeat counts, destination churn, time-to-first build) are **explanatory**.
  They help answer *why* results look the way they do, but are not primary outcomes.

## Inputs

- **Config** (`config.yaml` or similar)
  - `suite`: run-level settings (timeouts, trials, env, prewarm).
    - `scenario_timeouts_sec` can override `timeout_sec` per scenario.
    - `stall_timeout_sec` aborts a run if there is no output/tool activity for N seconds.
- `project`: repo path, base ref, simulator, build params.
  - `agents`: how to invoke Codex / Claude CLIs and pricing.
  - `mcp`: MCP start/stop commands + env.
- **Tasks** (`tasks.yaml`)
  - Prompt, setup commands, graders, reference patch, and task kind.
- **Task assets** (`task_assets/<task>/`)
  - Optional `TASK.md`, patches, or fixtures referenced by setup.

## Step-by-step pipeline

1. **Load config + tasks**
   - `run_suite.py` loads YAML config and task list.
   - The suite expands the run matrix: `{agent} × {scenario} × {task} × {trial}`.
   - Optional baselines are added per scenario when `suite.run_baselines: true`.
   - `--task-ids` can be used to run a subset of tasks by id.
   - When `suite.plan_mode: blocked_by_scenario`, runs are grouped by scenario→agent
     and baselines are executed before that agent’s task trials to reduce
     cross‑scenario cache leakage.

2. **(Optional) Validate reference solutions**
   - If `suite.validate_tasks: true` or `--validate-tasks`, each task’s
     `reference_patch` is applied to a clean worktree and graded.
   - Failures stop the run early so we don’t measure unsolvable tasks.

3. **Per-trial setup**
   - **Worktree isolation**: create a fresh git worktree per trial.
   - **Command shims**: a PATH overlay logs `xcodebuild`/`xcrun` invocations.
   - **Environment**:
     - `suite.env` applied to both agent and MCP processes.
     - `EVAL_*` vars injected (paths, run ids, repo root/workdir).
     - MCP-only scenarios enable MCP config; shell scenarios hard-disable it.

4. **Task setup**
   - Run `setup_commands` inside the worktree.
   - For `git_diff_forbidden`, a baseline snapshot of forbidden files is recorded
     **after setup**.
   - For install/launch graders, simulator state is reset per trial.

5. **Prompt construction**
   - `shell_unprimed`: prompt only, no build params.
   - `shell_primed`: prompt + explicit build params.
   - `mcp_unprimed`: prompt + MCP preference (no build params).

6. **Agent execution**
   - CLI runs in streaming mode; output is parsed live.
   - **Transcripts** are recorded (minimal by default).
   - Tool usage is logged from stream events:
     - MCP tool calls → `mcp_tool_calls.jsonl`
     - Shell tool calls → `cmd_log.jsonl`
   - **Tool errors** are counted (MCP vs non‑MCP) and summarized in
     `tool_error_summary.json`.

7. **Grading**
   - Graders run locally via deterministic checks (tests, install/launch checks,
     forbidden diff checks).
   - `success` is true only if **all graders pass** and the agent exit code is 0.

8. **Result capture**
   - Each trial is written to `runs.jsonl` and `runs.csv`.
   - Aggregates are computed into `summary.csv` and `summary.md`.

9. **Cleanup**
   - Worktrees are removed unless `suite.keep_workdirs: true`.

## Outputs (per run)

- `runs/<timestamp>/runs.jsonl`  
  Full trial records, including nested `grader_results`.
- `runs/<timestamp>/runs.csv`  
  Flat per‑trial table for analysis.
- `runs/<timestamp>/summary.csv`  
  Aggregates per `{agent, scenario, task}`.
- `runs/<timestamp>/summary.md`  
  Markdown version of the summary table.
- `runs/<timestamp>/summary_cold.csv`  
  Aggregates per `{agent, scenario, task}` for runs where `cached_read_tokens == 0`
  (only emitted when `suite.summary_cache_strata: true`).
- `runs/<timestamp>/summary_cold.md`  
  Markdown version of the cold‑only summary.

## Rebuilding reports

If parsing or aggregation logic changes, you can rebuild outputs for a prior run
without re-running evals:

```bash
python3 rebuild_reports.py --run runs/<timestamp> --config config.yaml --in-place
```
- `runs/<timestamp>/trials/<run_id>/…`  
  Per‑trial artifacts: transcripts, tool logs, command logs, etc.
- `runs/<timestamp>/tool_error_report_manifest.json` + `tool_error_report.md`  
  Optional post‑run agent summary when `suite.post_run_report: true`.

## How metrics are computed

### Success
- `success = all_graders_passed AND exit_code == 0`
- `failure_reason` is filled from grader output or agent exit status.
  - Agent timeouts are reported as `timeout_hard` (overall runtime) or `timeout_stall` (no output/tool activity).

### Time
- `wall_time_sec`: elapsed runtime for the agent process only.
- Summary stats per cell:
  - `time_p10`, `time_median`, `time_p90`
  - `time_mean`, `time_std`, `time_cv` (CV = std/mean)

### Cost
- Usage is parsed from streamed events or `{OUT_JSON}` output.
  - Claude: use provider `total_cost_usd` if present; else compute.
  - Codex: prefer `@ccusage/codex` when enabled; else compute.
- Two cost views are reported:
  - **Billed cost** (`cost_*`, `billed_cost_usd`): what the provider charged, including
    cache discounts and any upfront tool schema overhead (e.g., MCP schema tokens).
  - **Cold-equivalent cost** (`cold_cost_*`, `cold_equivalent_cost_usd`): treats cached
    reads as uncached to make like-for-like A/B/C comparisons less sensitive to
    infra-level caching outside our control.
- Summary stats:
  - `cost_p10`, `cost_median`, `cost_p90`
  - `cost_mean`, `cost_std`, `cost_cv`
  - `cost_per_success_mean = sum(costs) / successes`
- Cache-aware diagnostics (per trial, in `runs.csv`/`runs.jsonl`):
  - `cold_equivalent_cost_usd`: billed as if cached reads were uncached.
  - `cache_savings_usd = cold_equivalent_cost_usd - billed_cost_usd`
  - `cache_read_rate = cached_read_tokens / (uncached_input_tokens + cached_read_tokens)`
- Cache-aware aggregates (per cell, in `summary.csv`/`summary.md`):
  - `cold_cost_median`, `cold_cost_p90`, `cold_cost_cv`
  - `cache_savings_mean`, `cache_read_rate_mean`

### Reliability
Let `p = success_rate`.
- `pass_at_1 = p`
- `pass_at_3 = 1 - (1 - p)^3`
- `pass_pow_3 = p^3`

### Tool usage
Per trial:
- `xcodebuild_calls`, `xcrun_calls` are **agent shell calls** (source=agent).
- `simctl_calls` derived from `xcrun simctl` invocations.
- `mcp_tool_calls` counts MCP tool calls from stream events.
- `mcp_xcodebuild_calls` / `mcp_simctl_calls` split by MCP tool type.
- Efficiency/thrash signals (derived from `cmd_log.jsonl`):
  - `time_to_first_xcodebuild_sec`: seconds from trial start to first `xcodebuild`.
  - `xcodebuild_repeat_count`: repeated `xcodebuild` invocations with the same
    normalized argument signature.
  - `destination_count` and `destination_churn`: distinct `-destination` values
    (churn = count − 1).

### Tool errors
Per trial:
- `tool_error_total`, `tool_error_mcp`, `tool_error_non_mcp`
  are counted from stream events (tool results with error flags/status).
- Per-trial details (if any) are written to `tool_errors.jsonl`.
 - `tool_error_context.jsonl` captures the next N tool calls after each error
   (default 2; override via `EVAL_TOOL_ERROR_CONTEXT_CALLS`).

## What the results mean

- **Success rate**: fraction of trials that fully passed grading.
- **Pass@k**: expected chance of success if you allow k independent retries.
- **Time/cost CV**: higher CV means higher variance / instability.
- **Tool usage**: whether MCP calls actually replace shell calls.
- **Tool errors**: indicator of MCP/shell reliability; use in comparisons.

### Comparing MCP vs shell (cost)
- **Overall cost comparison (includes MCP schema overhead):**
  - Use `billed_cost_usd` per trial or `cost_*` in summaries.
- **Like-for-like comparison (minimize cache variance):**
  - Use `cold_equivalent_cost_usd` per trial or `cold_cost_*` in summaries.
  - Optionally compare `summary_cold.*` (runs with `cached_read_tokens == 0`).
- **Work-only cost (exclude tool schema overhead):**
  - Use `marginal_cost_usd` (total minus baseline for the same `{agent, scenario}`).

## Notes on determinism

The suite measures outcomes under nondeterminism:
- Multiple trials per cell are required to estimate variance.
- Baseline runs quantify MCP schema overhead without task execution.

## Graders and required config

Each grader is generic and uses config to know what to check. Task YAML selects graders and
can pass per-grader options.

### ios_test_pass
- Uses `project.build_params` for `scheme`, `workspace`/`project`, and `destination`.
- Required in config:
  - `project.build_params.scheme`
  - `project.build_params.project` or `project.build_params.workspace`
  - `project.simulator_name` (fallback when destination not provided)
- Optional per‑grader options in task YAML:
  - `only_testing`, `skip_testing`, `test_plan`, `xcodebuild_args`, `timeout_sec`

### ios_install_check / ios_launch_check
- Uses `project.build_params.bundle_id` and simulator info.
- Required in config:
  - `project.build_params.bundle_id`
  - `project.simulator_name`
  - Optional: `project.build_params.destination` (if you want a fixed destination)

### git_diff_forbidden
- Uses `forbidden_globs` in task YAML to snapshot and compare file contents.
- Required in task YAML:
  - `forbidden_globs` (list of glob patterns)
