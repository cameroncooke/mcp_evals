# Setup and Usage Guide

This document covers installation, configuration, and running the evaluation suite.

## Prerequisites

- Python 3.10+ (recommended: use `.venv`)
- Node.js + npm/npx (for `@ccusage/codex` and npx-based XcodeBuildMCP)
- Xcode + iOS simulator runtime (for iOS build/install/test graders)
- Codex CLI and/or Claude Code CLI installed and authenticated

## Installation

```bash
# Clone system under test (HackerNews app)
bash clone_sut.sh

# Set up Python environment
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Optional: dev dependencies for type checking
pip install -r requirements-dev.txt
```

### Codex cost accounting

Uses `@ccusage/codex` for accurate cost tracking:
- **Option A (no install):** Uses `npx` on demand (requires Node.js)
- **Option B (global install):** `npm install -g @ccusage/codex`

## Configuration

Copy `config.example.yaml` to `config.yaml` and configure:

### Required settings

| Setting | Description |
|---------|-------------|
| `project.repo_path` | Path to the repo under test |
| `project.base_ref` | Git commit/branch to reset to |
| `project.build_params.*` | Scheme, workspace/project, bundle ID, simulator |
| `agents.*` | CLI invocation commands and pricing |

### Optional settings

| Setting | Description |
|---------|-------------|
| `suite.env` | Environment vars for agent + MCP processes |
| `suite.prewarm_spm` | Pre-resolve SwiftPM dependencies |
| `suite.plan_mode: blocked_by_scenario` | Reduce cross-scenario cache leakage |
| `suite.shuffle_within_scenario` | Randomize task order within scenario blocks |
| `suite.scenario_timeouts_sec` | Per-scenario hard timeouts |
| `suite.stall_timeout_sec` | Abort if no activity for N seconds |
| `suite.post_run_report` | Generate tool-error summary after run |
| `suite.summary_cache_strata` | Emit cold-only summaries |
| `suite.validate_tasks` | Validate reference solutions before trials |
| `mcp.*` | MCP server start/stop commands |

**Notes:**
- `repo_path` may be a subdirectory; the harness detects the git root
- Relative paths are resolved from the config file location

## Defining tasks

Tasks are defined in `tasks.yaml`:

```yaml
- id: my_task
  prompt: "Fix the failing test..."
  graders:
    - ios_test_pass
  setup_commands:
    - "cp $EVAL_SUITE_ROOT/task_assets/my_task/TASK.md ."
  reference_patch: "task_assets/my_task/reference.patch"
  kind: capability  # or regression
```

Environment variables available to setup commands:
- `EVAL_SUITE_ROOT` — Eval suite directory
- `EVAL_RUN_ID` — Current trial ID
- `EVAL_REPO_ROOT` — Worktree root
- `EVAL_REPO_WORKDIR` — Working directory within repo

## Running the suite

### Basic run

```bash
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 10
```

### Run specific tasks

```bash
python3 run_suite.py --config config.yaml --tasks tasks.yaml \
    --task-ids hn_api_cache_ttl --trials 1
```

### Streaming output

For real-time agent output during debugging:

```bash
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 1 --stream
```

The harness prints transcript paths per trial for `tail -f` monitoring.

### Run specific agents

```bash
# Run just Codex
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 30 --agents codex

# Run just Claude (use agent ID from config)
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 30 --agents claude-sonnet
```

## Resuming interrupted runs

Long runs can be interrupted. Results are written incrementally to `runs.jsonl`:

```bash
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 30 \
    --resume runs/20260120_202830
```

Resume into the same directory when running agents separately:

```bash
# First agent
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 30 --agents codex

# Second agent (resume)
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 30 \
    --agents claude-sonnet --resume runs/<timestamp>
```

## Estimating cost and time

Before a large run, estimate based on a previous run:

```bash
python3 estimate_run.py --run runs/20260120_202830 --trials 30
```

Use `--json` for machine-readable output.

## Task validation

Validate reference solutions before running trials:

```bash
# Validate and run
python3 run_suite.py --config config.yaml --tasks tasks.yaml --validate-tasks

# Validate only (no trials)
python3 run_suite.py --config config.yaml --tasks tasks.yaml --validate-tasks-only
```

Results are written to `runs/<timestamp>/reference_checks/reference_checks.jsonl`.

## Outputs

All outputs go to `./runs/<timestamp>/`:

| File | Description |
|------|-------------|
| `runs.jsonl` | Full trial records (JSON per line) |
| `runs.csv` | Flat per-trial table |
| `summary.csv` | Aggregates per {agent, scenario, task} |
| `summary.md` | Markdown summary table |
| `summary_cold.*` | Cold-only aggregates (when enabled) |
| `run_metadata.json` | Host + toolchain info |
| `trials/<run_id>/` | Per-trial artifacts |

### Rebuilding reports

Recompute summaries for an existing run after parser/analysis fixes:

```bash
python3 rebuild_reports.py --run runs/<timestamp> --config config.yaml --in-place
```

## Entry points

| Script | Purpose |
|--------|---------|
| `run_suite.py` | Run the evaluation suite |
| `rebuild_reports.py` | Recompute summaries for existing run |
| `estimate_run.py` | Estimate cost/time from previous run |

## Makefile shortcuts

```bash
make deps-dev              # Install dependencies
make test                  # Run unit tests
make typecheck             # Run pyright
make check                 # Run all checks
make run CONFIG=... TASKS=... TRIALS=10
make rebuild RUN=runs/<timestamp>
make estimate RUN=runs/<timestamp> TRIALS=30
make smoke                 # Quick sanity check
```

## Implementation notes

### Isolation and measurement

- **Git worktrees** isolate each trial (clean environment per run)
- **Command shims** (PATH overlay) count `xcodebuild`/`xcrun` without parsing transcripts
- **Outcome-based grading** — Harness runs tests/checks, not transcript analysis

### Transcripts

Transcripts are **minimal by default** (assistant messages + tool calls/results, truncated to 2000 chars). Options:
- `suite.transcript_mode: "raw"` — Full stream JSON
- `suite.transcript_mode: "none"` — Skip transcripts

### Agent environment isolation

When `suite.clean_agent_env: true`:
- **Codex**: Runs with isolated `CODEX_HOME` (copies auth only)
- **Claude**: Uses `--disable-slash-commands` and `--setting-sources local`

### MCP configuration

- Shell scenarios: Empty MCP config (prevents leakage)
- MCP scenarios: Uses `mcp_configs/xcodebuildmcp_only.json`

### Graders

- `ios_test_pass` validates tests actually executed (non-zero test count)
- `git_diff_forbidden` snapshots files **after setup** (catches injected fixtures)
- Per-trial `-derivedDataPath` ensures stable builds

### Cost tracking

- **Claude**: Uses provider-reported `total_cost_usd` when available
- **Codex**: Uses `@ccusage/codex` (reads session logs); falls back to computed cost

Cache diagnostics per trial:
- `cold_equivalent_cost_usd` — Cost if cached reads were billed at full rate
- `cache_savings_usd` — Cold-equivalent minus billed
- `cache_read_rate` — Fraction of input from cache

For detailed metric definitions, see [EVAL_GUIDE.md](EVAL_GUIDE.md).
