"""
Eval Suite Package

A modular evaluation framework for AI coding agents.
"""

from evals.config import (
    AgentConfig,
    ProjectConfig,
    MCPConfig,
    SuiteConfig,
    TaskSpec,
    load_config,
    load_tasks,
    validate_project_config,
    validate_suite_config,
)

from evals.infrastructure import (
    run_cmd,
    safe_mkdir,
    now_ts,
    resolve_simulator_udid,
    resolve_simulator_udid_for_project,
    resolve_repo_layout,
    resolve_developer_dir,
    ensure_simulator_booted,
    reset_simulator_app_state,
    scrub_env,
    toml_string,
    toml_array,
    extract_udid_from_destination,
)

from evals.xcresult import (
    read_xcresult_json,
    read_xcresult_test_summary,
    extract_xcresult_id,
    find_xcresult_tests_ref_id,
    count_xcresult_tests_node,
    count_xcresult_tests,
)

from evals.metrics import (
    count_invocations,
    count_simctl_invocations,
    read_command_log_entries,
    parse_log_timestamp,
    extract_xcodebuild_destination,
    normalize_xcodebuild_argv,
    compute_xcodebuild_repeat_count,
    compute_time_to_first_xcodebuild_sec,
    count_mcp_tool_usage,
    count_mcp_tool_invocations,
    compute_time_to_first_mcp_build_sec,
    log_stream_command_invocation,
    log_stream_mcp_invocation,
)

from evals.worktrees import (
    make_worktree,
    remove_worktree,
)

from evals.graders import (
    GraderResult,
    sha256_file,
    normalize_repo_paths,
    path_matches,
    git_repo_root,
    snapshot_forbidden_files,
    capture_forbidden_baseline,
    grader_ios_install_check,
    grader_ios_launch_check,
    grader_ios_test_pass,
    grader_git_diff_forbidden,
    grader_screenshot_exists,
    grader_screenshot_compare,
    run_graders,
)

from evals.agents import (
    AgentAdapter,
    TemplateCommandAgent,
    make_agent,
    minimal_transcript_lines_claude,
    minimal_transcript_lines_codex,
    minimal_transcript_lines,
    iter_claude_tool_calls,
    log_claude_mcp_invocations,
    ccusage_codex_cost,
    read_tool_error_summary,
    build_prompt,
    make_command_shims,
    codex_mcp_args_from_json,
)

from evals.trial import (
    TrialResult,
    run_one_trial,
)

from evals.reporting import (
    results_to_rows,
    write_jsonl,
    append_jsonl,
    load_jsonl,
    get_completed_trial_keys,
    select_post_run_agent,
    prepare_report_env,
    build_tool_error_report_manifest,
    run_post_run_report,
    build_failure_analysis_manifest,
    run_failure_analysis_report,
    load_prompt,
)

SCENARIOS = ["shell_unprimed", "shell_primed", "mcp_unprimed", "mcp_unprimed_v2"]

__all__ = [
    # Constants
    "SCENARIOS",
    # Config
    "AgentConfig",
    "ProjectConfig",
    "MCPConfig",
    "SuiteConfig",
    "TaskSpec",
    "load_config",
    "load_tasks",
    "validate_project_config",
    "validate_suite_config",
    # Infrastructure
    "run_cmd",
    "safe_mkdir",
    "now_ts",
    "resolve_simulator_udid",
    "resolve_simulator_udid_for_project",
    "resolve_repo_layout",
    "resolve_developer_dir",
    "ensure_simulator_booted",
    "reset_simulator_app_state",
    "scrub_env",
    "toml_string",
    "toml_array",
    "extract_udid_from_destination",
    # XCResult
    "read_xcresult_json",
    "read_xcresult_test_summary",
    "extract_xcresult_id",
    "find_xcresult_tests_ref_id",
    "count_xcresult_tests_node",
    "count_xcresult_tests",
    # Metrics
    "count_invocations",
    "count_simctl_invocations",
    "read_command_log_entries",
    "parse_log_timestamp",
    "extract_xcodebuild_destination",
    "normalize_xcodebuild_argv",
    "compute_xcodebuild_repeat_count",
    "compute_time_to_first_xcodebuild_sec",
    "count_mcp_tool_usage",
    "count_mcp_tool_invocations",
    "compute_time_to_first_mcp_build_sec",
    "log_stream_command_invocation",
    "log_stream_mcp_invocation",
    # Worktrees
    "make_worktree",
    "remove_worktree",
    # Graders
    "GraderResult",
    "sha256_file",
    "normalize_repo_paths",
    "path_matches",
    "git_repo_root",
    "snapshot_forbidden_files",
    "capture_forbidden_baseline",
    "grader_ios_install_check",
    "grader_ios_launch_check",
    "grader_ios_test_pass",
    "grader_git_diff_forbidden",
    "grader_screenshot_exists",
    "grader_screenshot_compare",
    "run_graders",
    # Agents
    "AgentAdapter",
    "TemplateCommandAgent",
    "make_agent",
    "minimal_transcript_lines_claude",
    "minimal_transcript_lines_codex",
    "minimal_transcript_lines",
    "iter_claude_tool_calls",
    "log_claude_mcp_invocations",
    "ccusage_codex_cost",
    "read_tool_error_summary",
    "build_prompt",
    "make_command_shims",
    "codex_mcp_args_from_json",
    # Trial
    "TrialResult",
    "run_one_trial",
    # Reporting
    "results_to_rows",
    "write_jsonl",
    "append_jsonl",
    "load_jsonl",
    "get_completed_trial_keys",
    "select_post_run_agent",
    "prepare_report_env",
    "build_tool_error_report_manifest",
    "run_post_run_report",
    "build_failure_analysis_manifest",
    "run_failure_analysis_report",
    "load_prompt",
]
