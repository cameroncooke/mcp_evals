# Summary

Cost interpretation:
- `cost_*` = billed cost (includes cache discounts and any upfront tool schema overhead).
- `cold_cost_*` = cold-equivalent cost (treat cached reads as uncached for like-for-like A/B/C comparisons).
- `summary_cold.*` (when enabled) filters to runs where `cached_read_tokens == 0`.

MCP vs shell comparison:
- Use `cost_*` for overall cost (includes MCP schema overhead).
- Use `cold_cost_*` or `summary_cold.*` for like-for-like cost.
- Use `marginal_cost_usd` for work-only cost (baseline subtracted).

| agent_id | scenario | task_id | task_kind | runs | success_rate | time_median | time_p90 | time_cv | cost_median | cost_p90 | cost_cv | cold_cost_median | cold_cost_p90 | cold_cost_cv | cache_savings_mean | cache_read_rate_mean | pass_at_3 | pass_pow_3 | cost_per_success_mean | xcodebuild_calls_mean | xcrun_calls_mean | simctl_calls_mean | mcp_tool_calls_mean | time_to_first_xcodebuild_mean | time_to_first_mcp_build_mean | xcodebuild_repeat_mean | destination_count_mean | destination_churn_mean | tool_error_mean | tool_error_mcp_mean | tool_error_non_mcp_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet | mcp_unprimed | hn_settings_deeplink | capability | 1 | 0.0% | 480.0 | 480.0 |  |  |  |  |  |  |  |  |  | 0.0% | 0.0% |  | 0.000000 | 0.000000 | 0.000000 | 35.000000 |  | 200.5 | 0.000000 | 0.000000 | 0.000000 | 11.000000 | 0.000000 | 11.000000 |
| codex | shell_unprimed | hn_settings_deeplink | capability | 1 | 0.0% | 63.0 | 63.0 |  | 0.106894 | 0.106894 |  |  |  |  |  |  | 0.0% | 0.0% |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
