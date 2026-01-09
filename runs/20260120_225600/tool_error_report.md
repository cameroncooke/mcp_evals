# Tool Error Report

**Run:** `runs/20260120_225600`
**Generated:** 2026-02-02

## Summary

| Metric | Value |
|--------|-------|
| Total Runs | 1,575 |
| Runs with Errors | 842 (53.5%) |
| Runs with Real Errors | 591 (37.5%) |
| Real Errors (after exclusions) | 970 |

**Exclusions applied:** 484 session_defaults (MCP discovery), 39 sibling_cascade, 5 build_test_failure.

---

## Error Rate by Agent

| Agent | Trials | Runs w/ Real Errors | Real Errors | Error Rate |
|-------|--------|---------------------|-------------|------------|
| codex | 525 | 369 (70.3%) | 593 | 113% |
| claude-opus | 525 | 142 (27.0%) | 257 | 49% |
| claude-sonnet | 525 | 80 (15.2%) | 120 | 23% |

**Key finding:** Codex has 3-5x the error rate of Claude agents. Claude-sonnet performs best.

---

## Errors by Scenario

| Scenario | Real Errors | Notes |
|----------|-------------|-------|
| shell_unprimed | 468 | Highest - no session warmup |
| mcp_unprimed | 250 | After excluding MCP discovery |
| shell_primed | 142 | Priming helps but doesn't eliminate |
| mcp_unprimed_v2 | 110 | Improved MCP handling |

---

## Error Pattern Breakdown

### By Agent and Type

| Agent | Scenario | Pattern | Count |
|-------|----------|---------|-------|
| codex | shell_unprimed | shell_exit_nonzero | 258 |
| claude-opus | shell_unprimed | file_not_found (Read) | 160 |
| codex | shell_primed | shell_exit_nonzero | 136 |
| codex | mcp_unprimed_v2 | MCP null-error failure | 114 |
| codex | mcp_unprimed | MCP null-error failure | 113 |
| codex | mcp_unprimed | shell_exit_nonzero | 81 |
| claude-opus | shell_unprimed | xcodebuild_test_failure | 75 |
| codex | mcp_unprimed_v2 | shell_exit_nonzero | 50 |
| claude-opus | mcp_unprimed | session_defaults_missing | 30 |
| claude-sonnet | Read | file_not_found | 23 |
| claude-sonnet | shell_unprimed | xcodebuild_test_failure | 20 |
| claude-sonnet | mcp_unprimed | deeplink_openurl_failure | 17 |

---

## Agent Correction Behavior

After errors, agents exhibited the following follow-up patterns:

| Behavior | Count | % |
|----------|-------|---|
| Changed parameters (correction) | 638 | 52.8% |
| Switched to different tool | 456 | 37.7% |
| Repeated same call unchanged | 114 | 9.4% |

**90.6% of errors trigger adaptive behavior.** The 9.4% unchanged repeats suggest either transient failures or agent confusion.

---

## Agent Mistakes vs Environmental Failures

### Agent Mistakes (Actionable)

| Pattern | Count | Agent | Root Cause | Fix |
|---------|-------|-------|------------|-----|
| File Not Found (Read) | 187 | opus=160, sonnet=24 | Reads without path verification | Glob before Read |
| Shell Exit Nonzero | 525 | codex dominant | Command syntax/path errors | Command validation |
| MCP null-error | 227 | codex only | Malformed MCP requests | Input validation |
| Deep link failures | 17 | sonnet | URL opened before app ready | Wait for app launch |

### Environmental (Less Actionable)

| Pattern | Count | Notes |
|---------|-------|-------|
| MCP session_defaults | 484 | Expected discovery workflow |
| Sibling cascade | 39 | Downstream from prior error |
| xcodebuild test failures | ~100 | Test assertions, not tool errors |
| Build failures | 5 | Code issues, not agent mistakes |

---

## Recommendations

1. **Codex shell commands** - Investigate high failure rate (525 shell errors). Consider command syntax validation or linting before execution.

2. **File access pattern** - All agents, especially claude-opus, should verify file existence with Glob before Read to prevent 187 file-not-found errors.

3. **MCP input validation** - Codex MCP calls fail silently with null error (227 cases). Add input validation or clearer error messages.

4. **Deep link sequencing** - Add explicit wait/verification after app launch before attempting URL schemes.

5. **Test vs tool errors** - Consider separating "test assertion failures" from "tool errors" in reporting to reduce noise.

---

*Source: tool_error_report_manifest.json*
