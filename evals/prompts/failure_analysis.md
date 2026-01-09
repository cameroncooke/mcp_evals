Analyze failed evaluation runs and produce a detailed failure analysis report.

## Evaluation Context

This evaluation compares AI coding agents (Codex CLI vs Claude Code CLI) to understand:
1. How effective MCP tool servers (like XcodeBuildMCP) are for building iOS apps compared to agents using native shell tools
2. Agent success rates, runtime/wall-time, and token usage/cost across different scenarios
3. Tool-calling behavior - how well agents use tools (both shell commands and MCP tools)

The three scenarios tested:
- shell_unprimed: No MCP tools, no build params provided - agent must discover everything
- shell_primed: No MCP tools, but exact build params provided in prompt
- mcp_unprimed: MCP tools available (XcodeBuildMCP), no build params - agent should use MCP tools

We need to understand WHY runs failed to properly interpret the evaluation results. ENVIRONMENTAL failures should be filtered out of analysis since they don't reflect agent quality - they're infrastructure issues.

## Classification Principles

### AGENT_MISTAKE
The agent's behavior caused the failure. This is what the eval is designed to measure:
- Tool calling errors (wrong parameters, invalid tool names, malformed calls)
- Bad shell commands (wrong flags, syntax errors, incorrect paths)
- Incorrect implementation that broke tests
- Inefficient approach that exhausted time budget
- Misunderstanding task requirements

### ENVIRONMENTAL
Infrastructure failed - NOT the agent's fault. These should be excluded from agent quality analysis:
- API rate limits (429 errors) preventing the agent from running
- MCP server crashes or hangs (server infrastructure, not agent's tool call)
- Simulator crashes, disk full, network failures
- Any infrastructure issue outside the agent's control

### TASK_ISSUE
The task definition or grading has a problem - agent did the right thing but was marked wrong:
- Tests passed in transcript but harness marked as failed
- Agent followed TASK.md but grader expected different behavior
- Ambiguous or contradictory task requirements

### UNKNOWN
Genuinely cannot determine from available evidence.

For scenarios not explicitly covered above, use your judgment based on the core question: "Did the AGENT do something wrong, or did something EXTERNAL fail?"

## Manifest Location
{manifest_path}

## Output Report
Write to: {report_path}

## Report Format

The report must be DETAILED and ACTIONABLE - not just summary tables. Include:

# Failure Analysis Report

## Executive Summary
- Total failures and breakdown by classification
- Key themes/patterns observed
- Which failures should be excluded from agent quality analysis (ENVIRONMENTAL)

## Error Themes

Group failures by common patterns. For each theme:
### Theme: [Descriptive Name]
- **Classification**: AGENT_MISTAKE | ENVIRONMENTAL | TASK_ISSUE
- **Affected runs**: List of run_ids
- **What happened**: Detailed explanation of the failure pattern
- **Example**: Show actual error output, tool call that failed, or code snippet demonstrating the issue

Example of what to include:
```
Tool call that failed:
mcp__XcodeBuildMCP__build_ios_sim {{"scheme": "WrongScheme", "project": "..."}}
Error: Scheme 'WrongScheme' not found in project

This occurred in 5 runs where the agent guessed the scheme name incorrectly.
```

## Detailed Run Analysis

For runs that don't fit into themes, or that need individual explanation:
### [run_id]
- **Classification**:
- **Agent/Scenario/Task**:
- **What went wrong**: Detailed explanation
- **Evidence**: Actual transcript excerpt, error message, or code snippet
- **Tool errors**: If tool_error_total > 0, what were the actual tool errors?

## Recommendations
Based on the failure analysis, what should be improved in:
- Task definitions
- Grading logic
- Agent prompts
- Infrastructure/harness

## Instructions
- Read each transcript file listed in the manifest
- Look for actual error messages, failed tool calls, test output
- Include SPECIFIC examples - code snippets, tool calls, error messages
- Group similar failures into themes rather than repeating the same analysis
- The goal is to understand WHAT actually went wrong, not just count failures
- Do not run builds/tests; only read files

Then exit.
