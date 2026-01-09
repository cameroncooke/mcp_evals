Generate a concise post-run report on tool errors.

## Manifest Location
{manifest_path}

## Output Report
Write report to: {report_path}

## Important Context
The manifest contains ONLY runs that had tool errors. Use the summary stats at the top of the manifest to report accurate percentages:
- `total_runs`: Total non-baseline runs in the eval
- `runs_with_errors`: Runs that had at least one tool error
- `runs_without_errors`: Runs with zero tool errors
- `error_rate_percent`: Percentage of runs with errors
- `total_runs_by_agent`: Total runs per agent
- `runs_with_errors_by_agent`: Runs with errors per agent
- `errors_by_scenario`: Error counts per scenario with MCP/non-MCP breakdown
  - Each scenario has: `{"total": N, "mcp": N, "non_mcp": N}`

Use these to calculate accurate error rates per agent (e.g., "X/Y agent runs (Z%) had tool errors").

When reporting errors by scenario, show the full breakdown:
```
| Scenario         | Total | MCP | Non-MCP |
|------------------|-------|-----|---------|
| mcp_unprimed     |   ... | ... |     ... |
| shell_unprimed   |   ... |   0 |     ... |
| shell_primed     |   ... |   0 |     ... |
```

### Adjusted MCP Error Analysis (Session-Defaults Discovery)
XcodeBuildMCP requires agents to call `set_session_defaults` before most other tools will work. Agents discovering the MCP for the first time don't know this prerequisite—they call a tool, get a helpful error explaining the requirement, then correct by setting defaults. These are "discovery errors," not failures.

**Identification:** Only the FIRST MCP error per trial containing `"Missing required session defaults"` counts as discovery. Subsequent occurrences in the same trial are agent errors (the agent should have learned after the first correction).

**Reporting requirement:** Calculate and report BOTH:
1. **Raw MCP error count/rate** - All MCP errors as recorded
2. **Adjusted MCP error count/rate** - Excluding session-defaults discovery errors

Include a section like:
```
### Adjusted MCP Error Rate

Of the N MCP-specific tool errors recorded, X (Y%) were "session defaults" discovery
errors—the agent calling a tool before setting the project path, getting a helpful
error, and immediately correcting. These aren't failures; they're the expected
learning curve for MCP discovery.

| Metric | Raw | Adjusted (excl. discovery) |
|--------|-----|----------------------------|
| MCP errors | N | N - X |
| MCP share of all errors | A% | B% |

**Raw conclusion:** [conclusion based on raw numbers]

**Adjusted conclusion:** [conclusion based on adjusted numbers, noting the discovery
errors represent expected agent learning behavior, not tool failures]
```

This allows readers to draw their own conclusions based on whether they consider discovery errors as "real" failures.

## Requirements

1. **Summarize tool errors by agent/scenario**
   - Report error RATE per agent (runs with errors / total runs)
   - Count total errors per agent
   - Break down by MCP vs non-MCP tools
   - Identify which scenarios have the most errors

2. **Analyze error patterns**
   - For each error with context, note whether follow-up calls repeat the same command/tool or change parameters (possible agent correction)
   - Identify repeated errors vs one-time failures
   - Note any error cascades (one error leading to many more)

3. **Classification**
   - Call out likely agent mistakes vs transient/environmental failures
   - Agent mistakes: wrong parameters, bad syntax, incorrect tool usage
   - Environmental: rate limits, timeouts, server errors

4. **Actionable insights**
   - What are the most common error patterns?
   - What could agents do differently?
   - Are there tool usability issues to address?

## Instructions
- Read the manifest and referenced log files
- Do not run builds/tests; only read files
- Keep it short and actionable
- Focus on patterns and insights, not exhaustive listings

Then exit.
