# Agent instructions (project maintained)

- Do not inject or rely on this file for eval harness runs; it is only for working on the eval suite itself.
- When creating patch files for eval tasks, validate them with `git apply --check` against the target repo to catch malformed hunks early.
- If transcript logs are too noisy for evaluation, default to a minimal structured transcript (tool calls/results + assistant text) and make raw logs opt-out.
- When using `git ls-files` to build repo-root paths (e.g., hashing/forbidden diffs), run it from the repo root or add `--full-name`; subdir workdirs otherwise produce paths relative to the subdir.
- When running evals, use `tmux` and provide the exact connection command: `tmux attach -t <session>`.
- Always update `PROGRESS.md` for eval-suite work, then report the next task explicitly.
- When evaluating MCP scenarios, do not bias behavior toward MCP usage; record actual tool usage as an outcome rather than enforcing it.
- When asked to check-in on evals, capture tmux output non-interactively with `tmux capture-pane` and summarize status.
- Never delete or directly modify eval data in the runs/ directory unless explicitly instructed to do so.

# Agent instructions (agent maintained)

- For MCP vs non‑MCP comparisons, block runs by scenario and report cache‑cold plus cache‑neutral cost views to reduce provider cache skew.
- Wall‑clock metrics should remain **agent‑only**; do not replace them with pipeline/setup/grader time. If end‑to‑end timing is added, keep it as an additional field.
