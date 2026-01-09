# Task: Make StoryContent.relativeDate deterministic in snapshot mode

## Goal

When the environment variable `EMERGE_IS_RUNNING_FOR_SNAPSHOTS` is set to `"1"`,
`StoryContent.relativeDate()` should return the fixed string **"10 minutes ago"**.
This mirrors the existing behavior in `Story.displayableDate` and keeps snapshot
tests deterministic.

The harness has already added a failing unit test for this behavior. Your job is
to update the app code so the test passes, then run tests until green.

## Requirements

- Only change app code (do not modify tests added by the harness).
- Keep the change minimal and consistent with existing patterns.
- Run only the relevant test: `HackerNewsTests/StoryContentRelativeDateTests`.
- Once this test passes, **exit immediately**. Do not run additional test runners or rerun tests.

## Definition of Done

- All tests pass on the configured simulator.
- `StoryContent.relativeDate()` returns `"10 minutes ago"` when
  `EMERGE_IS_RUNNING_FOR_SNAPSHOTS=1`.
