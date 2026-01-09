# Task: Make HNApi testable offline with fixtures + deterministic tests

## Goal

Refactor the HackerNews iOS codebase so **HNApi can be tested offline** using local JSON fixtures.
Deterministic unit tests are already provided by the harness; your job is to make the
app code changes needed for those tests to pass.

The harness has already added failing tests and fixture files.

## Requirements

1) **Inject networking into `HNApi`**
   - Replace hard-coded `URLSession.shared` usage.
   - Add an initializer that accepts a `URLSession` (default to `.shared`).
   - Ensure `fetchStories`, `fetchPage`, and `fetchItems` use the injected session.

2) **Use fixtures from the test directory**
   - Fixtures are in `HackerNewsTests/Fixtures/`.
   - Tests load fixtures directly from disk (no bundle resource wiring required).

3) **Tests must be deterministic**
   - Do not hit the network.
   - Use the provided URLProtocol stub in tests.
   - Tests should pass with the fixture data and always return the same results.

4) **Run the relevant tests until green**
   - Run only the new test suite: `HackerNewsTests/HNApiOfflineTests`.
   - For efficiency, avoid running additional test suites once these tests pass.

## Definition of Done

- `HNApi` uses an injected `URLSession`.
- Fixture files are present in `HackerNewsTests/Fixtures/` and loaded from disk by the tests.
- The new tests pass reliably without network access.
