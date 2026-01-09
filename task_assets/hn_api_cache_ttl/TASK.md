# Task: Add deterministic in-memory response caching to HNApi

## Goal

Add a deterministic, testable response cache to `HNApi` so repeat requests within a TTL
return cached data (no network), while expired entries trigger fresh network calls.

Deterministic unit tests and fixtures are provided by the harness.

## Requirements

1) **Introduce cache + clock types in Common**
   - Add a `HNApiCache` protocol with:
     - `func get(_ key: String) -> Data?`
     - `func set(_ key: String, data: Data)`
   - Add a `HNApiClock` protocol with:
     - `func now() -> Date`
   - Add a `SystemClock` that returns `Date()`.
   - Add `HNApiInMemoryCache`:
     - Initialized with `ttlSeconds: TimeInterval` and `clock: HNApiClock`.
     - Stores raw `Data` and expires entries based on `clock.now()`.

2) **Update HNApi initializer**
   - `init(session: URLSession = .shared, cache: HNApiCache? = nil)`.
   - Preserve existing behavior when `cache == nil`.

3) **Use cache for all request paths**
   - Use cache for `fetchStories` (feed URL).
   - Use cache for item URLs in both `fetchPage` and `fetchItems`.
   - Cache key must be the **full request URL string**.

4) **Behavioral guarantees**
   - Cache hits must avoid network calls and decode from cached data.
   - Expired cache entries must trigger a network call.

5) **Run the relevant tests**
   - Run only `HackerNewsTests/HNApiCacheTests` while iterating.

## Definition of Done

- All `HNApiCacheTests` pass reliably.
- HNApi uses the injected cache for feed + item requests.
- Cache TTL behavior is deterministic via the injected clock.
