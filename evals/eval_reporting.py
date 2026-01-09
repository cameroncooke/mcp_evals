from __future__ import annotations

import csv
import json
import math
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple


def percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs_sorted[int(k)])
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return float(d0 + d1)


def mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def stdev(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def cv(xs: List[float]) -> Optional[float]:
    m = mean(xs)
    s = stdev(xs)
    if m is None or s is None or m == 0:
        return None
    return float(s / m)


SESSION_DEFAULTS_DISCOVERY_PATTERNS = [
    "Missing required session defaults",
]


def count_session_defaults_discovery(tool_error_log_path: Optional[str]) -> int:
    if not tool_error_log_path:
        return 0
    log_path = pathlib.Path(tool_error_log_path)
    if not log_path.exists():
        return 0
    try:
        with log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    err = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if err.get("tool_kind") != "mcp":
                    continue
                payload = err.get("payload", "")
                if any(p in payload for p in SESSION_DEFAULTS_DISCOVERY_PATTERNS):
                    return 1
    except Exception:
        return 0
    return 0


def parse_usage(provider_usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert provider usage into unified fields.

    Supports:
    - OpenAI-ish: {prompt_tokens, completion_tokens, total_tokens, ...}
    - Anthropic-ish: {inputTokens|input_tokens, outputTokens|output_tokens,
                      cacheReadTokens|cacheReadInputTokens|cache_read_input_tokens,
                      cacheWriteTokens|cache_creation_input_tokens|cache_write_tokens, ...}
    """
    out = {
        "model": None,
        "uncached_input_tokens": None,
        "output_tokens": None,
        "cached_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_write_ttl": None,
        "billed_cost_usd": None,
    }
    if not provider_usage:
        return out

    # model fields
    for k in ["model", "model_id", "modelName"]:
        if k in provider_usage and provider_usage[k]:
            out["model"] = provider_usage[k]
            break
    if out["model"] is None and isinstance(provider_usage.get("modelUsage"), dict):
        # Claude stream format can provide modelUsage as a map.
        keys = list(provider_usage["modelUsage"].keys())
        if keys:
            out["model"] = keys[0]

    # cost sometimes already computed/reported
    for k in ["cost_usd", "billed_cost_usd", "total_cost_usd", "usd_cost"]:
        v = provider_usage.get(k)
        if isinstance(v, (int, float)):
            out["billed_cost_usd"] = float(v)
            break

    usage_src = provider_usage
    if isinstance(provider_usage.get("usage"), dict):
        usage_src = provider_usage["usage"]

    # OpenAI Responses-style usage
    if (
        ("input_tokens" in usage_src or "output_tokens" in usage_src)
        and not any(
            k in usage_src
            for k in [
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "cacheReadInputTokens",
                "cache_creation",
            ]
        )
    ):
        input_tokens = int(usage_src.get("input_tokens") or 0)
        out["output_tokens"] = int(usage_src.get("output_tokens") or 0)
        if usage_src.get("cached_input_tokens") is not None:
            try:
                out["cached_read_tokens"] = int(usage_src.get("cached_input_tokens") or 0)
            except Exception:
                pass
        details = usage_src.get("input_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            try:
                out["cached_read_tokens"] = int(details["cached_tokens"])
            except Exception:
                pass
        if out["cached_read_tokens"]:
            out["uncached_input_tokens"] = max(
                0, input_tokens - int(out["cached_read_tokens"])
            )
        else:
            out["uncached_input_tokens"] = input_tokens
        return out

    # OpenAI
    if "prompt_tokens" in usage_src or "completion_tokens" in usage_src:
        prompt_tokens = int(usage_src.get("prompt_tokens") or 0)
        out["output_tokens"] = int(usage_src.get("completion_tokens") or 0)
        # Some clients expose cached tokens separately; treat as cached_read if present.
        for k in ["cached_tokens", "cached_input_tokens", "cached_input"]:
            if k in usage_src and usage_src[k] is not None:
                try:
                    out["cached_read_tokens"] = int(usage_src[k])
                    break
                except Exception:
                    pass
        # OpenAI-style nested cached tokens
        if not out["cached_read_tokens"]:
            details = usage_src.get("prompt_tokens_details")
            if isinstance(details, dict) and details.get("cached_tokens") is not None:
                try:
                    out["cached_read_tokens"] = int(details["cached_tokens"])
                except Exception:
                    pass
        if out["cached_read_tokens"]:
            out["uncached_input_tokens"] = max(
                0, prompt_tokens - int(out["cached_read_tokens"])
            )
        else:
            out["uncached_input_tokens"] = prompt_tokens
        return out

    # Anthropic / Claude
    def first_int(keys: List[str]) -> Optional[int]:
        for k in keys:
            if k in usage_src and usage_src[k] is not None:
                try:
                    return int(usage_src[k])
                except Exception:
                    continue
        return None

    out["uncached_input_tokens"] = first_int(
        ["inputTokens", "input_tokens", "input_tokens_total", "input"]
    )
    out["output_tokens"] = first_int(["outputTokens", "output_tokens", "output"])
    out["cached_read_tokens"] = (
        first_int(
            [
                "cacheReadTokens",
                "cacheReadInputTokens",
                "cache_read_input_tokens",
                "cache_read_tokens",
            ]
        )
        or 0
    )
    out["cache_write_tokens"] = (
        first_int(
            [
                "cacheWriteTokens",
                "cache_write_tokens",
                "cache_creation_input_tokens",
                "cache_write_input_tokens",
            ]
        )
        or 0
    )
    # TTL if reported
    ttl = provider_usage.get("cache_write_ttl") or provider_usage.get("cache_ttl")
    if isinstance(ttl, str):
        out["cache_write_ttl"] = ttl
    return out


def compute_cost(unified: Dict[str, Any], pricing: Dict[str, float]) -> Optional[float]:
    """
    Compute billed cost (USD) from unified token components and pricing.
    If unified already contains billed_cost_usd, return it.

    pricing expects:
    - input_per_token
    - output_per_token
    - (optional) cache_read_multiplier (default 1.0)
    - (optional) cache_write_multiplier_5m / _1h
    """
    if unified.get("billed_cost_usd") is not None:
        return float(unified["billed_cost_usd"])

    inp = unified.get("uncached_input_tokens")
    outp = unified.get("output_tokens")
    if inp is None or outp is None:
        return None

    input_rate = float(pricing.get("input_per_token", 0.0))
    output_rate = float(pricing.get("output_per_token", 0.0))

    cache_read_tokens = int(unified.get("cached_read_tokens") or 0)
    cache_write_tokens = int(unified.get("cache_write_tokens") or 0)
    cache_read_mult = float(pricing.get("cache_read_multiplier", 1.0))

    ttl = unified.get("cache_write_ttl")
    if ttl == "1h":
        cache_write_mult = float(pricing.get("cache_write_multiplier_1h", 1.0))
    else:
        # default 5m
        cache_write_mult = float(pricing.get("cache_write_multiplier_5m", 1.0))

    cost = 0.0
    cost += int(inp) * input_rate
    cost += cache_read_tokens * input_rate * cache_read_mult
    cost += cache_write_tokens * input_rate * cache_write_mult
    cost += int(outp) * output_rate
    return float(cost)


def compute_cold_equivalent_cost(
    unified: Dict[str, Any], pricing: Dict[str, float]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute cost as if all input tokens were billed at full input rate (no caching).
    Returns (cold_equivalent_cost_usd, cache_read_rate).

    Total input tokens = uncached_input + cached_read + cache_write
    - uncached_input: tokens not in cache, billed at full rate
    - cached_read: tokens read from cache (would be full rate if cold)
    - cache_write: tokens written to cache (still input tokens, billed at write rate)

    cache_read_rate = cached_read / total_input (proportion that benefited from cache)
    """
    inp = unified.get("uncached_input_tokens")
    outp = unified.get("output_tokens")
    if inp is None or outp is None:
        return None, None
    cached_read_tokens = int(unified.get("cached_read_tokens") or 0)
    cache_write_tokens = int(unified.get("cache_write_tokens") or 0)
    total_input = int(inp) + cached_read_tokens + cache_write_tokens
    input_rate = float(pricing.get("input_per_token", 0.0))
    output_rate = float(pricing.get("output_per_token", 0.0))
    cold_cost = total_input * input_rate + int(outp) * output_rate
    cache_read_rate = None
    if total_input > 0:
        cache_read_rate = cached_read_tokens / total_input
    return float(cold_cost), cache_read_rate


def is_cold_row(row: Dict[str, Any]) -> bool:
    return int(row.get("cached_read_tokens") or 0) == 0


def aggregate(
    rows: List[Dict[str, Any]],
    row_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregates non-baseline rows by {agent, scenario, task}.
    Also computes pass@k and pass^k based on success_rate, plus CV for time/cost.
    Optional row_filter can be used to compute stratified summaries (e.g., cache-cold only).
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("baseline_run"):
            continue
        if row_filter and not row_filter(row):
            continue
        grouped.setdefault((row["agent_id"], row["scenario"], row["task_id"]), []).append(
            row
        )

    out: List[Dict[str, Any]] = []
    for (agent_id, scenario, task_id), rs in sorted(grouped.items()):
        n = len(rs)
        succ = sum(1 for r in rs if r["success"])
        p = succ / n if n else 0.0

        times = [
            float(r["wall_time_sec"]) for r in rs if r["wall_time_sec"] is not None
        ]
        costs = [
            float(r["billed_cost_usd"])
            for r in rs
            if r["billed_cost_usd"] is not None
        ]
        cold_costs = [
            float(r["cold_equivalent_cost_usd"])
            for r in rs
            if r.get("cold_equivalent_cost_usd") is not None
        ]
        cache_savings = [
            float(r["cache_savings_usd"])
            for r in rs
            if r.get("cache_savings_usd") is not None
        ]
        cache_read_rates = [
            float(r["cache_read_rate"])
            for r in rs
            if r.get("cache_read_rate") is not None
        ]
        mcp_calls = [
            float(r["mcp_tool_calls"])
            for r in rs
            if r.get("mcp_tool_calls") is not None
        ]
        time_to_first = [
            float(r["time_to_first_xcodebuild_sec"])
            for r in rs
            if r.get("time_to_first_xcodebuild_sec") is not None
        ]
        time_to_first_mcp_build = [
            float(r["time_to_first_mcp_build_sec"])
            for r in rs
            if r.get("time_to_first_mcp_build_sec") is not None
        ]
        repeat_counts = [
            float(r["xcodebuild_repeat_count"])
            for r in rs
            if r.get("xcodebuild_repeat_count") is not None
        ]
        destination_counts = [
            float(r["destination_count"])
            for r in rs
            if r.get("destination_count") is not None
        ]
        destination_churn = [
            float(r["destination_churn"])
            for r in rs
            if r.get("destination_churn") is not None
        ]
        tool_errors: List[float] = []
        tool_errors_mcp: List[float] = []
        tool_errors_non_mcp: List[float] = []
        for r in rs:
            if r.get("tool_error_total") is None:
                continue
            total = float(r["tool_error_total"])
            mcp = float(r.get("tool_error_mcp") or 0)
            non_mcp = float(r.get("tool_error_non_mcp") or 0)
            if str(r.get("scenario", "")).startswith("mcp_"):
                discovery = count_session_defaults_discovery(r.get("tool_error_log_path"))
                if discovery:
                    total = max(0.0, total - discovery)
                    mcp = max(0.0, mcp - discovery)
            tool_errors.append(total)
            tool_errors_mcp.append(mcp)
            tool_errors_non_mcp.append(non_mcp)

        out.append(
            {
                "agent_id": agent_id,
                "scenario": scenario,
                "task_id": task_id,
                "task_kind": rs[0].get("task_kind"),
                "runs": n,
                "success_rate": p,
                "pass_at_1": p,
                "pass_at_3": 1 - (1 - p) ** 3,
                "pass_pow_3": p**3,
                "time_p10": percentile(times, 10),
                "time_median": percentile(times, 50),
                "time_p90": percentile(times, 90),
                "time_mean": mean(times),
                "time_std": stdev(times),
                "time_cv": cv(times),
                "cost_p10": percentile(costs, 10),
                "cost_median": percentile(costs, 50),
                "cost_p90": percentile(costs, 90),
                "cost_mean": mean(costs),
                "cost_std": stdev(costs),
                "cost_cv": cv(costs),
                "cold_cost_median": percentile(cold_costs, 50),
                "cold_cost_p90": percentile(cold_costs, 90),
                "cold_cost_cv": cv(cold_costs),
                "cache_savings_mean": mean(cache_savings),
                "cache_read_rate_mean": mean(cache_read_rates),
                "cost_per_success_mean": (sum(costs) / succ) if succ > 0 else None,
                "time_per_success_mean": (sum(times) / succ) if succ > 0 else None,
                "xcodebuild_calls_mean": mean([r["xcodebuild_calls"] for r in rs]),
                "xcrun_calls_mean": mean([r["xcrun_calls"] for r in rs]),
                "simctl_calls_mean": mean([r["simctl_calls"] for r in rs]),
                "mcp_tool_calls_mean": mean(mcp_calls),
                "time_to_first_xcodebuild_mean": mean(time_to_first),
                "time_to_first_mcp_build_mean": mean(time_to_first_mcp_build),
                "xcodebuild_repeat_mean": mean(repeat_counts),
                "destination_count_mean": mean(destination_counts),
                "destination_churn_mean": mean(destination_churn),
                "tool_error_mean": mean(tool_errors),
                "tool_error_mcp_mean": mean(tool_errors_mcp),
                "tool_error_non_mcp_mean": mean(tool_errors_non_mcp),
            }
        )
    return out


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_markdown_summary(
    path: pathlib.Path, summary_rows: List[Dict[str, Any]]
) -> None:
    if not summary_rows:
        path.write_text("No results.\n")
        return
    preface = [
        "# Summary",
        "",
        "Cost interpretation:",
        "- `cost_*` = billed cost (includes cache discounts and any upfront tool schema overhead).",
        "- `cold_cost_*` = cold-equivalent cost (treat cached reads as uncached for like-for-like A/B/C comparisons).",
        "- `summary_cold.*` (when enabled) filters to runs where `cached_read_tokens == 0`.",
        "",
        "MCP vs shell comparison:",
        "- Use `cost_*` for overall cost (includes MCP schema overhead).",
        "- Use `cold_cost_*` or `summary_cold.*` for like-for-like cost.",
        "- Use `marginal_cost_usd` for work-only cost (baseline subtracted).",
        "",
    ]
    cols = [
        "agent_id",
        "scenario",
        "task_id",
        "task_kind",
        "runs",
        "success_rate",
        "time_median",
        "time_p90",
        "time_cv",
        "cost_median",
        "cost_p90",
        "cost_cv",
        "cold_cost_median",
        "cold_cost_p90",
        "cold_cost_cv",
        "cache_savings_mean",
        "cache_read_rate_mean",
        "pass_at_3",
        "pass_pow_3",
        "cost_per_success_mean",
        "xcodebuild_calls_mean",
        "xcrun_calls_mean",
        "simctl_calls_mean",
        "mcp_tool_calls_mean",
        "time_to_first_xcodebuild_mean",
        "time_to_first_mcp_build_mean",
        "xcodebuild_repeat_mean",
        "destination_count_mean",
        "destination_churn_mean",
        "tool_error_mean",
        "tool_error_mcp_mean",
        "tool_error_non_mcp_mean",
    ]

    def fmt(v: Any, col: str) -> str:
        if v is None:
            return ""
        if col in ("success_rate", "pass_at_3", "pass_pow_3"):
            return f"{float(v)*100:.1f}%"
        if col.startswith("time_"):
            return f"{float(v):.1f}"
        if col.endswith("_rate") or col.endswith("_rate_mean"):
            return f"{float(v)*100:.1f}%"
        if (
            col.startswith("cost_")
            or col.startswith("cold_cost_")
            or col.endswith("_usd")
            or col.endswith("_mean")
        ):
            return f"{float(v):.6f}"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = []
    lines.extend(preface)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in summary_rows:
        lines.append("| " + " | ".join(fmt(r.get(c), c) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def recompute_rows(
    rows: List[Dict[str, Any]], pricing_by_agent: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    # Update unified usage fields (including cached_read_tokens) and cache-aware costs.
    for r in rows:
        provider_usage = r.get("provider_usage")
        unified = None
        if provider_usage:
            unified = parse_usage(provider_usage)
            for key in (
                "model",
                "uncached_input_tokens",
                "output_tokens",
                "cached_read_tokens",
                "cache_write_tokens",
                "cache_write_ttl",
            ):
                if key in unified:
                    r[key] = unified.get(key)

        pricing = pricing_by_agent.get(r.get("agent_id") or "", {})
        unified_for_cost = {
            "uncached_input_tokens": r.get("uncached_input_tokens"),
            "cached_read_tokens": r.get("cached_read_tokens"),
            "output_tokens": r.get("output_tokens"),
        }
        cold_equiv, cache_read_rate = compute_cold_equivalent_cost(
            unified_for_cost, pricing
        )
        r["cold_equivalent_cost_usd"] = cold_equiv
        r["cache_read_rate"] = cache_read_rate
        billed = r.get("billed_cost_usd")
        r["cache_savings_usd"] = (
            float(cold_equiv) - float(billed)
            if cold_equiv is not None and billed is not None
            else None
        )

    # Recompute baseline + marginal costs
    baseline_cost: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if r.get("baseline_run") and r.get("billed_cost_usd") is not None:
            agent_id = r.get("agent_id")
            scenario = r.get("scenario")
            if agent_id is None or scenario is None:
                continue
            baseline_cost[(str(agent_id), str(scenario))] = float(r["billed_cost_usd"])
    for r in rows:
        if r.get("baseline_run"):
            continue
        agent_id = r.get("agent_id")
        scenario = r.get("scenario")
        bc = (
            baseline_cost.get((str(agent_id), str(scenario)))
            if agent_id is not None and scenario is not None
            else None
        )
        r["baseline_cost_usd"] = bc
        if bc is not None and r.get("billed_cost_usd") is not None:
            r["marginal_cost_usd"] = float(r["billed_cost_usd"]) - float(bc)
    return rows
