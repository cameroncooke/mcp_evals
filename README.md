# XcodeBuildMCP Agent Evaluation Suite

An evaluation harness for comparing AI coding agents—**OpenAI Codex CLI** and **Claude Code CLI**—with and without **MCP tools** (specifically [XcodeBuildMCP](https://github.com/cameroncooke/XcodeBuildMCP)).

The goal is **apples-to-apples measurement** of **cost, speed, and reliability** across vendors and configurations, with explicit accounting for hidden overhead (system prompts, tool schemas, caching) and robust statistics under nondeterminism.

## Why this evaluation exists

AI coding agents are increasingly used in workflows that are **interactive** and **tool-driven** (builds, tests, simulator management). Two recurring questions come up in production:

1. **Cost and speed trade-offs** — Which agent is cheaper/faster *to complete a real session* (not just a single turn)? How much variability do we see across retries?

2. **Do tools pay for themselves?** — Tool schemas (MCP) add **context overhead** (tokens, latency). But tools can reduce guesswork, avoid dead-ends, and improve first-try success. We want to know if **adding tools is net-positive** in **total session cost** and **success probability**.

3. **How much does "priming" matter?** — If we provide exact Xcode build parameters up front, does it materially reduce failures and wasted work?

## Core hypotheses

We test these hypotheses with controlled scenarios:

- **H1 — Priming reduces cost/time and improves reliability**
  Providing correct build parameters up front reduces tool-less thrash and repeated build failures.

- **H2 — MCP tools reduce *total session* cost/time despite schema overhead**
  Even though tool definitions add tokens, tool access can reduce retries and shorten the path to passing tests.

- **H3 — Cross-vendor differences are dominated by session dynamics, not single-turn token counts**
  Caching behavior, hidden prompts, and multi-step loops can outweigh naive "tokens per response" comparisons.

## Evaluation scenarios

Each run is a **fresh session** (no prior chat history). The suite tests three scenarios:

| Scenario | MCP Tools | Build Params in Prompt | What it measures |
|----------|-----------|------------------------|------------------|
| **Shell (unprimed)** | No | No | Agent's ability to discover/guess `xcodebuild` parameters via shell |
| **Shell (primed)** | No | Yes | Best possible prompt-only performance without tools |
| **MCP (unprimed)** | Yes | No | Whether MCP tools justify schema overhead when agent is nudged to prefer them |

> MCP tools are intended to be *model-controlled* and discoverable. See [MCP Tools spec](https://modelcontextprotocol.io/specification/2025-06-18/server/tools).

## What we measure

### Primary outcomes
- **Success rate** — Deterministic graders (tests, state checks)
- **Agent runtime** — Wall-clock time for the agent process
- **Token cost** — USD cost (normalizes tokenizers, pricing, caching)

### Secondary/diagnostic metrics
- Token breakdown (uncached, cached, output)
- Tool call counts and errors
- Cache read rates
- Efficiency signals (time-to-first-build, destination churn, repeat calls)
- Variance statistics (p10/p50/p90, std, CV)
- Reliability metrics (pass@k, pass^k)

### Why USD cost is primary

Tokenizers differ across vendors—the same string yields different counts. USD cost normalizes:
- Different tokenizers and pricing
- Caching discounts
- Vendor-specific usage field shapes

For pricing details:
- [OpenAI pricing](https://platform.openai.com/docs/pricing)
- [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)

## Reliability under nondeterminism

LLM agents can fail for non-deterministic reasons. We treat evaluation as a distribution:

- **pass@k** — Probability of at least one success in k attempts
- **pass^k** — Probability of k consecutive successes (consistency)
- **Expected cost to success** — `total_cost / successes` (captures unreliability cost)

See [pass@k vs pass^k discussion](https://www.philschmid.de/agents-pass-at-k-pass-power-k).

## Fairness principles

- **Scenario isolation** — Only one factor changes at a time
- **Clean sessions** — Each run starts fresh (no prior history)
- **Baseline overhead** — "Do-nothing" runs measure fixed MCP schema cost
- **Marginal analysis** — Report both total and marginal (task-only) costs
- **Reproducibility** — Pin CLI versions, record host environment

## Documentation

| Document | Purpose |
|----------|---------|
| **[SETUP.md](SETUP.md)** | Prerequisites, installation, configuration, running the suite |
| **[EVAL_GUIDE.md](EVAL_GUIDE.md)** | Step-by-step pipeline, metric definitions, grader details |
| **[config.example.yaml](config.example.yaml)** | Configuration reference |

## Quick start

```bash
# 1. Clone and set up
cp config.example.yaml config.yaml  # Edit with your paths/settings
bash clone_sut.sh                    # Clone system under test

# 2. Install dependencies
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 3. Run
python3 run_suite.py --config config.yaml --tasks tasks.yaml --trials 10
```

See [SETUP.md](SETUP.md) for full setup instructions.

## References

- [XcodeBuildMCP](https://github.com/cameroncooke/XcodeBuildMCP)
- [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching)
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Claude Code costs](https://code.claude.com/docs/en/costs)
- [MCP Tools specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
