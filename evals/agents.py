"""
Agent adapters for the eval suite.

Provides the AgentAdapter base class and TemplateCommandAgent implementation
for running AI coding agents (Claude Code CLI, Codex CLI, etc.).
"""

from __future__ import annotations

import json
import os
import pathlib
import select
import shlex
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from evals.infrastructure import run_cmd, safe_mkdir, now_ts, toml_string, toml_array
from evals.metrics import (
    count_invocations,
    log_stream_command_invocation,
    log_stream_mcp_invocation,
)

if TYPE_CHECKING:
    from evals.config import AgentConfig


def format_compact_json(value: Any) -> str:
    """Format a value as compact JSON."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def extract_message_content(
    msg: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract text, tool calls, and tool results from a message.

    Returns (text, tool_calls, tool_results).
    """
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    content = msg.get("content")
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text") or "")
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "name": block.get("name"),
                            "id": block.get("id"),
                            "input": block.get("input"),
                        }
                    )
                elif btype == "tool_result":
                    tool_results.append(
                        {
                            "id": block.get("tool_use_id") or block.get("id"),
                            "content": block.get("content"),
                            "is_error": block.get("is_error"),
                        }
                    )
    return "".join(text_parts), tool_calls, tool_results


def format_tool_result(content: Any, max_chars: int = 2000) -> str:
    """Format a tool result for transcript output, truncating if needed."""
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    else:
        text = format_compact_json(content)
    if len(text) > max_chars:
        return text[:max_chars] + "...(truncated)"
    return text


def extract_mcp_error_payload(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract error payload from a Codex MCP tool result item.

    Returns None if not an error, otherwise returns the payload dict with
    error, status, and message (if available from result.content).

    This function is used both at runtime (stream processing) and for
    backfilling tool_errors.jsonl from transcripts.
    """
    if item.get("error") is None and item.get("status") not in {"error", "failed"}:
        return None

    error_payload: Dict[str, Any] = {
        "error": item.get("error"),
        "status": item.get("status"),
    }
    result = item.get("result")
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            texts = [
                c.get("text")
                for c in content
                if isinstance(c, dict) and c.get("text")
            ]
            if texts:
                error_payload["message"] = "\n".join(texts)
    return error_payload


def minimal_transcript_lines_claude(
    obj: Optional[Dict[str, Any]], raw_line: str
) -> List[str]:
    """Convert Claude Code CLI JSON output to minimal transcript lines."""
    if not isinstance(obj, dict):
        stripped = raw_line.strip()
        return [stripped] if stripped else []

    obj_type = obj.get("type")
    if obj_type == "stream_event":
        return []

    lines: List[str] = []

    if obj_type in ("assistant", "user") and isinstance(obj.get("message"), dict):
        msg = obj["message"]
        role = msg.get("role") or obj_type
        text, tool_calls, tool_results = extract_message_content(msg)
        if text:
            lines.append(f"{role.upper()}: {text}")
        for tc in tool_calls:
            lines.append(
                f"TOOL_CALL {tc.get('name')} {format_compact_json(tc.get('input'))}"
            )
        for tr in tool_results:
            prefix = "TOOL_RESULT"
            tool_id = tr.get("id")
            if tool_id:
                prefix = f"{prefix} {tool_id}"
            if tr.get("is_error"):
                prefix = f"{prefix} ERROR"
            payload = format_tool_result(tr.get("content"))
            lines.append(f"{prefix} {payload}".rstrip())
        return lines

    if obj_type == "result":
        lines.append(
            "RESULT "
            + format_compact_json(
                {
                    "subtype": obj.get("subtype"),
                    "is_error": obj.get("is_error"),
                    "duration_ms": obj.get("duration_ms"),
                    "total_cost_usd": obj.get("total_cost_usd"),
                }
            )
        )
        result_text = obj.get("result")
        if result_text:
            lines.append(f"RESULT_TEXT: {result_text}")
        return lines

    if obj_type == "error":
        lines.append(f"ERROR: {format_compact_json(obj)}")
        return lines

    # Some providers emit a top-level "message" without a wrapper type.
    if isinstance(obj.get("message"), dict):
        msg = obj["message"]
        role = msg.get("role") or "message"
        text, tool_calls, tool_results = extract_message_content(msg)
        if text:
            lines.append(f"{role.upper()}: {text}")
        for tc in tool_calls:
            lines.append(
                f"TOOL_CALL {tc.get('name')} {format_compact_json(tc.get('input'))}"
            )
        for tr in tool_results:
            prefix = "TOOL_RESULT"
            tool_id = tr.get("id")
            if tool_id:
                prefix = f"{prefix} {tool_id}"
            if tr.get("is_error"):
                prefix = f"{prefix} ERROR"
            payload = format_tool_result(tr.get("content"))
            lines.append(f"{prefix} {payload}".rstrip())
        return lines

    return []


def minimal_transcript_lines_codex(
    obj: Optional[Dict[str, Any]], raw_line: str
) -> List[str]:
    """Convert Codex CLI JSON output to minimal transcript lines."""
    if not isinstance(obj, dict):
        stripped = raw_line.strip()
        return [stripped] if stripped else []

    obj_type = obj.get("type")
    lines: List[str] = []

    if obj_type == "item.started":
        return []

    if obj_type == "item.completed":
        item = obj.get("item") or {}
        item_type = item.get("type")
        if item_type == "agent_message":
            text = item.get("text") or ""
            if text:
                lines.append(f"ASSISTANT: {text}")
        elif item_type == "command_execution":
            cmd = item.get("command")
            if cmd:
                lines.append(
                    f"TOOL_CALL Bash {format_compact_json({'command': cmd})}"
                )
            output = item.get("aggregated_output")
            result_payload = {
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
                "output": output,
            }
            lines.append(f"TOOL_RESULT {format_tool_result(result_payload)}")
        elif item_type == "mcp_tool_call":
            server = item.get("server")
            tool = item.get("tool")
            args = item.get("arguments")
            tool_name = None
            if server and tool:
                tool_name = f"mcp__{server}__{tool}"
            if tool_name:
                lines.append(
                    f"TOOL_CALL {tool_name} {format_compact_json(args)}".rstrip()
                )
            result = item.get("result")
            if result is not None or item.get("error") is not None:
                payload = {
                    "result": result,
                    "error": item.get("error"),
                    "status": item.get("status"),
                }
                lines.append(f"TOOL_RESULT {format_tool_result(payload)}")
        elif item_type == "tool_call":
            name = item.get("name")
            tool_input = item.get("input")
            lines.append(
                f"TOOL_CALL {name} {format_compact_json(tool_input)}".rstrip()
            )
        elif item_type == "tool_result":
            payload = {
                "tool_name": item.get("name"),
                "is_error": item.get("is_error"),
                "output": item.get("output"),
            }
            lines.append(f"TOOL_RESULT {format_tool_result(payload)}")
        return lines

    if obj_type in {"turn.failed", "error"}:
        lines.append(f"ERROR: {format_compact_json(obj)}")
        return lines

    return []


def minimal_transcript_lines(
    agent_kind: str, obj: Optional[Dict[str, Any]], raw_line: str
) -> List[str]:
    """Convert agent output to minimal transcript lines based on agent type."""
    if agent_kind == "codex_cli":
        return minimal_transcript_lines_codex(obj, raw_line)
    return minimal_transcript_lines_claude(obj, raw_line)


def iter_claude_tool_calls(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from a Claude message object."""
    msg: Optional[Dict[str, Any]] = None
    obj_type = obj.get("type")
    if obj_type in ("assistant", "user") and isinstance(obj.get("message"), dict):
        msg = obj["message"]
    elif isinstance(obj.get("message"), dict):
        msg = obj["message"]
    if not msg:
        return []
    _, tool_calls, _ = extract_message_content(msg)
    return tool_calls


def log_claude_mcp_invocations(
    mcp_log_path: pathlib.Path, obj: Dict[str, Any]
) -> None:
    """Log MCP tool invocations from a Claude message to the MCP log."""
    for tc in iter_claude_tool_calls(obj):
        name = tc.get("name") or ""
        if not name.startswith("mcp__"):
            continue
        parts = name.split("__", 2)
        server = parts[1] if len(parts) > 1 else None
        tool = parts[2] if len(parts) > 2 else name
        log_stream_mcp_invocation(mcp_log_path, server, tool, tc.get("input"))


def ccusage_codex_cost(codex_home: Optional[str]) -> Optional[float]:
    """Get cost from ccusage for a Codex session."""
    if not codex_home:
        return None
    cmd = ["npx", "-y", "@ccusage/codex@latest", "session", "--json"]
    env = dict(os.environ)
    env["CODEX_HOME"] = codex_home
    rc, out, _ = run_cmd(cmd, env=env, timeout=120)
    if rc != 0:
        return None
    try:
        payload = json.loads(out)
    except Exception:
        return None
    totals = payload.get("totals") if isinstance(payload, dict) else None
    if isinstance(totals, dict) and totals.get("costUSD") is not None:
        try:
            return float(totals["costUSD"])
        except Exception:
            return None
    return None


def read_tool_error_summary(
    summary_path: Optional[pathlib.Path],
) -> Dict[str, int]:
    """Read tool error counts from a summary file."""
    counts = {"total": 0, "mcp": 0, "non_mcp": 0}
    if not summary_path or not summary_path.exists():
        return counts
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return counts
    for key in counts:
        val = payload.get(key)
        if isinstance(val, int):
            counts[key] = val
    return counts


def build_prompt(
    task_prompt: str,
    scenario: str,
    build_params: Dict[str, Any],
    simulator_name: str,
) -> str:
    """
    Build the prompt for an agent based on scenario.

    Only shell_primed includes build parameters in the prompt.
    MCP scenarios include a hint to prefer MCP tools.
    """
    if scenario in ("mcp_unprimed", "mcp_unprimed_v2"):
        return (
            task_prompt.strip()
            + "\n\nIf XcodeBuildMCP tools are available, prefer them for build/test/"
            "install/launch. Use raw xcodebuild/xcrun only if the MCP tool cannot "
            "perform the task.\n"
        )
    if scenario != "shell_primed":
        return task_prompt.strip() + "\n"

    # shell_primed: add deterministic build params for shell usage.
    lines = [
        task_prompt.strip(),
        "",
        "Build parameters (provided):",
    ]
    for k in [
        "workspace",
        "project",
        "scheme",
        "bundle_id",
        "configuration",
        "simulator_name",
        "destination",
    ]:
        if k == "simulator_name":
            v = build_params.get(k) or simulator_name
        else:
            v = build_params.get(k)
        if v:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("You may use these parameters directly in shell commands.")
    lines.append("When complete, exit.")
    return "\n".join(lines) + "\n"


def make_command_shims(
    shim_dir: pathlib.Path, log_path: pathlib.Path, commands: List[str]
) -> Dict[str, str]:
    """
    Create wrapper scripts for commands that log invocations.

    The shim_dir is prepended to PATH so invocations are logged.
    Returns environment variable updates to apply.
    """
    safe_mkdir(shim_dir)
    env_updates = {
        "EVAL_CMD_LOG": str(log_path),
    }
    for cmd in commands:
        real = shutil.which(cmd)
        if not real:
            real = cmd
        shim_path = shim_dir / cmd
        shim_path.write_text(
            f"""#!/bin/bash
set -euo pipefail
source_tag="agent"
if [ "${{EVAL_MCP_PROCESS:-}}" = "1" ]; then
  source_tag="mcp"
else
  # Fallback: detect MCP parent process if env var is missing.
  ppid="$PPID"
  depth=0
  while [ -n "$ppid" ] && [ "$ppid" -gt 1 ] && [ "$depth" -lt 6 ]; do
    cmdline="$(ps -p "$ppid" -o command= 2>/dev/null || true)"
    if echo "$cmdline" | grep -Ei "xcodebuildmcp" >/dev/null 2>&1; then
      source_tag="mcp"
      break
    fi
    ppid="$(ps -p "$ppid" -o ppid= 2>/dev/null | tr -d ' ')"
    depth=$((depth+1))
  done
fi
ts="$(python3 - <<'PY'\nimport datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'))\nPY\n)"
argv_json="$(python3 - "$@" <<'PY'\nimport json,sys; print(json.dumps(sys.argv[1:]))\nPY\n)"
echo "{{\\"ts\\": \\"$ts\\", \\"cmd\\": \\"{cmd}\\", \\"argv\\": $argv_json, \\"source\\": \\"$source_tag\\"}}" >> "{log_path}"
exec "{real}" "$@"
"""
        )
        shim_path.chmod(0o755)
    return env_updates


def codex_mcp_args_from_json(cfg_path: pathlib.Path) -> List[str]:
    """Convert MCP config JSON to Codex CLI TOML config arguments."""
    if not cfg_path.exists():
        return []
    try:
        payload = json.loads(cfg_path.read_text())
    except Exception:
        return []
    servers = payload.get("mcpServers") or {}
    args: List[str] = []
    for name, info in servers.items():
        if not isinstance(info, dict):
            continue
        command = info.get("command")
        if command:
            args.extend(["-c", f"mcp_servers.{name}.command={toml_string(command)}"])
        raw_args = info.get("args")
        if isinstance(raw_args, list) and raw_args:
            args.extend(["-c", f"mcp_servers.{name}.args={toml_array(raw_args)}"])
        env_map = info.get("env")
        if isinstance(env_map, dict):
            for key, value in env_map.items():
                if value is None:
                    continue
                args.extend(
                    [
                        "-c",
                        f"mcp_servers.{name}.env.{key}={toml_string(str(value))}",
                    ]
                )
        # Handle tool_timeout_sec for Codex MCP calls
        tool_timeout = info.get("tool_timeout_sec")
        if tool_timeout is not None:
            args.extend(["-c", f"mcp_servers.{name}.tool_timeout_sec={int(tool_timeout)}"])
        # Enable the MCP server
        args.extend(["-c", f"mcp_servers.{name}.enabled=true"])
    return args


class AgentAdapter:
    """Base class for agent adapters."""

    def __init__(self, cfg: "AgentConfig"):
        self.cfg = cfg

    def run(
        self,
        prompt: str,
        workdir: pathlib.Path,
        out_json: pathlib.Path,
        env: Dict[str, str],
        timeout_sec: int,
        stall_timeout_sec: Optional[int],
        transcript_path: pathlib.Path,
        transcript_mode: str = "minimal",
        stream_output: bool = False,
        cmd_log_path: Optional[pathlib.Path] = None,
        tool_error_log_path: Optional[pathlib.Path] = None,
        tool_error_summary_path: Optional[pathlib.Path] = None,
        tool_error_context_log_path: Optional[pathlib.Path] = None,
    ) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        """
        Run the agent with the given prompt.

        Returns (exit_code, usage_data, timeout_reason).
        """
        raise NotImplementedError


class TemplateCommandAgent(AgentAdapter):
    """
    Generic adapter that runs a configured command template.

    The command should write a JSON usage/output payload to {OUT_JSON} if possible.
    """

    def run(
        self,
        prompt: str,
        workdir: pathlib.Path,
        out_json: pathlib.Path,
        env: Dict[str, str],
        timeout_sec: int,
        stall_timeout_sec: Optional[int],
        transcript_path: pathlib.Path,
        transcript_mode: str = "minimal",
        stream_output: bool = False,
        cmd_log_path: Optional[pathlib.Path] = None,
        tool_error_log_path: Optional[pathlib.Path] = None,
        tool_error_summary_path: Optional[pathlib.Path] = None,
        tool_error_context_log_path: Optional[pathlib.Path] = None,
    ) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        cmd: List[str] = []
        extra_args = shlex.split(env.get("EVAL_AGENT_EXTRA_ARGS", ""))
        for token in self.cfg.command:
            if token == "{EXTRA_ARGS}":
                cmd.extend(extra_args)
                continue
            cmd.append(self._subst(token, prompt, workdir, out_json))

        mode = (transcript_mode or "minimal").lower()
        if mode not in {"minimal", "raw", "none"}:
            mode = "minimal"

        safe_mkdir(transcript_path.parent)
        mcp_log_path: Optional[pathlib.Path] = None
        mcp_log_val = env.get("EVAL_MCP_TOOL_LOG")
        if mcp_log_val:
            mcp_log_path = pathlib.Path(mcp_log_val)
            safe_mkdir(mcp_log_path.parent)
        if tool_error_log_path:
            safe_mkdir(tool_error_log_path.parent)
        if tool_error_summary_path:
            safe_mkdir(tool_error_summary_path.parent)
        if tool_error_context_log_path:
            safe_mkdir(tool_error_context_log_path.parent)

        tool_error_counts = {"total": 0, "mcp": 0, "non_mcp": 0}
        claude_tool_ids: Dict[str, str] = {}
        context_limit = 2
        if env.get("EVAL_TOOL_ERROR_CONTEXT_CALLS"):
            try:
                context_limit = max(
                    0, int(env.get("EVAL_TOOL_ERROR_CONTEXT_CALLS", "2"))
                )
            except Exception:
                context_limit = 2
        pending_contexts: List[Dict[str, Any]] = []

        def classify_tool_kind(tool_name: Optional[str]) -> str:
            if tool_name and tool_name.startswith("mcp__"):
                return "mcp"
            return "non_mcp"

        def write_tool_error_context(entry: Dict[str, Any]) -> None:
            if not tool_error_context_log_path:
                return
            with open(tool_error_context_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        def note_tool_call_context(
            tool_name: Optional[str],
            tool_kind: str,
            tool_input: Any,
            source: str,
        ) -> None:
            if not pending_contexts:
                return
            event = {
                "ts": now_ts().replace("+00:00", "Z"),
                "tool_name": tool_name,
                "tool_kind": tool_kind,
                "source": source,
                "input": tool_input,
            }
            flushed: List[Dict[str, Any]] = []
            for pending in pending_contexts:
                if pending["remaining"] <= 0:
                    continue
                pending["entry"]["context"].append(event)
                pending["remaining"] -= 1
                if pending["remaining"] <= 0:
                    flushed.append(pending)
            if flushed:
                for item in flushed:
                    write_tool_error_context(item["entry"])
                pending_contexts[:] = [
                    item for item in pending_contexts if item not in flushed
                ]

        def record_tool_error(
            tool_name: Optional[str],
            tool_kind: str,
            payload: Any,
            source: str,
        ) -> None:
            tool_error_counts["total"] += 1
            tool_error_counts[tool_kind] += 1
            if not tool_error_log_path:
                return
            entry = {
                "ts": now_ts().replace("+00:00", "Z"),
                "tool_name": tool_name,
                "tool_kind": tool_kind,
                "source": source,
                "payload": format_tool_result(payload),
            }
            with open(tool_error_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if context_limit <= 0:
                write_tool_error_context({**entry, "context": []})
                return
            pending_contexts.append(
                {"entry": {**entry, "context": []}, "remaining": context_limit}
            )

        with open(transcript_path, "w", encoding="utf-8") as tf:
            if mode != "none":
                tf.write(
                    f"## COMMAND\n{cmd}\n\n## START {now_ts()}\n"
                    f"## transcript_mode={mode}\n\n"
                )
                tf.flush()
            p = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            start = time.time()
            last_heartbeat = start
            last_usage_obj: Optional[Dict[str, Any]] = None
            last_activity = start
            last_activity_check = start
            last_cmd_log_size = 0
            last_mcp_log_size = 0
            if cmd_log_path and cmd_log_path.exists():
                try:
                    last_cmd_log_size = cmd_log_path.stat().st_size
                except Exception:
                    last_cmd_log_size = 0
            if mcp_log_path and mcp_log_path.exists():
                try:
                    last_mcp_log_size = mcp_log_path.stat().st_size
                except Exception:
                    last_mcp_log_size = 0
            assert p.stdout is not None
            while True:
                now = time.time()
                if stall_timeout_sec and (now - last_activity_check) >= 1.0:
                    if cmd_log_path and cmd_log_path.exists():
                        try:
                            size = cmd_log_path.stat().st_size
                            if size > last_cmd_log_size:
                                last_activity = now
                                last_cmd_log_size = size
                        except Exception:
                            pass
                    if mcp_log_path and mcp_log_path.exists():
                        try:
                            size = mcp_log_path.stat().st_size
                            if size > last_mcp_log_size:
                                last_activity = now
                                last_mcp_log_size = size
                        except Exception:
                            pass
                    last_activity_check = now
                if timeout_sec and (now - start) > timeout_sec:
                    p.kill()
                    tf.write(f"\n## TIMEOUT after {timeout_sec}s\n")
                    tf.flush()
                    if tool_error_summary_path:
                        tool_error_summary_path.write_text(
                            json.dumps(tool_error_counts, ensure_ascii=False)
                        )
                    if pending_contexts:
                        for pending in pending_contexts:
                            write_tool_error_context(pending["entry"])
                    return 124, None, "timeout_hard"
                if stall_timeout_sec and (now - last_activity) > stall_timeout_sec:
                    p.kill()
                    tf.write(
                        f"\n## STALL TIMEOUT after {stall_timeout_sec}s (no output/tool activity)\n"
                    )
                    tf.flush()
                    if tool_error_summary_path:
                        tool_error_summary_path.write_text(
                            json.dumps(tool_error_counts, ensure_ascii=False)
                        )
                    if pending_contexts:
                        for pending in pending_contexts:
                            write_tool_error_context(pending["entry"])
                    return 124, None, "timeout_stall"
                ready, _, _ = select.select([p.stdout], [], [], 0.1)
                if ready:
                    line = p.stdout.readline()
                    if line:
                        last_activity = time.time()
                        try:
                            obj = json.loads(line)
                        except Exception:
                            obj = None
                        # Codex CLI streaming handling
                        if (
                            self.cfg.kind == "codex_cli"
                            and cmd_log_path is not None
                            and isinstance(obj, dict)
                            and obj.get("type") == "item.completed"
                        ):
                            item = obj.get("item") or {}
                            if item.get("type") == "command_execution":
                                log_stream_command_invocation(
                                    cmd_log_path, item.get("command")
                                )
                                note_tool_call_context(
                                    "bash",
                                    "non_mcp",
                                    {"command": item.get("command")},
                                    "codex",
                                )
                                exit_code = item.get("exit_code")
                                status = item.get("status")
                                if exit_code not in (0, None) or status in {
                                    "error",
                                    "failed",
                                }:
                                    record_tool_error(
                                        "bash",
                                        "non_mcp",
                                        {
                                            "exit_code": exit_code,
                                            "status": status,
                                        },
                                        "codex",
                                    )
                            if (
                                mcp_log_path is not None
                                and item.get("type") == "mcp_tool_call"
                            ):
                                log_stream_mcp_invocation(
                                    mcp_log_path,
                                    item.get("server"),
                                    item.get("tool"),
                                    item.get("arguments"),
                                )
                                server = item.get("server")
                                tool = item.get("tool")
                                tool_name = (
                                    f"mcp__{server}__{tool}" if server and tool else tool
                                )
                                note_tool_call_context(
                                    tool_name,
                                    "mcp",
                                    item.get("arguments"),
                                    "codex",
                                )
                                error_payload = extract_mcp_error_payload(item)
                                if error_payload is not None:
                                    record_tool_error(
                                        tool_name,
                                        "mcp",
                                        error_payload,
                                        "codex",
                                    )
                            if item.get("type") == "tool_result":
                                if item.get("is_error"):
                                    name = item.get("name")
                                    tool_kind = classify_tool_kind(name)
                                    record_tool_error(
                                        name,
                                        tool_kind,
                                        {"output": item.get("output")},
                                        "codex",
                                    )
                            if item.get("type") == "tool_call":
                                name = item.get("name")
                                tool_input = item.get("input")
                                tool_kind = classify_tool_kind(name)
                                note_tool_call_context(
                                    name, tool_kind, tool_input, "codex"
                                )
                        # Claude Code CLI MCP logging
                        if (
                            self.cfg.kind == "claude_code_cli"
                            and mcp_log_path is not None
                            and isinstance(obj, dict)
                        ):
                            log_claude_mcp_invocations(mcp_log_path, obj)
                        # Claude Code CLI tool error tracking
                        if self.cfg.kind == "claude_code_cli" and isinstance(obj, dict):
                            obj_type = obj.get("type")
                            msg = None
                            if obj_type in ("assistant", "user") and isinstance(
                                obj.get("message"), dict
                            ):
                                msg = obj.get("message")
                            elif isinstance(obj.get("message"), dict):
                                msg = obj.get("message")
                            if msg:
                                _, tool_calls, tool_results = extract_message_content(msg)
                                for tc in tool_calls:
                                    tool_id = tc.get("id")
                                    tool_name = tc.get("name")
                                    if tool_id and tool_name:
                                        claude_tool_ids[tool_id] = tool_name
                                    tool_kind = classify_tool_kind(tool_name)
                                    note_tool_call_context(
                                        tool_name, tool_kind, tc.get("input"), "claude"
                                    )
                                for tr in tool_results:
                                    if not tr.get("is_error"):
                                        continue
                                    tool_id = tr.get("id")
                                    tool_name = (
                                        claude_tool_ids.get(tool_id) if tool_id else None
                                    )
                                    tool_kind = classify_tool_kind(tool_name)
                                    record_tool_error(
                                        tool_name,
                                        tool_kind,
                                        tr.get("content"),
                                        "claude",
                                    )
                        # Write to transcript
                        if mode == "raw":
                            if mode != "none":
                                tf.write(line)
                                tf.flush()
                            if stream_output:
                                print(line, end="", flush=True)
                        else:
                            lines = minimal_transcript_lines(self.cfg.kind, obj, line)
                            if mode != "none":
                                for out_line in lines:
                                    tf.write(out_line + "\n")
                                tf.flush()
                            if stream_output:
                                for out_line in lines:
                                    print(out_line, flush=True)
                        # Track usage info
                        if isinstance(obj, dict):
                            if any(
                                k in obj
                                for k in (
                                    "usage",
                                    "total_cost_usd",
                                    "billed_cost_usd",
                                    "cost_usd",
                                    "prompt_tokens",
                                    "completion_tokens",
                                    "input_tokens",
                                    "output_tokens",
                                )
                            ):
                                last_usage_obj = obj
                    elif p.poll() is not None:
                        break
                elif p.poll() is not None:
                    break
                if stream_output and (time.time() - last_heartbeat) >= 30:
                    counts = {}
                    if cmd_log_path and cmd_log_path.exists():
                        counts = count_invocations(cmd_log_path)
                    elapsed = int(time.time() - start)
                    print(
                        f"[agent] still running ({elapsed}s elapsed, "
                        f"xcodebuild={counts.get('xcodebuild', 0)}, "
                        f"xcrun={counts.get('xcrun', 0)})",
                        flush=True,
                    )
                    last_heartbeat = time.time()

        def has_usage_fields(obj: Dict[str, Any]) -> bool:
            return any(
                k in obj
                for k in (
                    "usage",
                    "total_cost_usd",
                    "billed_cost_usd",
                    "cost_usd",
                    "prompt_tokens",
                    "completion_tokens",
                    "input_tokens",
                    "output_tokens",
                )
            )

        usage_obj: Optional[Dict[str, Any]] = None
        if out_json.exists():
            try:
                usage_obj = json.loads(out_json.read_text())
            except Exception:
                usage_obj = None

        if usage_obj is None and last_usage_obj is not None:
            usage_obj = last_usage_obj
        elif usage_obj is not None and last_usage_obj is not None:
            if not has_usage_fields(usage_obj) and has_usage_fields(last_usage_obj):
                usage_obj = last_usage_obj

        if tool_error_summary_path:
            tool_error_summary_path.write_text(
                json.dumps(tool_error_counts, ensure_ascii=False)
            )
        if pending_contexts:
            for pending in pending_contexts:
                write_tool_error_context(pending["entry"])

        return int(p.returncode), usage_obj, None

    @staticmethod
    def _subst(
        s: str, prompt: str, workdir: pathlib.Path, out_json: pathlib.Path
    ) -> str:
        return (
            s.replace("{PROMPT}", prompt)
            .replace("{WORKDIR}", str(workdir))
            .replace("{OUT_JSON}", str(out_json))
        )


def make_agent(cfg: "AgentConfig") -> AgentAdapter:
    """Create an agent adapter from configuration."""
    return TemplateCommandAgent(cfg)
