"""
Metrics extraction and command log parsing for the eval suite.

Provides functions for analyzing command invocations, MCP tool usage,
and computing derived metrics from trial logs.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import shlex
from typing import Any, Dict, List, Optional, Tuple

from evals.infrastructure import now_ts


# xcodebuild flags that take path values (to be normalized in repeats analysis)
XCODEBUILD_PATH_VALUE_FLAGS = {
    "-derivedDataPath",
    "-resultBundlePath",
    "-clonedSourcePackagesDirPath",
    "-archivePath",
    "-exportPath",
    "-resultStreamPath",
    "-logPath",
}

# xcodebuild actions that actually perform builds (excludes -list, -showBuildSettings, clean)
XCODEBUILD_BUILD_ACTIONS = {
    "build",
    "test",
    "archive",
    "build-for-testing",
    "install",
}

# MCP tools that wrap xcodebuild
XCODEBUILD_MCP_TOOLS = {
    "list_schemes",
    "show_build_settings",
    "build_sim",
    "build_run_sim",
    "test_sim",
    "build_macos",
    "test_macos",
    "archive_macos",
}

# MCP tools that wrap simctl
SIMCTL_MCP_TOOLS = {
    "list_sims",
    "boot_sim",
    "shutdown_sim",
    "erase_sim",
    "open_sim",
    "install_sim",
    "launch_app_sim",
    "uninstall_sim",
    "get_sim_app_path",
    "get_sim_app_paths",
    "get_app_bundle_id",
    "start_simulator_log_capture",
    "launch_app_with_logs_in_simulator",
    "stop_sim_log_cap",
}

# MCP tools that actually invoke xcodebuild (excludes informational tools)
MCP_BUILD_TOOLS = {
    "build_sim",
    "build_run_sim",
    "test_sim",
    "build_macos",
    "test_macos",
    "archive_macos",
}


def count_invocations(
    command_log_path: pathlib.Path, source: Optional[str] = None
) -> Dict[str, int]:
    """
    Count command invocations from a command log file.

    Args:
        command_log_path: Path to the JSONL command log
        source: Optional filter for source field ('agent' or 'mcp')

    Returns:
        Dict mapping command names to invocation counts
    """
    counts: Dict[str, int] = {}
    if not command_log_path.exists():
        return counts
    for line in command_log_path.read_text().splitlines():
        try:
            obj = json.loads(line)
            entry_source = obj.get("source") or "agent"
            if source and entry_source != source:
                continue
            cmd = obj.get("cmd")
            if cmd:
                counts[cmd] = counts.get(cmd, 0) + 1
        except Exception:
            continue
    return counts


def count_simctl_invocations(
    command_log_path: pathlib.Path, source: Optional[str] = None
) -> int:
    """
    Count simctl invocations (xcrun simctl ...) from a command log.

    Args:
        command_log_path: Path to the JSONL command log
        source: Optional filter for source field

    Returns:
        Number of simctl invocations
    """
    if not command_log_path.exists():
        return 0
    count = 0
    for line in command_log_path.read_text().splitlines():
        try:
            obj = json.loads(line)
            entry_source = obj.get("source") or "agent"
            if source and entry_source != source:
                continue
            if obj.get("cmd") != "xcrun":
                continue
            argv = obj.get("argv") or []
            if argv and argv[0] == "simctl":
                count += 1
        except Exception:
            continue
    return count


def read_command_log_entries(
    command_log_path: pathlib.Path, source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Read all entries from a command log file.

    Args:
        command_log_path: Path to the JSONL command log
        source: Optional filter for source field

    Returns:
        List of log entry dicts
    """
    if not command_log_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in command_log_path.read_text().splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        entry_source = obj.get("source") or "agent"
        if source and entry_source != source:
            continue
        entries.append(obj)
    return entries


def parse_log_timestamp(value: Optional[str]) -> Optional[dt.datetime]:
    """Parse an ISO timestamp string to datetime."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.datetime.fromisoformat(value)
    except Exception:
        return None


def extract_xcodebuild_destination(argv: List[str]) -> Optional[str]:
    """Extract the -destination value from xcodebuild arguments."""
    for i, arg in enumerate(argv):
        if arg == "-destination" and i + 1 < len(argv):
            return str(argv[i + 1])
        if arg.startswith("-destination="):
            return arg.split("=", 1)[1]
    return None


def normalize_xcodebuild_argv(argv: List[str]) -> Tuple[str, ...]:
    """
    Normalize xcodebuild arguments for repeat detection.

    Path-valued flags are replaced with '<path>' placeholder.
    """
    normalized: List[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in XCODEBUILD_PATH_VALUE_FLAGS and i + 1 < len(argv):
            normalized.append(arg)
            normalized.append("<path>")
            skip_next = True
            continue
        normalized.append(arg)
    return tuple(normalized)


def is_xcodebuild_build_action(argv: List[str]) -> bool:
    """Check if xcodebuild argv contains an actual build action."""
    for arg in argv:
        if arg in XCODEBUILD_BUILD_ACTIONS:
            return True
    return False


def compute_xcodebuild_repeat_count(entries: List[Dict[str, Any]]) -> int:
    """
    Compute how many times equivalent xcodebuild commands were repeated.

    Returns the count of duplicate invocations (total repeats - 1 per unique command).
    """
    counts: Dict[Tuple[str, ...], int] = {}
    for entry in entries:
        if entry.get("cmd") != "xcodebuild":
            continue
        argv = entry.get("argv") or []
        sig = normalize_xcodebuild_argv([str(x) for x in argv])
        counts[sig] = counts.get(sig, 0) + 1
    repeats = 0
    for count in counts.values():
        if count > 1:
            repeats += count - 1
    return repeats


def compute_time_to_first_xcodebuild_sec(
    ts_start: str, entries: List[Dict[str, Any]]
) -> Optional[float]:
    """
    Compute seconds from trial start to first xcodebuild build action.

    Only counts actual build actions, not informational commands like -list.
    """
    start_dt = parse_log_timestamp(ts_start)
    if not start_dt:
        return None
    first_dt: Optional[dt.datetime] = None
    for entry in entries:
        if entry.get("cmd") != "xcodebuild":
            continue
        argv = entry.get("argv", [])
        if not is_xcodebuild_build_action(argv):
            continue
        ts_val = parse_log_timestamp(entry.get("ts"))
        if not ts_val:
            continue
        if first_dt is None or ts_val < first_dt:
            first_dt = ts_val
    if not first_dt:
        return None
    delta = (first_dt - start_dt).total_seconds()
    return max(0.0, float(delta))


def count_mcp_tool_usage(log_path: Optional[pathlib.Path]) -> Dict[str, int]:
    """
    Count MCP tool usage from an MCP tool log.

    Returns dict with 'total', 'xcodebuild', and 'simctl' counts.
    """
    counts = {"total": 0, "xcodebuild": 0, "simctl": 0}
    if not log_path or not log_path.exists():
        return counts
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        tool = (
            obj.get("tool")
            or obj.get("name")
            or obj.get("method")
            or obj.get("tool_name")
        )
        if not tool:
            continue
        counts["total"] += 1
        if tool in XCODEBUILD_MCP_TOOLS:
            counts["xcodebuild"] += 1
        if tool in SIMCTL_MCP_TOOLS:
            counts["simctl"] += 1
    return counts


def count_mcp_tool_invocations(log_path: Optional[pathlib.Path]) -> Optional[int]:
    """Count total MCP tool invocations from a log file."""
    if not log_path or not log_path.exists():
        return None
    count = 0
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        count += 1
    return count


def compute_time_to_first_mcp_build_sec(
    ts_start: str, log_path: Optional[pathlib.Path]
) -> Optional[float]:
    """Compute time from trial start to first MCP build tool call."""
    if not log_path or not log_path.exists():
        return None
    start_dt = parse_log_timestamp(ts_start)
    if not start_dt:
        return None
    first_dt: Optional[dt.datetime] = None
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        tool = (
            obj.get("tool")
            or obj.get("name")
            or obj.get("method")
            or obj.get("tool_name")
        )
        if tool not in MCP_BUILD_TOOLS:
            continue
        ts_val = parse_log_timestamp(obj.get("ts"))
        if not ts_val:
            continue
        if first_dt is None or ts_val < first_dt:
            first_dt = ts_val
    if not first_dt:
        return None
    delta = (first_dt - start_dt).total_seconds()
    return max(0.0, float(delta))


def extract_command_tokens(command_str: str) -> List[str]:
    """
    Extract command tokens from a shell command string.

    Handles shell wrappers like 'bash -lc "actual command"'.
    """
    try:
        tokens = shlex.split(command_str)
    except Exception:
        return [command_str]
    if len(tokens) >= 3 and tokens[1] == "-lc":
        if tokens[0].endswith("zsh") or tokens[0].endswith("bash"):
            inner = tokens[2]
            try:
                return shlex.split(inner)
            except Exception:
                return inner.split()
    return tokens


def log_stream_command_invocation(
    cmd_log_path: pathlib.Path, command_str: Optional[str], source: str = "agent"
) -> None:
    """
    Log a command invocation to the command log file.

    Only logs xcodebuild and xcrun commands.
    """
    if not command_str:
        return
    tokens = extract_command_tokens(command_str)
    if not tokens:
        return
    cmd = tokens[0]
    base = os.path.basename(cmd)
    if base not in {"xcodebuild", "xcrun"}:
        return
    entry = {
        "ts": now_ts().replace("+00:00", "Z"),
        "cmd": base,
        "argv": tokens[1:],
        "source": source,
    }
    with open(cmd_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def log_stream_mcp_invocation(
    mcp_log_path: pathlib.Path, server: Optional[str], tool: Optional[str], args: Any
) -> None:
    """Log an MCP tool invocation to the MCP log file."""
    if not server or not tool:
        return
    entry = {
        "ts": now_ts().replace("+00:00", "Z"),
        "server": server,
        "tool": tool,
        "arguments": args,
    }
    with open(mcp_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
