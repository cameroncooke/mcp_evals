"""
Infrastructure utilities for the eval suite.

Contains low-level utilities for running commands, managing simulators,
resolving paths, and other foundational operations.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from evals.config import ProjectConfig


def now_ts() -> str:
    """Return current UTC timestamp in ISO format."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    capture: bool = True,
) -> Tuple[int, str, str]:
    """
    Run a command and return (return_code, stdout, stderr).

    Returns exit code 124 on timeout (matching shell convention).
    """
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out or "", err or ""
    return p.returncode, out or "", err or ""


def safe_mkdir(p: pathlib.Path) -> None:
    """Create directory and parents if they don't exist."""
    p.mkdir(parents=True, exist_ok=True)


def resolve_simulator_udid(simulator_name: str) -> Optional[str]:
    """Resolve a simulator name to its UDID."""
    rc, out, err = run_cmd(["xcrun", "simctl", "list", "devices", "-j"])
    if rc != 0:
        return None
    try:
        obj = json.loads(out)
    except Exception:
        return None
    devices = obj.get("devices", {})
    matches: List[Dict[str, Any]] = []
    for _, devs in devices.items():
        for dev in devs:
            if dev.get("name") != simulator_name:
                continue
            if dev.get("isAvailable") is False:
                continue
            matches.append(dev)
    if not matches:
        return None
    booted = [d for d in matches if d.get("state") == "Booted"]
    pick = booted[0] if booted else matches[0]
    return pick.get("udid")


def extract_udid_from_destination(dest: str) -> Optional[str]:
    """Extract simulator UDID from an xcodebuild destination string."""
    if not dest:
        return None
    m = re.search(r"(?:id|udid|UDID)=([A-F0-9-]+)", dest)
    if not m:
        return None
    return m.group(1)


def resolve_simulator_udid_for_project(
    project: "ProjectConfig", destination: Optional[str] = None
) -> Optional[str]:
    """Resolve simulator UDID for a project, checking build params first."""
    primed = project.build_params or {}
    for k in ["simulator_udid", "simulator_id", "udid"]:
        v = primed.get(k)
        if v:
            return str(v)
    if destination:
        parsed = extract_udid_from_destination(destination)
        if parsed:
            return parsed
    return resolve_simulator_udid(project.simulator_name)


def resolve_repo_layout(repo_path: str) -> Tuple[str, str]:
    """
    Resolve the git repository root and subdirectory.

    Returns (repo_root, repo_subdir) where repo_subdir is empty if
    repo_path is at the root of the git repo.
    """
    rc, out, err = run_cmd(["git", "-C", repo_path, "rev-parse", "--show-toplevel"])
    if rc != 0:
        raise SystemExit(
            f"repo_path is not inside a git repo: {repo_path}\n{err or out}"
        )
    repo_root = out.strip()
    rel = os.path.relpath(repo_path, repo_root)
    repo_subdir = "" if rel == "." else rel
    return repo_root, repo_subdir


def resolve_developer_dir() -> Optional[str]:
    """Get the active Xcode developer directory."""
    rc, out, err = run_cmd(["xcode-select", "-p"], timeout=30)
    if rc != 0:
        return None
    path = out.strip()
    return path or None


def ensure_simulator_booted(udid: str) -> None:
    """Best-effort boot and wait for a simulator."""
    run_cmd(["xcrun", "simctl", "boot", udid], timeout=60)
    run_cmd(["xcrun", "simctl", "bootstatus", udid, "-b"], timeout=300)


def reset_simulator_app_state(udid: str, bundle_id: str) -> None:
    """Best-effort cleanup of app state in simulator."""
    run_cmd(["xcrun", "simctl", "terminate", udid, bundle_id], timeout=60)
    run_cmd(["xcrun", "simctl", "uninstall", udid, bundle_id], timeout=120)


def scrub_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Redact sensitive values from environment dict for logging."""
    if not env:
        return {}
    redacted: Dict[str, str] = {}
    for k, v in env.items():
        if re.search(r"(key|token|secret|password)", k, re.IGNORECASE):
            redacted[k] = "<redacted>"
        else:
            redacted[k] = v
    return redacted


def toml_string(value: str) -> str:
    """Encode a string as a TOML string literal."""
    return json.dumps(value)


def toml_array(values: List[str]) -> str:
    """Encode a list of strings as a TOML array literal."""
    return "[" + ",".join(json.dumps(v) for v in values) + "]"
