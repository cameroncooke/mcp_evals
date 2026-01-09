"""
Configuration dataclasses and loading functions for the eval suite.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`"
    ) from e


SCENARIOS = ["shell_unprimed", "shell_primed", "mcp_unprimed", "mcp_unprimed_v2"]


@dataclasses.dataclass
class AgentConfig:
    id: str
    kind: str
    command: List[str]
    env: Dict[str, str]
    pricing: Dict[str, float]


@dataclasses.dataclass
class ProjectConfig:
    repo_path: str
    base_ref: str
    simulator_name: str
    build_params: Dict[str, Any]
    repo_root: Optional[str] = None
    repo_subdir: Optional[str] = None


@dataclasses.dataclass
class MCPConfig:
    enabled: bool
    start_command: Optional[List[str]]
    stop_command: Optional[List[str]]
    env: Dict[str, str]


@dataclasses.dataclass
class SuiteConfig:
    output_root: str
    timeout_sec: int
    trials_per_cell: int
    random_seed: int
    run_baselines: bool
    keep_workdirs: bool
    scenario_timeouts_sec: Optional[Dict[str, int]] = None
    stall_timeout_sec: Optional[int] = None
    plan_mode: str = "random"  # random | blocked_by_scenario
    shuffle_within_scenario: bool = True
    baseline_trials_per_scenario: int = 1
    summary_cache_strata: bool = True
    transcript_mode: str = "minimal"  # minimal | raw | none
    fetch_remote: bool = False
    validate_tasks: bool = False
    clean_agent_env: bool = False
    use_ccusage_for_codex: bool = True
    prewarm_spm: bool = False
    spm_cache_dir: Optional[str] = None
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    post_run_report: bool = False
    post_run_report_agent: Optional[str] = None
    post_run_report_path: Optional[str] = None


@dataclasses.dataclass
class TaskSpec:
    id: str
    description: str
    prompt: str
    setup_commands: List[List[str]]
    graders: List[Dict[str, Any]]
    kind: str = "capability"
    reference_patch: Optional[str] = None


def _resolve_path(value: Optional[str], base_dir: pathlib.Path) -> Optional[str]:
    if value is None:
        return None
    p = pathlib.Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def load_config(
    path: str,
) -> Tuple[SuiteConfig, ProjectConfig, List[AgentConfig], MCPConfig]:
    """Load configuration from a YAML file."""
    config_path = pathlib.Path(path).expanduser().resolve()
    obj = yaml.safe_load(config_path.read_text())
    suite = SuiteConfig(**obj["suite"])
    project = ProjectConfig(**obj["project"])
    agents = [AgentConfig(**a) for a in obj["agents"]]
    mcp = MCPConfig(
        **obj.get(
            "mcp",
            {"enabled": False, "start_command": None, "stop_command": None, "env": {}},
        )
    )
    base_dir = config_path.parent
    suite.output_root = _resolve_path(suite.output_root, base_dir) or suite.output_root
    project.repo_path = _resolve_path(project.repo_path, base_dir) or project.repo_path
    return suite, project, agents, mcp


def load_tasks(path: str) -> List[TaskSpec]:
    """Load task specifications from a YAML file."""
    obj = yaml.safe_load(pathlib.Path(path).read_text())
    tasks = []
    for t in obj["tasks"]:
        tasks.append(
            TaskSpec(
                id=t["id"],
                description=t.get("description", ""),
                prompt=t["prompt"],
                setup_commands=t.get("setup_commands", []),
                graders=t.get("graders", []),
                kind=t.get("kind", "capability"),
                reference_patch=t.get("reference_patch"),
            )
        )
    return tasks


def validate_project_config(project: ProjectConfig) -> None:
    """Validate project configuration and resolve repo layout."""
    from evals.infrastructure import resolve_repo_layout

    repo = pathlib.Path(project.repo_path)
    if not repo.exists():
        raise SystemExit(f"Project repo_path does not exist: {repo}")
    repo_root, repo_subdir = resolve_repo_layout(project.repo_path)
    project.repo_root = repo_root
    project.repo_subdir = repo_subdir
    repo_effective = (
        pathlib.Path(repo_root) / repo_subdir if repo_subdir else pathlib.Path(repo_root)
    )
    primed = project.build_params or {}
    workspace = primed.get("workspace")
    proj = primed.get("project")
    if workspace:
        ws_path = repo_effective / workspace
        if not ws_path.exists():
            raise SystemExit(f"Workspace not found: {ws_path}")
    if proj:
        proj_path = repo_effective / proj
        if not proj_path.exists():
            raise SystemExit(f"Project not found: {proj_path}")


def validate_suite_config(suite: SuiteConfig) -> None:
    """Validate suite configuration values."""
    if suite.transcript_mode not in {"minimal", "raw", "none"}:
        raise SystemExit(
            f"Invalid suite.transcript_mode: {suite.transcript_mode} "
            "(expected: minimal | raw | none)"
        )
    if suite.plan_mode not in {"random", "blocked_by_scenario"}:
        raise SystemExit(
            f"Invalid suite.plan_mode: {suite.plan_mode} "
            "(expected: random | blocked_by_scenario)"
        )
    if suite.baseline_trials_per_scenario < 1:
        raise SystemExit("suite.baseline_trials_per_scenario must be >= 1")
    if suite.scenario_timeouts_sec:
        if not isinstance(suite.scenario_timeouts_sec, dict):
            raise SystemExit("suite.scenario_timeouts_sec must be a mapping")
        for k, v in suite.scenario_timeouts_sec.items():
            if k not in SCENARIOS:
                raise SystemExit(
                    f"Invalid scenario timeout key: {k} (expected one of {SCENARIOS})"
                )
            try:
                val = int(v)
            except Exception:
                raise SystemExit(
                    f"Invalid scenario timeout for {k}: {v} (expected int)"
                )
            if val <= 0:
                raise SystemExit(
                    f"Invalid scenario timeout for {k}: {v} (expected > 0)"
                )
            suite.scenario_timeouts_sec[k] = val
    if suite.stall_timeout_sec is not None:
        try:
            stall_val = int(suite.stall_timeout_sec)
        except Exception:
            raise SystemExit(
                f"Invalid suite.stall_timeout_sec: {suite.stall_timeout_sec} (expected int)"
            )
        if stall_val <= 0:
            raise SystemExit(
                f"Invalid suite.stall_timeout_sec: {suite.stall_timeout_sec} (expected > 0)"
            )
        suite.stall_timeout_sec = stall_val
