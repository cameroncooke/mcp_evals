"""
Git worktree management for the eval suite.

Provides functions for creating and removing isolated git worktrees
for trial execution.
"""

from __future__ import annotations

import pathlib

from evals.infrastructure import run_cmd, safe_mkdir


def make_worktree(
    repo: pathlib.Path,
    base_ref: str,
    worktrees_root: pathlib.Path,
    run_id: str,
    fetch_remote: bool,
) -> pathlib.Path:
    """
    Create a git worktree at a specific ref to isolate each trial.

    Args:
        repo: Path to the main git repository
        base_ref: Git ref (branch/tag/commit) to check out
        worktrees_root: Directory where worktrees will be created
        run_id: Unique identifier for this worktree
        fetch_remote: Whether to fetch from remote before creating worktree

    Returns:
        Path to the created worktree

    Raises:
        RuntimeError: If worktree creation fails
    """
    wt = worktrees_root / f"wt_{run_id}"
    safe_mkdir(worktrees_root)
    # Optionally fetch to ensure we have the latest refs
    if fetch_remote:
        run_cmd(["git", "fetch", "--all", "--prune"], cwd=str(repo), timeout=600)
    rc, out, err = run_cmd(
        ["git", "worktree", "add", "--force", str(wt), base_ref],
        cwd=str(repo),
        timeout=600,
    )
    if rc != 0:
        raise RuntimeError(f"git worktree add failed: {out}\n{err}")
    return wt


def remove_worktree(repo: pathlib.Path, wt: pathlib.Path) -> None:
    """
    Remove a git worktree and prune stale worktree references.

    Args:
        repo: Path to the main git repository
        wt: Path to the worktree to remove
    """
    run_cmd(
        ["git", "worktree", "remove", "--force", str(wt)], cwd=str(repo), timeout=300
    )
    run_cmd(["git", "worktree", "prune"], cwd=str(repo), timeout=300)
