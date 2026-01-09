"""
Grading logic for the eval suite.

Provides deterministic graders for evaluating agent task completion,
including iOS install/launch checks, test execution, and file diff checks.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from evals.infrastructure import (
    run_cmd,
    resolve_simulator_udid_for_project,
    ensure_simulator_booted,
)
from evals.xcresult import count_xcresult_tests

if TYPE_CHECKING:
    from evals.config import ProjectConfig


class GraderResult(Tuple[bool, Optional[str]]):
    """Result from a grader: (passed, failure_reason)."""
    pass


def sha256_file(path: pathlib.Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_repo_paths(path: str, repo_subdir: str) -> List[str]:
    """
    Normalize a path to account for repo subdirectory.

    Returns both the original path and the path with the subdir prefix removed.
    """
    paths = [path]
    if repo_subdir:
        prefix = repo_subdir.strip("/") + "/"
        if path.startswith(prefix):
            paths.append(path[len(prefix):])
    return paths


def path_matches(patterns: List[str], path: str, repo_subdir: str) -> bool:
    """Check if a path matches any of the given glob patterns."""
    for candidate in normalize_repo_paths(path, repo_subdir):
        for pattern in patterns:
            if fnmatch.fnmatch(candidate, pattern):
                return True
    return False


def git_repo_root(workdir: pathlib.Path) -> Optional[pathlib.Path]:
    """Get the git repository root for a working directory."""
    rc, out, err = run_cmd(
        ["git", "rev-parse", "--show-toplevel"], cwd=str(workdir), timeout=30
    )
    if rc != 0:
        return None
    root = out.strip()
    if not root:
        return None
    return pathlib.Path(root)


def snapshot_forbidden_files(
    patterns: List[str], workdir: pathlib.Path, project: "ProjectConfig"
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Create a snapshot of files matching forbidden patterns.

    Returns (files_dict, error) where files_dict maps paths to SHA-256 hashes.
    """
    repo_root = git_repo_root(workdir)
    if not repo_root:
        return None, "forbidden_repo_root_missing"
    rc, out, err = run_cmd(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--full-name",
        ],
        cwd=str(repo_root),
        timeout=30,
    )
    if rc != 0:
        return None, "forbidden_ls_files_failed"
    files: Dict[str, str] = {}
    repo_subdir = project.repo_subdir or ""
    for rel in out.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        if not path_matches(patterns, rel, repo_subdir):
            continue
        file_path = repo_root / rel
        if not file_path.is_file():
            continue
        files[rel] = sha256_file(file_path)
    return files, None


def capture_forbidden_baseline(
    graders: List[Dict[str, Any]],
    workdir: pathlib.Path,
    project: "ProjectConfig",
    baseline_path: Optional[pathlib.Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Capture baseline state of forbidden files before agent runs.

    Returns (success, error_reason).
    """
    patterns: List[str] = []
    for g in graders:
        if g.get("type") == "git_diff_forbidden":
            patterns.extend([p for p in (g.get("forbidden_globs") or []) if p])
    patterns = sorted(set(patterns))
    if not patterns:
        return True, None
    snapshot, err = snapshot_forbidden_files(patterns, workdir, project)
    if err or snapshot is None:
        return False, err or "forbidden_snapshot_failed"
    payload = {"patterns": patterns, "files": snapshot}
    if baseline_path is None:
        baseline_path = workdir / ".eval_forbidden_baseline.json"
    baseline_path.write_text(json.dumps(payload, indent=2))
    try:
        baseline_path.chmod(0o444)
    except Exception:
        pass
    return True, None


def grader_ios_install_check(
    workdir: pathlib.Path, project: "ProjectConfig"
) -> Tuple[bool, Optional[str]]:
    """
    Deterministic install verification using simctl get_app_container.
    """
    bundle_id = project.build_params.get("bundle_id")
    if not bundle_id:
        return False, "grader_missing_bundle_id"
    udid = resolve_simulator_udid_for_project(project)
    if not udid:
        return False, "grader_missing_simulator_udid"
    rc, out, err = run_cmd(
        ["xcrun", "simctl", "get_app_container", udid, bundle_id, "app"],
        cwd=str(workdir),
        timeout=120,
    )
    if rc != 0:
        return False, "app_not_installed"
    return True, None


def grader_ios_launch_check(
    workdir: pathlib.Path, project: "ProjectConfig"
) -> Tuple[bool, Optional[str]]:
    """
    Deterministic launch check using simctl launch.
    """
    bundle_id = project.build_params.get("bundle_id")
    if not bundle_id:
        return False, "grader_missing_bundle_id"
    udid = resolve_simulator_udid_for_project(project)
    if not udid:
        return False, "grader_missing_simulator_udid"
    ensure_simulator_booted(udid)
    rc, out, err = run_cmd(
        ["xcrun", "simctl", "launch", udid, bundle_id],
        cwd=str(workdir),
        timeout=120,
    )
    if rc != 0:
        return False, "app_launch_failed"
    return True, None


def project_timeout_hint() -> int:
    """Default timeout hint for grader operations."""
    return 1800


def grader_ios_test_pass(
    workdir: pathlib.Path,
    project: "ProjectConfig",
    opts: Dict[str, Any],
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Deterministic correctness grading via xcodebuild test.
    """
    primed = project.build_params
    scheme = opts.get("scheme") or primed.get("scheme")
    workspace = opts.get("workspace") or primed.get("workspace")
    proj = opts.get("project") or primed.get("project")
    dest = opts.get("destination") or primed.get("destination")
    udid = resolve_simulator_udid_for_project(project, dest)
    if dest and "id=" in dest:
        pass
    elif udid:
        dest = f"platform=iOS Simulator,id={udid}"
    else:
        dest = dest or f"platform=iOS Simulator,name={project.simulator_name}"

    if not scheme:
        return False, "grader_missing_scheme"

    cmd = ["xcodebuild"]
    if workspace:
        cmd += ["-workspace", workspace]
    elif proj:
        cmd += ["-project", proj]
    cmd += ["-scheme", scheme, "-destination", dest]
    derived = workdir / ".derivedData"
    result_bundle = derived / "TestResults.xcresult"
    cmd += ["-derivedDataPath", str(derived)]
    cmd += ["-resultBundlePath", str(result_bundle)]

    test_plan = opts.get("test_plan")
    if test_plan:
        cmd += ["-testPlan", str(test_plan)]

    only_testing = opts.get("only_testing") or []
    for t in only_testing:
        cmd += ["-only-testing", str(t)]

    skip_testing = opts.get("skip_testing") or []
    for t in skip_testing:
        cmd += ["-skip-testing", str(t)]

    extra_args = opts.get("xcodebuild_args") or []
    cmd += [str(a) for a in extra_args]

    cmd += ["test"]

    timeout_sec = opts.get("timeout_sec") or project_timeout_hint()
    rc, out, err = run_cmd(cmd, cwd=str(workdir), env=env, timeout=timeout_sec)
    log_path = opts.get("log_path")
    if log_path:
        try:
            pathlib.Path(str(log_path)).write_text(out + "\n" + err)
        except Exception:
            pass
    if rc != 0:
        return False, "tests_failed"
    # Ensure tests actually executed
    if not result_bundle.exists():
        return False, "tests_no_result_bundle"
    test_count = count_xcresult_tests(result_bundle)
    if test_count is None:
        return False, "tests_result_parse_failed"
    if test_count == 0:
        return False, "tests_not_executed"
    return True, None


def grader_git_diff_forbidden(
    workdir: pathlib.Path, project: "ProjectConfig", opts: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Check that files matching forbidden patterns were not modified.
    """
    patterns = [p for p in (opts.get("forbidden_globs") or []) if p]
    if not patterns:
        return False, "grader_missing_forbidden_globs"

    baseline_path = opts.get("baseline_path")
    if baseline_path:
        baseline_path = pathlib.Path(str(baseline_path))
    else:
        baseline_path = workdir / ".eval_forbidden_baseline.json"
    if not baseline_path.exists():
        return False, "forbidden_baseline_missing"
    try:
        baseline = json.loads(baseline_path.read_text())
    except Exception:
        return False, "forbidden_baseline_unreadable"

    baseline_files_all = baseline.get("files", {}) or {}
    repo_subdir = project.repo_subdir or ""
    baseline_files = {
        path: digest
        for path, digest in baseline_files_all.items()
        if path_matches(patterns, path, repo_subdir)
    }

    snapshot, err = snapshot_forbidden_files(patterns, workdir, project)
    if err or snapshot is None:
        return False, err or "forbidden_snapshot_failed"

    for path, digest in baseline_files.items():
        current_digest = snapshot.get(path)
        if current_digest is None:
            return False, f"forbidden_removed:{path}"
        if current_digest != digest:
            return False, f"forbidden_modified:{path}"

    for path in snapshot.keys():
        if path not in baseline_files:
            return False, f"forbidden_added:{path}"

    return True, None


def grader_screenshot_exists(
    workdir: pathlib.Path, project: "ProjectConfig", opts: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Check that a screenshot file exists at the specified path.
    """
    screenshot_path = opts.get("path", "settings_screenshot.png")
    full_path = workdir / screenshot_path
    if not full_path.exists():
        return False, f"screenshot_not_found:{screenshot_path}"
    # Verify it's a valid PNG file
    try:
        with open(full_path, "rb") as f:
            header = f.read(8)
        if header[:8] != b'\x89PNG\r\n\x1a\n':
            return False, "screenshot_invalid_png"
    except Exception as e:
        return False, f"screenshot_read_error:{e}"
    return True, None


def grader_screenshot_compare(
    workdir: pathlib.Path, project: "ProjectConfig", opts: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Compare a screenshot to a reference image using perceptual hashing.
    """
    screenshot_path = opts.get("path", "settings_screenshot.png")
    reference_path = opts.get("reference_path")
    threshold = opts.get("threshold", 10)

    if not reference_path:
        return False, "grader_missing_reference_path"

    full_screenshot = workdir / screenshot_path
    if not full_screenshot.exists():
        return False, f"screenshot_not_found:{screenshot_path}"

    # Resolve reference path relative to eval suite root
    ref_path = pathlib.Path(reference_path)
    if not ref_path.is_absolute():
        ref_path = pathlib.Path(__file__).resolve().parent.parent / reference_path
    if not ref_path.exists():
        return False, f"reference_not_found:{reference_path}"

    try:
        from PIL import Image
        import imagehash
    except ImportError:
        # Gracefully degrade: just check file exists and is valid PNG
        try:
            with open(full_screenshot, "rb") as f:
                header = f.read(8)
            if header[:8] != b'\x89PNG\r\n\x1a\n':
                return False, "screenshot_invalid_png"
            return True, None
        except Exception as e:
            return False, f"screenshot_read_error:{e}"

    try:
        screenshot_img = Image.open(full_screenshot)
        reference_img = Image.open(ref_path)

        screenshot_hash = imagehash.phash(screenshot_img)
        reference_hash = imagehash.phash(reference_img)

        distance = screenshot_hash - reference_hash

        if distance > threshold:
            return False, f"screenshot_mismatch:distance={distance},threshold={threshold}"

        return True, None
    except Exception as e:
        return False, f"screenshot_compare_error:{e}"


def run_graders(
    graders: List[Dict[str, Any]],
    workdir: pathlib.Path,
    project: "ProjectConfig",
    timeout_sec: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    """
    Run all graders for a task and return results.

    Returns (all_passed, first_failure_reason, detailed_results).
    """
    results: List[Dict[str, Any]] = []
    for g in graders:
        gtype = g.get("type")
        t0 = time.time()
        if gtype == "ios_install_check":
            ok, reason = grader_ios_install_check(workdir, project)
        elif gtype == "ios_launch_check":
            ok, reason = grader_ios_launch_check(workdir, project)
        elif gtype == "ios_test_pass":
            gg = g
            if timeout_sec is not None and g.get("timeout_sec") is None:
                gg = dict(g)
                gg["timeout_sec"] = timeout_sec
            ok, reason = grader_ios_test_pass(workdir, project, gg, env=env)
        elif gtype == "git_diff_forbidden":
            ok, reason = grader_git_diff_forbidden(workdir, project, g)
        elif gtype == "screenshot_exists":
            ok, reason = grader_screenshot_exists(workdir, project, g)
        elif gtype == "screenshot_compare":
            ok, reason = grader_screenshot_compare(workdir, project, g)
        else:
            reason = f"unknown_grader:{gtype}"
            ok = False
        results.append(
            {
                "type": gtype,
                "ok": bool(ok),
                "reason": reason,
                "duration_sec": round(time.time() - t0, 3),
            }
        )
        if not ok:
            return False, reason or f"grader_failed:{gtype}", results
    return True, None, results
