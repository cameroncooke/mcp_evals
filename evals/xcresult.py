"""
XCResult parsing utilities for the eval suite.

Provides functions for reading and parsing Xcode result bundles (.xcresult).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

from evals.infrastructure import run_cmd


def read_xcresult_json(
    result_path: pathlib.Path, result_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Read JSON data from an xcresult bundle.

    Args:
        result_path: Path to the .xcresult bundle
        result_id: Optional result ID for specific nested data

    Returns:
        Parsed JSON data or None if reading fails
    """
    base_cmd = [
        "xcrun",
        "xcresulttool",
        "get",
        "object",
        "--format",
        "json",
        "--path",
        str(result_path),
    ]
    if result_id:
        if not isinstance(result_id, str):
            result_id = extract_xcresult_id(result_id)
        if result_id:
            base_cmd += ["--id", result_id]
    # Xcode 26+ requires --legacy; try without first for older installs.
    for extra in ([], ["--legacy"]):
        cmd = base_cmd + extra
        rc, out, err = run_cmd(cmd, timeout=60)
        if rc != 0:
            continue
        try:
            return json.loads(out)
        except Exception:
            continue
    return None


def read_xcresult_test_summary(
    result_path: pathlib.Path,
) -> Optional[Dict[str, Any]]:
    """
    Read test summary from an xcresult bundle using the newer API.

    Returns:
        Test summary JSON or None if not available
    """
    cmd = [
        "xcrun",
        "xcresulttool",
        "get",
        "test-results",
        "summary",
        "--path",
        str(result_path),
    ]
    rc, out, err = run_cmd(cmd, timeout=60)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def extract_xcresult_id(value: Any) -> Optional[str]:
    """
    Extract an ID string from various xcresult ID formats.

    Handles both plain strings and nested dict structures.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "_value" in value and isinstance(value["_value"], str):
            return value["_value"]
        if "id" in value:
            return extract_xcresult_id(value.get("id"))
    return None


def find_xcresult_tests_ref_id(root: Dict[str, Any]) -> Optional[str]:
    """
    Find the tests reference ID in an xcresult root object.

    Returns:
        The tests reference ID or None if not found
    """
    actions = root.get("actions", {}).get("_values") or []
    for action in actions:
        tests_ref = (action.get("actionResult") or {}).get("testsRef")
        if not tests_ref:
            continue
        if isinstance(tests_ref, dict):
            found = extract_xcresult_id(tests_ref.get("id"))
        else:
            found = extract_xcresult_id(tests_ref)
        if found:
            return found
    return None


def count_xcresult_tests_node(node: Any) -> int:
    """
    Recursively count test summaries in an xcresult node.

    Counts nodes where _type._name == 'ActionTestSummary'.
    """
    if isinstance(node, dict):
        count = 0
        t = node.get("_type", {})
        if isinstance(t, dict) and t.get("_name") == "ActionTestSummary":
            count += 1
        for v in node.values():
            count += count_xcresult_tests_node(v)
        return count
    if isinstance(node, list):
        return sum(count_xcresult_tests_node(v) for v in node)
    return 0


def count_xcresult_tests(result_path: pathlib.Path) -> Optional[int]:
    """
    Count the total number of tests in an xcresult bundle.

    Tries the newer test-results summary API first, falls back to
    parsing the full result structure.

    Returns:
        Number of tests or None if parsing fails
    """
    summary = read_xcresult_test_summary(result_path)
    if summary:
        total = summary.get("totalTestCount")
        if isinstance(total, int):
            return total
        if isinstance(total, str) and total.isdigit():
            return int(total)
    root = read_xcresult_json(result_path)
    if not root:
        return None
    tests_ref_id = find_xcresult_tests_ref_id(root)
    if not tests_ref_id:
        return 0
    tests_root = read_xcresult_json(result_path, tests_ref_id)
    if not tests_root:
        return None
    return count_xcresult_tests_node(tests_root)
