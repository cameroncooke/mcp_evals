"""
Tests for xcresult parsing.

Uses a real xcresult fixture from HackerNews project to verify parsing.
"""

from __future__ import annotations

import pathlib
import unittest

from evals.xcresult import (
    extract_xcresult_id,
    find_xcresult_tests_ref_id,
    count_xcresult_tests_node,
    count_xcresult_tests,
    read_xcresult_json,
    read_xcresult_test_summary,
)


FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
XCRESULT_PATH = FIXTURES_DIR / "sample.xcresult"


class TestExtractXcresultId(unittest.TestCase):
    """Test ID extraction from various formats."""

    def test_extract_plain_string(self) -> None:
        """Plain string ID returned as-is."""
        self.assertEqual(extract_xcresult_id("abc123"), "abc123")

    def test_extract_from_value_dict(self) -> None:
        """Extract ID from _value nested dict."""
        value = {"_value": "nested-id-456"}
        self.assertEqual(extract_xcresult_id(value), "nested-id-456")

    def test_extract_from_id_dict(self) -> None:
        """Extract ID from nested id field."""
        value = {"id": {"_value": "deeply-nested-789"}}
        self.assertEqual(extract_xcresult_id(value), "deeply-nested-789")

    def test_extract_none_from_invalid(self) -> None:
        """Return None for invalid input."""
        self.assertIsNone(extract_xcresult_id(None))
        self.assertIsNone(extract_xcresult_id(123))
        self.assertIsNone(extract_xcresult_id([]))
        self.assertIsNone(extract_xcresult_id({"other": "field"}))


class TestFindXcresultTestsRefId(unittest.TestCase):
    """Test finding tests reference ID in xcresult structure."""

    def test_find_tests_ref_in_actions(self) -> None:
        """Find testsRef in typical xcresult structure."""
        root = {
            "actions": {
                "_values": [
                    {
                        "actionResult": {
                            "testsRef": {"id": {"_value": "test-ref-123"}}
                        }
                    }
                ]
            }
        }
        self.assertEqual(find_xcresult_tests_ref_id(root), "test-ref-123")

    def test_find_tests_ref_string_format(self) -> None:
        """Find testsRef when it's a plain string."""
        root = {
            "actions": {
                "_values": [
                    {
                        "actionResult": {
                            "testsRef": "plain-ref-456"
                        }
                    }
                ]
            }
        }
        self.assertEqual(find_xcresult_tests_ref_id(root), "plain-ref-456")

    def test_returns_none_when_no_actions(self) -> None:
        """Return None when no actions present."""
        self.assertIsNone(find_xcresult_tests_ref_id({}))
        self.assertIsNone(find_xcresult_tests_ref_id({"actions": {}}))
        self.assertIsNone(find_xcresult_tests_ref_id({"actions": {"_values": []}}))

    def test_returns_none_when_no_tests_ref(self) -> None:
        """Return None when actions have no testsRef."""
        root = {
            "actions": {
                "_values": [
                    {"actionResult": {"buildResult": {}}}
                ]
            }
        }
        self.assertIsNone(find_xcresult_tests_ref_id(root))


class TestCountXcresultTestsNode(unittest.TestCase):
    """Test recursive test counting."""

    def test_count_single_test_summary(self) -> None:
        """Count a single ActionTestSummary."""
        node = {
            "_type": {"_name": "ActionTestSummary"},
            "name": {"_value": "testExample"}
        }
        self.assertEqual(count_xcresult_tests_node(node), 1)

    def test_count_nested_test_summaries(self) -> None:
        """Count multiple nested ActionTestSummary nodes."""
        node = {
            "_type": {"_name": "ActionTestPlanRunSummaries"},
            "summaries": {
                "_values": [
                    {
                        "_type": {"_name": "ActionTestSummary"},
                        "name": {"_value": "test1"}
                    },
                    {
                        "_type": {"_name": "ActionTestSummary"},
                        "name": {"_value": "test2"}
                    },
                    {
                        "_type": {"_name": "ActionTestSummaryGroup"},
                        "subtests": {
                            "_values": [
                                {
                                    "_type": {"_name": "ActionTestSummary"},
                                    "name": {"_value": "test3"}
                                }
                            ]
                        }
                    }
                ]
            }
        }
        self.assertEqual(count_xcresult_tests_node(node), 3)

    def test_count_returns_zero_for_empty(self) -> None:
        """Return 0 for empty or non-test nodes."""
        self.assertEqual(count_xcresult_tests_node({}), 0)
        self.assertEqual(count_xcresult_tests_node([]), 0)
        self.assertEqual(count_xcresult_tests_node(None), 0)
        self.assertEqual(count_xcresult_tests_node("string"), 0)

    def test_count_ignores_non_test_types(self) -> None:
        """Don't count nodes that aren't ActionTestSummary."""
        node = {
            "_type": {"_name": "ActionTestSummaryGroup"},
            "name": {"_value": "TestGroup"}
        }
        self.assertEqual(count_xcresult_tests_node(node), 0)


@unittest.skipUnless(XCRESULT_PATH.exists(), "xcresult fixture not available")
class TestXcresultIntegration(unittest.TestCase):
    """Integration tests using real xcresult fixture."""

    def test_read_xcresult_json(self) -> None:
        """Read JSON from real xcresult bundle."""
        result = read_xcresult_json(XCRESULT_PATH)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        # Should have standard xcresult structure
        assert result is not None
        self.assertIn("_type", result)

    def test_read_xcresult_test_summary(self) -> None:
        """Read test summary from real xcresult bundle."""
        result = read_xcresult_test_summary(XCRESULT_PATH)
        # May return None on older Xcode versions
        if result is not None:
            self.assertIsInstance(result, dict)

    def test_count_xcresult_tests(self) -> None:
        """Count tests in real xcresult bundle."""
        count = count_xcresult_tests(XCRESULT_PATH)
        # Should return a non-negative integer
        self.assertIsNotNone(count)
        self.assertIsInstance(count, int)
        assert count is not None
        self.assertGreaterEqual(count, 0)

    def test_read_xcresult_json_nonexistent(self) -> None:
        """Return None for nonexistent xcresult."""
        result = read_xcresult_json(pathlib.Path("/nonexistent/path.xcresult"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
