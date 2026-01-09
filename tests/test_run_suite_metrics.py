import datetime as dt
import pathlib
import tempfile
import unittest

from evals import (
    compute_time_to_first_mcp_build_sec,
    compute_time_to_first_xcodebuild_sec,
    compute_xcodebuild_repeat_count,
    extract_xcodebuild_destination,
    normalize_xcodebuild_argv,
    parse_log_timestamp,
)


class TestRunSuiteMetrics(unittest.TestCase):
    def test_parse_log_timestamp_z_suffix(self) -> None:
        ts = "2026-01-20T12:34:56.000000Z"
        parsed = parse_log_timestamp(ts)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.tzinfo, dt.timezone.utc)
        self.assertIsNone(parse_log_timestamp("not-a-timestamp"))

    def test_extract_xcodebuild_destination_flags(self) -> None:
        argv = [
            "-scheme",
            "HackerNews",
            "-destination",
            "platform=iOS Simulator,name=iPhone 17 Pro",
            "test",
        ]
        self.assertEqual(
            extract_xcodebuild_destination(argv),
            "platform=iOS Simulator,name=iPhone 17 Pro",
        )
        argv2 = [
            "-scheme",
            "HackerNews",
            "-destination=platform=iOS Simulator,id=ABC",
            "test",
        ]
        self.assertEqual(
            extract_xcodebuild_destination(argv2),
            "platform=iOS Simulator,id=ABC",
        )
        self.assertIsNone(extract_xcodebuild_destination(["-scheme", "HackerNews"]))
        self.assertIsNone(extract_xcodebuild_destination(["-destination"]))

    def test_normalize_xcodebuild_argv_masks_paths(self) -> None:
        argv = [
            "-scheme",
            "HackerNews",
            "-derivedDataPath",
            "/tmp/derived",
            "-resultBundlePath",
            "/tmp/result.xcresult",
            "test",
        ]
        normalized = normalize_xcodebuild_argv(argv)
        self.assertIn("<path>", normalized)
        self.assertEqual(
            normalized,
            (
                "-scheme",
                "HackerNews",
                "-derivedDataPath",
                "<path>",
                "-resultBundlePath",
                "<path>",
                "test",
            ),
        )
        # Missing path value should leave the flag untouched.
        normalized_missing = normalize_xcodebuild_argv(["-derivedDataPath"])
        self.assertEqual(normalized_missing, ("-derivedDataPath",))

    def test_compute_xcodebuild_repeat_count(self) -> None:
        entries = [
            {
                "cmd": "xcodebuild",
                "argv": [
                    "-scheme",
                    "HackerNews",
                    "-derivedDataPath",
                    "/tmp/one",
                    "test",
                ],
            },
            {
                "cmd": "xcodebuild",
                "argv": [
                    "-scheme",
                    "HackerNews",
                    "-derivedDataPath",
                    "/tmp/two",
                    "test",
                ],
            },
            {
                "cmd": "xcodebuild",
                "argv": [
                    "-scheme",
                    "Other",
                    "-derivedDataPath",
                    "/tmp/three",
                    "test",
                ],
            },
            {
                "cmd": "xcrun",
                "argv": ["simctl", "list"],
            },
        ]
        # First two normalize to the same signature; third is distinct.
        self.assertEqual(compute_xcodebuild_repeat_count(entries), 1)

    def test_time_to_first_xcodebuild_sec(self) -> None:
        ts_start = "2026-01-20T00:00:00+00:00"
        entries = [
            {
                "cmd": "xcrun",
                "ts": "2026-01-20T00:00:03+00:00",
                "argv": ["simctl", "list"],
            },
            {
                "cmd": "xcodebuild",
                "ts": "2026-01-20T00:00:05+00:00",
                "argv": ["-list"],  # Informational, should be ignored
            },
            {
                "cmd": "xcodebuild",
                "ts": "2026-01-20T00:00:07+00:00",
                "argv": ["-scheme", "App", "build"],
            },
            {
                "cmd": "xcodebuild",
                "ts": "2026-01-20T00:00:06+00:00",
                "argv": ["-scheme", "App", "test"],
            },
        ]
        # Earliest build action is test at t+6s (ignores -list at t+5s).
        self.assertEqual(compute_time_to_first_xcodebuild_sec(ts_start, entries), 6.0)
        # No xcodebuild entries returns None.
        self.assertIsNone(
            compute_time_to_first_xcodebuild_sec(
                ts_start, [{"cmd": "xcrun", "ts": "2026-01-20T00:00:01+00:00", "argv": []}]
            )
        )
        # Only informational xcodebuild commands returns None.
        self.assertIsNone(
            compute_time_to_first_xcodebuild_sec(
                ts_start, [{"cmd": "xcodebuild", "ts": "2026-01-20T00:00:01+00:00", "argv": ["-list"]}]
            )
        )

    def test_time_to_first_mcp_build_sec(self) -> None:
        ts_start = "2026-01-20T00:00:00+00:00"
        log_content = (
            '{"ts": "2026-01-20T00:00:03Z", "tool": "list_schemes"}\n'
            '{"ts": "2026-01-20T00:00:10Z", "tool": "build_sim"}\n'
            '{"ts": "2026-01-20T00:00:08Z", "tool": "test_sim"}\n'
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(log_content)
            log_path = pathlib.Path(f.name)
        try:
            # Earliest build tool (test_sim) is at t+8s.
            result = compute_time_to_first_mcp_build_sec(ts_start, log_path)
            self.assertEqual(result, 8.0)
        finally:
            log_path.unlink()

        # No build tools returns None.
        log_content_no_build = '{"ts": "2026-01-20T00:00:03Z", "tool": "list_schemes"}\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(log_content_no_build)
            log_path = pathlib.Path(f.name)
        try:
            result = compute_time_to_first_mcp_build_sec(ts_start, log_path)
            self.assertIsNone(result)
        finally:
            log_path.unlink()

        # None path returns None.
        self.assertIsNone(compute_time_to_first_mcp_build_sec(ts_start, None))


if __name__ == "__main__":
    unittest.main()
