import json
import unittest
from types import SimpleNamespace

from workloads.generators.session_kv import generate_sessions, run


class SessionKVGeneratorTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "num_sessions": 10,
            "session_rate": 2.0,
            "seed": 7,
            "output": "unused.jsonl",
            "gap_profile": "mixed",
            "first_arrival_sec": 0.0,
            "max_turns": 12,
            "turn_stop_prob": 0.25,
            "max_context_toks": 32768,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_generation_is_deterministic_and_append_only(self):
        first = generate_sessions(self._args())
        second = generate_sessions(self._args())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        previous_arrival = -1
        for session in first:
            self.assertGreaterEqual(session["arrival_time_ns"], previous_arrival)
            previous_arrival = session["arrival_time_ns"]
            self.assertTrue(session["reuse_previous_kv"])
            requests = session["sub_requests"]
            self.assertEqual(requests[-1]["tool_duration_ns"], 0)
            for predecessor, continuation in zip(requests, requests[1:]):
                self.assertGreater(
                    continuation["input_toks"],
                    predecessor["input_toks"] + predecessor["output_toks"],
                )
                self.assertGreater(predecessor["tool_duration_ns"], 0)

    def test_run_writes_jsonl_and_summary(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "sessions.jsonl"
            self.assertEqual(run(self._args(output=str(output))), 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            summary = json.loads(
                output.with_suffix(".jsonl.summary.json").read_text())
        self.assertEqual(len(rows), 10)
        self.assertEqual(summary["realized"]["sessions"], 10)
        self.assertGreater(summary["realized"]["requests"], 10)

    def test_invalid_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "session_rate"):
            generate_sessions(self._args(session_rate=0))


if __name__ == "__main__":
    unittest.main()
