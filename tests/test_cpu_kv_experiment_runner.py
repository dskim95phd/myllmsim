import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_cpu_kv_experiment as runner


class CPUKVExperimentRunnerTest(unittest.TestCase):
    def test_stage_and_load_subsets_are_parsed(self):
        self.assertEqual(
            runner._parse_stages("calibration,screen"),
            ("calibration", "screen"))
        self.assertEqual(runner._parse_loads("low,overload"), ("low", "overload"))

    def test_prepare_configs_builds_requested_session_capacities(self):
        with tempfile.TemporaryDirectory(dir=runner.REPO_ROOT) as directory:
            run_root = Path(directory)
            (run_root / "configs").mkdir()
            configs = runner._prepare_configs(run_root, (4, 32))

            config4 = json.loads(configs["session4"].read_text())
            config32 = json.loads(configs["session32"].read_text())

        self.assertEqual(config4["nodes"][0]["cpu_mem"]["mem_size"], 4)
        self.assertEqual(config32["nodes"][0]["cpu_mem"]["mem_size"], 32)
        self.assertEqual(configs["active16"], runner.ACTIVE_CONFIG)
        self.assertEqual(configs["capacity_oracle"], runner.CAPACITY_ORACLE_CONFIG)

    def test_request_metrics_reconstruct_session_makespan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "requests.csv"
            with output.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=[
                    "session id", "arrival", "end_time", "latency", "TTFT"])
                writer.writeheader()
                writer.writerows([
                    {"session id": "a", "arrival": 10, "end_time": 30,
                     "latency": 20, "TTFT": 5},
                    {"session id": "a", "arrival": 40, "end_time": 80,
                     "latency": 40, "TTFT": 7},
                    {"session id": "b", "arrival": 20, "end_time": 50,
                     "latency": 30, "TTFT": 6},
                ])
            metrics = runner._request_metrics(output)

        self.assertEqual(metrics["request_rows"], 3)
        self.assertEqual(metrics["session_rows"], 2)
        self.assertEqual(metrics["session_makespan_p50_ns"], 50)

    def test_oracle_comparison_rejects_digest_mismatch(self):
        direct = {
            "completed": True,
            "requests_sha256": "a",
            "sidecar_sha256": "b",
            "total_clocks_ns": 10,
            "batch_done_digest": "c",
            "run_batch_digest": "d",
        }
        oracle = dict(direct)
        runner._assert_oracle_match(direct, oracle)
        oracle["total_clocks_ns"] = 11
        with self.assertRaisesRegex(RuntimeError, "total_clocks_ns"):
            runner._assert_oracle_match(direct, oracle)

    def test_completed_record_revalidates_output_hashes(self):
        with tempfile.TemporaryDirectory(dir=runner.REPO_ROOT) as directory:
            case_dir = Path(directory)
            requests = case_dir / "requests.csv"
            timing = case_dir / "host_timing.json"
            record_path = case_dir / "record.json"
            requests.write_text(
                "session id,sub request index,session kv hit tier,"
                "session kv hit tokens\n",
                encoding="utf-8")
            timing.write_text("{}\n", encoding="utf-8")
            record = {
                "completed": True,
                "requests_sha256": runner._sha256(requests),
                "sidecar_sha256": None,
                "host_timing_sha256": runner._sha256(timing),
                "paths": {
                    "requests": runner._repo_relative(requests),
                    "sidecar": runner._repo_relative(case_dir / "sidecar.csv"),
                    "host_timing": runner._repo_relative(timing),
                },
            }
            runner._write_json(record_path, record)
            self.assertTrue(runner._case_complete(record_path))

            timing.write_text('{"changed": true}\n', encoding="utf-8")
            self.assertFalse(runner._case_complete(record_path))

    def test_summary_accepts_failed_record_before_successful_record(self):
        with tempfile.TemporaryDirectory(dir=runner.REPO_ROOT) as directory:
            run_root = Path(directory)
            failed_dir = run_root / "runs" / "a_failed"
            success_dir = run_root / "runs" / "b_success"
            failed_dir.mkdir(parents=True)
            success_dir.mkdir(parents=True)

            requests = success_dir / "requests.csv"
            requests.write_text(
                "session id,arrival,end_time,latency,TTFT\n"
                "session-0,10,30,20,5\n",
                encoding="utf-8")

            common = {
                "stage": "screen",
                "policy": "recompute",
                "capacity_gib": None,
                "sessions": 1,
                "session_rate": 1.0,
                "seed": 7,
                "execution": "direct",
                "returncode": 1,
                "wall_seconds": 1.0,
                "simulated_seconds": None,
                "completed_requests": None,
            }
            failed = {
                **common,
                "label": "a_failed",
                "completed": False,
                "paths": {
                    "requests": runner._repo_relative(failed_dir / "requests.csv"),
                    "sidecar": runner._repo_relative(failed_dir / "sidecar.csv"),
                },
            }
            success = {
                **common,
                "label": "b_success",
                "completed": True,
                "returncode": 0,
                "completed_requests": 1,
                "paths": {
                    "requests": runner._repo_relative(requests),
                    "sidecar": runner._repo_relative(success_dir / "sidecar.csv"),
                },
            }
            runner._write_json(failed_dir / "record.json", failed)
            runner._write_json(success_dir / "record.json", success)

            rows = runner._write_summary(run_root)
            with (run_root / "summary.csv").open(newline="", encoding="utf-8") as input_file:
                csv_rows = list(csv.DictReader(input_file))

        self.assertEqual(len(rows), 2)
        self.assertIn("request_rows", csv_rows[0])
        self.assertEqual(csv_rows[0]["request_rows"], "")
        self.assertEqual(csv_rows[1]["request_rows"], "1")


if __name__ == "__main__":
    unittest.main()
