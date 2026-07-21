#!/usr/bin/env python3
"""Run correctness-qualified Chakra pipeline benchmarks in Docker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time


SCENARIOS = {
    "dense1": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "workloads/example_trace.jsonl",
        "num_reqs": 1,
    },
    "dense3": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "workloads/example_trace.jsonl",
        "num_reqs": 3,
    },
    "dense100": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl",
        "num_reqs": 100,
    },
    "chunked1": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl",
        "num_reqs": 1,
    },
    "dense_tp2": {
        "cluster": "configs/cluster/single_node_tensor_parallel_instance.json",
        "dataset": "workloads/example_trace.jsonl",
        "num_reqs": 1,
    },
    "agentic1": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 1,
    },
    "agentic10": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 10,
    },
    "agentic50": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 50,
    },
    "agentic100": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 100,
    },
    "agentic300": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 300,
    },
    "agentic10-sequential": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:sequential",
        "num_reqs": 10,
    },
    "agentic10-burst": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:burst",
        "num_reqs": 10,
    },
    "agentic50-sequential": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:sequential",
        "num_reqs": 50,
    },
    "agentic50-burst": {
        "cluster": "configs/cluster/single_node_single_instance.json",
        "dataset": "generated:burst",
        "num_reqs": 50,
    },
    "pd10": {
        "cluster": "configs/cluster/single_node_pd_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 10,
    },
    "pd50": {
        "cluster": "configs/cluster/single_node_pd_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 50,
    },
    "pd100-burst": {
        "cluster": "configs/cluster/single_node_pd_instance.json",
        "dataset": "generated:burst",
        "num_reqs": 100,
    },
    "session_kv1": {
        "cluster": (
            "configs/cluster/"
            "single_node_session_kv_experiment_offload_16gb.json"
        ),
        "dataset": "workloads/example_session_retention_zero_wait.jsonl",
        "num_reqs": 1,
    },
    "session_kv10": {
        "cluster": (
            "configs/cluster/"
            "single_node_session_kv_experiment_offload_16gb.json"
        ),
        "dataset": "generated:moderate",
        "num_reqs": 10,
    },
    "dp_ep10": {
        "cluster": "configs/cluster/single_node_moe_dp_ep_instance.json",
        "dataset": "generated:moderate",
        "num_reqs": 10,
    },
    "local_ep1": {
        "cluster": "configs/cluster/single_node_moe_single_instance.json",
        "dataset": "workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl",
        "num_reqs": 1,
    },
    "pim1": {
        "cluster": "configs/cluster/single_node_pim_instance.json",
        "dataset": "workloads/example_trace.jsonl",
        "num_reqs": 1,
        "extra_args": ["--enable-attn-offloading"],
    },
    "pim_subbatch3": {
        "cluster": "configs/cluster/single_node_pim_instance.json",
        "dataset": "workloads/example_trace.jsonl",
        "num_reqs": 3,
        "extra_args": [
            "--enable-attn-offloading",
            "--enable-sub-batch-interleaving",
        ],
    },
    "pp1": {
        "cluster": (
            "configs/cluster/single_node_pipeline_parallel_instance.json"
        ),
        "dataset": "workloads/example_trace.jsonl",
        "num_reqs": 1,
    },
}

GENERATED_WORKLOAD_SEED = 20260721
GENERATED_DATASET_MOUNT = "/benchmark-input/chakra-benchmark-agentic.jsonl"
BASELINE_GENERATED_DATASET_MOUNT = (
    "/repo/workloads/chakra-benchmark-agentic.jsonl")

TOTAL_CLOCKS_RE = re.compile(r"Total clocks \(ns\):\s+(\d+)")
SIMULATION_TIME_RE = re.compile(
    r"Total simulation time:\s+(\d+)h\s+(\d+)m\s+([0-9.]+)s"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare an isolated pre-optimization checkout with the current "
            "Chakra IPC implementation."
        )
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--current-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Defaults to outputs/chakra-plan-benchmark/<timestamp>.",
    )
    parser.add_argument(
        "--image", default="llmservingsim-plan-dev:latest")
    parser.add_argument(
        "--scenario", action="append", choices=sorted(SCENARIOS),
        help="May be repeated. Defaults to dense1.",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--modes", default="baseline,direct",
        help=("Comma-separated subset of baseline,direct,direct-cold,"
              "oracle,file."),
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip the untimed warm-up pair.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse successful per-run records already present in output-root.",
    )
    parser.add_argument(
        "--cpus", type=float, default=None,
        help="Optional Docker CPU limit applied identically to every run.",
    )
    parser.add_argument(
        "--tracemalloc-frames", type=int, default=0,
        help="Enable Python allocation tracing with this many stack frames.",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.tracemalloc_frames < 0:
        parser.error("--tracemalloc-frames must be non-negative")
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    invalid_modes = set(modes) - {
        "baseline", "direct", "direct-cold", "oracle", "file"
    }
    if invalid_modes:
        parser.error(f"Unsupported modes: {sorted(invalid_modes)}")
    if len(modes) < 2:
        parser.error("At least two modes are required for a comparison")
    args.modes = modes
    args.scenario = args.scenario or ["dense1"]
    return args


def run_command(command, cwd=None, timeout=None, check=True):
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        rendered = subprocess.list2cmdline([str(value) for value in command])
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{rendered}\n{result.stdout[-4000:]}"
        )
    return result


def git_state(root):
    commit = run_command(
        ["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    status = run_command(
        ["git", "status", "--short"], cwd=root).stdout
    submodule = run_command(
        ["git", "-C", "astra-sim", "rev-parse", "HEAD"],
        cwd=root,
    ).stdout.strip()
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "astra_sim_commit": submodule,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_agentic_inputs(result_root):
    """Create small deterministic workloads for repeatable session scaling."""
    profiles = {
        "sequential": {
            "arrival_spacing_ns": 5_000_000_000,
            "tool_gaps_ns": (2_000_000_000, 3_000_000_000),
        },
        "moderate": {
            "arrival_spacing_ns": 500_000_000,
            "tool_gaps_ns": (100_000_000, 800_000_000),
        },
        "burst": {
            "arrival_spacing_ns": 100_000_000,
            "tool_gaps_ns": (1_000_000_000, 2_500_000_000),
        },
    }
    generated = {}
    for profile_index, (profile, settings) in enumerate(profiles.items()):
        rng = random.Random(GENERATED_WORKLOAD_SEED + profile_index)
        path = result_root / f"agentic-{profile}-seed{GENERATED_WORKLOAD_SEED}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for session_index in range(300):
                turns = []
                for turn_index in range(3):
                    input_toks = 24 + turn_index * 8 + rng.randrange(0, 17)
                    tool_duration_ns = (
                        settings["tool_gaps_ns"][turn_index]
                        if turn_index < 2 else 0
                    )
                    turns.append({
                        "input_toks": input_toks,
                        "output_toks": 2,
                        "tool_duration_ns": tool_duration_ns,
                    })
                row = {
                    "session_id": f"{profile}-{session_index:03d}",
                    "arrival_time_ns": (
                        session_index * settings["arrival_spacing_ns"]),
                    "reuse_previous_kv": True,
                    "sub_requests": turns,
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
        generated[profile] = path
    return generated


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_simulation_seconds(stdout):
    match = SIMULATION_TIME_RE.search(stdout)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def csv_summary(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    artifacts = {"requests": sha256_file(path)}
    kv_offload_path = path.with_name(
        f"{path.stem}_kv_offload{path.suffix or '.csv'}")
    if kv_offload_path.is_file():
        artifacts["kv_offload"] = sha256_file(kv_offload_path)
    combined = hashlib.sha256()
    for name, digest in sorted(artifacts.items()):
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "sha256": sha256_file(path),
        "artifact_sha256": artifacts,
        "combined_sha256": combined.hexdigest(),
        "rows": len(rows),
        "request_ids": [row.get("request id") for row in rows],
        "end_times": [row.get("end_time") for row in rows],
    }


def source_input_manifest(root, scenario, generated_inputs):
    result = {}
    for key in ("cluster", "dataset"):
        if key == "dataset" and scenario[key].startswith("generated:"):
            profile = scenario[key].split(":", 1)[1]
            path = generated_inputs[profile]
            result[key] = {
                "path": path.name,
                "profile": profile,
                "seed": GENERATED_WORKLOAD_SEED,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            continue
        relative = Path(scenario[key])
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key} input: {path}")
        result[key] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return result


def docker_command(args, mode, source_root, result_root, scenario, stem,
                   generated_inputs):
    output_path = f"/results/{stem}.csv"
    support_root = (
        args.current_root.resolve() / "scripts" / "benchmark_support")
    command = ["docker", "run", "--rm"]
    if args.cpus is not None:
        command.extend(["--cpus", str(args.cpus)])
    dataset = scenario["dataset"]
    generated_mount = None
    if dataset.startswith("generated:"):
        profile = dataset.split(":", 1)[1]
        dataset_mount = (
            BASELINE_GENERATED_DATASET_MOUNT
            if mode == "baseline" else GENERATED_DATASET_MOUNT
        )
        generated_mount = (
            f"{generated_inputs[profile].resolve()}:"
            f"{dataset_mount}:ro"
        )
        dataset = (
            "workloads/chakra-benchmark-agentic.jsonl"
            if mode == "baseline" else dataset_mount
        )
    command.extend([
        "-v", f"{source_root}:/repo",
        "-v", f"{result_root}:/results",
        "-v", f"{support_root}:/benchmark-support:ro",
        "-e", "PYTHONPATH=/benchmark-support",
        "-e", "CHAKRA_ROOT=/repo/astra-sim/extern/graph_frontend/chakra",
    ])
    if args.tracemalloc_frames:
        command.extend([
            "-e", f"PYTHONTRACEMALLOC={args.tracemalloc_frames}",
        ])
    if generated_mount is not None:
        command.extend(["-v", generated_mount])
    if mode != "baseline":
        cache_directory = (
            f"/results/profile-cache-cold-{stem}"
            if mode == "direct-cold" else
            "/results/profile-cache-warm"
        )
        command.extend([
            "-e", f"LLMSERVINGSIM_PROFILE_CACHE_DIR={cache_directory}",
        ])
    command.extend([
        "-w", "/repo",
        args.image,
        "python3", "/benchmark-support/run_with_resource.py",
        "--output", f"/results/{stem}-resource.json",
        "--",
        "python3", "-m", "serving",
        "--cluster-config", scenario["cluster"],
        "--dataset", dataset,
        "--num-reqs", str(scenario["num_reqs"]),
        "--dtype", "bfloat16",
        "--block-size", "16",
        "--log-level", "WARNING",
        "--output", output_path,
    ])
    command.extend(scenario.get("extra_args", ()))
    if mode in ("direct", "direct-cold"):
        command.extend([
            "--workload-transport", "ipc",
            "--ipc-execution", "direct",
            "--chakra-converter", "in-process",
            "--host-timing-output", f"/results/{stem}-timing.json",
        ])
    elif mode == "oracle":
        command.extend([
            "--workload-transport", "ipc",
            "--ipc-execution", "oracle",
            "--chakra-converter", "in-process",
            "--host-timing-output", f"/results/{stem}-timing.json",
        ])
    elif mode == "file":
        command.extend([
            "--workload-transport", "file",
            "--chakra-converter", "in-process",
            "--host-timing-output", f"/results/{stem}-timing.json",
        ])
    elif mode != "baseline":
        raise ValueError(f"Unknown mode: {mode}")
    return command


def execute_run(args, scenario_name, mode, run_index, warmup, roots,
                result_root, generated_inputs):
    scenario = SCENARIOS[scenario_name]
    source_root = roots["baseline" if mode == "baseline" else "current"]
    suffix = "warmup" if warmup else f"run{run_index:02d}"
    stem = f"{scenario_name}-{mode}-{suffix}"
    record_path = result_root / f"{stem}-record.json"
    if args.resume and record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        identity = (
            record.get("scenario"), record.get("mode"),
            record.get("run_index"), record.get("warmup"),
        )
        expected = (scenario_name, mode, run_index, warmup)
        if (identity != expected or record.get("returncode") != 0 or
                record.get("correctness") is None):
            raise RuntimeError(
                f"Cannot resume invalid benchmark record: {record_path}")
        return record
    command = docker_command(
        args, mode, source_root, result_root, scenario, stem,
        generated_inputs)
    started_at = time.time()
    before = time.perf_counter()
    timed_out = False
    try:
        result = run_command(
            command, timeout=args.timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        result = subprocess.CompletedProcess(
            command, 124, stdout=(error.stdout or "") + (error.stderr or ""))
    wall_seconds = time.perf_counter() - before
    stdout = result.stdout or ""
    log_path = result_root / f"{stem}.log"
    log_path.write_text(stdout, encoding="utf-8")

    csv_path = result_root / f"{stem}.csv"
    clock_match = TOTAL_CLOCKS_RE.search(stdout)
    record = {
        "scenario": scenario_name,
        "mode": mode,
        "run_index": run_index,
        "warmup": warmup,
        "started_at_unix": started_at,
        "wall_seconds": wall_seconds,
        "returncode": result.returncode,
        "timed_out": timed_out,
        "command": [str(value) for value in command],
        "log": log_path.name,
        "total_clocks_ns": int(clock_match.group(1)) if clock_match else None,
        "reported_simulation_loop_seconds": parse_simulation_seconds(stdout),
    }
    if csv_path.is_file():
        record["output"] = csv_path.name
        record["correctness"] = csv_summary(csv_path)
    else:
        record["output"] = None
        record["correctness"] = None

    timing_path = result_root / f"{stem}-timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        record["host_timing"] = timing_path.name
        record["transport_counters"] = timing.get("counters", {})
        record["host_timing_metadata"] = timing.get("metadata", {})
        record["workload_metrics"] = timing.get(
            "metadata", {}).get("workload_metrics", {})
        record["host_metrics"] = timing.get("metrics", {})
        record["sequence_digests"] = timing.get("digests", {})
        record["stage_timing"] = timing.get("stages", {})
        record["rss_checkpoints"] = timing.get("rss_checkpoints", [])
        record["tracemalloc_top"] = timing.get("tracemalloc_top", [])
        stages = record["stage_timing"]
        frontend_seconds = stages.get(
            "frontend_elapsed", {}).get("total_seconds")
        if frontend_seconds is not None:
            loop_seconds = stages.get(
                "simulation_loop", {}).get("total_seconds", 0.0)
            cleanup_seconds = stages.get(
                "input_cleanup", {}).get("total_seconds", 0.0)
            outside_seconds = max(
                0.0, frontend_seconds - loop_seconds - cleanup_seconds)
            component_sum = loop_seconds + cleanup_seconds + outside_seconds
            record["stage_reconciliation"] = {
                "frontend_seconds": frontend_seconds,
                "simulation_loop_seconds": loop_seconds,
                "input_cleanup_seconds": cleanup_seconds,
                "outside_loop_and_cleanup_seconds": outside_seconds,
                "relative_error": (
                    abs(component_sum - frontend_seconds) / frontend_seconds
                    if frontend_seconds else 0.0),
            }
    else:
        record["host_timing"] = None

    resource_path = result_root / f"{stem}-resource.json"
    if resource_path.is_file():
        record["resource"] = resource_path.name
        record["resource_usage"] = json.loads(
            resource_path.read_text(encoding="utf-8"))
    else:
        record["resource"] = None

    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if result.returncode != 0 or record["correctness"] is None:
        raise RuntimeError(
            f"Benchmark run failed: {stem}; see {log_path}")
    return record


def aggregate(records, scenarios, modes):
    summary = {"scenarios": {}}
    for scenario_name in scenarios:
        scenario_summary = {"modes": {}, "comparisons": {}}
        measured = [
            record for record in records
            if record["scenario"] == scenario_name and not record["warmup"]
        ]
        for mode in modes:
            selected = [record for record in measured if record["mode"] == mode]
            values = [record["wall_seconds"] for record in selected]
            fingerprints = sorted({
                (
                    record["correctness"]["combined_sha256"],
                    record["total_clocks_ns"],
                    record["correctness"]["rows"],
                )
                for record in selected
            })
            sequence_digests = sorted({
                tuple(sorted(
                    (name, value["sha256"])
                    for name, value in
                    record.get("sequence_digests", {}).items()
                ))
                for record in selected
            })
            scenario_summary["modes"][mode] = {
                "runs": len(values),
                "median_wall_seconds": statistics.median(values),
                "iqr_wall_seconds": (
                    percentile(values, 0.75) - percentile(values, 0.25)
                ),
                "min_wall_seconds": min(values),
                "max_wall_seconds": max(values),
                "fingerprints": fingerprints,
                "internally_deterministic": (
                    len(fingerprints) == 1 and len(sequence_digests) == 1),
                "sequence_digests": sequence_digests,
            }
            simulation_loop_values = [
                record["reported_simulation_loop_seconds"]
                for record in selected
                if record["reported_simulation_loop_seconds"] is not None
            ]
            resource_values = [
                record["resource_usage"] for record in selected
                if record.get("resource_usage") is not None
            ]
            counters = sorted({
                counter
                for record in selected
                for counter in record.get("transport_counters", {})
            })
            counter_medians = {
                counter: statistics.median([
                    record.get("transport_counters", {}).get(counter, 0)
                    for record in selected
                ])
                for counter in counters
            }
            hits = counter_medians.get("template_cache_hits", 0)
            misses = counter_medians.get("template_cache_misses", 0)
            stage_names = sorted({
                stage
                for record in selected
                for stage in record.get("stage_timing", {})
            })
            mode_summary = scenario_summary["modes"][mode]
            checkpoint_names = sorted({
                checkpoint.get("name")
                for record in selected
                for checkpoint in record.get("rss_checkpoints", [])
                if checkpoint.get("name")
            })
            median_rss_checkpoints = {}
            for checkpoint_name in checkpoint_names:
                matching = [
                    checkpoint
                    for record in selected
                    for checkpoint in record.get("rss_checkpoints", [])
                    if checkpoint.get("name") == checkpoint_name
                ]
                median_rss_checkpoints[checkpoint_name] = {}
                for process_name in ("frontend", "backend"):
                    process_values = [
                        checkpoint.get(process_name)
                        for checkpoint in matching
                        if checkpoint.get(process_name) is not None
                    ]
                    process_summary = {}
                    for field in ("rss_kib", "hwm_kib"):
                        field_values = [
                            value[field] for value in process_values
                            if value.get(field) is not None
                        ]
                        process_summary[field] = (
                            statistics.median(field_values)
                            if field_values else None
                        )
                    median_rss_checkpoints[checkpoint_name][
                        process_name] = process_summary
            mode_summary.update({
                "median_simulation_loop_seconds": (
                    statistics.median(simulation_loop_values)
                    if simulation_loop_values else None),
                "median_user_cpu_seconds": (
                    statistics.median([
                        value["user_cpu_seconds"] for value in resource_values
                    ]) if resource_values else None),
                "median_system_cpu_seconds": (
                    statistics.median([
                        value["system_cpu_seconds"] for value in resource_values
                    ]) if resource_values else None),
                "median_peak_rss_kib": (
                    statistics.median([
                        value["max_rss_kib"] for value in resource_values
                    ]) if resource_values else None),
                "max_peak_rss_kib": (
                    max(value["max_rss_kib"] for value in resource_values)
                    if resource_values else None),
                "median_rss_checkpoints": median_rss_checkpoints,
                "template_cache_hit_rate": (
                    hits / (hits + misses) if hits + misses else None),
                "median_counters": counter_medians,
                "median_stage_total_seconds": {
                    stage: statistics.median([
                        record.get("stage_timing", {}).get(
                            stage, {}).get("total_seconds", 0.0)
                        for record in selected
                    ])
                    for stage in stage_names
                },
                "workload_metrics": (
                    selected[0].get("workload_metrics", {})
                    if selected else {}),
                "stage_reconciliation": {
                    key: statistics.median([
                        record.get("stage_reconciliation", {}).get(key, 0.0)
                        for record in selected
                    ])
                    for key in (
                        "frontend_seconds",
                        "simulation_loop_seconds",
                        "input_cleanup_seconds",
                        "outside_loop_and_cleanup_seconds",
                        "relative_error",
                    )
                },
            })
            compute_batches = counter_medians.get("compute_batches", 0)
            turns = mode_summary["workload_metrics"].get(
                "agentic_turns", 0)
            sessions = mode_summary["workload_metrics"].get(
                "terminal_sessions", 0)
            mode_summary["host_rates"] = {
                "compute_batches_per_second": (
                    compute_batches / mode_summary["median_wall_seconds"]),
                "turns_per_second": (
                    turns / mode_summary["median_wall_seconds"]),
                "sessions_per_second": (
                    sessions / mode_summary["median_wall_seconds"]),
                "host_seconds_per_1000_compute_batches": (
                    mode_summary["median_wall_seconds"] * 1000 /
                    compute_batches if compute_batches else None),
            }
        for reference_index, reference_mode in enumerate(modes):
            reference = scenario_summary["modes"][reference_mode]
            for mode in modes[reference_index + 1:]:
                candidate = scenario_summary["modes"][mode]
                reference_sequences = dict(
                    reference["sequence_digests"][0]
                    if len(reference["sequence_digests"]) == 1 else ())
                candidate_sequences = dict(
                    candidate["sequence_digests"][0]
                    if len(candidate["sequence_digests"]) == 1 else ())
                common_sequences = (
                    reference_sequences.keys() & candidate_sequences.keys())
                sequence_match = all(
                    reference_sequences[name] == candidate_sequences[name]
                    for name in common_sequences
                )
                exact = (
                    reference["internally_deterministic"] and
                    candidate["internally_deterministic"] and
                    reference["fingerprints"] == candidate["fingerprints"] and
                    sequence_match
                )
                scenario_summary["comparisons"][
                    f"{reference_mode}_vs_{mode}"] = {
                    "correctness_exact": exact,
                    "speedup": (
                        reference["median_wall_seconds"] /
                        candidate["median_wall_seconds"]),
                    "correctness_qualified_speedup": exact,
                    }
        summary["scenarios"][scenario_name] = scenario_summary
    return summary


def main():
    args = parse_args()
    roots = {
        "baseline": args.baseline_root.resolve(),
        "current": args.current_root.resolve(),
    }
    for label, root in roots.items():
        if not (root / "serving" / "__main__.py").is_file():
            raise FileNotFoundError(f"Invalid {label} root: {root}")
        binary = (
            root / "astra-sim" / "build" / "astra_analytical" / "build" /
            "AnalyticalAstra" / "bin" / "AnalyticalAstra"
        )
        # The build creates a Linux symlink. Windows cannot stat its target,
        # but Docker resolves it correctly inside the mounted workspace.
        if not os.path.lexists(binary):
            raise FileNotFoundError(
                f"Build the analytical backend before benchmarking: {binary}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else roots["current"] / "outputs" / "chakra-plan-benchmark" / timestamp
    )
    result_root.mkdir(parents=True, exist_ok=args.resume)
    generated_inputs = generate_agentic_inputs(result_root)

    image = run_command([
        "docker", "image", "inspect", args.image,
        "--format", "{{.Id}}",
    ]).stdout.strip()
    inputs = {}
    input_labels = (
        ("baseline", "current")
        if "baseline" in args.modes else ("current",)
    )
    for scenario_name in args.scenario:
        inputs[scenario_name] = {
            label: source_input_manifest(
                root, SCENARIOS[scenario_name], generated_inputs)
            for label, root in roots.items()
            if label in input_labels
        }
        if "baseline" in input_labels:
            for key in ("cluster", "dataset"):
                baseline_hash = (
                    inputs[scenario_name]["baseline"][key]["sha256"])
                current_hash = inputs[scenario_name]["current"][key]["sha256"]
                if baseline_hash != current_hash:
                    raise RuntimeError(
                        f"{scenario_name} {key} differs between baseline and current")

    manifest = {
        "schema_version": 1,
        "created_at": timestamp,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "docker_image": {"name": args.image, "id": image},
        "roots": {label: str(root) for label, root in roots.items()},
        "git": {label: git_state(root) for label, root in roots.items()},
        "inputs": inputs,
        "scenarios": args.scenario,
        "modes": args.modes,
        "repetitions": args.repetitions,
        "timeout_seconds": args.timeout_seconds,
        "cpus": args.cpus,
        "generated_workload_seed": GENERATED_WORKLOAD_SEED,
        "tracemalloc_frames": args.tracemalloc_frames,
        "resumed": args.resume,
    }
    (result_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    records = []
    for scenario_name in args.scenario:
        if not args.skip_warmup:
            for mode in args.modes:
                record = execute_run(
                    args, scenario_name, mode, 0, True, roots, result_root,
                    generated_inputs)
                records.append(record)
                print(
                    f"warmup {scenario_name}/{mode}: "
                    f"{record['wall_seconds']:.3f}s",
                    flush=True,
                )

        for run_index in range(1, args.repetitions + 1):
            ordered_modes = (
                args.modes if run_index % 2 else list(reversed(args.modes)))
            for mode in ordered_modes:
                record = execute_run(
                    args, scenario_name, mode, run_index, False,
                    roots, result_root, generated_inputs,
                )
                records.append(record)
                print(
                    f"run {run_index} {scenario_name}/{mode}: "
                    f"{record['wall_seconds']:.3f}s",
                    flush=True,
                )

    (result_root / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    summary = aggregate(records, args.scenario, args.modes)
    (result_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results: {result_root}")


if __name__ == "__main__":
    main()
