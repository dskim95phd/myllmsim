#!/usr/bin/env python3
"""Run the staged CPU KV offloading experiment reproducibly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.generators import session_kv  # noqa: E402


CONFIG_ROOT = REPO_ROOT / "configs" / "cluster"
RECOMPUTE_CONFIG = CONFIG_ROOT / "single_node_session_kv_experiment_recompute.json"
OFFLOAD_CONFIG = CONFIG_ROOT / "single_node_session_kv_experiment_offload_16gb.json"
ACTIVE_CONFIG = CONFIG_ROOT / "single_node_session_kv_experiment_active_offload_16gb.json"
NPU_RETENTION_CONFIG = CONFIG_ROOT / "single_node_session_kv_experiment_npu_retention.json"
CAPACITY_ORACLE_CONFIG = CONFIG_ROOT / "single_node_session_kv_experiment_capacity_oracle.json"
ASTRA_BINARY = (
    REPO_ROOT / "astra-sim" / "build" / "astra_analytical" / "build" /
    "AnalyticalAstra" / "bin" / "AnalyticalAstra"
)
PROFILE_ROOT = (
    REPO_ROOT / "profiler" / "perf" / "RTXPRO6000" / "meta-llama" /
    "Llama-3.1-8B" / "bf16" / "tp1"
)
DEFAULT_CALIBRATION_RATES = (0.5, 1.0, 1.5, 2.0)
DEFAULT_SCREEN_CAPACITIES = (4, 16, 64, 256)
DEFAULT_CONFIRM_CAPACITIES = (4, 8, 16, 32, 64, 128, 256)
DEFAULT_CONFIRM_SEEDS = (7, 17, 29, 43, 71)
PILOT_STAGES = ("timing", "calibration", "screen", "validation")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _read_json(path: Path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip()


def _rate_label(rate: float) -> str:
    return format(rate, ".8g").replace(".", "p")


def _parse_numbers(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _parse_stages(value: str):
    stages = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(stages) - set(PILOT_STAGES))
    if invalid:
        raise argparse.ArgumentTypeError(
            "unknown pilot stages: " + ", ".join(invalid))
    return stages


def _parse_loads(value: str):
    loads = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(loads) - {"low", "high", "overload"})
    if invalid:
        raise argparse.ArgumentTypeError(
            "unknown confirmation loads: " + ", ".join(invalid))
    return loads


def _sidecar_path(requests_path: Path) -> Path:
    return requests_path.with_name(
        f"{requests_path.stem}_kv_offload{requests_path.suffix}")


def _prepare_run_root(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for child in ("configs", "workloads", "runs"):
        (run_root / child).mkdir(exist_ok=True)
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        _write_json(manifest_path, {
            "created_at_epoch": time.time(),
            "root_commit": _git_output("rev-parse", "HEAD"),
            "root_status": _git_output("status", "--short"),
            "submodules": _git_output("submodule", "status", "--recursive"),
            "python": sys.version,
            "runner": _repo_relative(Path(__file__)),
        })


def _prepare_configs(run_root: Path, capacities) -> dict[str, Path]:
    paths = {
        "recompute": RECOMPUTE_CONFIG,
        "active16": ACTIVE_CONFIG,
        "npu_retention": NPU_RETENTION_CONFIG,
        "capacity_oracle": CAPACITY_ORACLE_CONFIG,
    }
    base = _read_json(OFFLOAD_CONFIG)
    for capacity in sorted(set(capacities)):
        config = json.loads(json.dumps(base))
        config["nodes"][0]["cpu_mem"]["mem_size"] = capacity
        path = run_root / "configs" / f"session_offload_{capacity}gb.json"
        _write_json(path, config)
        paths[f"session{capacity}"] = path
    return paths


def _validate_environment() -> None:
    required = [
        RECOMPUTE_CONFIG, OFFLOAD_CONFIG, ACTIVE_CONFIG,
        NPU_RETENTION_CONFIG, CAPACITY_ORACLE_CONFIG,
        PROFILE_ROOT / "dense.csv", PROFILE_ROOT / "per_sequence.csv",
        PROFILE_ROOT / "attention.csv", ASTRA_BINARY,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise RuntimeError(
            "The experiment environment is incomplete. Run "
            f"scripts/compile.sh after checkout. Missing:\n  - {formatted}")


def _ensure_workload(
        run_root: Path, name: str, sessions: int, rate: float, seed: int,
        gap_profile: str, resume: bool) -> Path:
    output = run_root / "workloads" / f"{name}.jsonl"
    summary = output.with_suffix(output.suffix + ".summary.json")
    expected = {
        "num_sessions": sessions,
        "session_rate": rate,
        "seed": seed,
        "gap_profile": gap_profile,
        "first_arrival_sec": 0.0,
        "max_turns": 12,
        "turn_stop_prob": 0.25,
        "max_context_toks": 32768,
    }
    if resume and output.is_file() and summary.is_file():
        parameters = _read_json(summary).get("parameters", {})
        if all(parameters.get(key) == value for key, value in expected.items()):
            return output
        raise RuntimeError(
            f"Existing workload parameters do not match: {summary}")

    args = SimpleNamespace(
        generator="session-kv",
        num_sessions=sessions,
        session_rate=rate,
        seed=seed,
        output=str(output.resolve()),
        gap_profile=gap_profile,
        first_arrival_sec=0.0,
        max_turns=12,
        turn_stop_prob=0.25,
        max_context_toks=32768,
    )
    session_kv.run(args)
    return output


def _case_complete(record_path: Path) -> bool:
    if not record_path.is_file():
        return False
    record = _read_json(record_path)
    if not record.get("completed"):
        return False
    requests = REPO_ROOT / record.get("paths", {}).get("requests", "")
    timing = REPO_ROOT / record.get("paths", {}).get("host_timing", "")
    if not requests.is_file() or not timing.is_file():
        return False
    sidecar = REPO_ROOT / record.get("paths", {}).get("sidecar", "")
    if record.get("requests_sha256") != _sha256(requests):
        return False
    if record.get("sidecar_sha256") != _sha256(sidecar):
        return False
    if record.get("host_timing_sha256") != _sha256(timing):
        return False
    with requests.open(newline="", encoding="utf-8") as input_file:
        columns = next(csv.reader(input_file), [])
    required = {
        "session id", "sub request index", "session kv hit tier",
        "session kv hit tokens",
    }
    return required.issubset(columns)


def _run_case(
        args, run_root: Path, label: str, stage: str, policy: str,
        config: Path, workload: Path, sessions: int, rate: float, seed: int,
        capacity_gib: int | None = None, execution: str = "direct"):
    case_dir = run_root / "runs" / label
    case_dir.mkdir(parents=True, exist_ok=True)
    record_path = case_dir / "record.json"
    if args.resume and _case_complete(record_path):
        record = _read_json(record_path)
        expected = {
            "stage": stage,
            "policy": policy,
            "capacity_gib": capacity_gib,
            "sessions": sessions,
            "session_rate": rate,
            "seed": seed,
            "gap_profile": args.gap_profile,
            "execution": execution,
            "config_sha256": _sha256(config),
            "workload_sha256": _sha256(workload),
        }
        mismatches = [
            key for key, value in expected.items()
            if record.get(key) != value]
        if mismatches:
            raise RuntimeError(
                f"Resume collision for {label}; changed fields: "
                + ", ".join(mismatches))
        print(f"SKIP {label}: completed")
        return record

    requests_path = case_dir / "requests.csv"
    timing_path = case_dir / "host_timing.json"
    log_path = case_dir / "run.log"
    for output in (requests_path, _sidecar_path(requests_path), timing_path):
        if output.exists():
            output.unlink()
    shutil.copy2(config, case_dir / "cluster_config.json")
    workload_summary = workload.with_suffix(workload.suffix + ".summary.json")
    shutil.copy2(workload_summary, case_dir / "workload.summary.json")

    run_id_digest = hashlib.sha256(
        f"{run_root}:{label}".encode("utf-8")).hexdigest()[:16]
    command = [
        args.python, "-m", "serving",
        "--cluster-config", _repo_relative(config),
        "--dtype", "bfloat16",
        "--block-size", "16",
        "--dataset", str(workload.resolve()),
        "--output", str(requests_path.resolve()),
        "--num-reqs", str(sessions),
        "--network-backend", "analytical",
        "--workload-transport", "ipc",
        "--ipc-execution", execution,
        "--chakra-converter", "in-process",
        "--host-timing-output", str(timing_path.resolve()),
        "--run-id", f"cpu-kv-{run_id_digest}",
        "--log-level", args.log_level,
        "--log-interval", "10",
    ]
    (case_dir / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["LLMSERVINGSIM_PROFILE_CACHE_DIR"] = str(
        (run_root / "profile-cache").resolve())
    for variable in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    print(f"RUN  {label}")
    start = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment,
            stdout=log_file, stderr=subprocess.STDOUT, text=True)
        try:
            returncode = process.wait(timeout=args.case_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
    wall_seconds = time.perf_counter() - start

    timing = _read_json(timing_path) if timing_path.is_file() else {}
    metadata = timing.get("metadata", {})
    digests = timing.get("digests", {})
    completed = (
        returncode == 0 and requests_path.is_file() and
        timing_path.is_file() and not timed_out)
    record = {
        "label": label,
        "stage": stage,
        "policy": policy,
        "capacity_gib": capacity_gib,
        "sessions": sessions,
        "session_rate": rate,
        "seed": seed,
        "gap_profile": args.gap_profile,
        "execution": execution,
        "runner_workers": args.workers,
        "command": command,
        "config_sha256": _sha256(config),
        "workload_sha256": _sha256(workload),
        "returncode": returncode,
        "timed_out": timed_out,
        "completed": completed,
        "wall_seconds": wall_seconds,
        "simulated_seconds": metadata.get("simulated_seconds"),
        "total_clocks_ns": metadata.get("total_clocks_ns"),
        "completed_requests": metadata.get("completed_requests"),
        "requests_sha256": _sha256(requests_path),
        "sidecar_sha256": _sha256(_sidecar_path(requests_path)),
        "host_timing_sha256": _sha256(timing_path),
        "batch_done_digest": digests.get("batch_done_sequence", {}).get("sha256"),
        "run_batch_digest": digests.get("run_batch_sequence", {}).get("sha256"),
        "paths": {
            "requests": _repo_relative(requests_path),
            "sidecar": _repo_relative(_sidecar_path(requests_path)),
            "host_timing": _repo_relative(timing_path),
            "log": _repo_relative(log_path),
        },
    }
    _write_json(record_path, record)
    state = "OK" if completed else f"FAILED({returncode})"
    print(f"{state:>9} {label}: {wall_seconds:.3f}s")
    return record


def _run_jobs(args, run_root: Path, jobs):
    """Run independent simulator cases with bounded process concurrency."""
    jobs = list(jobs)
    records = {}
    if not jobs:
        return records

    cache_root = run_root / "profile-cache"
    if args.workers > 1 and not any(cache_root.glob("*.gz")):
        key, function = jobs.pop(0)
        records[key] = function()

    if args.workers == 1:
        for key, function in jobs:
            records[key] = function()
        return records

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(function): key for key, function in jobs}
        for future in as_completed(futures):
            key = futures[future]
            records[key] = future.result()
    return records


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _request_metrics(path: Path):
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    latencies = [int(row["latency"]) for row in rows]
    ttfts = [int(row["TTFT"]) for row in rows]
    sessions = {}
    for row in rows:
        session_id = row.get("session id")
        if not session_id:
            continue
        start, end = sessions.get(session_id, (None, None))
        arrival = int(row["arrival"])
        end_time = int(row["end_time"])
        sessions[session_id] = (
            arrival if start is None else min(start, arrival),
            end_time if end is None else max(end, end_time),
        )
    makespans = [end - start for start, end in sessions.values()]
    return {
        "request_rows": len(rows),
        "session_rows": len(sessions),
        "latency_p50_ns": _percentile(latencies, 50),
        "latency_p95_ns": _percentile(latencies, 95),
        "latency_p99_ns": _percentile(latencies, 99),
        "ttft_p50_ns": _percentile(ttfts, 50),
        "ttft_p95_ns": _percentile(ttfts, 95),
        "ttft_p99_ns": _percentile(ttfts, 99),
        "session_makespan_p50_ns": _percentile(makespans, 50),
        "session_makespan_p95_ns": _percentile(makespans, 95),
        "session_makespan_p99_ns": _percentile(makespans, 99),
    }


def _sidecar_metrics(path: Path):
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    numeric = {}
    for row in rows:
        for key, value in row.items():
            try:
                numeric[key] = numeric.get(key, 0) + int(value)
            except (TypeError, ValueError):
                continue
    return numeric


def _write_summary(run_root: Path):
    rows = []
    for record_path in sorted((run_root / "runs").glob("*/record.json")):
        record = _read_json(record_path)
        requests = REPO_ROOT / record["paths"]["requests"]
        sidecar = REPO_ROOT / record["paths"]["sidecar"]
        request_metrics = _request_metrics(requests)
        sidecar_metrics = _sidecar_metrics(sidecar)
        simulated = record.get("simulated_seconds")
        completed_requests = record.get("completed_requests")
        row = {
            "label": record["label"],
            "stage": record["stage"],
            "policy": record["policy"],
            "capacity_gib": record.get("capacity_gib"),
            "sessions": record["sessions"],
            "session_rate": record["session_rate"],
            "seed": record["seed"],
            "execution": record["execution"],
            "completed": record["completed"],
            "returncode": record["returncode"],
            "wall_seconds": record["wall_seconds"],
            "simulated_seconds": simulated,
            "completed_requests": completed_requests,
            "completed_requests_per_second": (
                completed_requests / simulated
                if completed_requests is not None and simulated else None),
            **request_metrics,
        }
        for key in (
                "eviction batches", "reload batches", "evict bytes",
                "reload bytes", "migration time ns", "reload stall ns",
                "session npu hit count", "session npu hit tokens",
                "session cpu hit count", "session cpu hit tokens",
                "session miss count", "session recomputed prompt tokens",
                "session capacity drop count", "session capacity drop bytes"):
            row[key] = sidecar_metrics.get(key)
        rows.append(row)

    csv_path = run_root / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as output_file:
            fieldnames = []
            seen = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    _write_json(run_root / "summary.json", {"runs": rows})
    print(f"Summary: {csv_path}")
    return rows


def _assert_oracle_match(direct, oracle) -> None:
    fields = (
        "requests_sha256", "sidecar_sha256", "total_clocks_ns",
        "batch_done_digest", "run_batch_digest")
    mismatches = [
        field for field in fields if direct.get(field) != oracle.get(field)]
    if not direct.get("completed") or not oracle.get("completed") or mismatches:
        raise RuntimeError(
            "Direct/oracle validation failed; mismatched fields: "
            + ", ".join(mismatches or ["completion status"]))


def _pilot(args, run_root: Path) -> None:
    _validate_environment()
    stages = set(args.stages)
    capacities = tuple(sorted(set(args.screen_capacities) | {16}))
    configs = _prepare_configs(run_root, capacities)

    if "timing" in stages:
        timing_workload = _ensure_workload(
            run_root, f"timing_seed{args.seed}", args.timing_sessions,
            args.timing_rate, args.seed, args.gap_profile, args.resume)
        timing_rate_label = _rate_label(args.timing_rate)
        timing_jobs = []
        for policy, config in (
                ("recompute", configs["recompute"]),
                ("session-offload", configs["session16"])):
            capacity = 16 if policy == "session-offload" else None
            label = (
                f"timing_rate_{timing_rate_label}_"
                f"{policy.replace('-', '_')}_direct")
            timing_jobs.append((label, partial(
                _run_case, args, run_root, label, "timing", policy, config,
                timing_workload, args.timing_sessions, args.timing_rate,
                args.seed, capacity)))
        _run_jobs(args, run_root, timing_jobs)

    pilot_path = run_root / "pilot.json"
    if "calibration" in stages:
        calibration_jobs = []
        for rate in args.calibration_rates:
            rate_name = _rate_label(rate)
            workload = _ensure_workload(
                run_root, f"calibrate_rate_{rate_name}_seed{args.seed}",
                args.calibration_sessions, rate, args.seed,
                args.gap_profile, args.resume)
            for policy, config in (
                    ("recompute", configs["recompute"]),
                    ("session-offload", configs["session16"])):
                capacity = 16 if policy == "session-offload" else None
                label = (
                    f"calibrate_rate_{rate_name}_"
                    f"{policy.replace('-', '_')}_direct")
                key = (rate, policy)
                calibration_jobs.append((key, partial(
                    _run_case, args, run_root, label, "calibration", policy,
                    config, workload, args.calibration_sessions, rate,
                    args.seed, capacity)))
        calibration_records = _run_jobs(
            args, run_root, calibration_jobs)
        stable_rates = [
            rate for (rate, policy), record in calibration_records.items()
            if policy == "recompute" and record.get("completed")]
        if not stable_rates:
            _write_summary(run_root)
            raise RuntimeError("No Recompute calibration rate completed.")
        lambda_sat = max(stable_rates)
        upper_bound_reached = lambda_sat == max(args.calibration_rates)
        if upper_bound_reached:
            print(
                "WARNING: the highest calibration rate completed; lambda_sat "
                "is a provisional lower bound. Extend --calibration-rates if "
                "the capacity knee is sensitive to offered load.")
        pilot_state = {
            "lambda_sat": lambda_sat,
            "low_rate": 0.6 * lambda_sat,
            "high_rate": 0.9 * lambda_sat,
            "overload_rate": 1.1 * lambda_sat,
            "calibration_upper_bound_reached": upper_bound_reached,
            "seed": args.seed,
            "screen_capacities": list(args.screen_capacities),
            "direct_oracle_validation": "pending",
        }
        _write_json(pilot_path, pilot_state)
    elif stages & {"screen", "validation"}:
        if not pilot_path.is_file():
            raise RuntimeError(
                "Run the calibration stage first: missing " + str(pilot_path))
        pilot_state = _read_json(pilot_path)
    else:
        pilot_state = _read_json(pilot_path) if pilot_path.is_file() else None

    if "screen" in stages:
        screen_jobs = []
        for load, rate in (
                ("low", pilot_state["low_rate"]),
                ("high", pilot_state["high_rate"])):
            workload = _ensure_workload(
                run_root,
                f"screen_{load}_rate_{_rate_label(rate)}_seed{args.seed}",
                args.screen_sessions, rate, args.seed,
                args.gap_profile, args.resume)
            policies = [
                ("recompute", configs["recompute"], None),
                ("active-offload", configs["active16"], 16),
                ("capacity-oracle", configs["capacity_oracle"], 256),
            ]
            policies.extend(
                ("session-offload", configs[f"session{capacity}"], capacity)
                for capacity in args.screen_capacities)
            if load == "low":
                policies.append(
                    ("npu-retention", configs["npu_retention"], None))
            for policy, config, capacity in policies:
                capacity_label = (
                    f"_{capacity}gb" if capacity is not None else "")
                label = (
                    f"screen_{load}_rate_{_rate_label(rate)}_"
                    f"{policy.replace('-', '_')}"
                    f"{capacity_label}_direct")
                key = (load, policy, capacity)
                screen_jobs.append((key, partial(
                    _run_case, args, run_root, label, "screen", policy,
                    config, workload, args.screen_sessions, rate, args.seed,
                    capacity)))
        _run_jobs(args, run_root, screen_jobs)
        pilot_state["screen_capacities"] = list(args.screen_capacities)
        pilot_state["screen_sessions"] = args.screen_sessions
        pilot_state["screen_seed"] = args.seed
        pilot_state["screen_gap_profile"] = args.gap_profile
        _write_json(pilot_path, pilot_state)

    if "validation" in stages:
        high_rate = pilot_state["high_rate"]
        screen_sessions = pilot_state.get("screen_sessions", args.screen_sessions)
        screen_seed = pilot_state.get("screen_seed", args.seed)
        screen_gap_profile = pilot_state.get(
            "screen_gap_profile", args.gap_profile)
        direct_label = (
            f"screen_high_rate_{_rate_label(high_rate)}_"
            "session_offload_16gb_direct")
        direct_path = run_root / "runs" / direct_label / "record.json"
        if not _case_complete(direct_path):
            raise RuntimeError(
                "Run the screen stage first: missing completed " + direct_label)
        direct = _read_json(direct_path)
        high_workload = _ensure_workload(
            run_root,
            f"screen_high_rate_{_rate_label(high_rate)}_seed{screen_seed}",
            screen_sessions, high_rate, screen_seed,
            screen_gap_profile, args.resume)
        oracle_label = (
            f"validate_high_rate_{_rate_label(high_rate)}_"
            "session_offload_16gb_oracle")
        oracle = _run_case(
            args, run_root, oracle_label, "validation", "session-offload",
            configs["session16"], high_workload, screen_sessions,
            high_rate, screen_seed, 16, execution="oracle")
        _assert_oracle_match(direct, oracle)
        pilot_state["direct_oracle_validation"] = "passed"
        _write_json(pilot_path, pilot_state)

    _write_summary(run_root)
    if pilot_state is not None:
        print(
            f"Pilot stages complete ({','.join(args.stages)}): "
            f"lambda_sat={pilot_state['lambda_sat']:g}, "
            f"low={pilot_state['low_rate']:g}, "
            f"high={pilot_state['high_rate']:g}, "
            f"overload={pilot_state['overload_rate']:g}")
    else:
        print(f"Pilot stages complete ({','.join(args.stages)})")


def _confirm(args, run_root: Path) -> None:
    _validate_environment()
    pilot_path = run_root / "pilot.json"
    if not pilot_path.is_file():
        raise RuntimeError(f"Run the pilot first: missing {pilot_path}")
    pilot = _read_json(pilot_path)
    all_rates = {
        ("low", pilot["low_rate"]),
        ("high", pilot["high_rate"]),
        ("overload", pilot["overload_rate"]),
    }
    rates = sorted(
        (item for item in all_rates if item[0] in args.loads),
        key=lambda item: ("low", "high", "overload").index(item[0]))
    configs = _prepare_configs(run_root, args.capacities)
    jobs = []
    for load, rate in rates:
        for seed in args.seeds:
            workload = _ensure_workload(
                run_root,
                f"confirm_{load}_rate_{_rate_label(rate)}_"
                f"seed{seed}_sessions{args.sessions}",
                args.sessions, rate, seed, args.gap_profile, args.resume)
            policies = [
                ("recompute", configs["recompute"], None),
                ("active-offload", configs["active16"], 16),
                ("capacity-oracle", configs["capacity_oracle"], 256),
            ]
            policies.extend(
                ("session-offload", configs[f"session{capacity}"], capacity)
                for capacity in args.capacities)
            if load == "low":
                policies.append(("npu-retention", configs["npu_retention"], None))
            for policy, config, capacity in policies:
                capacity_label = f"_{capacity}gb" if capacity is not None else ""
                label = (
                    f"confirm_{load}_rate_{_rate_label(rate)}_seed{seed}_"
                    f"sessions{args.sessions}_"
                    f"{policy.replace('-', '_')}"
                    f"{capacity_label}_direct")
                jobs.append((label, partial(
                    _run_case, args, run_root, label, "confirmation", policy,
                    config, workload, args.sessions, rate, seed, capacity)))
    _run_jobs(args, run_root, jobs)
    _write_summary(run_root)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", required=True,
        help="result directory under the repository, reused across stages")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--case-timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="number of independent simulator cases to run concurrently")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gap-profile", choices=("tool-heavy", "mixed", "human-heavy"), default="mixed")
    parser.add_argument("--log-level", choices=("WARNING", "INFO", "DEBUG"), default="WARNING")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="write manifests and generated configs")

    pilot = subparsers.add_parser("pilot", help="run timing, calibration, screen, and oracle validation")
    pilot.add_argument(
        "--stages", type=_parse_stages, default=PILOT_STAGES,
        help="comma-separated subset: timing,calibration,screen,validation")
    pilot.add_argument("--seed", type=int, default=7)
    pilot.add_argument("--timing-sessions", type=int, default=10)
    pilot.add_argument("--timing-rate", type=float, default=1.0)
    pilot.add_argument("--calibration-sessions", type=int, default=20)
    pilot.add_argument(
        "--calibration-rates", type=lambda value: _parse_numbers(value, float),
        default=DEFAULT_CALIBRATION_RATES)
    pilot.add_argument("--screen-sessions", type=int, default=50)
    pilot.add_argument(
        "--screen-capacities", type=lambda value: _parse_numbers(value, int),
        default=DEFAULT_SCREEN_CAPACITIES)

    confirm = subparsers.add_parser("confirm", help="run the full seeded capacity matrix")
    confirm.add_argument(
        "--loads", type=_parse_loads, default=("low", "high", "overload"),
        help="comma-separated subset: low,high,overload")
    confirm.add_argument("--sessions", type=int, default=1000)
    confirm.add_argument(
        "--seeds", type=lambda value: _parse_numbers(value, int),
        default=DEFAULT_CONFIRM_SEEDS)
    confirm.add_argument(
        "--capacities", type=lambda value: _parse_numbers(value, int),
        default=DEFAULT_CONFIRM_CAPACITIES)

    subparsers.add_parser("summarize", help="rebuild summary.csv from run records")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.case_timeout_seconds <= 0:
        parser.error("--case-timeout-seconds must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    run_root = (REPO_ROOT / args.run_root).resolve()
    try:
        run_root.relative_to(REPO_ROOT)
    except ValueError:
        parser.error("--run-root must be inside the repository")
    _prepare_run_root(run_root)

    if args.command == "prepare":
        _prepare_configs(run_root, DEFAULT_CONFIRM_CAPACITIES)
        print(f"Prepared: {run_root}")
    elif args.command == "pilot":
        _pilot(args, run_root)
    elif args.command == "confirm":
        _confirm(args, run_root)
    elif args.command == "summarize":
        _write_summary(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
