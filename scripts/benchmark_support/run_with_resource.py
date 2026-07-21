#!/usr/bin/env python3
"""Run a command and save portable process resource accounting as JSON."""

import argparse
import json
import os
import resource
import subprocess
import time


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as source:
            return source.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def _process_snapshot(root_pid):
    pending = [root_pid]
    seen = set()
    snapshot = {}
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children = _read_text(
            f"/proc/{pid}/task/{pid}/children").split()
        pending.extend(int(child) for child in children if child.isdigit())
        status = _read_text(f"/proc/{pid}/status")
        if not status:
            continue
        fields = {}
        for line in status.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        def kib(name):
            value = fields.get(name, "0 kB").split()[0]
            return int(value) if value.isdigit() else 0
        cmdline = _read_text(f"/proc/{pid}/cmdline").replace("\0", " ").strip()
        snapshot[pid] = {
            "pid": pid,
            "name": fields.get("Name", ""),
            "cmdline": cmdline,
            "rss_kib": kib("VmRSS"),
            "hwm_kib": kib("VmHWM"),
        }
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    started = time.perf_counter()
    process = subprocess.Popen(command)
    peak_tree_rss_kib = 0
    process_peaks = {}
    rss_samples = 0
    while process.poll() is None:
        snapshot = _process_snapshot(process.pid)
        peak_tree_rss_kib = max(
            peak_tree_rss_kib,
            sum(item["rss_kib"] for item in snapshot.values()),
        )
        for pid, item in snapshot.items():
            identity = (pid, item["cmdline"])
            previous = process_peaks.get(identity)
            if previous is None or item["hwm_kib"] > previous["hwm_kib"]:
                process_peaks[identity] = item
        rss_samples += 1
        time.sleep(0.01)
    returncode = process.wait()
    snapshot = _process_snapshot(process.pid)
    peak_tree_rss_kib = max(
        peak_tree_rss_kib,
        sum(item["rss_kib"] for item in snapshot.values()),
    )
    elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "wall_seconds": elapsed,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "peak_tree_rss_kib": peak_tree_rss_kib,
        "process_peak_rss": sorted(
            process_peaks.values(), key=lambda item: item["hwm_kib"],
            reverse=True),
        "rss_sample_interval_seconds": 0.01,
        "rss_samples": rss_samples,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
        "returncode": returncode,
        "command": command,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
