"""Generate synthetic append-only agentic sessions for KV-retention studies."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


_GAP_PROFILES = {
    "tool-heavy": (
        (0.85, 0.2, 0.8, 0.01, 2.0),
        (0.14, 5.0, 1.0, 0.5, 60.0),
        (0.01, 120.0, 0.8, 10.0, 600.0),
    ),
    "mixed": (
        (0.65, 0.2, 0.8, 0.01, 2.0),
        (0.25, 5.0, 1.0, 0.5, 60.0),
        (0.10, 120.0, 0.8, 10.0, 600.0),
    ),
    "human-heavy": (
        (0.35, 0.2, 0.8, 0.01, 2.0),
        (0.30, 5.0, 1.0, 0.5, 60.0),
        (0.35, 120.0, 0.8, 10.0, 600.0),
    ),
}

_CONTEXT_PROFILES = {
    "standard": {
        "initial": (1024, 0.8, 256, 8192),
        "new": (256, 0.9, 16, 4096),
    },
    "long": {
        "initial": (8192, 0.55, 4096, 16384),
        "new": (2048, 0.7, 256, 8192),
    },
}


def register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-sessions", type=int, required=True)
    parser.add_argument("--session-rate", type=float, required=True,
                        help="Poisson arrival rate in sessions per second.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gap-profile", choices=sorted(_GAP_PROFILES),
                        default="mixed")
    parser.add_argument("--context-profile", choices=sorted(_CONTEXT_PROFILES),
                        default="standard")
    parser.add_argument("--first-arrival-sec", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--turn-stop-prob", type=float, default=0.25)
    parser.add_argument("--max-context-toks", type=int, default=32768)


def _bounded_lognormal(rng, median, sigma, lower, upper):
    value = rng.lognormvariate(math.log(median), sigma)
    return int(min(upper, max(lower, round(value))))


def _sample_turn_count(rng, max_turns, stop_probability):
    turns = 2
    while turns < max_turns and rng.random() > stop_probability:
        turns += 1
    return turns


def _sample_gap_ns(rng, profile_name):
    pick = rng.random()
    cumulative = 0.0
    for probability, median, sigma, lower, upper in _GAP_PROFILES[profile_name]:
        cumulative += probability
        if pick <= cumulative:
            seconds = rng.lognormvariate(math.log(median), sigma)
            seconds = min(upper, max(lower, seconds))
            return int(round(seconds * 1_000_000_000))
    raise AssertionError("Gap-profile probabilities must sum to one.")


def generate_sessions(args):
    if args.num_sessions <= 0:
        raise ValueError("num_sessions must be positive.")
    if args.session_rate <= 0:
        raise ValueError("session_rate must be positive.")
    if args.max_turns < 2:
        raise ValueError("max_turns must be at least two.")
    if not 0 < args.turn_stop_prob <= 1:
        raise ValueError("turn_stop_prob must be in (0, 1].")

    rng = random.Random(args.seed)
    context_profile = _CONTEXT_PROFILES[args.context_profile]
    initial_spec = context_profile["initial"]
    new_context_spec = context_profile["new"]
    arrival_ns = int(round(args.first_arrival_sec * 1_000_000_000))
    sessions = []
    for session_index in range(args.num_sessions):
        if session_index:
            arrival_ns += int(round(
                rng.expovariate(args.session_rate) * 1_000_000_000))
        target_turns = _sample_turn_count(
            rng, args.max_turns, args.turn_stop_prob)
        input_toks = _bounded_lognormal(rng, *initial_spec)
        sub_requests = []
        for turn_index in range(target_turns):
            output_toks = _bounded_lognormal(rng, 128, 0.7, 16, 1024)
            sub_request = {
                "input_toks": input_toks,
                "output_toks": output_toks,
                "tool_duration_ns": 0,
            }
            sub_requests.append(sub_request)
            if turn_index + 1 >= target_turns:
                break
            new_context = _bounded_lognormal(rng, *new_context_spec)
            next_input = input_toks + output_toks + new_context
            if next_input > args.max_context_toks:
                break
            sub_request["tool_duration_ns"] = _sample_gap_ns(
                rng, args.gap_profile)
            input_toks = next_input

        sessions.append({
            "session_id": f"synthetic-{args.seed}-{session_index}",
            "arrival_time_ns": arrival_ns,
            "reuse_previous_kv": True,
            "sub_requests": sub_requests,
        })
    return sessions


def _percentile(values, percentile):
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(args, sessions):
    turns = [len(session["sub_requests"]) for session in sessions]
    inputs = [
        request["input_toks"]
        for session in sessions for request in session["sub_requests"]
    ]
    outputs = [
        request["output_toks"]
        for session in sessions for request in session["sub_requests"]
    ]
    gaps = [
        request["tool_duration_ns"] / 1_000_000_000
        for session in sessions for request in session["sub_requests"]
        if request["tool_duration_ns"] > 0
    ]
    return {
        "generator": "session-kv",
        "parameters": vars(args),
        "realized": {
            "sessions": len(sessions),
            "requests": sum(turns),
            "turns_per_session_mean": sum(turns) / len(turns),
            "turns_per_session_p90": _percentile(turns, 90),
            "input_toks_p50": _percentile(inputs, 50),
            "input_toks_p90": _percentile(inputs, 90),
            "input_toks_p99": _percentile(inputs, 99),
            "output_toks_p50": _percentile(outputs, 50),
            "output_toks_p90": _percentile(outputs, 90),
            "gap_seconds_p50": _percentile(gaps, 50),
            "gap_seconds_p90": _percentile(gaps, 90),
            "gap_seconds_p99": _percentile(gaps, 99),
        },
    }


def run(args: argparse.Namespace) -> int:
    sessions = generate_sessions(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for session in sessions:
            output_file.write(json.dumps(session, separators=(",", ":")))
            output_file.write("\n")

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(_summary(args, sessions), summary_file, indent=2)
        summary_file.write("\n")
    print(f"Wrote {len(sessions)} sessions -> {output_path}")
    print(f"Wrote generator summary -> {summary_path}")
    return 0
