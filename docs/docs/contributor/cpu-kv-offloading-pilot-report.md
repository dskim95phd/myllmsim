---
title: CPU KV Offloading Pilot Readiness Report
---

# CPU KV Offloading Pilot Readiness Report

## Status

The CPU KV offloading implementation and experiment harness are ready for a
new timing and load-calibration pilot. The analytical backend uses Chakra
template IPC, protocol-v2 system targeting, and direct in-memory execution by
default. A transport-oracle run is used only to validate correctness.

No publication-scale capacity matrix should start until the pilot establishes
the stable offered-load range and current wall-clock cost.

## Experiment baseline

The pilot uses the following controlled setup:

| Parameter | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.1-8B` |
| Weight and KV dtype | BF16 |
| NPU count / TP / PP | 1 / 1 / 1 |
| NPU memory | 24 GiB |
| CPU KV capacity | 16 GiB for the Session-offload case |
| CPU memory bandwidth | 256 GB/s |
| Host-link bandwidth / latency | 64 GB/s / 800 ns |
| Host transfer model | `pipelined` |
| KV block size | 16 tokens |
| Session TTL | disabled |

Use the session-KV workload generator with seed 7 and the mixed gap profile.
Reuse the exact same JSONL file for Recompute and Session offload at each
arrival rate.

## Current execution path

Run performance cases with these settings recorded explicitly:

```text
--network-backend analytical
--workload-transport ipc
--ipc-execution direct
--host-timing-output <run-directory>/host_timing.json
```

Direct execution registers structure-keyed Chakra templates and sends compact
per-batch patches to ASTRA-Sim. It avoids matching per-batch trace and `.et`
materialization. The host-timing JSON records template-cache activity, skipped
graph work, IPC traffic, completion-sequence digests, and host-stage timing.

Run one representative Session-offload pressure case again with
`--ipc-execution oracle`. The direct and oracle cases must have identical
per-request output, final simulated time, KV-offload counters, and completion
sequence. Host timing and transport counters are expected to differ.

## Validation status

The focused generator, session-retention, and KV-offloading suites contain 72
tests. Run them before the pilot:

```bash
python3 -m unittest \
  tests.test_session_kv_generator \
  tests.test_session_kv_retention \
  tests.test_kv_offloading_foundation -q
```

Run `scripts/compile.sh` after checkout and whenever the ASTRA-Sim or Chakra
submodule revision changes. This installs the checked-out Chakra fork and
builds the analytical backend. NS-3 is not required for this experiment.

## Pilot sequence

Run the complete sequence with:

```bash
python3 scripts/run_cpu_kv_experiment.py \
  --run-root outputs/cpu_kv_experiment/server/RUN_ID \
  pilot
```

The runner generates all workloads and derived capacity configs, validates
direct execution against transport oracle, and writes `pilot.json`,
`summary.csv`, and resumable per-case records.

On a multicore server, keep Gate 0 serial and split the remaining pilot into
`--stages calibration`, `--stages screen`, and `--stages validation` commands
using the same run root. Calibration and screen accept `--workers N` before
the `pilot` subcommand. Confirmation can be split with `--loads low`,
`--loads high`, and `--loads overload`. The complete copy-and-run commands and
worker sizing guidance are in the
[server runbook](/docs/contributor/cpu-kv-offloading-server-runbook).

### Gate 0: current wall-clock timing

Generate one 10-session workload and run paired Recompute and 16 GiB Session-
offload cases using direct IPC. Record:

- total wall time and completed LLM calls;
- host-stage timing and Chakra graphs skipped;
- IPC bytes and template-cache hits, misses, registrations, and evictions;
- final simulated time and completion-sequence digest.

Use these measurements only to size the next server batch. Do not extrapolate
the publication matrix until the load-calibration cases finish.

### Gate 1: offered-load calibration

Generate 20-session workloads at 0.5, 1.0, 1.5, and 2.0 sessions/s. Run paired
Recompute and 16 GiB Session-offload cases for every rate. The largest
Recompute rate that completes without persistent no-progress behavior is the
provisional saturation rate, `lambda_sat`.

Select low and high rates near `0.6 * lambda_sat` and `0.9 * lambda_sat`.
Generate new workloads at those exact rates instead of reusing a workload with
a different arrival process.

### Gate 2: capacity screen

At the selected low and high loads, run 50 sessions with CPU capacities 4, 16,
64, and 256 GiB. Include paired Recompute and Active-offload references. Add
8, 32, and 128 GiB or more seeds only around the observed capacity knee.

## Required pilot outputs

Every case must preserve:

- workload JSONL and generator summary;
- cluster configuration and exact CLI command;
- root and recursive submodule revisions;
- per-request CSV and KV-offload sidecar;
- host-timing JSON;
- process log, wall time, exit status, and simulator completion status.

The per-request CSV includes `session_id`, `sub_request_index`,
`session_kv_hit_tier`, and `session_kv_hit_tokens`. The runner uses these fields
to report request/session latency and session makespan, while the aggregate
sidecar supplies migration, hit, miss, recomputation, and capacity-drop
counters.

## Decision gate

Proceed to longer workloads only when:

- direct and transport-oracle outputs match on the representative pressure
  case;
- the low and high stable offered loads are identified;
- all selected configurations complete without unexplained no-progress
  intervals;
- host timing confirms that the direct path is skipping matching per-batch
  Chakra graph generation; and
- the 50-session screen identifies a capacity knee worth confirming with more
  seeds.
