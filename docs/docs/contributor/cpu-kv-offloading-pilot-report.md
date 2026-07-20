---
title: CPU KV Offloading Timing Pilot Report
---

# CPU KV Offloading Timing Pilot Report

## Status

This is a timing and feasibility gate, not a publication result. It validates
the generated workload, demonstrates that CPU offloading changes progress
under NPU pressure, and estimates the cost of the proposed sweep. It does not
yet establish statistically significant throughput or latency improvements.

The pilot used commit state from the `agent/cpu-kv-offloading` branch on
July 20, 2026. The workload and completed CSV are under
`outputs/cpu_kv_experiment/pilot/seed7/`.

## Workload

The generator produced 100 sessions and 476 LLM calls at 2 sessions/s with
seed 7 and the mixed gap profile.

| Realized property | Value |
| --- | ---: |
| Mean turns/session | 4.76 |
| Turns/session p90 | 9 |
| Input tokens p50 / p90 / p99 | 2,412 / 6,178 / 9,440 |
| Output tokens p50 / p90 | 129 / 330 |
| Gap p50 / p90 / p99 | 0.38 / 17.34 / 208.37 s |

The same JSONL was used for every pilot. The setup was Llama-3.1-8B BF16 on
one simulated 24 GiB RTXPRO6000. The offload case used a 16 GiB CPU KV pool,
64 GB/s host link, 800 ns link latency, and 0.90/0.80 high/low watermarks.

## Runs and observations

| Run | Scope | Wall time | Outcome |
| --- | --- | ---: | --- |
| Recompute pressure pilot | 100 sessions | 214.4 s | Stopped after a persistent progress stall |
| Session offload pressure pilot | 100 sessions, 16 GiB CPU | 1,075.4 s | Stopped after reaching 150 simulated seconds |
| Recompute timing pilot | 10 sessions, 46 calls | 626.6 s | Simulation completed; legacy report code failed after summary |
| Recompute smoke run | 1 session, 3 calls | 49.8 s | Completed and wrote the per-request CSV |

### Recompute liveness at the offered load

The 100-session Recompute run reached 99.99% NPU memory at approximately 36
simulated seconds. It then reported zero running requests while the waiting
queue grew from 44 to 79 requests through simulated second 116. No tokens were
processed during that interval, so the run was stopped.

This means 2 sessions/s cannot be treated as a valid stable Recompute load.
It may also expose a scheduler liveness defect when all runnable KV has been
preempted but memory remains committed. The load calibration gate must find a
stable `lambda_sat`, and this state must be diagnosed before using overload
completion rate as a scientific result.

### Offload progress under the same pressure

The 16 GiB Session-offload run passed the Recompute stall point and continued
to 150 simulated seconds. Selected occupancy samples were:

| Simulated time | NPU used | CPU KV used | Waiting requests |
| ---: | ---: | ---: | ---: |
| 20 s | 89.4% | 3.63 GiB | 0 |
| 40 s | 88.5% | 12.76 GiB | 22 |
| 60 s | 82.0% | 15.34 GiB | 29 |
| 70 s | 83.4% | 11.34 GiB | 19 |
| 100 s | 81.7% | 6.27 GiB | 8 |
| 130 s | 74.1% | 3.78 GiB | 0 |

The CPU pool therefore filled and later drained as state was restored or
dropped, rather than growing monotonically. This is evidence that offloading
preserves forward progress under this pressure. Because the run was stopped
before all deferred turns completed, it is not valid to report an end-to-end
speedup from this run.

### Completed-run timing

The 10-session run completed 46 calls in 626.6 wall-clock seconds, or 13.6
seconds per call. The one-session smoke run completed three calls in 49.8
seconds, or 16.6 seconds per call. These similar values show that graph
generation and Chakra conversion per decode iteration dominate small-run wall
time. High-load batching should lower the per-call cost, but migration traces
add work in the offload cases.

The one-session CSV reports 5.462 simulated seconds for three calls, with mean
TTFT 68.21 ms and mean TPOT 11.42 ms. These values only validate output
generation; the sample is too small for a performance conclusion.

## Tooling findings

Two experiment-harness issues were found:

1. The simulator image contained an older installed Chakra converter than the
   checked-out submodule. Migration-only traces failed until the repository
   converter was copied into the ephemeral container. The image should be
   rebuilt before the main batch.
2. `--log-interval` values above one second used integer floor division for
   throughput scaling. That produced zero interval throughput and a division-
   by-zero failure during final reporting. The working tree now uses a floating
   scale, and a one-session end-to-end run with a 10-second interval completed
   successfully.

The generator, session retention, and offloading unit suites contain 72 tests
and all pass after the reporting fix.

## Runtime estimate

The original main matrix contains about 135 runs of 1,000 sessions each. A
linear estimate from the completed 10-session timing run is about 17.4 hours
per run and 98 days serially. High-load batching provides a more optimistic
lower bound of roughly 3 hours per run, based on the partial pressure pilot.
The practical planning range is therefore:

| Scope | Estimated serial time |
| --- | ---: |
| One 1,000-session run | 3-18 hours |
| Original 135-run matrix | 17-101 days |
| Original matrix with 8 ideal workers | 2-13 days, plus contention |
| Proposed 50-session screening batch | 2-12 hours |

The full matrix should not be started now. The recommended next action is the
load-calibration gate followed by the 50-session screening batch. Only the
capacity knee and adjacent points should then receive additional seeds and
longer workloads.

## Decision gate

Proceed only after reviewing these points:

- accept the staged 20/50-session design instead of immediately launching the
  1,000-session matrix;
- diagnose or explicitly classify the Recompute no-progress state;
- rebuild the simulator image with the current Chakra submodule;
- add per-turn session identifiers and hit-tier fields before the final report.
