---
title: CPU KV Offloading Capacity Experiment
---

# CPU KV Offloading Capacity Experiment

## Objective

This experiment measures when additional CPU DRAM improves agentic-session
serving and when the benefit saturates. The causal chain under test is:

```text
CPU KV capacity
  -> retained inactive-session working set
  -> NPU/CPU hits, capacity drops, and recomputed prompt tokens
  -> migration overhead
  -> throughput and tail latency
```

The first experiment uses one colocated instance. Same-node prefill/decode
disaggregation is evaluated only after the colocated capacity curve is
understood, because PD forces a D2H/H2D bridge between every pair of turns and
would otherwise confound the basic capacity result.

## Hypotheses

1. More CPU DRAM retains a larger inactive-session KV working set and reduces
   capacity drops and prompt recomputation.
2. Throughput and turn latency improve until the useful working set fits; more
   capacity beyond that knee has little value.
3. A very small CPU pool can be slower than recomputation because it pays
   migration cost but still drops state before reuse.
4. Longer think/tool times increase the number of simultaneously parked
   sessions and move the capacity knee to the right.
5. Host-link bandwidth determines whether additional cache hits translate into
   end-to-end performance rather than reload stalls.

## Controlled hardware setup

The primary setup is a capacity-constrained instance using the existing
RTXPRO6000 profile:

| Parameter | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.1-8B` |
| Weight and KV dtype | BF16 |
| NPU count / TP / PP | 1 / 1 / 1 |
| NPU memory | 24 GiB |
| Block size | 16 tokens |
| CPU memory bandwidth | 256 GB/s |
| Host-link bandwidth / latency | 64 GB/s / 800 ns |
| Host transfer model | `pipelined` |
| Session TTL | disabled for the primary capacity sweep |

The simulator reports approximately 14.96 GiB of model weights and 128 KiB
of full-cluster KV per token for this model. The 24 GiB NPU therefore has about
9 GiB, or roughly 74K tokens, available for live and parked KV. One thousand
tokens occupy about 125 MiB.

## Probabilistic workload

Every JSONL record is one append-only session with
`reuse_previous_kv: true`. A later turn's input is derived from the previous
turn rather than sampled independently:

```text
input[t + 1] = input[t] + output[t] + new_context[t]
```

This makes the declared reuse opportunity internally consistent without token
IDs. The primary workload uses the following bounded distributions:

| Variable | Distribution |
| --- | --- |
| Session arrivals | Poisson process; exponential inter-arrival time |
| Turns per session | `min(12, 2 + Geometric(p=0.25))` |
| Initial input | log-normal, median 1,024, sigma 0.8, range 256-8,192 tokens |
| Output per turn | log-normal, median 128, sigma 0.7, range 16-1,024 tokens |
| New context per turn | log-normal, median 256, sigma 0.9, range 16-4,096 tokens |
| Maximum context | 32,768 tokens; stop the session before exceeding it |

The mixed gap profile samples each non-terminal pause from:

| Class | Probability | Log-normal median | Sigma | Bounds |
| --- | ---: | ---: | ---: | ---: |
| Fast tool | 65% | 0.2 s | 0.8 | 0.01-2 s |
| Slow tool | 25% | 5 s | 1.0 | 0.5-60 s |
| Human thinking | 10% | 120 s | 0.8 | 10-600 s |

The last turn always has `tool_duration_ns: 0`. The generator writes a summary
manifest containing its arguments and realized p50/p90/p99 values. The same
generated workload and seed must be reused across every policy and capacity in
a comparison.

Context truncation and summarization are excluded from the primary experiment.
A later sensitivity sweep may set `reused_prefix_toks` on 10% or 30% of turns.

## Compared policies

| Label | Session retention | CPU offloading | Purpose |
| --- | ---: | ---: | --- |
| Recompute | off | off | Full-prefill reference for every turn |
| Active offload | off | on | Isolate request-level CPU swapping |
| Session offload | on | on | Evaluate the complete implementation |
| NPU-only retention | on | off | Transfer-free cache reference at safe low load |
| Capacity oracle | on | on | Large CPU pool and near-zero host-link cost upper bound |

NPU-only retention is not used after its working set exceeds physical NPU
capacity. A zero-capacity Session-offload run is also not used as the
Recompute baseline because these policies have different progress semantics.

## Primary sweeps

### Offered load

First estimate the Recompute saturation arrival rate, `lambda_sat`, with a
short pilot. Run the main comparison at:

- `0.6 * lambda_sat` (low load);
- `0.9 * lambda_sat` (high but stable load); and
- `1.1 * lambda_sat` (overload).

### CPU capacity

Sweep 4, 8, 16, 32, 64, 128, and 256 GiB. For Llama-3.1-8B BF16 these
capacities correspond to approximately 32K, 65K, 131K, 262K, 524K, 1.05M,
and 2.10M KV tokens. Always publish both units.

### Replication

- Pilot: 100-200 sessions and one seed.
- Main experiment: 1,000 sessions for each of five common-random-number seeds.
- Reuse an identical JSONL file for all configurations under one seed.
- Report paired differences and 95% bootstrap confidence intervals.

The main matrix has 105 Session-offload runs
(`3 loads * 7 capacities * 5 seeds`) plus roughly 30 Recompute and Active-
offload baselines. The full matrix is not launched until two pilot runs have
provided a wall-clock estimate.

### Pilot-gated execution

The publication matrix above is a target, not the first batch to launch. The
analytical backend uses Chakra template IPC and direct in-memory execution by
default. Retain the staged design so load stability and the capacity knee are
established before spending time on the full matrix:

1. **Load calibration:** use 20 sessions at session rates 0.5, 1.0, 1.5, and
   2.0 sessions/s for Recompute and 16 GiB Session offload. Stop any run that
   makes no request progress for 30 simulated seconds.
2. **Capacity screening:** use 50 sessions, one seed, the low and high stable
   loads, and capacities 4, 16, 64, and 256 GiB. Include paired Recompute and
   Active-offload references. This is the next batch after the pilot review.
3. **Confirmation:** run more sessions, seeds, and intermediate capacities
   only around the observed capacity knee. The 1,000-session, five-seed matrix
   is launched only if confidence intervals from the smaller runs require it.

Before Gate 1, rerun 10-session Recompute and Session-offload timing cases with
`--workload-transport ipc --ipc-execution direct` and record
`--host-timing-output`. Use those measurements to publish a new wall-clock
estimate. For one representative pressure workload, also run
`--ipc-execution oracle` and require identical per-request results, simulated
completion time, and KV-offload counters. Here, transport **oracle** is a
correctness path that materializes every graph; it is unrelated to the
Capacity oracle policy in the comparison table.

The separate [Chakra template IPC validation](./chakra-template-ipc-plan.md)
measured 4.92x and 7.12x direct-over-oracle median speedups on 50- and
100-session PD cases, respectively, with exact simulated outputs. Run the
CPU-KV-specific timing pilot before estimating the wall time of the capacity
matrix.

## Metrics

### User-visible performance

- completed requests/s and sessions/s;
- turn latency and TTFT p50/p95/p99;
- end-to-end session makespan;
- completion rate or SLO attainment under overload.

### Cache effectiveness

- NPU and CPU session hit count/tokens;
- session misses and recomputed prompt tokens;
- TTL expiration and capacity-drop count/bytes;
- average parked occupancy from byte-ns divided by simulated duration.

### Cost

- D2H/H2D bytes and migration time;
- reload stall and session reload wait;
- NPU/CPU peak used and reserved capacity;
- avoided prefill time minus migration and queueing overhead.

Define the capacity knee as the smallest CPU size that reaches at least 95% of
the maximum-capacity throughput without degrading p99 latency by more than 5%.

## Required plots

1. Throughput speedup versus CPU capacity, split by offered load.
2. Turn-latency p95/p99 versus CPU capacity.
3. NPU hit, CPU hit, and miss proportions as a stacked plot.
4. Recomputed prompt tokens and migrated bytes versus CPU capacity.
5. Capacity knee versus gap profile or offered load.
6. Marginal throughput gain per additional GiB of CPU DRAM.

## Sensitivity experiments

After selecting representative capacities of 8, 32, and 128 GiB, sweep:

- TTL: 0.5, 2, 10, and 60 seconds plus disabled;
- host link: 32, 64, and 128 GB/s;
- NPU memory: 24, 48, and 96 GiB;
- tool-heavy, mixed, and human-heavy gap profiles;
- summarization probability: 0%, 10%, and 30%;
- BF16 versus FP8 KV.

Finally repeat only the baseline, knee, and maximum-capacity points in the
same-node PD configuration.

## Reproducibility and reporting

Every result directory must retain:

- workload JSONL and generator summary manifest;
- cluster config, CLI arguments, seed, and git/submodule commits;
- per-request CSV and KV-offload sidecar;
- host-timing JSON, including transport/template counters and completion
  sequence digests;
- process wall time and simulator completion status.

Run `scripts/compile.sh` after checkout and whenever the ASTRA-Sim or Chakra
submodule revision changes. The script installs the checked-out Chakra fork
and builds the analytical backend. Do not repair a stale container with an
unrecorded manual converter copy.

The analytical experiment must explicitly record
`--network-backend analytical --workload-transport ipc --ipc-execution direct`
even though these are the current defaults. The file transport and transport-
oracle modes are compatibility and correctness references, not publication
performance configurations.

The per-request output includes `session_id`, `sub_request_index`,
`session_kv_hit_tier`, and `session_kv_hit_tokens`, allowing the experiment
runner to reconstruct turn-level metrics and end-to-end session makespan. The
aggregate sidecar remains the source for instance-level migration, occupancy,
drop, and recomputation counters.
