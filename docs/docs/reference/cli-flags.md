---
sidebar_position: 1
title: CLI flags
---

# `python -m serving` CLI flags

Complete reference for every command-line flag accepted by
`python -m serving`. For the conceptual side of each flag (what it
*does* internally), see **[Simulator](/docs/simulator/architecture)**.

## Cluster topology

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--cluster-config` | path | `configs/cluster/single_node_single_instance.json` | Path to a cluster-config JSON. See **[Cluster config](./cluster-config)** |
| `--network-backend` | choice | `analytical` | Network simulation backend. `analytical` (fast) or `ns3` (detailed, WIP) |

## Batching and scheduling

These flags are deployment defaults. A cluster config can override the
matching runtime knobs per `instances[i]`; see
**[Cluster config](./cluster-config#runtime-overrides-optional)**.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--max-num-seqs` | int | `128` | Max sequences in a batch. `0` = unlimited |
| `--max-num-batched-tokens` | int | `2048` | Max tokens per iteration across all requests (token budget) |
| `--long-prefill-token-threshold` | int | `0` | Per-request token cap per step for chunked prefill. `0` = disabled |
| `--enable-chunked-prefill` | bool | `True` | Split long prefill across iterations. Use `--no-enable-chunked-prefill` to disable |
| `--prioritize-prefill` | flag | off | Run prefill before decode in the same iteration |
| `--block-size` | int | `16` | KV cache block size in tokens |
| `--skip-prefill` | flag | off | Skip prefill, run decode only |

## Routing

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--request-routing-policy` | `LOAD` / `RR` / `RAND` / `CUSTOM` | `LOAD` | Cross-instance request routing |
| `--expert-routing-policy` | `BALANCED` / `RR` / `RAND` / `CUSTOM` | `BALANCED` | MoE expert token routing |
| `--enable-block-copy` | bool | `True` | Replay one block's trace across layers (set False for per-layer EP variance) |

## Precision

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--dtype` | `float16` / `bfloat16` / `float32` / `fp8` / `int8` | model's `torch_dtype`, fallback `bfloat16` | Model weight dtype |
| `--kv-cache-dtype` | `auto` / `fp8` | `auto` (inherits dtype) | KV cache dtype. `fp8` halves KV memory and selects a `*-kvfp8` profile variant |

## Prefix caching and offloading

| Flag | Default | Description |
| --- | --- | --- |
| `--enable-prefix-caching` | `True` | RadixAttention prefix caching. Use `--no-enable-prefix-caching` to disable |
| `--enable-session-kv-retention` | `False` | Reuse retained NPU or CPU KV across append-only turns of the same agentic session. Supports hard TTL, colocated instances, shared CPU KV offloading, and same-node PD continuation, but not generic prefix caching |
| `--session-kv-ttl-ns` | `0` | Default hard TTL in ns for inactive retained session KV. `0` disables time-based expiration |
| `--enable-kv-offloading` | `False` | Phase-1 request-level exclusive migration between NPU KV and node-shared CPU DRAM. Requires `--no-enable-prefix-caching`; same-node PD only |
| `--kv-offload-high-watermark` | `0.90` | NPU-KV pressure level that starts CPU eviction |
| `--kv-offload-low-watermark` | `0.80` | NPU-KV target after CPU eviction; must be no greater than the high watermark |
| `--kv-offload-victim-policy` | `lru` | Victim selector for CPU offload: `lru` or `largest-kv` |
| `--enable-prefix-sharing` | off | Second-tier prefix pool shared across instances within a node |
| `--prefix-storage` | `None` | Where the second-tier pool lives. `None` / `CPU` / `CXL` |
| `--enable-local-offloading` | off | Weight offloading to NPU (counts weight reads in profiling) |
| `--enable-attn-offloading` | off | Attention computation offloading to PIM |
| `--enable-sub-batch-interleaving` | off | Overlap GPU compute with PIM attention. Requires `--enable-attn-offloading` |

CPU KV offloading also requires `cpu_mem.host_link_bw` and
`cpu_mem.host_link_latency` in the cluster config. These values model the
CPU-to-NPU path independently from the top-level NPU collective link. See
**[Cluster config](./cluster-config#cpu_mem)**.

Session KV retention uses `session_id` rather than token hashes and never
shares KV across sessions. The workload must set `reuse_previous_kv: true` or
provide `reused_prefix_toks` on a continuation. In colocated mode, parked
state may remain on NPU or use the same node-shared CPU allocator as
active-request offloading. A CPU hit completes a modeled H2D reload before
suffix prefill begins. CPU pressure drops LRU parked session state before it
blocks correctness-owned active-request eviction. The later continuation is a
full-prefill miss. Generic prefix caching remains rejected.

Same-node PD mode requires both session retention and CPU KV offloading on
every prefill and decode instance on that node. Each non-terminal decode turn
is parked through a modeled D2H workload. Its continuation adopts the
node-shared CPU record on the prefill instance, completes a modeled H2D reload,
and then computes only the suffix. Cross-node PD session reuse is unsupported.

The instance default `--session-kv-ttl-ns` is measured from each non-terminal
turn's completion. A workload-level `session_kv_ttl_ns` overrides it for one
session. Expiry is a simulator event even while every instance is idle, and it
wins over an arrival at the same timestamp. State expiring during D2H or H2D
is hidden immediately and physically released when the synchronous transfer
finishes.

While a simulation is running, heartbeat lines report used and reserved NPU
memory per rank and used and reserved CPU memory per node. The final
per-instance CPU KV offloading summary reports preemptions, aggregate D2H/H2D
bytes, migration time, reload stall count and time, and peak used/reserved
occupancy. Migration byte and batch totals include both active requests and
parked session state, while `preemption count` counts active requests only.
A CPU-resident request remains in the swapped queue while any
NPU-resident work is runnable, preventing an immediate reload/evict cycle
under sustained watermark pressure.

For same-node prefill/decode disaggregation, every prefill and decode instance
on an offloading node must enable CPU KV offloading. Prompt KV remains charged
to the prefill instance until the decode scheduler reserves destination NPU
capacity. Under decode pressure, an ordinary CPU eviction completes before the
handoff commits. CPU-backed prefix caching on that node remains unsupported.

## Dataset and output

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | path | `None` | JSONL workload file. See **[Workloads → JSONL format](/docs/workloads/jsonl-format)** |
| `--num-reqs` | int | `0` | Entries to load from the dataset (`0` = all). For agentic, each entry is a session |
| `--output` | path | `None` | Per-request CSV output path. Stdout only if `None`. The literal `{run_id}` is replaced with the active run id. CPU KV offloading also writes a sibling `*_kv_offload.csv` with one metrics row per enabled instance |

## Run isolation

Each invocation writes ASTRA-Sim intermediates under a run-specific input
root so parallel simulations do not overwrite each other's generated
configs, traces, or Chakra workloads. Generated text traces are removed
after Chakra conversion by default, and the run-specific input root is
removed after a successful simulation by default.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--run-id` | string | auto-generated | Path-safe id for this simulation run. Used in `astra-sim/inputs/runs/<run-id>` and the `{run_id}` output placeholder |
| `--inputs-root` | path | `astra-sim/inputs/runs/<run-id>` | Override the generated ASTRA-Sim input root, for example to place intermediates on local SSD or tmpfs |
| `--cleanup-inputs` / `--no-cleanup-inputs` | bool | `true` | Remove generated trace files after Chakra conversion and remove the generated run directory after a successful simulation. Use `--no-cleanup-inputs` to preserve traces, Chakra workloads, and input configs for debugging |

## Graph conversion and host timing

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--chakra-converter` | `in-process` / `subprocess` | `in-process` | Select the Chakra text-to-protobuf converter path. `in-process` reuses the serving Python process and avoids one interpreter launch per graph. `subprocess` preserves the legacy reference path for comparisons |
| `--workload-transport` | `file` / `ipc` | `ipc` for analytical; `file` for ns-3 | Select the ASTRA-Sim control channel. Analytical execution registers static Chakra templates and sends compact per-batch patches over a Unix-domain socket by default. Use `file` to send `.et` paths over legacy stdin for compatibility or debugging. ns-3 remains file-only |
| `--ipc-execution` | `direct` / `oracle` | `direct` | Select how batches run when IPC transport is enabled. `direct` executes prepared in-memory iterations and skips matching per-batch trace and `.et` files. Colocated TP/PP, local-EP MoE, PIM attention, prefill/decode disaggregation, CPU KV migration, and atomic DP+EP waves use structure-keyed templates. `oracle` materializes and converts every batch, then extracts the complete dynamic graph patch before using the same prepared execution engine. It is the preferred correctness comparison for synchronized modes |
| `--template-cache-capacity` | Integer | `64` | Maximum number of shape-keyed compute-template bundles retained by the Python frontend. The cache is LRU and records class-specific hits, misses, registrations, and evictions in `--host-timing-output` |
| `--host-timing-output` | path | `None` | Write host-side stage distributions, workload metrics, sequence digests, lifecycle RSS checkpoints, and transport/template counters as JSON. The literal `{run_id}` is replaced with the active run id |

## Logging

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--log-interval` | float | `1.0` | Seconds between throughput / memory log lines |
| `--log-level` | choice | `WARNING` | `WARNING` (default) / `INFO` / `DEBUG` |

## Quick reference: which flag for which feature

| Feature | Flag(s) |
| --- | --- |
| Multi-instance (parallelism via cluster config) | (cluster config `num_instances`) |
| Tensor parallel | (cluster config `tp_size`) |
| MoE expert parallel | (cluster config `ep_size`) |
| DP+EP MoE | (cluster config `dp_group`) |
| Prefix caching | `--enable-prefix-caching` (default on), `--enable-prefix-sharing`, `--prefix-storage` |
| CPU KV offload | `--enable-kv-offloading --no-enable-prefix-caching` |
| Chunked prefill | `--enable-chunked-prefill` (default on), `--long-prefill-token-threshold` |
| PIM attention offload | `--enable-attn-offloading` (cluster config sets `pim_config`) |
| FP8 KV cache | `--kv-cache-dtype fp8` |
| ns3 backend | `--network-backend ns3` |
| Host-overhead comparison | `--host-timing-output timings-{run_id}.json`, optionally with `--chakra-converter subprocess` |
| Legacy file workload control | `--workload-transport file` (compatibility/debug path; required for ns-3) |

For the full conceptual treatment of each feature, browse the
**[Simulator](/docs/simulator/architecture)** section. For runnable
examples, see **[Examples](/docs/examples)**.
