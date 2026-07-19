---
sidebar_position: 7
title: CPU KV Cache Offloading Plan
---

# CPU KV Cache Offloading Plan

This document specifies the planned CPU-only KV-cache offloading model for
LLMServingSim. It is an implementation plan, not a description of behaviour
already available in released simulator runs.

The goal is to evaluate how host-DRAM KV offloading changes admission,
preemption, throughput, and tail latency. The model includes the end-to-end
NPU-to-host transfer delay, including the host interconnect and CPU DRAM
service time. CXL, remote-KV attention, and multi-tier placement are explicitly
out of scope for this first version.

## Scope and non-goals

The initial implementation models one NPU-memory tier and one CPU-memory
tier per node. CPU DRAM is shared by every instance on that node; it is not a
separate prefill- or decode-instance resource. A request's live KV state is in
exactly one tier at a time, except while a transfer is in flight. The
implementation must account for capacity and transfer time in both tiers.

The first version does not model:

- CXL as a KV destination;
- CPU-resident blocks directly read by attention;
- partially resident live requests;
- background speculative replication of every new KV block; or
- overlap between a KV transfer and compute.
- prefill-decode disaggregation across different nodes.

Those are useful follow-on experiments, but adding them before a correct swap
baseline makes it difficult to attribute performance changes.

## Terminology

**KV block** is a contiguous, page-aligned range of tokens for one request.
Its byte size includes the K and V tensors for every transformer layer and is
the same allocation unit used by the scheduler. It is not a single layer's
KV tensor.

**Live KV** is the mutable state required to continue generating a request.
It has one owner and is managed by the request scheduler.

**Prefix-cache KV** is an immutable, possibly shared, cache entry. Prefix
caching remains a separate concern from live-request preemption; it may retain
blocks after a request ends, but it is not the backing store for a swapped live
request in the first milestone.

**Migration** means copying a live block between NPU and CPU then changing its
owner. The default policy is *exclusive*: after a successful migration, the
source copy is released.

## Required invariants

The following invariants should be asserted in debug builds and checked by
tests.

1. A live KV block is resident in one of `NPU`, `CPU`,
   `NPU_TO_CPU`, or `CPU_TO_NPU`.
2. A block in either in-flight state consumes space in both source and
   destination tiers until the transfer completes.
3. NPU memory is not released until an NPU-to-CPU copy completes.
4. CPU memory is not released until a CPU-to-NPU copy completes.
5. A scheduled attention step may execute only when every live block needed by
   that request is NPU-resident.
6. Prefix-cache references never serve as mutable live-KV ownership.
7. `used_npu_bytes + reserved_npu_bytes` never exceeds NPU capacity, and the
   equivalent condition holds for CPU memory.

## Data model

Add explicit KV residency metadata instead of using only `Request.evict`.
The exact Python types may vary, but the state should have the following
shape.

```python
class KVResidency(Enum):
    NPU = auto()
    CPU = auto()
    NPU_TO_CPU = auto()
    CPU_TO_NPU = auto()

@dataclass
class KVBlock:
    request_id: int
    block_index: int
    token_start: int
    token_count: int
    bytes_per_rank: int
    residency: KVResidency
    owner_instance_id: int
    pinned: bool = False
    last_scheduled_ns: int = 0
    transfer_complete_ns: int | None = None
```

`Request` should own an ordered list of `KVBlock` objects. The existing
aggregate size helpers can remain for fast admission estimates, but they must
derive their result from the block list or be checked against it.

`MemoryModel` should expose tier-neutral APIs such as:

```python
reserve(size, tier)
commit_reservation(size, tier)
release(size, tier)
available(size, tier)
```

CPU sizes must be tracked as full-cluster bytes if that is the chosen ASTRA-Sim
memory-node convention; NPU sizes remain per rank. The conversion must happen
at one boundary only and be documented in the API contract.

## Baseline lifecycle

### Block creation

New prefill and decode tokens allocate newly completed KV blocks on the NPU.
The scheduler reserves their NPU capacity before launching the batch. A block
is not copied to CPU merely because it was created.

This is a write-back design: CPU traffic is paid only when a request is
preempted. A future write-through experiment can copy every new block to CPU,
but must be a separate policy because it measures replication bandwidth rather
than only offload pressure.

### NPU-to-CPU eviction

Eviction occurs before a batch is admitted when its NPU reservation cannot fit.
Use a high/low watermark policy:

1. Trigger eviction when `used + reserved` exceeds the NPU high watermark,
   initially 90% of capacity.
2. Evict enough data to make the batch fit and, where possible, reduce usage
   to a low watermark, initially 80%.
3. Reserve CPU destination capacity before the transfer is issued.
4. Mark every selected block `NPU_TO_CPU` and emit a D2H transfer event.
5. On completion, mark it `CPU`, release the NPU allocation, and make the
   request ineligible for scheduling until it has been reloaded.

The source NPU copy must not be removed at transfer submission time. It is
still required if a transfer fails or if the simulator models a non-zero copy
duration.

### CPU-to-NPU reload

Before admitting a swapped request, the scheduler reserves NPU space for all
of its live blocks. It then:

1. marks the blocks `CPU_TO_NPU`;
2. emits an H2D transfer event;
3. waits for that event before the request's first attention computation;
4. marks blocks `NPU` on completion; and
5. releases the CPU allocation.

This is also exclusive migration. Keeping a CPU replica after reload is a
different, explicitly configured policy; it needs ongoing coherence for newly
generated decode blocks and must count the duplicate capacity.

## Eviction unit and victim selection

The physical accounting unit is a KV block, but the initial scheduling policy
evicts **all live blocks of one request atomically**. This avoids an incomplete
request whose next attention step would need to fetch its missing blocks
individually.

Eligible victims are requests that are not in the batch being executed, not
currently transferring, and not otherwise pinned. The initial victim policy is
least-recently-scheduled request first. It should be isolated behind a policy
interface so experiments can add:

- largest-KV-first, to reclaim capacity quickly;
- longest-predicted-time-to-next-use first;
- priority-aware LRU; and
- cost-aware scoring, such as reclaimed bytes divided by estimated reload
  latency.

If evicting one request is more than needed, the excess capacity is intentional
in the baseline. Avoid partial-request eviction until remote or selective-block
attention is modeled.

## Scheduler integration

Admission should occur in this order:

1. Form a candidate batch using normal token and sequence limits.
2. Calculate new-KV allocation and reload reservations separately.
3. Reload already-selected CPU-resident requests if they fit after reservation.
4. If they do not fit, choose and submit victim migrations.
5. Re-evaluate admission after completed migrations.
6. Emit compute only after all selected requests are NPU-resident.

The scheduler must not treat a request as runnable merely because it has an
`evict` flag. It must consult block residency and transfer completion. A
request selected as a victim remains queued, but it is excluded from decode
until its reload is admitted.

Use explicit counters for `new_kv_bytes`, `evict_bytes`, and `reload_bytes` in
each batch. These counters must not double-count prefix-cache insertions.

## Transfer-delay model

CPU KV migration traverses a host interconnect such as PCIe or NVLink-C2C and
then accesses CPU DRAM. The cluster's top-level `link_bw` and `link_latency`
remain reserved for NPU collectives; CPU offloading has separate fields under
the node's `cpu_mem` configuration:

```json
{
  "cpu_mem": {
    "mem_size": 512,
    "mem_bw": 256,
    "mem_latency": 80,
    "host_link_bw": 64,
    "host_link_latency": 800,
    "host_transfer_model": "pipelined"
  }
}
```

Sizes are in GB, bandwidths are in GB/s, and latencies are in nanoseconds. The
Phase-1 symmetric H2D/D2H model uses the pipelined end-to-end runtime:

```text
effective_bw = min(cpu_mem_bw, host_link_bw)
transfer_ns = cpu_mem_latency + host_link_latency
            + ceil(bytes / effective_bw)
```

`host_transfer_model=serial` may be added as a diagnostic alternative, using
`bytes / cpu_mem_bw + bytes / host_link_bw`, but `pipelined` is the default.
Directional H2D/D2H bandwidths are deferred until measurements justify the
extra configuration surface.

The analytical remote-memory backend owns this timing calculation and the
per-node request queue. Consequently, migrations from multiple instances on
one node contend for the same modeled host-link/DRAM service resource. The
implementation must not add the same delay again in Python.

## Trace and ASTRA-Sim integration

Represent migration as dedicated trace operations rather than generic layer
weight accesses:

```text
KV_EVICT_CPU  NPU -> REMOTE:<node>  bytes=<per-rank bytes>
KV_RELOAD_CPU REMOTE:<node> -> NPU  bytes=<per-rank bytes>
```

The Chakra conversion must create a memory-store node for eviction and a
memory-load node for reload, with the CPU memory device as the target/source.
For the baseline, emit a migration-only workload and wait for that workload to
finish before issuing compute. This is necessary because the current Python
controller observes workload completion, not individual Chakra-node
completion. It also gives the scheduler an exact point at which reservations
can be committed and source capacity can be released.

The batch model therefore distinguishes `COMPUTE`, `KV_EVICT`, `KV_RELOAD`,
and `PD_HANDOFF`. A migration batch owns the affected request list and byte
counters, but it advances no model tokens. Overlap between migration and
compute is a later feature and requires an explicit ASTRA-Sim-to-Python node
completion event rather than speculative state changes in the scheduler.

Do not reuse the `kv_loc` configuration field for this path. In this milestone,
active KV is always NPU-resident. A future remote-KV design needs separate
per-layer attention-memory accesses and is not equivalent to preemption
migration.

## Prefill-decode disaggregation

Prefill-to-decode handoff and CPU offloading are distinct operations.

The first implementation supports only a prefill instance and its paired decode
instance on the **same node**. Both use the node's shared CPU DRAM allocator,
but PD handoff does not stage live KV through CPU by default.

When a prefill instance completes a request, it transfers the prompt KV
directly to the local decode instance using the existing PD handoff path. This
is not first written to CPU merely to use the offload mechanism. Once the
handoff completes:

1. the decode instance owns all prompt KV blocks and records them as
   NPU-resident;
2. the prefill instance releases its live request KV blocks;
3. any retained prefill prefix-cache entry follows prefix-cache policy, not
   live-request offload policy; and
4. every subsequent decode block is created under the decode instance's owner.

After handoff, the decode scheduler may preempt the request. At that point it
migrates the request's **entire current KV history**—both prompt blocks created
by prefill and output blocks created by decode—to the shared node CPU memory
together. It is therefore incorrect for the initial model to independently
offload “prefill KV” and “decode KV” based solely on where the tokens were
generated. Their owner is now the decode instance, and one attention step
requires both histories.

The prefill instance does not need a CPU-resident live copy after direct
handoff, so there is no later prefill-side reload to miss. If the decode
instance subsequently swaps the request to CPU, it stores and reloads the
complete KV history using the same node-level CPU allocator. Prefix-cache reuse
is separate: a future shared CPU prefix-cache pool can make completed prefill
prefixes visible to other local instances, but it is not needed for the live
KV handoff baseline.

The simulator should label PD handoff traffic separately from CPU offload
traffic, for example `kv_pd_send`/`kv_pd_recv` versus
`kv_evict_cpu`/`kv_load_cpu`. Report their bytes and delay independently.

## Prefix-cache interaction

For the first CPU-offload milestone, use one of these two controlled modes:

1. disable prefix caching for the offload baseline; or
2. leave prefix caching enabled, but never use it as the live-KV swap backing
   store.

In the second mode, a prefix hit reduces compute as usual. It does not mean a
swapped request is restored unless all of its required live blocks have been
explicitly reloaded. Completed-request prefix entries may be demoted to CPU in
a later prefix-cache-specific experiment, with separate accounting and cache
replacement policy.

## Configuration surface

Add the following instance-level runtime controls with CLI defaults. The names
are tentative, but their semantics should remain stable.

```text
--kv-offload-tier cpu
--enable-kv-offloading / --no-enable-kv-offloading
--kv-offload-high-watermark 0.90
--kv-offload-low-watermark 0.80
--kv-offload-victim-policy lru
--kv-offload-granularity request
--kv-offload-copy-policy write-back
```

Reject unsupported combinations explicitly, including CXL destinations,
block-selective live offload, and CPU-resident attention. Do not silently map
them to a CPU request-swap policy.

## Implementation phases

### Phase 1: Residency and reservation foundation

- Add four-state residency metadata and NPU/CPU reservations.
- Keep live-KV and prefix-cache CPU allocation APIs separate.
- Add migration records and migration-aware batch kinds.
- Assert capacity and state-transition invariants.

### Phase 2: Synchronous migration execution

- Emit migration-only eviction and reload workloads.
- Commit destination reservations and release the source only on completion.
- Replace boolean-only eviction scheduling with residency-aware admission.
- Ensure a failed admission plan changes no request or memory state.

### Phase 3: Host-link timing and contention

- Add host-link bandwidth and latency to the generated remote-memory config.
- Extend the analytical memory backend with the pipelined transfer formula.
- Validate per-node serialization/contention and exact byte counts.
- Keep collective-network and host-link configuration independent.

### Phase 4: Policy and observability

- Apply high/low watermarks as eviction targets, not hard capacity limits.
- Keep the LRU victim-policy interface and add largest-KV-first.
- Record preemption count, migrated bytes, reload stalls, migration time, and
  used/reserved tier occupancy.
- Preserve generated trace files for a small diagnostic run.

### Phase 5: PD ownership separation

- Make PD handoff establish decode-side ownership of prompt blocks.
- Release prefill live KV after handoff completion.
- Apply CPU offloading only through the decode scheduler after handoff.
- Report PD handoff and CPU migration metrics separately.

### Phase 6: Prefix-cache integration

- Re-enable prefix caching with independent live-KV and prefix-cache state.
- Verify that shared prefix references do not alter live-KV ownership.
- Add CPU prefix-cache demotion only as a separate, opt-in policy.

### Phase 7: Advanced experiments

- asynchronous transfer/compute overlap;
- predictive prefetch;
- write-through replicas;
- block-selective eviction; and
- CXL or remote-KV attention.

## Validation matrix

Every phase should be tested with a small deterministic workload and at least
one pressure workload whose total live KV exceeds NPU capacity.

| Scenario | Expected result |
| --- | --- |
| NPU-only capacity fits | No migrations; results match the existing baseline within trace changes. |
| CPU offload, one victim | One D2H eviction followed by one H2D reload; capacity never exceeds either tier. |
| Repeated pressure | Victim order follows configured policy; no negative counters or duplicate ownership. |
| CPU capacity exhausted | Admission stalls or fails with a clear error; no source NPU block is lost. |
| PD without pressure | Direct prompt-KV handoff; no CPU traffic solely due to handoff. |
| PD with decode pressure | Decode instance offloads the combined prompt-plus-decode history; prefill has no live copy after handoff. |
| PD across nodes | Rejected by the first implementation with a clear configuration error. |
| Prefix caching enabled | Prefix hits affect compute reuse only; live swap byte accounting remains correct. |

For each run, collect request throughput, TTFT/TPOT p50/p95/p99, preemption
count, eviction and reload bytes, migration time, NPU/CPU peak occupancy, and
reload-induced scheduling stall time. These metrics are required to distinguish
capacity gains from tail-latency regressions.

## Implementation progress

This table is updated in the same change that completes each implementation
step. A step is complete only after its focused validation passes.

| Step | Status | Validation gate |
| --- | --- | --- |
| 0. Consolidate the approved design in this document | Complete | Plan covers ownership, migration completion, host-link timing, PD, and observability |
| 1. Four-state residency and used/reserved capacity foundation | Complete | State-transition and capacity-invariant tests |
| 2. Migration-only eviction/reload workloads | Complete | One D2H followed by one H2D; no tokens advanced by migration batches |
| 3. Host-link timing in analytical remote memory | Complete | Size and bandwidth scaling; fixed-latency delta; per-node contention |
| 4. Watermark/LRU policy and atomic admission | Complete | Repeated pressure, no trace-free mutation, no high-watermark deadlock |
| 5. Same-node PD ownership and decode-pressure handling | Not started | Direct handoff without CPU traffic; decode eviction before constrained handoff |
| 6. Metrics, documentation, and end-to-end validation | Not started | Full validation matrix and output fields |

### Progress log

- 2026-07-19: Approved the synchronous migration-only baseline and the
  separate CPU host-link delay model.
- 2026-07-19: Completed Step 1. Added four residency states, migration and
  batch metadata, and used/reserved accounting for per-rank NPU capacity and
  full-cluster node CPU capacity. Focused foundation tests pass.
- 2026-07-19: Completed Step 2. Eviction and reload now use migration-only
  batches; source ownership changes only when ASTRA-Sim reports workload
  completion, and migration batches do not advance tokens. Added dedicated
  `KV_EVICT_CPU` / `KV_RELOAD_CPU` traces and conversion for memory-only
  Chakra graphs. Focused scheduler/trace tests and a two-NPU graph-conversion
  smoke test pass.
- 2026-07-19: Completed Step 3. Added explicit CPU-to-NPU host-link bandwidth,
  latency, and transfer-model fields without reusing the NPU collective link.
  Dedicated migration memory nodes now use the analytical backend's pipelined
  timing and existing per-node queue. ASTRA-Sim smoke tests report 944 cycles
  for 4 KiB and 1008 cycles for 8 KiB with an 80 ns DRAM latency, 800 ns link
  latency, and 64 GB/s bottleneck. Two simultaneous 4 KiB transfers on one
  node finish at 944 and 1888 cycles, confirming serialized contention.
- 2026-07-20: Completed Step 4. High-watermark pressure now evicts one or more
  request-level victims toward the low watermark using either LRU or
  largest-KV ordering, subject to CPU capacity. Admission plans remain
  non-mutating until reservations or allocations succeed, and the high
  watermark falls back to physical capacity when eviction cannot make
  progress. CPU-swapped requests wait until NPU-resident work drains, avoiding
  immediate reload/evict ping-pong. The focused suite passes 24 tests. A
  preserved two-request ASTRA-Sim run completes with nine batches in the order
  compute, evict, three computes, reload, and three computes; it reports one
  preemption and exactly 2 MiB evicted plus 2 MiB reloaded.
