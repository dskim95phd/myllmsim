---
sidebar_position: 10
title: Block-Granular KV Offloading Scheduler Plan
---

# Block-Granular KV Offloading Scheduler Plan

## Status and relationship to the baseline

This document specifies the next CPU KV-offloading and request-scheduling
policy for LLMServingSim. It is a proposed implementation plan, not a
description of the current scheduler.

The existing [CPU KV cache offloading plan](./cpu-kv-offloading-plan.md)
established the correctness-first baseline:

- one aggregate residency state per live request;
- request-at-a-time D2H and H2D migration;
- synchronous migration-only workloads;
- strict preference for NPU-resident work; and
- high/low watermarks that also influence eviction.

That baseline exposed two limitations in long-context agentic experiments:

1. CPU-resident unfinished decodes may starve while any NPU-resident request
   remains runnable.
2. Moving an entire request for a small allocation deficit creates more host
   traffic and migration stall than the current iteration requires.

This plan replaces those scheduling-policy sections with:

- explicit per-instance request queues and request execution states;
- request-level preemption with block-granular migration;
- a 10% admission reserve that is separate from the eviction trigger;
- enough D2H work to protect approximately ten future decode iterations;
- capacity reservation before every handoff or H2D reload;
- migration/compute overlap; and
- distinct behavior for colocated and prefill/decode-disaggregated serving.

The first implementation and all primary experiments disable generic prefix
caching. Session-scoped KV retention remains supported because it has explicit
ownership and does not use cross-session radix-cache sharing.

## Goals

The policy must:

1. prevent a new request or reload from consuming the last 10% of usable NPU
   KV capacity;
2. keep running decodes resident long enough to use that reserve before
   pressure forces another preemption;
3. stop one request at a time under pressure while allowing the remaining
   running requests to continue;
4. copy and release KV in block-sized units rather than moving a request's
   complete history in one atomic transfer;
5. reserve destination capacity before migration and never expose partially
   restored KV to attention;
6. avoid immediate D2H/H2D/D2H ping-pong without indefinitely starving
   swapped requests;
7. preserve one clear owner for every live or parked KV block;
8. model host-link and CPU-memory contention once; and
9. support both colocated serving and same-node PD disaggregation.

## Non-goals

The first version does not include:

- attention directly reading CPU-resident KV;
- cross-node PD session continuation;
- CXL or storage as a live-KV destination;
- cross-session prefix sharing;
- branching sessions with concurrent child turns;
- speculative write-through copies of every decode block;
- block compression or a different KV dtype during migration; or
- independent request queues on individual TP ranks.

All blocks required by one attention step must be NPU-resident before that
request can run.

## Terminology

**Scheduling instance** is one model replica controlled by one scheduler. A
TP instance has one logical request queue even though KV blocks are sharded
across several NPUs. A PP instance also has one admission decision, with
per-stage or per-rank memory accounting behind it.

**Usable KV capacity**, `C`, is the NPU memory available to live and parked KV
after model weights and fixed runtime allocations. The 10% reserve is measured
against `C`, not total device memory.

**Running request** is admitted work whose complete live KV is NPU-resident.
It may be omitted from one physical batch because of token or sequence limits,
but it remains eligible for compute.

**Waiting request** is not currently eligible for compute. New work, a
partially swapped request, and a request reloading from CPU are different
waiting states.

**Live KV** belongs to an unfinished LLM request. **Parked session KV** belongs
to a completed non-terminal session turn while the tool or human step runs.
These ownership classes use the same node CPU pool but different queues and
priorities.

**Admission** moves a request into the active NPU working set. It includes:

- starting a new colocated prefill;
- accepting a P-to-D handoff;
- reloading an unfinished swapped decode; and
- reloading parked session KV for a continuation.

**Pressure** occurs only when the next planned compute allocation does not fit
physical NPU capacity after counting all committed reservations. Crossing the
90% admission ceiling alone is not a pressure event.

## Core policy

### Separate admission and eviction thresholds

The policy intentionally separates two conditions:

```text
Admission ceiling:
    A new request, handoff, or reload may enter only if at least 10% of
    usable KV capacity remains after its complete reservation.

Eviction trigger:
    Evict only when the next compute iteration cannot reserve the KV blocks
    it needs from physical capacity.
```

With a default admission reserve ratio of `0.10`:

```text
admission_limit = 0.90 * C

admit(candidate) only if:
    npu_used
  + npu_reserved
  + candidate_npu_reservation
  <= admission_limit
```

All quantities are rounded to KV blocks. The check and reservation must be one
atomic scheduler operation. It must include reservations for:

- accepted but incomplete P-to-D handoffs;
- CPU-to-NPU live-decode reloads;
- CPU-to-NPU session reloads;
- new blocks reserved by in-flight compute; and
- any prefill reservation already granted to the request.

No incoming path may perform a stale read of the 10% reserve and reserve memory
later.

The 90-100% region is therefore reserved for growth of already admitted work.
A request reloaded to exactly 90% usage can continue decoding until the
running set consumes the remaining capacity. It is not evicted merely because
usage exceeded 90%.

### Ten-iteration eviction horizon

When the next compute allocation does not fit, the scheduler predicts the KV
blocks needed by the surviving running set over the next `H` decode
iterations. The initial value is:

```text
H = 10 iterations
```

For a decode request with one generated token per scheduled iteration, the
blocks required across a horizon starting at token position `t` are:

```text
horizon_blocks(req, t, steps) =
    ceil((t + min(steps, remaining_output_tokens)) / block_size)
  - ceil(t / block_size)
```

The scheduler first calculates the exact blocks required by the next batch.
It then projects the remaining `H - 1` iterations from the token positions
that would result after that batch. This avoids counting the next iteration's
new block twice.

The scheduler recomputes the projection after removing the selected victim
from the running set. The D2H reclaim target is:

```text
required_now = blocks required by the next selected compute batch

future_growth_after_next =
    sum(
        horizon_blocks(req, position_after_next_batch, H - 1)
        for surviving running requests
    )

reclaim_target =
    max(
        0,
        required_now
      + future_growth_after_next
      - currently_free_unreserved_blocks
    )
```

`reclaim_target` is a block count, not an unrounded byte estimate. Prefill
growth is included using the chunks that the scheduler expects to admit during
the same horizon.

The ten-iteration horizon amortizes migration setup and prevents a new
one-block D2H decision on every decode iteration. It is not a promise that all
ten iterations will run without another pressure event: a newly completed
handoff, a different prefill mix, or requests reaching block boundaries
together may change the forecast.

### Request-level preemption, block-level migration

The scheduler selects a victim request, removes it from the running set, and
then copies enough of that request's blocks to satisfy `reclaim_target`.

This distinction is required:

```text
Preemption unit: request
Migration and memory-release unit: KV block
```

An ordinary full-attention decode cannot run with missing historical blocks.
A partially offloaded request therefore stays waiting even if some of its
blocks remain on the NPU.

The initial block order is reverse logical order: the newest allocated block
is copied first. Full attention gives no computational benefit to keeping
either old or new blocks, but reverse order is deterministic and matches
append-oriented allocation. The policy interface must not rely on this order
for correctness.

If the first victim does not contain enough blocks, the scheduler fully drains
it and selects another victim. If later pressure occurs while the first victim
still has NPU-resident blocks, the scheduler continues draining that victim
before preempting another running request.

### Transfer aggregation

Residency and accounting remain block-granular, but issuing one ASTRA-Sim
workload per small block would add excessive frontend and IPC overhead.
Adjacent selected blocks should be aggregated into a transfer chunk.

The initial implementation should support either:

```text
--kv-offload-transfer-chunk-blocks <integer>
```

or an internal target such as 16-64 MiB per transfer, rounded to whole blocks.
Blocks in a chunk become reusable only when that chunk's D2H completion is
reported.

The simulator must record both block counts and aggregated transfer counts so
experiments can distinguish policy behavior from transport batching.

## Queue and state model

### Per-instance queues

Each scheduling instance owns the following logical queues:

| Queue | Meaning | Counts toward `max_num_seqs` |
| --- | --- | ---: |
| `waiting_new` | Arrived but not admitted | No |
| `waiting_handoff` | Completed prefill waiting for decode reservation | No |
| `running_prefill` | Admitted prefill with NPU reservation | Yes |
| `running_decode` | Runnable decode with complete NPU KV | Yes |
| `swapping_out` | Unfinished request with D2H in flight | No |
| `swapped_live` | Unfinished request with at least one CPU block | No |
| `swapping_in` | Unfinished request with reserved H2D destination | No |
| `ready_decode` | H2D complete; eligible next scheduling iteration | Yes |
| `parking_session` | Completed non-terminal turn moving to CPU | No |
| `parked_session` | Inactive session-owned state | No |
| `finished` | Terminal request with no live allocation | No |

The implementation may store these as deques, heaps, or indexed state sets,
but state transitions must be explicit. It must not reconstruct queue identity
from only `Request.evict` or aggregate residency.

### Request execution states

Add a state independent of individual block residency:

```python
class RequestState(Enum):
    WAITING_NEW = auto()
    RUNNING_PREFILL = auto()
    RUNNING_DECODE = auto()
    SWAPPING_OUT = auto()
    SWAPPED = auto()
    SWAPPING_IN = auto()
    READY_DECODE = auto()
    WAITING_PD_HANDOFF = auto()
    PARKING_SESSION = auto()
    FINISHED = auto()
```

The principal live-request transition is:

```text
RUNNING_DECODE
    -> SWAPPING_OUT
    -> SWAPPED
    -> SWAPPING_IN
    -> READY_DECODE
    -> RUNNING_DECODE
```

A request becomes ineligible for compute as soon as it enters
`SWAPPING_OUT`. It becomes eligible again only after every required block is
NPU-resident and the H2D reservation has committed.

### Block states

Each live or parked block has one of:

```python
class KVBlockState(Enum):
    NPU = auto()
    NPU_TO_CPU = auto()
    CPU = auto()
    CPU_TO_NPU = auto()
```

Each block record includes at least:

```python
@dataclass
class KVBlock:
    request_id: int | None
    session_id: str | None
    logical_index: int
    token_start: int
    token_count: int
    bytes_per_rank: int
    bytes_full_cluster: int
    state: KVBlockState
    owner_instance_id: int
    npu_allocation_id: int | None
    cpu_allocation_id: int | None
    migration_id: int | None
```

Request aggregate residency becomes a derived property:

- all required blocks `NPU`: runnable;
- any `NPU_TO_CPU`: swapping out;
- a mixture of `NPU` and `CPU`: partially swapped and not runnable;
- any `CPU_TO_NPU`: swapping in; and
- all required blocks `CPU`: fully swapped.

## Memory accounting invariants

The following invariants apply in colocated and PD modes.

1. `npu_used + npu_reserved <= usable_npu_capacity`.
2. `cpu_used + cpu_reserved <= node_cpu_capacity`.
3. D2H reserves the CPU destination before submission.
4. A D2H source block remains NPU-used until transfer completion.
5. H2D reserves the NPU destination before submission.
6. An H2D source block remains CPU-used until transfer completion.
7. A block in flight therefore consumes source-used and
   destination-reserved capacity simultaneously.
8. A request may run attention only if every required block is `NPU`.
9. `SWAPPING_IN` requests and their NPU reservations are not eviction
   candidates.
10. Terminal cleanup, cancellation, TTL expiry, and failed migration release
    each allocation exactly once.
11. CPU sizes use full-cluster bytes; NPU sizes use the existing per-rank
    convention. Conversion occurs at one memory-model boundary.
12. Prefix-cache allocations do not alias live or parked block ownership.

Assertions should report request or session identity, block index, owner
instance, migration ID, and both tier counters.

## Scheduling algorithm

### Iteration order

At every scheduling event, an instance performs:

```text
1. Apply completed compute and migration events.
2. Commit or release memory reservations.
3. Move requests to their next explicit queues.
4. Promote READY_DECODE requests to RUNNING_DECODE.
5. Build the candidate batch from the admitted running set.
6. Calculate exact block allocations required by that batch.
7. If it fits, reserve new blocks and issue compute.
8. If it does not fit:
   a. choose one request-level victim;
   b. remove it from the candidate and running set;
   c. recompute the immediate and ten-iteration block requirement;
   d. reserve CPU block destinations;
   e. issue enough D2H chunks to meet the reclaim target; and
   f. continue any capacity-feasible compute for the surviving running set.
9. Evaluate external admissions against the 90% ceiling.
10. Keep requests that fail admission in their current waiting queue.
```

Admission is deliberately after existing running work. This preserves the
10% reserve for requests that already own NPU KV.

### Victim eligibility

A live request is eligible for preemption only when:

- it is in `RUNNING_DECODE` or an explicitly supported prefill state;
- it has no compute batch in flight;
- it has no H2D or D2H chunk in flight;
- it is not a newly committed P-to-D handoff awaiting its first decode step;
- it is not `SWAPPING_IN`; and
- its removal leaves at least one runnable request or another progress path.

The first victim policy should remain configurable:

```text
lru
largest-kv
cost-aware
```

For a deterministic first implementation, use LRU with largest-KV as the
tie-breaker. Once a request becomes the active partial-swap victim, drain it
before choosing another request unless it has no remaining NPU blocks.

### Reload admission and fairness

An unfinished swapped decode receives priority over completely new work,
subject to the same 90% admission rule:

```text
1. oldest `swapped_live` request that fits;
2. waiting P-to-D handoff that fits;
3. CPU session continuation that fits;
4. new request or new colocated prefill that fits.
```

PD experiments may compare P-to-D handoff and session-continuation order, but
the chosen policy must be recorded.

Before H2D begins, the scheduler reserves NPU capacity for every missing block
of the request. It does not reserve again for blocks that remained NPU-resident
during a partial swap. While H2D is in flight:

- the request remains outside `running_decode`;
- its reserved NPU blocks cannot be used by another request;
- it cannot be selected as a victim; and
- its CPU source blocks remain valid.

The 10% admission reserve is the primary anti-ping-pong mechanism. A separate
minimum-residency lease is initially disabled. The implementation still
records reload-to-next-eviction distance in iterations and generated tokens.
If experiments show frequent re-eviction within the ten-iteration horizon, add
a configurable fallback:

```text
--kv-reload-min-resident-iters 10
```

This fallback must not deadlock the scheduler when all running requests are
protected.

### Admission behavior after pressure

Offloading only the ten-iteration reclaim target should normally leave total
usage above the 90% admission ceiling. New admissions therefore remain
blocked while surviving requests use the reclaimed space.

If a large transfer chunk or small victim unexpectedly drops usage below 90%,
the scheduler may admit a request only if its complete reservation still
leaves 10% free. A stricter completion-gated reopening policy may be retained
as an experiment, but it is not the default:

```text
--kv-admission-reopen-policy headroom
--kv-admission-reopen-policy completion
```

`completion` mode reopens admission only after a member of the pre-pressure
running set produces durable capacity:

- a terminal request releases KV; or
- a completed non-terminal session finishes its D2H park and releases NPU KV.

This mode is useful for determining whether headroom alone is sufficient.

## Migration and compute overlap

Block-granular offload is useful only if the other running requests can
continue while D2H/H2D work progresses. The current correctness-first global
migration barrier must therefore be replaced with separate logical resources:

```text
Compute lane
    model execution for capacity-feasible running requests

Migration lane
    D2H and H2D transfer chunks

Shared node service
    host-link and CPU-memory queue shared by all local instances
```

The memory backend may serialize migrations on the shared host service, but a
migration must not automatically serialize unrelated NPU compute. If the
ASTRA-Sim/controller interface cannot report concurrent workload completion,
implement this in two milestones:

1. block-correct synchronous state transitions; then
2. explicit compute/migration overlap with independent workload IDs.

Performance conclusions must use milestone 2. Milestone 1 is only a
correctness reference.

## PD disaggregation disabled

### Ownership and queues

One colocated instance owns prefill and decode for a request. It uses one
central scheduler with separate `running_prefill` and `running_decode` sets.
There is no P-to-D ownership transfer.

The request path is:

```text
router waiting
    -> colocated admission
    -> RUNNING_PREFILL
    -> RUNNING_DECODE
    -> FINISHED or PARKED_SESSION
```

### New-request admission

A new request consumes no NPU KV while it remains in `waiting_new`. Before
promoting it to `RUNNING_PREFILL`, the scheduler reserves the known prompt KV
footprint or an explicitly configured bounded prefill reservation. The
correctness-first option reserves the full known prompt footprint and admits
only if 10% remains.

Chunked prefill commits this reservation block by block as computation
finishes. If the full-prompt reservation is too conservative, a later policy
may reserve a bounded prefill horizon, but that policy must be evaluated
separately because an already-started prefill could then encounter pressure.

### Active decode pressure

When active decodes exhaust physical capacity:

1. choose one running request;
2. move it to `SWAPPING_OUT`;
3. offload enough blocks for the surviving set's next ten iterations;
4. keep the victim waiting until all missing blocks can be reserved and
   reloaded; and
5. prioritize that swapped live request over new arrivals once it fits under
   the admission ceiling.

### Agentic session completion

On a terminal turn, free active KV immediately.

On a non-terminal turn with session retention:

- transfer ownership from the completed request to `SessionKVState`;
- keep it parked on the colocated NPU while capacity allows;
- prefer parked session KV over active live KV as a pressure victim;
- D2H parked blocks through the node CPU pool when selected; and
- on continuation, claim NPU blocks directly or reserve and reload CPU blocks
  into the same colocated instance before suffix prefill.

The tool-wait interval does not consume a running sequence slot.

## PD disaggregation enabled

### Independent admission domains

Prefill and decode instances have separate NPU capacities, queues, and
admission decisions. A typical `1P + 3D` node therefore has:

```text
Prefill scheduler P0
    waiting_new
    running_prefill
    session_reload
    waiting_pd_handoff

Decode scheduler D0
    running_decode
    swapped_live
    migration queues

Decode scheduler D1
    running_decode
    swapped_live
    migration queues

Decode scheduler D2
    running_decode
    swapped_live
    migration queues

Shared node CPU KV pool and host-link service
```

A routing decision never substitutes for a capacity reservation. Each decode
candidate must atomically test its own 90% admission ceiling.

### First-turn P-to-D flow

The first turn follows:

```text
new request
    -> prefill admission on P
    -> prompt KV creation on P
    -> choose a decode instance that can reserve the complete handoff
    -> reserve destination blocks on D
    -> P-to-D handoff
    -> commit decode ownership
    -> release prefill live ownership
    -> RUNNING_DECODE on D
```

If no decode instance can preserve the 10% reserve:

- the request stays in `waiting_pd_handoff`;
- the prefill source allocation or an explicitly modeled handoff staging
  allocation remains valid;
- the decode request does not enter `running_decode`;
- no destination ownership is committed; and
- the router retries when decode capacity changes.

For multiple decode instances, routing first filters instances that can
reserve the handoff, then applies the configured load/RR policy within that
feasible set.

P-to-D handoff traffic remains distinct from CPU offload traffic. The current
atomic same-node handoff may be retained as a compatibility path, but reports
must not interpret its zero modeled transfer time as hardware behavior. A
timed direct NPU-to-NPU handoff is a separate integration step.

### Unfinished decode preemption

An unfinished decode swapped under pressure remains owned by its decode
instance:

```text
RUNNING_DECODE on D1
    -> block D2H to shared CPU
    -> SWAPPED_LIVE owned by D1
    -> block H2D to D1
    -> RUNNING_DECODE on D1
```

It does not return to prefill because the same LLM call is still generating
tokens. Its complete prompt-plus-output KV history is one live block table.

The decode instance uses the same request-level victim and ten-iteration
block-reclaim policy as colocated mode. Other decode instances continue
independently, subject to contention on the node-shared CPU/host service.

### Completed non-terminal session bridge

A completed agentic turn is different from an unfinished swapped decode. Its
next turn has new input and must visit prefill again.

For same-node PD with session retention:

```text
decode turn completes on D
    -> request ownership becomes parked session ownership
    -> mandatory D2H park to shared CPU
    -> release decode NPU blocks after D2H completion
    -> tool or human gap
    -> continuation arrives
    -> choose prefill instance
    -> reserve complete CPU-to-P reload while preserving P's 10% reserve
    -> H2D session KV to P
    -> suffix prefill on P
    -> capacity-reserved P-to-D handoff
    -> next decode turn
```

The completion event itself does not free decode NPU memory. Capacity becomes
available only when the terminal cleanup occurs or the non-terminal session's
D2H park completes.

If the shared CPU pool cannot admit the parked session:

- never discard unfinished live decode state to preserve optional parked
  state;
- drop optional LRU parked-session state when policy allows;
- record a session miss; and
- full-prefill the next turn.

Same-node PD session retention continues to require CPU KV offloading.
Cross-node continuation remains rejected until the inter-node ownership and
transfer path is modeled.

### PD scheduling priorities

Each decode scheduler first protects its admitted running set. Capacity-aware
router priorities are:

```text
1. resume an unfinished swapped decode on its owner decode instance;
2. accept an already-computed P-to-D handoff;
3. accept a new decode handoff.
```

The prefill scheduler priorities are:

```text
1. finish already admitted prefill chunks;
2. reload an arrived session continuation that fits;
3. start a new prefill that fits.
```

These priorities should be configurable for TTFT-versus-TPOT experiments, but
no policy may bypass destination reservation.

## Configuration surface

Tentative instance-level controls:

```text
--enable-kv-offloading / --no-enable-kv-offloading
--kv-offload-granularity block
--kv-admission-reserve-ratio 0.10
--kv-offload-lookahead-iters 10
--kv-offload-transfer-chunk-blocks <auto-or-integer>
--kv-offload-victim-policy lru
--kv-admission-reopen-policy headroom
--kv-reload-min-resident-iters 0
```

The existing high/low watermark flags require migration:

- the old `high_watermark=0.90` must not remain the eviction trigger;
- `kv_admission_reserve_ratio=0.10` replaces its admission meaning; and
- the lookahead reclaim target replaces a fixed 80% low-watermark target.

For compatibility, old flags may map to the baseline request-granular policy
only. The simulator should reject ambiguous combinations instead of silently
changing semantics.

## Metrics and diagnostics

### Queue and latency metrics

Record per request:

- time in every queue and execution state;
- first admission time;
- swap-out start and completion;
- reload reservation, start, and completion;
- number of decode iterations and tokens between reload and next eviction;
- TTFT, TPOT, ITL, and end-to-end latency; and
- owner instance transitions.

### Memory and migration metrics

Record per instance and node:

- NPU/CPU used and reserved occupancy over time;
- admission rejections caused by the 10% reserve;
- physical-capacity pressure events;
- predicted versus actual blocks needed over the ten-iteration horizon;
- victim count and victim identities;
- partial- and full-swap block counts;
- D2H/H2D chunks, blocks, bytes, service time, and queueing time;
- time requests retain mixed NPU/CPU residency;
- reload-to-re-eviction distance; and
- same-request D2H/H2D/D2H ping-pong count within `H` iterations.

### PD-specific metrics

Record separately:

- P-to-D handoff count, bytes, reservation wait, and transfer time;
- handoff rejection by decode instance;
- decode-instance selection and load distribution;
- non-terminal D-to-CPU session park bytes and wait;
- CPU-to-P continuation reload bytes and wait;
- suffix-prefill computed versus reused tokens; and
- shared host-link contention split by live swap, session park, and session
  reload.

Do not combine live-decode swapping with completed-session retention in one hit
or wait counter.

## Validation plan

### Unit tests

Add deterministic tests for:

1. the exact 90% admission boundary;
2. atomic failure when two admissions compete for the same reserve;
3. physical-capacity eviction trigger independent of the 90% ceiling;
4. exact ten-iteration block prediction at every offset within a block;
5. one victim leaving the running queue before its first D2H submission;
6. partial D2H completion releasing only completed source blocks;
7. a partially swapped request remaining non-runnable;
8. H2D reserving every missing destination block before submission;
9. `SWAPPING_IN` requests being ineligible as victims;
10. continuation of surviving compute during D2H;
11. CPU exhaustion and transfer cancellation without leaked reservations;
12. terminal completion freeing immediately;
13. non-terminal PD completion freeing only after session D2H;
14. no duplicate KV owner during P-to-D handoff; and
15. TP/PP instances keeping one logical request state across ranks/stages.

### Deterministic scenario tests

#### PD disabled

Use ten running decodes with synchronized block boundaries:

1. force the next iteration to exceed physical capacity;
2. verify exactly one victim leaves `running_decode`;
3. verify only the calculated ten-iteration block target is copied;
4. allow the other nine requests to progress during D2H;
5. finish one running request;
6. reload the oldest swapped request only if the complete reservation leaves
   10% free; and
7. verify no request executes with a CPU block.

Repeat with a non-terminal agentic turn and verify NPU and CPU session hits.

#### PD enabled

Use one prefill and three decode instances:

1. fill one decode instance above its handoff-admission limit;
2. verify the router chooses another capacity-feasible decode;
3. fill all decode instances and verify the handoff waits at prefill;
4. trigger active decode pressure on one decode instance without stopping
   compute on the other two;
5. complete a non-terminal turn and verify D-to-CPU park;
6. release the continuation and verify CPU-to-P reload, suffix prefill, and
   a newly capacity-checked P-to-D handoff; and
7. verify live swapped decode state never routes through prefill.

### End-to-end experiment matrix

Keep prefix caching disabled and run the same long-context agentic workload
under:

| Dimension | Values |
| --- | --- |
| PD mode | colocated, `1P + 3D` |
| CPU KV capacity | 4, 16, 48 GiB |
| Host-link bandwidth | 64, 256 GB/s |
| Admission reserve | 5%, 10%, 15% |
| Lookahead | 1, 10, 32 iterations |
| Seed | 7 plus two confirmation seeds |

Run the existing request-granular scheduler as the baseline. Report:

- generated-token throughput;
- TTFT and TPOT p50/p95/p99;
- D2H/H2D bytes and host-service time;
- active live-swap and session-hit/miss counts;
- pressure events;
- queue time by state; and
- ping-pong count and reload-to-re-eviction distance.

### Acceptance criteria

The implementation is accepted only if:

1. all deterministic requests and sessions complete;
2. no memory counter becomes negative or exceeds capacity;
3. no block has duplicate committed ownership;
4. no partially swapped request executes attention;
5. all external admissions leave the configured NPU reserve;
6. every pressure event either makes compute progress or reports a finite
   capacity blocker;
7. block migration reduces D2H bytes for small deficits relative to the
   request-granular baseline;
8. surviving requests progress while migration is in flight;
9. immediate ping-pong within the lookahead horizon is absent or explicitly
   explained by an unavoidable capacity event; and
10. PD metrics distinguish active decode swap, completed-session bridge, and
    P-to-D handoff traffic.

Performance acceptance should focus on reducing p95/p99 TPOT without
materially reducing generated-token throughput. A throughput/latency tradeoff
must be reported rather than hidden by one aggregate score.

## Implementation phases

### Phase 0: Freeze semantics and add diagnostics

- Approve this plan and the default 10%/10-iteration values.
- Add queue-state and reload-to-re-eviction diagnostics to the existing
  request-granular baseline.
- Preserve prefix-off behavior in all validation configs.

### Phase 1: Explicit queues and request states

- Add `RequestState`.
- Replace strict aggregate resident filtering with explicit running, waiting,
  swapped, and migration queues.
- Keep request-granular migration temporarily.
- Validate fairness and state transitions before changing granularity.

### Phase 2: Separate admission reserve from eviction

- Add the 10% atomic admission check.
- Apply it to colocated prefill, P-to-D handoff, live reload, and session
  reload.
- Trigger eviction only on an actual next-allocation deficit.
- Add exact block-boundary lookahead calculations.

### Phase 3: KV block table and partial swap

- Add per-block ownership and residency.
- Select victims by request but submit only the required D2H blocks.
- Add transfer chunk aggregation.
- Support partial NPU/CPU residency only for non-runnable waiting requests.

### Phase 4: Reload reservation and anti-ping-pong policy

- Reserve every missing NPU block before H2D.
- Protect `SWAPPING_IN` state.
- Prioritize oldest unfinished swapped decode.
- Measure headroom-only reopening and compare completion-gated reopening if
  necessary.

### Phase 5: Compute/migration overlap

- Split compute and migration lanes in the scheduler/controller.
- Preserve node-wide host-link contention.
- Allow surviving requests and unaffected instances to compute during D2H/H2D.
- Require explicit completion events before releasing source blocks.

### Phase 6: Colocated integration

- Integrate new-request prefill reservation.
- Integrate active decode pressure.
- Integrate NPU/CPU parked-session retention.
- Run deterministic and long-context colocated tests.

### Phase 7: Same-node PD integration

- Add capacity-aware P-to-D routing and waiting handoffs.
- Keep live swapped decodes on their decode owner.
- Integrate mandatory D-to-CPU-to-P continuation bridging.
- Run `1P + 3D` pressure tests and verify independent decode queues.

### Phase 8: Policy sweep and report

- Run the end-to-end matrix.
- Compare request-granular and block-granular policies.
- Select admission reserve, lookahead, chunk size, and fairness defaults from
  measured TPOT, TTFT, throughput, and migration traffic.
- Update the public simulator documentation only after defaults are stable.

## Expected code touch points

| File | Planned responsibility |
| --- | --- |
| `serving/core/request.py` | Request state and per-block migration metadata |
| `serving/core/memory_model.py` | Block allocator, reservations, and tier invariants |
| `serving/core/scheduler.py` | Queues, admission, victim selection, lookahead, reload |
| `serving/core/router.py` | Capacity-aware P-to-D routing and waiting handoffs |
| `serving/__main__.py` | Independent compute/migration event progress |
| `serving/core/trace_generator.py` | Aggregated block migration workloads |
| `serving/core/controller.py` | Concurrent workload IDs and completion events |
| `serving/core/config_builder.py` | New policy controls and validation |
| `astra-sim/` analytical backend | Overlapped workload completion and shared host queue |
| `tests/test_kv_offloading_foundation.py` | State, capacity, block, and PD invariants |

## Open decisions

The following decisions require measurements rather than assumptions:

1. whether 10% of usable KV capacity is sufficient across model sizes and
   batch sizes;
2. whether ten iterations is a better reclaim horizon than a byte-based
   transfer target;
3. whether reverse logical block order affects fragmentation or reload cost;
4. whether headroom-only admission reopening fully eliminates practical
   ping-pong;
5. whether swapped live decode should always outrank a waiting P-to-D handoff;
6. the transfer chunk size that balances migration responsiveness and IPC
   overhead; and
7. how much compute/migration overlap the analytical backend can represent
   without double-counting contention.

These are explicit experiment axes. They must not be silently fixed by
incidental queue order.
