---
sidebar_position: 4
title: Agentic sessions
---

# Agentic sessions

A standard inference benchmark like ShareGPT models *independent*
prompts: each request is one prompt → one response, and the next
request is unrelated to the previous one. Real production traffic
for **agents** doesn't look like this.

A coding agent (Cursor, Aider, or SWE-bench solvers) runs a tight
loop: ask the LLM what to do → run a tool (compile, test, search) →
feed the result back → ask the LLM the next thing → run another tool
→ ... A request budget for "1000 SWE-bench problems" is really 1000
*sessions*, each with 5–50 chained LLM calls and tool waits in
between.

That's what the **agentic** workload format is for.

## The format

Each JSONL line is one session:

```json
{
  "session_id": "session_42",
  "arrival_time_ns": 4059740,
  "sub_requests": [
    {"input_toks": 1472, "output_toks": 133, "tool_duration_ns": 127348767},
    {"input_toks": 1582, "output_toks": 125, "tool_duration_ns": 197295027},
    {"input_toks": 1734, "output_toks": 77,  "tool_duration_ns": 0}
  ]
}
```

Three sub-requests, with `tool_duration_ns` between each, that's the
simulated time spent running tools (test runner, web fetch, file
search) between LLM calls. The simulator doesn't simulate the tool
itself, it just waits.

Full schema reference is on
**[JSONL format → Agentic format](./jsonl-format#agentic-format)**.

## How the simulator handles dependency chains

When the workload is loaded, **only the first sub-request** of each
session is added to `Router._pending_requests`. The rest live in
`Router._deferred_sessions`, keyed by session id.

```mermaid
sequenceDiagram
    autonumber
    participant L as Loader
    participant R as Router
    participant Sc as Scheduler
    participant Clock as Simulated clock
    L->>R: load (only sub_request[0] enqueued)
    Note over R: sub_request[1..] deferred
    Clock->>R: arrival_time_ns reached
    R->>Sc: add_request(sub_request[0])
    Sc->>Sc: schedule, run, finish
    Sc->>R: notify_request_completed(sub_0)
    Note over R: release sub_request[1] with<br/>arrival = completion + tool_duration_ns
    Clock->>R: that arrival reached
    R->>Sc: add_request(sub_request[1])
    Note over R,Sc: ...continue until sub_requests empty
```

`Router.has_deferred_sessions()` keeps the main loop from exiting
while sessions are still active (otherwise a workload with a long
final tool_duration could exit prematurely between sub-requests).

For the full lifecycle, see
**[Simulator → Request lifecycle](/docs/simulator/request-lifecycle#agentic-sessions-when-stage-10-is-not-the-end)**.

## Reusing KV within one session

For append-only agent or conversation turns, add
`"reuse_previous_kv": true` to the session and run with
`--enable-session-kv-retention --no-enable-prefix-caching`. The router pins
the session to its first colocated scheduler. A completed non-terminal turn
parks its existing allocation, and the next turn claims it and computes only
the uncached input suffix.

Add `--enable-kv-offloading` to let memory pressure move parked session KV to
the node-shared CPU pool. The migration uses the configured host-link and CPU
memory timing. A continuation that hits CPU-resident state waits for its H2D
reload to complete before computing the suffix; it never attends directly to
CPU memory. If active-request eviction needs the remaining CPU capacity, the
simulator drops LRU parked session state first and records a future miss.
Generic prefix caching remains unsupported in this mode.

For same-node prefill/decode disaggregation, enable both session retention and
CPU KV offloading on every prefill and decode instance. A non-terminal decode
turn always parks through a modeled D2H workload. The next turn reloads the
CPU record into its affinitized prefill instance through a modeled H2D
workload before computing the suffix. This decode-to-prefill bridge is never a
free ownership change. Cross-node PD session reuse is not supported.

Use `--session-kv-ttl-ns` to set a default maximum idle lifetime, or add
`session_kv_ttl_ns` to one session record to override it. The timer starts
when a non-terminal turn completes. `0` disables expiry. The simulator wakes
at the expiry event during long tool or human pauses, and expiration wins if
the next turn arrives at the exact same timestamp.

This mode does not require `input_tok_ids` and never shares KV across two
different session ids. Use a per-turn `reused_prefix_toks` scalar when a
context was truncated or summarized. See the
**[session-retention schema](./jsonl-format#session-kv-retention-without-token-ids)**
for fields and current implementation limits.

## Runnable session-retention examples

The bundled colocated example retains KV on the NPU across a 100 ms tool wait:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_session_kv_retention.json \
  --dtype bfloat16 --block-size 16 \
  --dataset workloads/example_session_retention.jsonl \
  --output outputs/session_retention.csv \
  --num-reqs 1
```

The second turn has 15 input tokens and reuses 13 predecessor tokens, so its
prefill computes only the 2-token suffix. The companion
`outputs/session_retention_kv_offload.csv` sidecar records the NPU hit.

To exercise hard expiry, set a TTL shorter than the 100 ms tool wait:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_session_kv_retention.json \
  --dtype bfloat16 --block-size 16 \
  --dataset workloads/example_session_retention.jsonl \
  --output outputs/session_retention_ttl.csv \
  --num-reqs 1 --session-kv-ttl-ns 20000000
```

The 20 ms TTL expires during the wait, so the second turn is a full-prefill
miss. For the same-node PD CPU bridge, including the zero-wait boundary:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_pd_session_kv_retention.json \
  --dtype bfloat16 --block-size 16 \
  --dataset workloads/example_session_retention_zero_wait.jsonl \
  --output outputs/session_retention_pd.csv \
  --num-reqs 1
```

This run performs a decode-side D2H park and a prefill-side H2D reload before
the 2-token suffix. Inspect the sidecar to distinguish these migration bytes
from the ordinary forward prefill-to-decode handoffs.

## Bundled SWE-bench example

The repo ships
`workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl`: 50 SWE-bench
sessions for `Qwen3-30B-A3B-Instruct-2507`, arriving at 0.2
sessions/second.

A typical session in this file has 8-15 sub-requests with input
lengths in the 1000-3000 token range and tool durations of 50-300 ms
(the wait while pytest runs, etc).

Run it with the bundled DP+EP MoE config:

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_moe_dp_ep_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl' \
  --output 'outputs/swebench_run.csv' \
  --num-reqs 1
```

`--num-reqs 1` means one *session* (which expands to 8-15
sub-requests). Bump it for longer runs.

## Building your own agentic workload

There's no bundled generator for agentic format, chain extraction
depends on your data source. The pattern:

1. **Extract sessions from your trace source.** For SWE-bench, that's
   one session per problem; for browser-agent traces, one session per
   user task.
2. **For each session, extract the per-call (prompt, response) pairs
   and tool durations.** Tool duration is wall-clock time between
   the assistant message and the next user message in the trace.
3. **Tokenize prompts** with the simulator's target model's
   tokenizer. Optionally tokenize responses too if you want
   downstream analysis.
4. **Write one JSONL line per session** with the schema from
   [JSONL format → Agentic](./jsonl-format#agentic-format).

A minimal Python sketch:

```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")

with open("workloads/my-agentic.jsonl", "w") as f:
    for session_id, calls in extract_sessions_from_my_data():
        sub_requests = []
        for prompt, response, next_call_delay_ns in calls:
            ids_in = tok.encode(prompt)
            ids_out = tok.encode(response)
            sub_requests.append({
                "input_toks": len(ids_in),
                "output_toks": len(ids_out),
                "input_tok_ids": ids_in,
                "output_tok_ids": ids_out,
                "tool_duration_ns": next_call_delay_ns,
            })
        # last sub-request has no follow-up
        if sub_requests:
            sub_requests[-1]["tool_duration_ns"] = 0

        f.write(json.dumps({
            "session_id": session_id,
            "arrival_time_ns": session_start_ns(session_id),
            "sub_requests": sub_requests,
        }) + "\n")
```

Adjust the `extract_sessions_from_my_data()` and
`session_start_ns()` to your dataset.

## Picking arrival rates

Agentic workloads are usually **much sparser** than ShareGPT-style
workloads in arrival rate, because each session lasts much longer
in simulator-time:

| Workload | Typical sps | Why |
| --- | --- | --- |
| ShareGPT | 5-20 | Each request finishes in 1-5 seconds; high arrival rate keeps the scheduler busy |
| Agentic SWE-bench | 0.1-0.5 | Each session can run for 30-120 seconds; even 0.2 sps overlaps many sessions |

The bundled SWE-bench file uses `sps=0.2`. With 50 sessions arriving
over 250 simulator-seconds and each running ~60 seconds, you get
~12 sessions active concurrently, a realistic load.

## Mixing flat + agentic in one file

The loader handles per-line auto-detection, so you can have:

```jsonl
{"input_toks": 100, "output_toks": 50, "arrival_time_ns": 0}
{"session_id": "s0", "arrival_time_ns": 1000000, "sub_requests": [{"input_toks": 200, "output_toks": 100, "tool_duration_ns": 0}]}
{"input_toks": 150, "output_toks": 80, "arrival_time_ns": 2000000}
```

Useful when you want a sanity-baseline of independent prompts mixed
with agentic sessions.

## Gotchas

1. **Last sub-request's `tool_duration_ns` should be 0** (or just
   omitted in your generator if you treat 0 as default). Non-zero
   keeps the session "alive" past its real end and the simulator
   waits unnecessarily.
2. **Session arrival_time_ns is for the *first* sub-request.**
   Subsequent sub-requests have their arrival times computed at run
   time as `previous_completion + tool_duration_ns`.
3. **Choose one cache model.** Generic cross-session prefix caching needs
   `input_tok_ids`. Same-session KV retention does not: it uses session order
   and scalar token counts, and must run with generic prefix caching disabled.
4. **Affinity depends on the mode.** Session-retention continuations remain
   affinitized to their original colocated prefill scheduler. Without session
   retention, independent turns are routed by the selected request policy.

## What's next

- **[Simulator → Request lifecycle](/docs/simulator/request-lifecycle)**
  what happens at runtime when the simulator processes a session.
- **[Examples → DP+EP MoE](/docs/examples/parallelism/dp-ep-moe)** -
  uses the bundled SWE-bench agentic workload.
