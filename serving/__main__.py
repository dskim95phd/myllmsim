"""Simulation entry point: ``python -m serving --cluster-config <...> [...]``.

Parses CLI args, generates ASTRA-Sim input files via ``serving.core.config_builder``,
spawns the ASTRA-Sim subprocess, and runs the iteration loop:
``router.route -> scheduler.schedule -> trace_generator -> graph -> ASTRA-Sim
-> scheduler.add_done`` until every request completes.
"""

import os
import subprocess
import argparse
from collections import deque
import hashlib
import json
import shutil
import tempfile
from time import perf_counter, time
from collections import defaultdict

from serving.core.scheduler import *
from serving.core.request import *
from serving.core.utils import *
from serving.core.controller import *
from serving.core.memory_model import *
from serving.core.graph_generator import *
from serving.core.trace_generator import *
from serving.core.pim_model import *
from serving.core.config_builder import *
from serving.core.router import *
from serving.core.power_model import *
from serving.core.logger import *
from serving.core.run_paths import build_run_paths, resolve_run_id
from serving.core.host_timing import HostTimingRecorder
from serving.core.chakra_template import (
    BoundedTemplateCache,
    ChakraTemplateBundle,
    RuntimeTraceSnapshot,
)
from serving.core.workload_transport import (
    FileWorkloadTransport,
    IpcFileWorkloadTransport,
)
import sys as flush

from pyinstrument import Profiler


def _pad_batch_to_max(batch, max_len):
    """Pad a batch up to ``max_len`` for DP-sync.

    Mirrors vLLM's CUDA-graph DP padding: every DP rank's forward runs at
    ``max(num_tokens_across_dp)``. We bump the high-level counters so
    dense layers, lm_head, and the MoE compute path all reflect the
    padded shape — but we deliberately leave ``decode_k_list`` /
    prefill lists untouched so attention continues to see only the real
    decodes. FlashAttention's varlen kernel gives padded ``seq_len=0``
    entries zero compute in real vLLM, and extending ``decode_k_list``
    with ``kv=1`` dummies would instead collapse ``kv_decode_mean``
    toward 1 and push the attention lookup far outside the profiled
    sweep.

    MoE AG/RS comm size is anchored separately to ``max_total_len`` (no
    ``× group_size``) in the iteration loop — that calibrates the
    bandwidth model against the same ``link_bw`` AllReduce already uses.

    Request-completion accounting (`scheduler.add_done`) reads
    ``batch.requests`` and ``batch.end``, not these mutated token-list
    fields, so it is unaffected.
    """
    pad = max_len - batch.total_len
    if pad <= 0:
        return
    batch.total_len = max_len
    batch.kv_len += pad                  # each dummy contributes kv=1
    batch.num_decode += pad              # counted for lm_head / dense shape


def _runtime_limit(value):
    return float('inf') if value == 0 else value


def _cluster_config_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join("..", path)


def _load_cluster_config_for_overrides(path):
    with open(_cluster_config_path(path), "r") as f:
        return json.load(f)


def _resolve_output_file(path, run_id):
    if path is None:
        return None
    return path.replace("{run_id}", run_id)


def _workload_ipc_socket_path(run_id):
    name = f"llmservingsim-{run_id}.sock"
    path = os.path.join(tempfile.gettempdir(), name)
    if len(os.fsencode(path)) <= 100:
        return path
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]
    return os.path.join(tempfile.gettempdir(), f"llmservingsim-{digest}.sock")


def _read_chakra_graph_bundle(workload_prefix, system_ids):
    graphs = {}
    for system_id in system_ids:
        graph_path = f"{workload_prefix}.{system_id}.et"
        with open(graph_path, "rb") as graph_file:
            graphs[system_id] = graph_file.read()
    return graphs


def _kv_offload_output_file(path):
    """Return the instance-level CPU KV offload sidecar path."""
    if path is None:
        return None
    stem, extension = os.path.splitext(path)
    return f"{stem}_kv_offload{extension or '.csv'}"


def _next_idle_event(router, schedulers):
    """Return the earliest request-arrival or parked-session-expiry event."""
    events = [router.get_next_pending_arrival()]
    events.extend(
        scheduler.get_next_session_kv_expiry()
        for scheduler in schedulers
    )
    events = [event for event in events if event is not None]
    return min(events) if events else None


def _cleanup_inputs_root(run_paths, logger):
    """Remove generated ASTRA-Sim inputs after a completed simulation."""
    runs_root = os.path.abspath(os.path.join("inputs", "runs"))
    inputs_root = os.path.abspath(run_paths.inputs_root)
    if inputs_root in (os.path.abspath("inputs"), runs_root):
        raise RuntimeError(f"Refusing to remove broad inputs root: {inputs_root}")
    if not inputs_root.startswith(runs_root + os.sep):
        logger.warning(
            "Skipping ASTRA-Sim inputs cleanup because inputs_root is outside %s: %s",
            runs_root, inputs_root,
        )
        return
    shutil.rmtree(inputs_root, ignore_errors=True)
    logger.info("Removed ASTRA-Sim inputs root: %s", inputs_root)


def _prepare_ns3_config(astra_sim, run_paths):
    template = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
    output_dir = os.path.join(run_paths.inputs_root, "ns3", "output")
    config_path = os.path.join(run_paths.inputs_root, "ns3", "config.txt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    replacements = {
        "FLOW_FILE": os.path.join(output_dir, "flow.txt"),
        "TRACE_FILE": os.path.join(output_dir, "trace.txt"),
        "TRACE_OUTPUT_FILE": os.path.join(output_dir, "mix.tr"),
        "FCT_OUTPUT_FILE": os.path.join(output_dir, "fct.txt"),
        "PFC_OUTPUT_FILE": os.path.join(output_dir, "pfc.txt"),
        "QLEN_MON_FILE": os.path.join(output_dir, "qlen.txt"),
    }

    for path in (replacements["FLOW_FILE"], replacements["TRACE_FILE"]):
        open(path, "w").close()

    with open(template, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in replacements:
                f.write(f"{parts[0]} {replacements[parts[0]]}\n")
            else:
                f.write(line)
    return config_path


def _iter_raw_instances(cluster_config):
    for node in cluster_config.get("nodes", []):
        for instance in node.get("instances", []):
            yield instance


def _resolve_instance_dtype(instance, cli_dtype, dtype_to_bits):
    dtype = instance.get("dtype", cli_dtype)
    if dtype is None:
        config = get_config(instance["model_name"])
        torch_dtype = config.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype in dtype_to_bits:
            dtype = torch_dtype
        else:
            dtype = "bfloat16"
    if dtype not in dtype_to_bits:
        raise ValueError(f"Unsupported dtype '{dtype}' for instance {instance.get('instance_id')}")
    return dtype


def _build_instance_runtime_configs(instances, args, dtype_to_bits):
    runtime_configs = []
    for instance_id, instance in enumerate(instances):
        dtype = _resolve_instance_dtype(instance, args.dtype, dtype_to_bits)
        kv_cache_dtype = instance.get("kv_cache_dtype", args.kv_cache_dtype)
        if kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError(f"Unsupported kv_cache_dtype '{kv_cache_dtype}' for instance {instance_id}")

        enable_attn_offloading = instance.get("enable_attn_offloading", args.enable_attn_offloading)
        enable_sub_batch_interleaving = instance.get(
            "enable_sub_batch_interleaving", args.enable_sub_batch_interleaving)
        if enable_sub_batch_interleaving and not enable_attn_offloading:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving without attention offloading")

        runtime_configs.append({
            "max_num_seqs": _runtime_limit(instance.get("max_num_seqs", args.max_num_seqs)),
            "max_num_batched_tokens": _runtime_limit(
                instance.get("max_num_batched_tokens", args.max_num_batched_tokens)),
            "long_prefill_token_threshold": instance.get(
                "long_prefill_token_threshold", args.long_prefill_token_threshold),
            "block_size": instance.get("block_size", args.block_size),
            "dtype": dtype,
            "fp": dtype_to_bits[dtype],
            "kv_cache_dtype": kv_cache_dtype,
            "enable_chunked_prefill": instance.get(
                "enable_chunked_prefill", args.enable_chunked_prefill),
            "enable_prefix_caching": instance.get(
                "enable_prefix_caching", args.enable_prefix_caching),
            "enable_session_kv_retention": instance.get(
                "enable_session_kv_retention",
                args.enable_session_kv_retention),
            "session_kv_ttl_ns": instance.get(
                "session_kv_ttl_ns", args.session_kv_ttl_ns),
            "enable_kv_offloading": instance.get(
                "enable_kv_offloading", args.enable_kv_offloading),
            "kv_offload_high_watermark": instance.get(
                "kv_offload_high_watermark", args.kv_offload_high_watermark),
            "kv_offload_low_watermark": instance.get(
                "kv_offload_low_watermark", args.kv_offload_low_watermark),
            "kv_offload_victim_policy": instance.get(
                "kv_offload_victim_policy", args.kv_offload_victim_policy),
            "prioritize_prefill": instance.get("prioritize_prefill", args.prioritize_prefill),
            "enable_local_offloading": instance.get(
                "enable_local_offloading", args.enable_local_offloading),
            "enable_attn_offloading": enable_attn_offloading,
            "enable_sub_batch_interleaving": enable_sub_batch_interleaving,
            "enable_block_copy": instance.get("enable_block_copy", args.enable_block_copy),
        })
        cfg = runtime_configs[-1]
        if cfg["session_kv_ttl_ns"] < 0:
            raise ValueError(
                "Session KV TTL must be non-negative; got "
                f"{cfg['session_kv_ttl_ns']} for instance {instance_id}.")
        high = cfg["kv_offload_high_watermark"]
        low = cfg["kv_offload_low_watermark"]
        if not 0 < low <= high <= 1:
            raise ValueError(
                "KV offload watermarks must satisfy 0 < low <= high <= 1; "
                f"got low={low}, high={high} for instance {instance_id}")
        if cfg["kv_offload_victim_policy"] not in {"lru", "largest-kv"}:
            raise ValueError(
                "Unsupported KV offload victim policy "
                f"'{cfg['kv_offload_victim_policy']}' for instance {instance_id}. "
                "Supported policies: lru, largest-kv.")
        if cfg["enable_kv_offloading"] and cfg["enable_prefix_caching"]:
            raise ValueError(
                "CPU KV offloading Phase 1 does not support prefix caching. "
                "Use --no-enable-prefix-caching or disable it for the instance.")
        if (cfg["enable_session_kv_retention"] and
                cfg["enable_prefix_caching"]):
            raise ValueError(
                "Session KV retention does not support generic prefix "
                "caching. Use --no-enable-prefix-caching or disable it for "
                "the instance.")
        if (cfg["enable_session_kv_retention"] and
                instances[instance_id].get("pd_type") is not None):
            if not cfg["enable_kv_offloading"]:
                raise ValueError(
                    "PD session KV retention requires CPU KV offloading "
                    "on every PD instance so decode-to-prefill reuse has "
                    "modeled D2H/H2D transfers.")
        if cfg["enable_kv_offloading"] and cfg["enable_attn_offloading"]:
            raise ValueError(
                "CPU KV offloading Phase 1 cannot be combined with PIM attention offloading.")
    return runtime_configs


def _validate_kv_offload_node_scope(instances, runtime_configs, prefix_storage):
    """Validate node-shared resources and return nodes using live-KV offload."""
    offload_nodes = {
        instances[instance_id]["node_id"]
        for instance_id, cfg in enumerate(runtime_configs)
        if cfg["enable_kv_offloading"]
    }
    for node_id in offload_nodes:
        pd_instance_ids = [
            instance_id for instance_id, instance in enumerate(instances)
            if (instance["node_id"] == node_id and
                instance["pd_type"] in {"prefill", "decode"})
        ]
        disabled_pd_ids = [
            instance_id for instance_id in pd_instance_ids
            if not runtime_configs[instance_id]["enable_kv_offloading"]
        ]
        if disabled_pd_ids:
            raise ValueError(
                "CPU KV offloading requires every prefill/decode "
                f"instance on node {node_id} to enable offloading; disabled "
                f"instances: {disabled_pd_ids}.")
        session_pd_ids = [
            instance_id for instance_id in pd_instance_ids
            if runtime_configs[instance_id]["enable_session_kv_retention"]
        ]
        disabled_session_pd_ids = [
            instance_id for instance_id in pd_instance_ids
            if not runtime_configs[instance_id]["enable_session_kv_retention"]
        ]
        if session_pd_ids and disabled_session_pd_ids:
            raise ValueError(
                "PD session KV retention requires every prefill/decode "
                f"instance on node {node_id} to enable session retention; "
                f"disabled instances: {disabled_session_pd_ids}.")
        cpu_prefix_ids = [
            instance_id for instance_id, instance in enumerate(instances)
            if (instance["node_id"] == node_id and
                runtime_configs[instance_id]["enable_prefix_caching"] and
                prefix_storage == "CPU")
        ]
        if cpu_prefix_ids:
            raise ValueError(
                "CPU KV offloading cannot share a node with CPU prefix "
                f"cache capacity; node {node_id}, prefix instances: "
                f"{cpu_prefix_ids}.")
    return offload_nodes


def main():
    process_start = perf_counter()
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(prog='python -m serving',
                                     description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, default='configs/cluster/single_node_single_instance.json',
                        help='path to cluster config JSON defining node topology, instance layout, hardware, and memory hierarchy')
    parser.add_argument('--max-num-seqs', type=int, default=128,
                        help='maximum number of sequences in a batch (0 = unlimited)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048,
                        help='maximum number of tokens processed per iteration across all requests (the total token budget). '
                        'With chunked prefill, long inputs are split across iterations; '
                        'without chunked prefill, this effectively caps max input length')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=0,
                        help='per-request token cap per step for chunked prefill (0 = disabled). '
                        'Limits how many tokens a single prefill request consumes per iteration, '
                        'preventing long prompts from monopolizing the token budget. '
                        'When 0, a single prefill can consume the entire budget')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32', 'fp8', 'int8'], default=None,
                        help='model weight data type (vLLM-style). When omitted, defaults to the model config\'s '
                        '``torch_dtype`` (falling back to bfloat16). Overrides only take effect if the profiler '
                        'produced matching data under perf/<hw>/<model>/<variant>/tp<N>/')
    parser.add_argument('--request-routing-policy', type=str, choices=['LOAD', 'RR', 'RAND', 'CUSTOM'], default='LOAD',
                        help='request routing policy across instances: LOAD (vLLM-style weighted least-loaded, default), '
                        'RR (round-robin), RAND (random), CUSTOM (user-defined)')
    parser.add_argument('--expert-routing-policy', type=str,
                        choices=['BALANCED', 'RR', 'RAND', 'CUSTOM'],
                        default='BALANCED',
                        help='expert token routing policy for MoE models: '
                        'BALANCED (default; analytical pigeonhole approximation of '
                        'a trained load-balanced learned gate), '
                        'RR (round-robin), RAND (uniform random per token), '
                        'CUSTOM (user-defined)')
    parser.add_argument('--enable-block-copy', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Replay one transformer block\'s trace across every '
                        'layer instead of re-computing the routing per layer — '
                        'cuts trace-generation time roughly num_hidden_layers× '
                        'on MoE models. Safe with BALANCED (deterministic); '
                        'RR/RAND get a small per-layer variance averaged out. '
                        'Disable only for CUSTOM policies that need faithful '
                        'per-layer variance.')
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True,
                        help='enable prefix caching via RadixAttention to reuse KV cache across requests '
                        'with shared prefixes (default: enabled). Use --no-enable-prefix-caching to disable')
    parser.add_argument('--enable-session-kv-retention',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='retain KV between append-only turns of the same '
                        'agentic session. Supports colocated and '
                        'same-node PD NPU/CPU hits and requires '
                        '--no-enable-prefix-caching')
    parser.add_argument('--session-kv-ttl-ns', type=int, default=0,
                        help='default hard TTL in ns for inactive retained '
                        'session KV; 0 disables time-based expiration')
    parser.add_argument('--enable-kv-offloading', action=argparse.BooleanOptionalAction, default=False,
                        help='enable Phase-1 request-level exclusive KV migration between NPU memory and '
                        'the node-shared CPU DRAM pool. Requires --no-enable-prefix-caching')
    parser.add_argument('--kv-offload-high-watermark', type=float, default=0.90,
                        help='NPU KV usage watermark that triggers CPU eviction (0 < value <= 1)')
    parser.add_argument('--kv-offload-low-watermark', type=float, default=0.80,
                        help='target NPU KV usage watermark after eviction (0 < value <= high watermark)')
    parser.add_argument('--kv-offload-victim-policy', choices=['lru', 'largest-kv'], default='lru',
                        help='request victim policy for CPU KV offloading (lru or largest-kv)')
    parser.add_argument('--enable-chunked-prefill', action=argparse.BooleanOptionalAction, default=True,
                        help='enable chunked prefill to split long prefill requests across multiple iterations, '
                        'matching vLLM v1 behavior (default: enabled). Use --no-enable-chunked-prefill to disable')
    parser.add_argument('--enable-prefix-sharing', action='store_true', default=False,
                        help='enable second-tier prefix cache pooling across instances within a node')
    parser.add_argument('--prefix-storage', type=str, choices=['None', 'CPU', 'CXL'], default='None',
                        help='storage medium for the second-tier prefix cache pool: None (NPU only), CPU, or CXL')
    parser.add_argument('--enable-local-offloading', action='store_true', default=False,
                        help='enable weight offloading to local (NPU) memory. '
                        'Recommended to disable unless weight memory access is not counted in profiling')
    parser.add_argument('--enable-attn-offloading', action='store_true', default=False,
                        help='enable attention computation offloading to PIM (Processing-In-Memory) devices')
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', default=False,
                        help='enable sub-batch interleaving to overlap XPU and PIM computation. '
                        'Requires --enable-attn-offloading')
    parser.add_argument('--prioritize-prefill', action='store_true', default=False,
                        help='prioritize prefill requests over decode requests in scheduling')
    parser.add_argument('--block-size', type=int, default=16,
                        help='KV cache block size in tokens (number of tokens per block)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='path to .jsonl dataset file with request traces. '
                        'If None, requests must be added manually in serving/__main__.py')
    parser.add_argument('--output', type=str, default=None,
                        help='path for per-request CSV output with latency metrics (TTFT, TPOT, ITL). '
                        'If None, results are printed to stdout only. Supports {run_id} placeholder')
    parser.add_argument('--run-id', type=str, default=None,
                        help='unique id for this simulation run. Intermediate ASTRA-Sim inputs are written under '
                        'astra-sim/inputs/runs/<run-id>. If omitted, a process-unique id is generated')
    parser.add_argument('--inputs-root', type=str, default=None,
                        help='override the root directory for generated ASTRA-Sim inputs. Defaults to '
                        'astra-sim/inputs/runs/<run-id>')
    parser.add_argument('--cleanup-inputs', action=argparse.BooleanOptionalAction, default=True,
                        help='remove generated ASTRA-Sim inputs under astra-sim/inputs/runs/<run-id> '
                        'after a successful simulation (default: enabled). Use --no-cleanup-inputs '
                        'to preserve generated trace files, Chakra workloads, and input configs for debugging')
    parser.add_argument('--skip-prefill', action='store_true', default=False,
                        help='skip the prefill phase, running decode only')
    parser.add_argument('--num-reqs', type=int, default=0,
                        help='number of entries (requests or sessions) to load from the dataset. '
                        'For agentic datasets, each entry is a session with multiple sub-requests. '
                        '0 = load all entries')
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='interval in seconds between throughput/memory usage log messages')
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], default='WARNING',
                        help='logging verbosity: WARNING (minimal), INFO (per-iteration details), DEBUG (per-layer memory)')
    parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'fp8'], default='auto',
                        help='KV cache data type: auto (use default profile.csv) or fp8 (use profile_fp8.csv, halves KV cache memory)')
    parser.add_argument('--network-backend', type=str, choices=['analytical', 'ns3'], default='analytical',
                        help='network simulation backend: analytical (fast, default) or ns3 (detailed, WIP)')
    parser.add_argument('--chakra-converter', choices=['in-process', 'subprocess'], default='in-process',
                        help='Chakra conversion mode. in-process avoids launching a Python interpreter for every batch; '
                        'subprocess preserves the legacy reference path')
    parser.add_argument('--workload-transport', choices=['file', 'ipc'], default=None,
                        help='ASTRA-Sim workload control transport. Defaults to ipc for analytical and file for ns3. '
                        'file preserves the legacy stdin/path workflow; ipc sends framed commands over a run-specific Unix domain socket')
    parser.add_argument('--ipc-execution', choices=['direct', 'oracle'], default='direct',
                        help='execution mode used with --workload-transport ipc. direct executes the '
                        'prepared in-memory iteration; oracle materializes and converts every batch '
                        'before extracting a full-graph patch for correctness comparison')
    parser.add_argument('--template-cache-capacity', type=int, default=64,
                        help='maximum number of shape-keyed compute templates retained by '
                        'the serving frontend (default: 64; must be positive)')
    parser.add_argument('--host-timing-output', type=str, default=None,
                        help='optional JSON output for host-side stage timings and counters. '
                        'Supports the {run_id} placeholder')

    args = parser.parse_args()
    if args.template_cache_capacity <= 0:
        parser.error('--template-cache-capacity must be positive')
    if args.workload_transport is None:
        args.workload_transport = (
            'ipc' if args.network_backend == 'analytical' else 'file')
    
    args.run_id = resolve_run_id(args.run_id)
    run_paths = build_run_paths(astra_sim, args.run_id, args.inputs_root)
    args.inputs_root = run_paths.inputs_root
    args.output = _resolve_output_file(args.output, args.run_id)
    args.host_timing_output = _resolve_output_file(
        args.host_timing_output, args.run_id)
    args.workload_ipc_socket = (
        _workload_ipc_socket_path(args.run_id)
        if args.workload_transport == "ipc" else None
    )
    host_timing = HostTimingRecorder()
    host_timing.record_rss_checkpoint("frontend_initialized")

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    print_banner()
    
    _dtype_to_bits = {'float16': 16, 'bfloat16': 16, 'float32': 32, 'fp8': 8, 'int8': 8}
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    dataset=args.dataset
    output_file=args.output
    is_init = not args.skip_prefill
    num_req=args.num_reqs
    log_interval=args.log_interval
    network_backend = args.network_backend
    raw_cluster_config = _load_cluster_config_for_overrides(args.cluster_config)
    raw_instances = list(_iter_raw_instances(raw_cluster_config))
    build_enable_local_offloading = args.enable_local_offloading or any(
        inst.get("enable_local_offloading", False) for inst in raw_instances)
    build_enable_attn_offloading = args.enable_attn_offloading or any(
        inst.get("enable_attn_offloading", False) for inst in raw_instances)
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(
        astra_sim, args.cluster_config, build_enable_local_offloading, build_enable_attn_offloading,
        enable_kv_offloading=args.enable_kv_offloading,
        inputs_root=run_paths.inputs_root)
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    if (args.workload_transport == "file" and
            any(instance.get("dp_group") is not None
                for instance in instances)):
        raise ValueError(
            "Cross-instance DP+EP requires atomic wave submission. Use "
            "--workload-transport ipc --ipc-execution oracle for a fully "
            "materialized Chakra correctness run, or --ipc-execution direct "
            "for template execution.")
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    instance_runtime_configs = _build_instance_runtime_configs(instances, args, _dtype_to_bits)
    any_prefix_caching = any(cfg["enable_prefix_caching"] for cfg in instance_runtime_configs)
    offload_nodes = _validate_kv_offload_node_scope(
        instances, instance_runtime_configs, prefix_storage)
    print_input_config(args=args, runtime_configs=instance_runtime_configs)
    print_markup("[sim.heading]▶ Starting simulation...[/]\n")
    flush.stdout.flush()
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend == 'analytical':
        network=run_paths.network_config
        binary=os.path.join(astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
    elif network_backend == 'ns3':
        network=_prepare_ns3_config(astra_sim, run_paths)
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    else:
        raise NotImplementedError("Only analytical and ns3 network backend are supported")
    if args.workload_transport == "ipc" and network_backend != "analytical":
        raise ValueError("IPC workload transport currently supports only the analytical backend.")
    memory=run_paths.memory_config
    system=run_paths.system_config
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    pool_device = None

    if prefix_storage == "CPU":
        pool_device = Device.CPU
    elif prefix_storage == "CXL":
        pool_device = Device.CXL

    if any_prefix_caching and enable_prefix_sharing and prefix_storage != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []

        def _pool_kv_bytes_per_token(inst_ids):
            """KV bytes per token for a shared pool."""
            kv_shapes = {
                (
                    instances[i]["model_name"],
                    instance_runtime_configs[i]["fp"],
                    instance_runtime_configs[i]["kv_cache_dtype"],
                )
                for i in inst_ids
            }
            if len(kv_shapes) > 1:
                raise RuntimeError(
                    "Shared prefix pool requires instances to share model, "
                    f"dtype, and kv_cache_dtype; got {kv_shapes}"
                )
            model = instances[inst_ids[0]]['model_name']
            cfg = instance_runtime_configs[inst_ids[0]]
            return full_cluster_kv_bytes_per_token(model, cfg["fp"], cfg["kv_cache_dtype"])

        if prefix_storage == 'CPU':
            for i in range(num_prefix_pool):
                if cpu_mem_size[i] > 0:
                    new_prefix_pool = RadixCache(
                                                node_id=0,
                                                device=prefix_storage,
                                                page_size=256,
                                                capacity = cpu_mem_size[i] * GB_TO_BYTE,
                                                kv_size=_pool_kv_bytes_per_token(node2inst_mapping[i]),
                                                enable_kv_cache_events=True)
                    prefix_pools.append(new_prefix_pool)
                else:
                    raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
            # This means one node shares one prefix pool
            prefix_pool_inst_mapping = inst2node_mapping

        elif prefix_storage == 'CXL':
            if cluster["cxl_mem_size"] > 0:
                new_prefix_pool = RadixCache(
                                            node_id=None,
                                            device=prefix_storage,
                                            page_size=1,
                                            capacity = cluster["cxl_mem_size"] * GB_TO_BYTE,
                                            kv_size=_pool_kv_bytes_per_token(list(range(num_instances))),
                                            enable_kv_cache_events=True)
                prefix_pools.append(new_prefix_pool)
                # This means every instance shares the same universal prefix pool (maybe fixed later)
                prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
            else:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    # Phase-1 CPU KV offloading uses one live-KV allocator per node. It is
    # deliberately separate from prefix-cache pools, which are not supported
    # in this phase.
    cpu_kv_pools = {}
    if offload_nodes:
        prefill_nodes = {
            instance["node_id"] for instance in instances
            if (instance["pd_type"] == "prefill" and
                instance["node_id"] in offload_nodes)
        }
        decode_nodes = {
            instance["node_id"] for instance in instances
            if instance["pd_type"] == "decode"
        }
        missing_decode_nodes = sorted(prefill_nodes - decode_nodes)
        if missing_decode_nodes:
            raise ValueError(
                "CPU KV offloading Phase 1 supports only same-node prefill/decode "
                f"disaggregation; nodes without a decode instance: {missing_decode_nodes}")
        cpu_kv_pools = {
            node_id: NodeCPUKVPool(node_id, cpu_mem_size[node_id] * GB_TO_BYTE)
            for node_id in offload_nodes
        }

    schedulers = []
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        
        # Make scheduler for each instance

        inst_cfg = instance_runtime_configs[instance_id]

        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id,
            inst_cfg["max_num_seqs"], inst_cfg["max_num_batched_tokens"],
            instance["num_npus"], instance["tp_size"], instance["pp_size"],
            instance["npu_mem"]["mem_size"], cpu_mem_size[instance["node_id"]],
            inst2npu_mapping[instance_id], instance["pd_type"],
            inst_cfg["fp"], inst_cfg["block_size"], num_req,
            inst_cfg["prioritize_prefill"], inst_cfg["enable_prefix_caching"],
            enable_prefix_sharing, prefix_pool, pool_device, inst_cfg["enable_chunked_prefill"],
            inst_cfg["long_prefill_token_threshold"],
            cxl_mem,
            ep_size=instance.get("ep_total", 1),
            kv_cache_dtype=inst_cfg["kv_cache_dtype"],
            enable_kv_offloading=inst_cfg["enable_kv_offloading"],
            kv_offload_high_watermark=inst_cfg["kv_offload_high_watermark"],
            kv_offload_low_watermark=inst_cfg["kv_offload_low_watermark"],
            kv_offload_victim_policy=inst_cfg["kv_offload_victim_policy"],
            cpu_kv_pool=cpu_kv_pools.get(instance["node_id"]),
            enable_session_kv_retention=inst_cfg[
                "enable_session_kv_retention"],
            session_kv_ttl_ns=inst_cfg["session_kv_ttl_ns"],
        ))

    # Controller for astra-sim process communication
    controller = Controller(total_npu, host_timing=host_timing)
    # Global Request Router
    router = Router(
        num_instances, schedulers, num_req, request_routing_policy,
        same_node_pd_only=bool(cpu_kv_pools))
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None
    # Load requests into router (routed in real-time during simulation)
    if dataset != None:
        router.load_requests(dataset, enable_prefix_caching=any_prefix_caching, is_init=is_init)
    else:
        # Manually adding request (legacy: route all upfront)
        for i in range(16):
            for sched in schedulers:
                sched.add_request([i, sched.model, 64, 128, 0, i % num_instances])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL = log_interval*FREQ
    throughput_scale = FREQ / INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = router.get_first_arrival_time()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    generate_event(int(event_time), inputs_root=run_paths.inputs_root,
                   host_timing=host_timing)
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True, inputs_root=run_paths.inputs_root,
                   cleanup_trace=args.cleanup_inputs,
                   converter_mode=args.chakra_converter,
                   host_timing=host_timing)
    # set first workload file
    workload = get_workload(None, None, event=True, inputs_root=run_paths.inputs_root)
    # run subprocess
    astra_args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        astra_args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        astra_args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        astra_args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    if args.workload_ipc_socket is not None:
        astra_args.append("--workload-ipc-socket="+args.workload_ipc_socket)
    host_timing.record_rss_checkpoint("before_backend_start")
    p = subprocess.Popen(astra_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    host_timing.record_rss_checkpoint("backend_started", backend_pid=p.pid)
    if args.workload_transport == "ipc":
        workload_transport = IpcFileWorkloadTransport(
            args.workload_ipc_socket, process=p, host_timing=host_timing)
        event_graphs = _read_chakra_graph_bundle(
            workload, range(total_npu))
        workload_transport.register_template("event-handler", event_graphs)
        host_timing.record_rss_checkpoint(
            "event_template_registered", backend_pid=p.pid)
    else:
        workload_transport = FileWorkloadTransport(
            controller, p, host_timing=host_timing)
    registered_compute_templates = BoundedTemplateCache(
        args.template_cache_capacity)
    registered_template_ids = {}
    compute_template_rss_recorded = False
    session_rss_thresholds = (10, 50, 100, 300)
    recorded_session_rss_thresholds = set()

    def record_compute_template_rss():
        nonlocal compute_template_rss_recorded
        if compute_template_rss_recorded:
            return
        host_timing.record_rss_checkpoint(
            "compute_template_registered", backend_pid=p.pid)
        compute_template_rss_recorded = True

    def record_session_rss():
        completed = router.terminal_session_count
        for threshold in session_rss_thresholds:
            if (completed >= threshold and
                    threshold not in recorded_session_rss_thresholds):
                host_timing.record_rss_checkpoint(
                    f"terminal_sessions_{threshold}",
                    backend_pid=p.pid,
                    metadata={"terminal_sessions": completed},
                )
                recorded_session_rss_thresholds.add(threshold)

    def get_compute_template(cache_key):
        if cache_key is None:
            return None
        return registered_compute_templates.get(cache_key)

    def cache_compute_template(cache_key, template):
        if cache_key is None:
            return
        evicted = registered_compute_templates.put(cache_key, template)
        if evicted is not None:
            host_timing.increment("template_cache_evictions")
            _, (_, evicted_template) = evicted
            evicted_trace = evicted_template.runtime_trace
            evicted_class = (
                evicted_trace.template_class
                if evicted_trace is not None else "unknown")
            host_timing.increment(
                f"template_cache_evictions_{evicted_class}")
    completion_driven = False
    direct_submission_started = False
    bootstrap_systems_remaining = {
        int(value)
        for encoded in (start_npu_ids, end_npu_ids)
        for value in encoded.split(',')
        if value.strip()
    }
    ready_systems = deque()
    queued_ready_systems = set()
    last_astra_iteration = {system_id: 0 for system_id in range(total_npu)}

    def queue_ready_system(system_id):
        if system_id not in queued_ready_systems:
            ready_systems.append(system_id)
            queued_ready_systems.add(system_id)
            logger.debug("Queued logical wake-up for NPU[%d]", system_id)

    def instance_end_system(instance_id):
        instance = instances[instance_id]
        system_count = instance["num_npus"]
        if instance["pd_type"] == "prefill":
            system_count *= 2
        return inst2npu_mapping[instance_id] + system_count - 1

    # DP group synchronization: defer trace generation until all members have scheduled
    # dp_groups maps dp_group_name -> list of instance_ids
    dp_groups = {}
    for inst in instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst["instance_id"])
    # Reverse lookup: instance_id -> dp_group_name
    inst_dp_group = {}
    for dg, members in dp_groups.items():
        for inst_id in members:
            inst_dp_group[inst_id] = dg
    # Pending batches per DP group (waiting for all members to schedule)
    dp_pending = {dg: {} for dg in dp_groups}  # dp_group -> {instance_id: (new_req, sys)}
    # Direct waves include dummy batches that are intentionally absent from
    # Scheduler.inflight. Track physical completions separately so an idle
    # peer is not rescheduled or put to sleep before its wave has completed.
    dp_active_systems = {dg: set() for dg in dp_groups}
    # Pre-generated workloads ready to submit on next "Waiting"
    dp_ready_workloads = {}  # instance_id -> workload_path
    next_wave_id = 1

    def record_dp_system_completion(system_id):
        completed_instance_id = npu2inst_mapping[system_id]
        completed_dp_group = inst_dp_group.get(completed_instance_id)
        if completed_dp_group is None:
            return
        dp_active_systems[completed_dp_group].discard(system_id)
        completed_start = inst2npu_mapping[completed_instance_id]
        completed_end = instance_end_system(completed_instance_id) + 1
        member_still_active = any(
            active_system in dp_active_systems[completed_dp_group]
            for active_system in range(completed_start, completed_end)
        )
        if (dp_pending[completed_dp_group] and
                not member_still_active):
            queue_ready_system(completed_start)

    def submit_dp_wave(dp_group, responder_instance_id):
        """Submit one complete DP wave through the selected execution path."""
        nonlocal direct_submission_started, next_wave_id

        pending = dp_pending[dp_group]
        if len(pending) != len(dp_groups[dp_group]):
            raise RuntimeError(
                f"DP group {dp_group} is incomplete: "
                f"{len(pending)}/{len(dp_groups[dp_group])} participants")

        max_total_len = max(batch.total_len for batch, _ in pending.values())
        for batch, _ in pending.values():
            _pad_batch_to_max(batch, max_total_len)

        first_instance_id = dp_groups[dp_group][0]
        first_batch = pending[first_instance_id][0]
        workload_name = (
            f'{instances[first_instance_id]["hardware"]}/'
            f'{instances[first_instance_id]["model_name"]}/'
            f'dp_{dp_group}_batch{first_batch.batch_id}'
        )
        prepared_execution = args.workload_transport == "ipc"
        direct_patch_mode = (
            prepared_execution and args.ipc_execution == "direct")

        if prepared_execution:
            wave_runs = []
            # Preserve the legacy wave's deterministic launch order: the
            # controller that completed the barrier fires first, followed by
            # its peers. Equal-timestamp collective callbacks can otherwise
            # differ by a few cycles even though the graph is identical.
            wave_member_ids = [responder_instance_id] + [
                member_instance_id
                for member_instance_id in dp_groups[dp_group]
                if member_instance_id != responder_instance_id
            ]
            for member_instance_id in wave_member_ids:
                batch, member_node_id = pending[member_instance_id]
                member = instances[member_instance_id]
                member_config = instance_runtime_configs[member_instance_id]
                generated_trace = generate_trace(
                    batch, member["hardware"], member["tp_size"],
                    member["pp_size"], member["local_ep"],
                    member["ep_total"], member["pd_type"], member_node_id,
                    member_instance_id,
                    member_config["max_num_batched_tokens"],
                    member_config["max_num_seqs"],
                    placement[member_instance_id],
                    block_mode_on[member_instance_id],
                    expert_routing_policy,
                    member_config["enable_prefix_caching"],
                    member_config["enable_attn_offloading"], power_model,
                    pim_models[member_node_id],
                    member_config["enable_sub_batch_interleaving"],
                    member_config["fp"], dtype=member_config["dtype"],
                    kv_cache_dtype=member_config["kv_cache_dtype"],
                    tp_dim=member.get("tp_dim"),
                    ep_dim=member.get("ep_dim"),
                    dp_sum_total_len=max_total_len,
                    enable_block_copy=member_config["enable_block_copy"],
                    inputs_root=run_paths.inputs_root,
                    kv_offload_cpu=member_config["enable_kv_offloading"],
                    host_timing=host_timing,
                    materialize=not direct_patch_mode,
                    render_text=not direct_patch_mode,
                )
                trace_structure = generated_trace.runtime_trace
                if trace_structure is None:
                    with host_timing.measure("runtime_trace_parsing"):
                        trace_structure = RuntimeTraceSnapshot.parse(
                            generated_trace.ensure_text())
                template_cache_key = (
                    member_instance_id, trace_structure.structure_key)
                registered_template = get_compute_template(
                    template_cache_key)
                cache_result = (
                    "hits" if registered_template is not None else "misses")
                host_timing.increment(f"template_cache_{cache_result}")
                host_timing.increment(
                    f"template_cache_{cache_result}_"
                    f"{trace_structure.template_class}")

                patch = None
                if (direct_patch_mode and
                        registered_template is not None and
                        registered_template[1].supports_trace_patch):
                    template_id, template = registered_template
                    try:
                        with host_timing.measure("batch_patch_generation"):
                            patch = template.build_patch_from_snapshot(
                                template_id, batch.batch_id,
                                trace_structure, host_timing=host_timing)
                        host_timing.increment("direct_trace_patches")
                        host_timing.increment("trace_files_skipped")
                        host_timing.increment("chakra_graphs_skipped")
                    except ValueError as error:
                        logger.warning(
                            "Direct DP trace patch fallback for instance %d: %s",
                            member_instance_id, error)
                        host_timing.increment("direct_trace_patch_fallbacks")

                if patch is None:
                    generated_trace = materialize_generated_trace(
                        generated_trace)
                    generate_graph(
                        batch, member["hardware"], member["num_npus"],
                        member_node_id, member_instance_id,
                        inst2npu_mapping[member_instance_id],
                        member_config["enable_local_offloading"],
                        workload_name=workload_name,
                        inputs_root=run_paths.inputs_root,
                        cleanup_trace=args.cleanup_inputs,
                        converter_mode=args.chakra_converter,
                        host_timing=host_timing,
                    )
                    workload = get_workload(
                        batch, member["hardware"], member_instance_id,
                        workload_name=workload_name,
                        inputs_root=run_paths.inputs_root,
                    )
                    first_system = inst2npu_mapping[member_instance_id]
                    system_ids = range(
                        first_system,
                        first_system + member["num_npus"],
                    )
                    graphs = _read_chakra_graph_bundle(workload, system_ids)
                    if registered_template is None:
                        template = ChakraTemplateBundle(
                            f"instance-{member_instance_id}-"
                            f"{trace_structure.structure_key}",
                            graphs, runtime_trace=trace_structure,
                            host_timing=host_timing,
                        )
                        template_id = registered_template_ids.get(
                            template_cache_key)
                        if template_id is None:
                            template_id = workload_transport.register_template(
                                template.template_key, graphs,
                                bindings=template.bindings,
                            )
                            registered_template_ids[
                                template_cache_key] = template_id
                            record_compute_template_rss()
                            host_timing.increment(
                                f"template_registrations_"
                                f"{trace_structure.template_class}")
                        else:
                            host_timing.increment(
                                "template_cache_rehydrations")
                            host_timing.increment(
                                f"template_cache_rehydrations_"
                                f"{trace_structure.template_class}")
                        registered_template = (template_id, template)
                        cache_compute_template(
                            template_cache_key, registered_template)
                    template_id, template = registered_template
                    with host_timing.measure("batch_patch_generation"):
                        patch = template.build_patch(
                            template_id, batch.batch_id, graphs)

                wave_runs.append((member_instance_id, patch))

            wave_id = next_wave_id
            workload_transport.prepare_wave(wave_id, wave_runs)
            dp_active_systems[dp_group] = {
                system.system_id
                for _, patch in wave_runs
                for system in patch.systems
            }
            next_wave_id += 1
            direct_submission_started = True
            pending.clear()
            return

        for member_instance_id in dp_groups[dp_group]:
            batch, member_node_id = pending[member_instance_id]
            member = instances[member_instance_id]
            member_config = instance_runtime_configs[member_instance_id]
            generate_trace(
                batch, member["hardware"], member["tp_size"],
                member["pp_size"], member["local_ep"], member["ep_total"],
                member["pd_type"], member_node_id, member_instance_id,
                member_config["max_num_batched_tokens"],
                member_config["max_num_seqs"],
                placement[member_instance_id], block_mode_on[member_instance_id],
                expert_routing_policy,
                member_config["enable_prefix_caching"],
                member_config["enable_attn_offloading"], power_model,
                pim_models[member_node_id],
                member_config["enable_sub_batch_interleaving"],
                member_config["fp"], dtype=member_config["dtype"],
                kv_cache_dtype=member_config["kv_cache_dtype"],
                tp_dim=member.get("tp_dim"), ep_dim=member.get("ep_dim"),
                dp_sum_total_len=max_total_len,
                enable_block_copy=member_config["enable_block_copy"],
                inputs_root=run_paths.inputs_root,
                kv_offload_cpu=member_config["enable_kv_offloading"],
                host_timing=host_timing,
            )
            generate_graph(
                batch, member["hardware"], member["num_npus"],
                member_node_id, member_instance_id,
                inst2npu_mapping[member_instance_id],
                member_config["enable_local_offloading"],
                workload_name=workload_name,
                inputs_root=run_paths.inputs_root,
                cleanup_trace=args.cleanup_inputs,
                converter_mode=args.chakra_converter,
                host_timing=host_timing,
            )
        responder_batch = pending[responder_instance_id][0]
        responder = instances[responder_instance_id]
        workload = get_workload(
            responder_batch, responder["hardware"], responder_instance_id,
            workload_name=workload_name, inputs_root=run_paths.inputs_root)
        wave_systems = []
        wave_member_ids = [responder_instance_id] + [
            member_instance_id
            for member_instance_id in dp_groups[dp_group]
            if member_instance_id != responder_instance_id
        ]
        for member_instance_id in wave_member_ids:
            member = instances[member_instance_id]
            member_start = inst2npu_mapping[member_instance_id]
            member_end = instance_end_system(member_instance_id)
            wave_systems.extend(range(member_start, member_end + 1))
            member_batch = pending[member_instance_id][0]
            for submitted_system in range(member_start, member_end + 1):
                if submitted_system not in member_batch.fired:
                    member_batch.fired.append(submitted_system)
        workload_transport.run_wave(workload, wave_systems)
        dp_active_systems[dp_group] = set(wave_systems)
        pending.clear()

    # ----------------------------------- Start simulation loop ------------------------------------
    # Starting simulation, one while loop processes one iteration
    while True:
        reported_completion = False
        if ready_systems:
            ready_system = ready_systems.popleft()
            queued_ready_systems.discard(ready_system)
            logger.debug("Processing logical wake-up for NPU[%d]", ready_system)
            out_dict = {
                'sys': ready_system,
                'id': last_astra_iteration[ready_system],
                'cycle': current,
            }
        elif completion_driven:
            completion = workload_transport.wait_batch_done()
            reported_completion = True
            out_dict = {
                'sys': completion.system_id,
                # Scheduler batch ids start at zero while ASTRA iteration
                # zero belongs to the bootstrap event graph.
                'id': completion.batch_id + 1,
                'cycle': completion.cycles,
            }
            last_astra_iteration[completion.system_id] = out_dict['id']
        else:
            out = controller.read_wait(p)
            out_dict = controller.parse_output(out[-2])
            if out_dict is not None:
                reported_completion = True
                bootstrap_systems_remaining.discard(out_dict['sys'])
                last_astra_iteration[out_dict['sys']] = out_dict['id']
                logger.debug(
                    "Bootstrap report from NPU[%d]; remaining=%s",
                    out_dict['sys'], sorted(bootstrap_systems_remaining))
        
        if out_dict != None:
            sys = out_dict['sys']
            id = out_dict['id']
            current = out_dict['cycle']
            router.observe_concurrency(current)
            workload_transport.set_system(sys)
            if reported_completion:
                record_dp_system_completion(sys)

        # Hard expiry wins when it shares a timestamp with an arrival or a
        # migration completion.
        for scheduler in schedulers:
            scheduler.expire_session_kv(current)

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current,
                        last_end_time[instance_id], last_calc_time[instance_id], num_npus=instances[instance_id]["num_npus"])
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        # An instance can span multiple NPUs. Only update end-time when sys is the first NPU of the instance.
        # waiting_request[instance_id] = True means the instance has no batch to run (idle).
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # check request is done
        inflight_count_before_completion = len(
            schedulers[instance_id].inflight)
        prompt_t, gen_t, finished_reqs = schedulers[instance_id].add_done(id, sys, current)
        batch_completed = (
            len(schedulers[instance_id].inflight) <
            inflight_count_before_completion)
        if (not bootstrap_systems_remaining and batch_completed and
                sys != inst2npu_mapping[instance_id]):
            queue_ready_system(inst2npu_mapping[instance_id])
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(finished_reqs) if instances[instance_id]["pd_type"] != "prefill" else 0

        # Notify router of completed requests for dependency chain release
        if instances[instance_id]["pd_type"] != "prefill":
            for req in finished_reqs:
                router.notify_request_completed(req.id, current)
            record_session_rss()
            if (finished_reqs and not router.has_pending_requests() and
                    not router.has_deferred_sessions()):
                for idle_scheduler in schedulers:
                    if (idle_scheduler.instance_id != instance_id and
                            idle_scheduler.is_request_empty() and
                            not idle_scheduler.inflight):
                        queue_ready_system(inst2npu_mapping[
                            idle_scheduler.instance_id])

        # Add prefill ended requests to decode instance
        if instances[instance_id]["pd_type"] == "prefill" and len(finished_reqs) > 0:
            router.transfer_prefill_request(finished_reqs, current)
            for decode_id in decode_instance:
                queue_ready_system(inst2npu_mapping[decode_id])

        # Apply batch completion and dependency-chain releases before routing
        # equal-time arrivals. This lets a zero-think-time PD continuation see
        # the CPU session record committed by a just-finished D2H migration.
        with host_timing.measure("scheduling_and_routing"):
            if dataset is not None:
                routed_requests = router.route_arrived_requests(current)
                if routed_requests:
                    for routed_scheduler in schedulers:
                        if (routed_scheduler.instance_id != instance_id and
                                not routed_scheduler.inflight and
                                any(request.arrival <= current
                                    for request in routed_scheduler.request)):
                            queue_ready_system(inst2npu_mapping[
                                routed_scheduler.instance_id])

            # schedule requests
            new_req = schedulers[instance_id].schedule(current, sys, id)
        if new_req is not None:
            host_timing.observe("batch_size", len(new_req.requests))
            host_timing.increment(
                f"{new_req.kind.name.lower()}_batches")
        router.observe_concurrency(current)
        responded = False  # track whether we already sent a response to ASTRA-Sim

        # Check if a pre-generated workload is ready for this instance (from DP sync)
        if new_req is None and instance_id in dp_ready_workloads:
            workload_transport.run_batch(dp_ready_workloads.pop(instance_id))
            responded = True
        # DP group: truly idle instance (no inflight batch) — create dummy batch so ALLTOALL syncs
        elif new_req is None and instance_id in inst_dp_group and sys == inst2npu_mapping[instance_id] and len(schedulers[instance_id].inflight) == 0:
            dg = inst_dp_group[instance_id]
            if dp_pending[dg]:
                # Emit a 1-token dummy; the uniform pad-to-max pass below
                # brings it (and any undersized real peers) up to the
                # group's max_total_len, matching vLLM's CUDA-graph DP padding.
                logger.debug(f"Instance {instance_id} is idle but DP group {dg} has pending batches. Creating dummy batch for synchronization.")
                dummy = Batch(schedulers[instance_id].get_batch_id(), instances[instance_id]["model_name"],
                              1, 1, [1], [], 0, 1, [], [], [1], current, 0)
                host_timing.increment("dummy_batches")
                dummy.fired.append(sys)
                dp_pending[dg][instance_id] = (dummy, inst2node_mapping[instance_id])

                if len(dp_pending[dg]) == len(dp_groups[dg]):
                    submit_dp_wave(dg, instance_id)
                    responded = True
                else:
                    workload_transport.pass_system()
                    responded = True
        # runnable batch exists
        elif new_req is not None:
            if sys == inst2npu_mapping[instance_id]:  # first NPU of the instance
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                dg = inst_dp_group.get(instance_id)
                if new_req.kind is not BatchKind.COMPUTE:
                    # CPU KV migration is node-local and must not wait for or
                    # create dummy work in the DP collective barrier.
                    dg = None

                if dg is not None:
                    # DP group: defer trace generation until all members scheduled
                    dp_pending[dg][instance_id] = (new_req, node_id)

                    if len(dp_pending[dg]) == len(dp_groups[dg]):
                        submit_dp_wave(dg, instance_id)
                        responded = True
                    else:
                        # Waiting for other DP members — send pass
                        workload_transport.pass_system()
                        responded = True
                        if not bootstrap_systems_remaining:
                            for peer_instance_id in dp_groups[dg]:
                                if (peer_instance_id not in dp_pending[dg] and
                                        not schedulers[peer_instance_id].inflight and
                                        not any(
                                            system_id in dp_active_systems[dg]
                                            for system_id in range(
                                                inst2npu_mapping[peer_instance_id],
                                                inst2npu_mapping[peer_instance_id] +
                                                instances[peer_instance_id]["num_npus"])
                                        ) and
                                        peer_instance_id not in done_instance):
                                    queue_ready_system(
                                        inst2npu_mapping[peer_instance_id])
                else:
                    # Independent instance: generate trace immediately
                    inst_cfg = instance_runtime_configs[instance_id]
                    prepared_execution = args.workload_transport == "ipc"
                    direct_ipc = (
                        prepared_execution and
                        args.ipc_execution == "direct"
                    )
                    generated_trace = generate_trace(
                                   new_req, instance["hardware"], instance["tp_size"], instance["pp_size"],
                                   instance["local_ep"], instance["ep_total"],
                                   instance["pd_type"],
                                   node_id, instance_id,
                                   inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                   placement[instance_id], block_mode_on[instance_id],
                                   expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                   inst_cfg["enable_attn_offloading"], power_model, pim_models[node_id],
                                   inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                   dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                    tp_dim=instance["tp_dim"], ep_dim=instance["ep_dim"],
                                    enable_block_copy=inst_cfg["enable_block_copy"],
                                    inputs_root=run_paths.inputs_root,
                                    kv_offload_cpu=inst_cfg["enable_kv_offloading"],
                                    host_timing=host_timing,
                                    materialize=not direct_ipc,
                                    render_text=not direct_ipc)

                    trace_structure = None
                    template_cache_key = None
                    if args.workload_transport == "ipc":
                        try:
                            trace_structure = generated_trace.runtime_trace
                            if trace_structure is None:
                                with host_timing.measure("runtime_trace_parsing"):
                                    trace_structure = RuntimeTraceSnapshot.parse(
                                        generated_trace.ensure_text())
                            template_cache_key = (
                                instance_id, trace_structure.structure_key)
                        except ValueError as error:
                            logger.warning(
                                "Runtime trace cannot use a structural template "
                                "for instance %d: %s", instance_id, error)
                    registered_template = get_compute_template(
                        template_cache_key)
                    if template_cache_key is not None:
                        cache_result = (
                            "hits" if registered_template is not None
                            else "misses")
                        host_timing.increment(
                            f"template_cache_{cache_result}")
                        host_timing.increment(
                            f"template_cache_{cache_result}_"
                            f"{trace_structure.template_class}")
                    direct_trace_patch = (
                        direct_ipc and
                        registered_template is not None and
                        registered_template[1].supports_trace_patch
                    )

                    patch = None
                    if direct_trace_patch:
                        template_id, template = registered_template
                        try:
                            with host_timing.measure("batch_patch_generation"):
                                patch = template.build_patch_from_snapshot(
                                    template_id, new_req.batch_id,
                                    generated_trace.runtime_trace,
                                    host_timing=host_timing)
                            host_timing.increment("direct_trace_patches")
                            host_timing.increment("trace_files_skipped")
                            host_timing.increment("chakra_graphs_skipped")
                        except ValueError as error:
                            logger.warning(
                                "Direct trace patch fallback for instance %d: %s",
                                instance_id, error)
                            host_timing.increment("direct_trace_patch_fallbacks")
                            generated_trace = materialize_generated_trace(
                                generated_trace)

                    if patch is None:
                        generated_trace = materialize_generated_trace(
                            generated_trace)
                        generate_graph(new_req, instance["hardware"], instance["num_npus"], node_id,
                                       instance_id, inst2npu_mapping[instance_id],
                                       inst_cfg["enable_local_offloading"],
                                       inputs_root=run_paths.inputs_root,
                                       cleanup_trace=args.cleanup_inputs,
                                       converter_mode=args.chakra_converter,
                                       host_timing=host_timing)
                        workload = get_workload(new_req, instance["hardware"], instance_id,
                                                inputs_root=run_paths.inputs_root)
                        if args.workload_transport == "ipc":
                            first_system = inst2npu_mapping[instance_id]
                            system_ids = list(range(
                                first_system,
                                first_system + instance["num_npus"],
                            ))
                            if instance["pd_type"] == "prefill":
                                system_ids.extend(range(
                                    first_system + instance["num_npus"],
                                    first_system + 2 * instance["num_npus"],
                                ))
                            graphs = _read_chakra_graph_bundle(
                                workload, system_ids)
                            if registered_template is None:
                                structure_suffix = (
                                    trace_structure.structure_key
                                    if trace_structure is not None
                                    else f"batch-{new_req.batch_id}"
                                )
                                template = ChakraTemplateBundle(
                                    f"instance-{instance_id}-{structure_suffix}", graphs,
                                    runtime_trace=generated_trace.runtime_trace,
                                    host_timing=host_timing)
                                template_id = registered_template_ids.get(
                                    template_cache_key)
                                if template_id is None:
                                    template_id = workload_transport.register_template(
                                        template.template_key,
                                        graphs,
                                        bindings=template.bindings,
                                    )
                                    registered_template_ids[
                                        template_cache_key] = template_id
                                    record_compute_template_rss()
                                    host_timing.increment(
                                        f"template_registrations_"
                                        f"{trace_structure.template_class}")
                                else:
                                    host_timing.increment(
                                        "template_cache_rehydrations")
                                    host_timing.increment(
                                        f"template_cache_rehydrations_"
                                        f"{trace_structure.template_class}")
                                registered_template = (template_id, template)
                                if template_cache_key is not None:
                                    cache_compute_template(
                                        template_cache_key,
                                        registered_template)
                            template_id, template = registered_template
                            with host_timing.measure("batch_patch_generation"):
                                patch = template.build_patch(
                                    template_id, new_req.batch_id, graphs)

                    if args.workload_transport == "ipc":
                        workload_transport.prepare_batch(
                            instance_id, patch, execute=prepared_execution)
                        if prepared_execution:
                            direct_submission_started = True
                    if not prepared_execution:
                        workload_transport.run_batch(workload)
                        end_system = instance_end_system(instance_id)
                        for submitted_system in range(
                                inst2npu_mapping[instance_id] + 1,
                                end_system + 1):
                            if submitted_system not in new_req.fired:
                                new_req.fired.append(submitted_system)
            elif new_req is not None:
                # Non-first NPU: pick up existing batch workload
                workload = get_workload(new_req, instances[instance_id]["hardware"], instance_id,
                                        inputs_root=run_paths.inputs_root)
                prepared_execution = args.workload_transport == "ipc"
                if prepared_execution:
                    workload_transport.pass_system()
                else:
                    workload_transport.run_batch(workload)

        # check time to store throughput (only print on start NPU to avoid transient states)
        if current > last_log + INTERVAL and sys == inst2npu_mapping[instance_id]:
            # store the prompt
            throughput.append((prompt_th * throughput_scale, gen_th * throughput_scale))
            last_log += INTERVAL
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            # Heartbeat timestamp stays in the terminal's default
            # colour — bright enough to scan, not so dim that it
            # disappears. (The per-log-record [HH:MM:SS.mmm] stays
            # dim via sim.time because it appears every other line.)
            print_markup(
                f"{log_time_str} "
                f"[blue]Avg prompt throughput: {prompt_th * throughput_scale:.1f} tokens/s,[/] "
                f"[blue]Avg generation throughput: {gen_th * throughput_scale:.1f} tokens/s[/]"
            )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                running_reqs = sum(len(batch.requests) for batch in schedulers[inst_id].inflight)
                waiting_reqs = len([req for req in schedulers[inst_id].request if req.arrival <= current])

                mem = schedulers[inst_id].memory
                npu_used_mb = mem.npu_used / MB_TO_BYTE
                npu_reserved_mb = mem.npu_reserved / MB_TO_BYTE
                npu_util = (
                    (mem.npu_used + mem.npu_reserved) / mem.npu_mem * 100.0
                    if mem.npu_mem else 0.0)

                line = (
                    f"{log_indent+tree_indent}Running Instance\\[{inst_id}]: "
                    f"{running_reqs} reqs, Waiting: {waiting_reqs} reqs, "
                    f"Total # {schedulers[inst_id].num_npus} NPUs, "
                    f"Each NPU Memory Used/Reserved "
                    f"{npu_used_mb:.2f}/{npu_reserved_mb:.2f} MB "
                    f"({npu_util:.3f} % Committed)"
                )
                if schedulers[inst_id].enable_prefix_caching:
                    line += schedulers[inst_id].memory.npu_prefix_cache.format_prefix_info()
                print_markup(line)

            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    node_cpu_reserved = 0
                    inst_usage = []
                    if node_id in cpu_kv_pools:
                        node_cpu_usage = cpu_kv_pools[node_id].used
                        node_cpu_reserved = cpu_kv_pools[node_id].reserved
                    elif any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        node_cpu_usage = prefix_pools[node_id].total_size() * prefix_pools[node_id].kv_size
                    else:
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].memory.cpu_used
                            node_cpu_usage += inst_cpu_usage
                            node_cpu_reserved += schedulers[inst_id].memory.cpu_reserved
                            inst_usage.append(inst_cpu_usage)

                    cpu_util = (
                        (node_cpu_usage + node_cpu_reserved) /
                        (cpu_mem_size[node_id] * GB_TO_BYTE) * 100)
                    if prefix_storage != "CXL" and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    line = (
                        f"{log_indent+tree_indent}Node\\[{node_id}]: "
                        f"Total CPU Memory Used/Reserved "
                        f"{node_cpu_usage/MB_TO_BYTE:.2f}/"
                        f"{node_cpu_reserved/MB_TO_BYTE:.2f} MB, "
                        f"{cpu_util:.3f} % Committed "
                    )
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        line += prefix_pools[node_id].format_prefix_info()

                    if (any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU") or (len(inst_ids) == 1):
                        print_markup(line)
                    else:
                        parts = []
                        for j, inst_cpu_usage in enumerate(inst_usage):
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            parts.append(f"Instance\\[{inst_ids[j]}]: {inst_cpu_util:.2f} %")
                        print_markup(line + "(" + ", ".join(parts) + ")")

            ######### Per CXL Metrics #########
            if any_prefix_caching and prefix_storage == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = cxl_pool.total_size() * cxl_pool.kv_size
                        cxl_util = cxl_usage / cxl_pool.capacity
                        if not power_modeling and cxl_id == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[{cxl_id}]: "
                            f"Total CXL Device Memory Usage "
                            f"{cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used"
                        )
                else:
                    enabled_inst_ids = [
                        inst_id for inst_id, sched in enumerate(schedulers)
                        if sched.enable_prefix_caching
                    ]
                    for pos, inst_id in enumerate(enabled_inst_ids):
                        second_tier = getattr(
                            schedulers[inst_id].memory, "second_tier_prefix_cache", None)
                        if second_tier is None:
                            continue
                        cxl_usage = second_tier.total_size() * second_tier.kv_size
                        cxl_util = cxl_usage / second_tier.capacity
                        if not power_modeling and pos == len(enabled_inst_ids) - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[0]/Instance\\[{inst_id}]: "
                            f"Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, "
                            f"{cxl_util:.3f} % Used"
                        )

            ######### Power Modeling #########
            if power_modeling:
                tree_indent = '└─'
                print_markup(
                    f"{log_indent+tree_indent}"
                    f"Avg power consumption: {power_model.get_current_power(current)} W"
                )
        # check if all requests are done for current instance#
        # NOTE: 'instance_id' could occur in duplicate, because 'npu2inst_mapping[sys]' is not one-to-one mapping
        if (instance_id not in decode_instance or is_prefill_done) and instance_id not in done_instance and schedulers[instance_id].is_request_empty() and not router.has_pending_requests() and not router.has_deferred_sessions():
            # For DP groups: only mark done when ALL members of the group are empty
            dg = inst_dp_group.get(instance_id)
            if dg is not None:
                all_dp_empty = all(
                    schedulers[inst_id].is_request_empty() and len(schedulers[inst_id].inflight) == 0
                    for inst_id in dp_groups[dg]
                ) and not dp_active_systems[dg]
                if not all_dp_empty:
                    # Other DP members still have work — keep this instance alive for dummy waves
                    if not responded:
                        workload_transport.pass_system()
                    flush.stdout.flush()
                    continue
                if not bootstrap_systems_remaining:
                    for peer_instance_id in dp_groups[dg]:
                        if peer_instance_id in done_instance:
                            continue
                        peer_start = inst2npu_mapping[peer_instance_id]
                        peer_end = (
                            peer_start +
                            instances[peer_instance_id]["num_npus"] - 1)
                        for peer_system in {peer_start, peer_end}:
                            if (peer_system != sys and
                                    peer_system not in
                                    done_inst_npus[peer_instance_id]):
                                queue_ready_system(peer_system)

            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)
            required_boundary_reports = (
                1 if instances[instance_id]["num_npus"] == 1 else 2)
            if (len(done_inst_npus[instance_id]) <
                    required_boundary_reports and
                    sys == inst2npu_mapping[instance_id]):
                # In TP, the end rank can finish just before the controller
                # rank makes the batch terminal. Give that already-reported
                # boundary one logical wake-up so shutdown can account for
                # both sides without waiting for a duplicate C++ report.
                queue_ready_system(instance_end_system(instance_id))
            if (len(done_inst_npus[instance_id]) ==
                    required_boundary_reports):
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True
                for decode_id in decode_instance:
                    if decode_id not in done_instance:
                        queue_ready_system(inst2npu_mapping[decode_id])

            # check if all instances are done
            if len(done_instance) == num_instances:
                for inst_idx in range(num_instances):
                    schedulers[inst_idx].memory.free_prefix_cache()
                    schedulers[inst_idx].memory.free_weight()
                
                # check memory leak before exit
                schedulers[inst_idx].memory.is_free()

                print_rule()
                print_markup("[sim.heading]▶ Exiting simulation...[/]\n")
                workload_transport.close()
                break
            workload_transport.sleep_system() # make done instances to sleep
        elif new_req == None and not responded:
            # If all instances are idle but deferred sessions have pending
            # requests with future arrival times (tool calls still running),
            # advance current time so the next iteration can pick them up.
            next_event = _next_idle_event(router, schedulers)
            globally_idle = (
                all(not scheduler.inflight for scheduler in schedulers) and
                all(not any(request.arrival <= current
                            for request in scheduler.request)
                    for scheduler in schedulers)
            )
            if (globally_idle and next_event is not None and
                    next_event > current):
                current = next_event
                router.observe_concurrency(current)
                for scheduler in schedulers:
                    scheduler.expire_session_kv(current)
                workload_transport.advance_time(current)
                queue_ready_system(sys)
            else:
                workload_transport.pass_system()
        
        # flush
        flush.stdout.flush()
        if (direct_submission_started and not completion_driven and
                not bootstrap_systems_remaining):
            controller.start_drain(p)
            completion_driven = True

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    host_timing.record("simulation_loop", total_time)
    host_timing.record_rss_checkpoint(
        "simulation_loop_complete", backend_pid=p.pid,
        metadata={"terminal_sessions": router.terminal_session_count},
    )
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)

    # calcuate prefix caching metrics
    total_requested_tokens = 0
    total_npu_hit_tokens = 0
    total_cpu_hit_tokens = 0
    if any_prefix_caching:
        for i in range(num_instances):
            if not schedulers[i].enable_prefix_caching:
                continue
            (temp_npu_a, temp_npu_b), (temp_cpu_a, temp_cpu_b) = schedulers[i].memory.return_prefix_info()
            if (not enable_prefix_sharing) and (prefix_storage != "None") and (temp_npu_a != temp_cpu_a):
                raise RuntimeError(f"Instance[{i}] prefix caching requested tokens mismatch between NPU ({temp_npu_a}) and CPU ({temp_cpu_a})")
            total_requested_tokens += temp_npu_a
            total_npu_hit_tokens += temp_npu_b
            if not enable_prefix_sharing:
                total_cpu_hit_tokens += temp_cpu_b
        
        if enable_prefix_sharing:
            for pool in prefix_pools:
                _, temp_cpu_b = pool.return_prefix_info()
                total_cpu_hit_tokens += temp_cpu_b
    
    # This is total system's throughput
    total_latency = current/FREQ
    print_rule()
    print_markup("[sim.heading]▶ Simulation results...[/]\n")
    print_markup(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print_rule("[sim.tagline]Throughput Results[/]")
    print_markup(f"Total requests:                                                     {req_cnt}")
    print_markup(f"Total clocks (ns):                                                  {current}")
    print_markup(f"Total latency (s):                                                  {total_latency:.3f}")
    print_markup(f"Total input tokens:                                                 {total_prompt}")
    print_markup(f"Total generated tokens:                                             {total_gen}")
    print_markup(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print_markup(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print_markup(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print_markup(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print_markup(f"Throughput per {log_interval:g} sec (\\[prompt_throughput], \\[gen_throughput]): {throughput}")
    print_rule()
    if any_prefix_caching:
        print_rule("[sim.tagline]Prefix Caching Results[/]")
        print_markup(f"Total requested prompt tokens:                                      {total_requested_tokens}")
        print_markup(f"NPU prefix hit prompt tokens:                                       {total_npu_hit_tokens}")
        if total_requested_tokens > 0:
            print_markup(f"NPU prefix hit ratio (%):                                           {(total_npu_hit_tokens/total_requested_tokens)*100:.2f}")
            if prefix_storage != "None":
                print_markup(f"{prefix_storage} prefix hit prompt tokens:                                       {total_cpu_hit_tokens}")
                print_markup(f"{prefix_storage} prefix hit ratio (%):                                           {(total_cpu_hit_tokens/total_requested_tokens)*100:.2f}")
            print_markup(f"Total prefix hit ratio (%):                                         {((total_npu_hit_tokens+total_cpu_hit_tokens)/total_requested_tokens)*100:.2f}")
        else:
            print_markup("NPU prefix hit ratio (%):                                           N/A (no requests tracked)")
        print_rule()
    if power_modeling:
        print_rule("[sim.tagline]Power Modeling Results[/]")
        total_energy = power_model.get_final_energy(current)
        print_markup(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print_markup(f"Power per {log_interval:g} sec (W): {power_model.power_time_series}")
        print_rule()
    # Each instacne results
    for i in range(num_instances):
        print_rule(f"[sim.tagline]Instance \\[{i}][/]")
        schedulers[i].print_result()
        print_rule()
    
    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i in range(num_instances):
            schedulers[i].save_output(output_file, is_append=False if i == 0 else True)
        offload_schedulers = [
            schedulers[i] for i, cfg in enumerate(instance_runtime_configs)
            if (cfg["enable_kv_offloading"] or
                cfg["enable_session_kv_retention"])
        ]
        if offload_schedulers:
            kv_offload_output = _kv_offload_output_file(output_file)
            print(
                "Saving instance-level KV migration/session metrics to output file: "
                f"{kv_offload_output}")
            for i, scheduler in enumerate(offload_schedulers):
                scheduler.save_kv_offload_output(
                    kv_offload_output, is_append=i != 0)

    if args.cleanup_inputs:
        with host_timing.measure("input_cleanup"):
            _cleanup_inputs_root(run_paths, logger)

    if args.host_timing_output is not None:
        host_timing.record("frontend_elapsed", perf_counter() - process_start)
        conversion_count = host_timing.sample_count("chakra_conversion")
        host_timing.increment("chakra_conversions", conversion_count)
        host_timing.increment(
            "converter_process_launches",
            conversion_count if args.chakra_converter == "subprocess" else 0,
        )
        host_timing.write_json(
            args.host_timing_output,
            metadata={
                "run_id": args.run_id,
                "network_backend": network_backend,
                "chakra_converter": args.chakra_converter,
                "workload_transport": args.workload_transport,
                "ipc_execution": args.ipc_execution,
                "total_clocks_ns": current,
                "simulated_seconds": total_latency,
                "completed_requests": req_cnt,
                "input_tokens": total_prompt,
                "generated_tokens": total_gen,
                "workload_metrics": router.concurrency_summary(current),
            },
        )
        print(f"Saving host timing summary to: {args.host_timing_output}")
    

if __name__ == "__main__": 
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
