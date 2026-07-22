import bisect
import pandas as pd
from time import time
import csv
import os
from dataclasses import asdict, dataclass

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .graph_generator import *
from .trace_generator import *
from .logger import print_markup, print_rule
from .pim_model import *
import numpy as np


@dataclass
class KVOffloadStats:
    preemption_count: int = 0
    evict_bytes: int = 0
    reload_bytes: int = 0
    eviction_batches: int = 0
    reload_batches: int = 0
    reload_stall_count: int = 0
    reload_stall_ns: int = 0
    migration_time_ns: int = 0
    eviction_time_ns: int = 0
    reload_time_ns: int = 0
    npu_peak_used_bytes_per_rank: int = 0
    npu_peak_reserved_bytes_per_rank: int = 0
    cpu_peak_used_bytes: int = 0
    cpu_peak_reserved_bytes: int = 0
    pd_handoff_count: int = 0
    pd_handoff_bytes: int = 0
    pd_handoff_wait_ns: int = 0
    session_npu_hit_count: int = 0
    session_npu_hit_tokens: int = 0
    session_cpu_hit_count: int = 0
    session_cpu_hit_tokens: int = 0
    session_cpu_reload_bytes: int = 0
    session_cpu_reload_time_ns: int = 0
    session_miss_count: int = 0
    session_recomputed_prompt_tokens: int = 0
    session_ttl_expiration_count: int = 0
    session_ttl_npu_bytes_freed: int = 0
    session_ttl_cpu_bytes_freed: int = 0
    session_capacity_drop_count: int = 0
    session_capacity_drop_bytes: int = 0
    session_parked_npu_byte_ns: int = 0
    session_parked_cpu_byte_ns: int = 0
    session_peak_parked_npu_bytes_per_rank: int = 0
    session_peak_parked_cpu_bytes: int = 0
    session_reload_wait_ns: int = 0
    session_current_parked: int = 0
    session_terminal_cleanup_count: int = 0
    session_terminal_cleanup_bytes: int = 0


@dataclass
class PendingPDHandoff:
    request: Request
    source_scheduler: object
    submit_time_ns: int

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, node_id, instance_id, max_num_seqs, max_num_batched_tokens,
                 num_npus, tp_size, pp_size, npu_mem, cpu_mem,
                 start_npu, pd_type, fp, block_size, req_num,
                 prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, enable_chunked_prefill=False,
                 long_prefill_token_threshold=0, cxl_mem=0, ep_size=1, kv_cache_dtype='auto',
                 enable_kv_offloading=False, kv_offload_high_watermark=0.90,
                 kv_offload_low_watermark=0.80, kv_offload_victim_policy='lru',
                 cpu_kv_pool=None, enable_session_kv_retention=False,
                 session_kv_ttl_ns=0):
        self.model = model
        self.config = get_config(model)
        self.node_id = node_id
        self.instance_id = instance_id
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = min(max_num_batched_tokens, self.config['max_position_embeddings'])
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefix_storage = prefix_storage
        self.prioritize_prefill = prioritize_prefill
        self.enable_kv_offloading = enable_kv_offloading
        self.kv_offload_high_watermark = kv_offload_high_watermark
        self.kv_offload_low_watermark = kv_offload_low_watermark
        self.kv_offload_victim_policy = kv_offload_victim_policy
        self.enable_session_kv_retention = enable_session_kv_retention
        self.session_kv_ttl_ns = session_kv_ttl_ns
        # lists are sorted in arrival time manner
        self.request = []
        self.inflight = []
        self.done = []
        self.pending_pd_handoffs = []
        self.session_kv_states = {}
        self.batch_ids = -1
        self.migration_ids = -1

        # memory model
        self.memory = MemoryModel(model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem, ep_size=ep_size, pp_size=pp_size, kv_cache_dtype=kv_cache_dtype, cpu_kv_pool=cpu_kv_pool)

        self.kv_offload_stats = KVOffloadStats()
        self._record_kv_occupancy()

        # logger
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)
    
 
    def schedule(self, current, sys, batch_id=-1):
        if self.enable_prefix_caching:
            return self.schedule_with_prefix(current, sys, batch_id)
        else:
            return self.schedule_base(current, sys, batch_id)

    def enqueue_pd_handoff(self, req, source_scheduler, submit_time_ns):
        """Queue a completed prefill without changing ownership or capacity."""
        if self.pd_type != "decode":
            raise RuntimeError(
                f"PD handoff target instance {self.instance_id} is not a decode instance.")
        if source_scheduler.pd_type != "prefill":
            raise RuntimeError(
                f"PD handoff source instance {source_scheduler.instance_id} is not a prefill instance.")
        if self.node_id != source_scheduler.node_id:
            raise RuntimeError(
                "CPU KV offloading supports same-node PD handoff only: "
                f"source node {source_scheduler.node_id}, destination node {self.node_id}.")
        if req.kv_owner_instance_id != source_scheduler.instance_id:
            raise RuntimeError(
                f"Request #{req.id} is owned by instance "
                f"{req.kv_owner_instance_id}, not source instance "
                f"{source_scheduler.instance_id}.")
        if not req.is_kv_on_npu():
            raise RuntimeError(
                f"Request #{req.id} PD handoff requires NPU-resident source KV.")
        if any(item.request.id == req.id for item in self.pending_pd_handoffs):
            raise ValueError(f"Request #{req.id} already has a pending PD handoff.")
        self.pending_pd_handoffs.append(PendingPDHandoff(
            request=req,
            source_scheduler=source_scheduler,
            submit_time_ns=submit_time_ns,
        ))

    def _commit_pd_handoff(self, handoff, current):
        """Atomically transfer live-KV capacity and ownership to this instance."""
        req = handoff.request
        source = handoff.source_scheduler
        source_size = source.memory.get_total_kv(req)
        destination_size = self.memory.get_total_kv(req)
        self.memory.reserve_live_kv(destination_size, Device.NPU)
        destination_committed = False
        source_released = False
        owner_transferred = False
        try:
            self.memory.commit_live_kv_reservation(destination_size, Device.NPU)
            destination_committed = True
            source.memory.free(source_size, Device.NPU)
            source_released = True
            req.transfer_kv_owner(source.instance_id, self.instance_id)
            owner_transferred = True
            bisect.insort(self.request, req, key=lambda item: (item.arrival, item.id))
            self.pending_pd_handoffs.pop(0)
        except Exception:
            self.request = [item for item in self.request if item is not req]
            if owner_transferred:
                req.transfer_kv_owner(self.instance_id, source.instance_id)
            if source_released:
                source.memory.allocate(source_size, Device.NPU)
            if destination_committed:
                self.memory.free(destination_size, Device.NPU)
            else:
                self.memory.cancel_live_kv_reservation(destination_size, Device.NPU)
            raise

        stats = self._get_kv_offload_stats()
        stats.pd_handoff_count += 1
        stats.pd_handoff_bytes += destination_size * self.num_npus
        stats.pd_handoff_wait_ns += max(0, current - handoff.submit_time_ns)
        source._record_kv_occupancy()
        self._record_kv_occupancy()
        self.logger.info(
            "Admitted PD handoff for request #%d from instance %d, %.2fMB per rank",
            req.id, source.instance_id, destination_size / MB_TO_BYTE)

    def _admit_pending_pd_handoff(self, current, sys):
        """Admit one handoff or submit the eviction needed to make it fit."""
        if not self.pending_pd_handoffs or self.inflight:
            return None
        handoff = self.pending_pd_handoffs[0]
        handoff_size = self.memory.get_total_kv(handoff.request)
        projected = self.memory.npu_used + self.memory.npu_reserved + handoff_size
        high_limit = self.memory.npu_mem * self.kv_offload_high_watermark
        low_limit = self.memory.npu_mem * self.kv_offload_low_watermark

        selected_victims = []
        if self.enable_kv_offloading and projected > high_limit:
            candidates = [
                req for req in self.request
                if (req.arrival <= current and not req.is_prefill() and
                    req.is_kv_on_npu())
            ]
            cpu_available = self._cpu_kv_capacity_with_session_drops()
            selected_cpu_bytes = 0
            while projected > low_limit and candidates:
                victim = self._select_offload_victim(candidates)
                candidates.remove(victim)
                victim_size = self.memory.get_evict_kv(victim)
                victim_cpu_size = victim_size * self.num_npus
                if victim_size <= 0:
                    continue
                if selected_cpu_bytes + victim_cpu_size > cpu_available:
                    continue
                selected_victims.append(victim)
                selected_cpu_bytes += victim_cpu_size
                projected -= victim_size

        if selected_victims:
            return self._start_kv_eviction(selected_victims, current, sys)

        if not self.memory.is_avail(handoff_size, Device.NPU):
            return None

        self._commit_pd_handoff(handoff, current)
        return None

    def _park_completed_session_kv(self, req, completion_time_ns):
        """Transfer a completed non-terminal turn's allocation to its session."""
        if not self.enable_session_kv_retention:
            raise RuntimeError("Session KV retention is not enabled.")
        if self.pd_type == "prefill":
            raise RuntimeError(
                "A PD prefill instance cannot park completed session KV.")
        if req.session_id in self.session_kv_states:
            raise RuntimeError(
                f"Session {req.session_id} already owns a parked KV state.")
        bytes_per_rank = self.memory.get_evict_kv(req)
        ttl_ns = (
            req.session_kv_ttl_ns
            if req.session_kv_ttl_ns is not None
            else self.session_kv_ttl_ns
        )
        expires_at_ns = None
        if ttl_ns is not None and ttl_ns > 0:
            expires_at_ns = completion_time_ns + ttl_ns
        req.transfer_kv_to_session()
        state = SessionKVState(
            session_id=req.session_id,
            model_name=req.model,
            sub_request_index=req.sub_request_index,
            source_request_id=req.id,
            cached_tokens=req.num_computed_tokens,
            bytes_per_rank=bytes_per_rank,
            bytes_full_cluster=bytes_per_rank * self.num_npus,
            residency=req.kv_residency,
            owner_node_id=self.node_id,
            owner_instance_id=self.instance_id,
            num_npus=self.num_npus,
            tp_size=self.tp_size,
            block_size=self.memory.block_size,
            kv_fp=self.memory.kv_fp,
            parked_at_ns=completion_time_ns,
            expires_at_ns=expires_at_ns,
            last_access_ns=completion_time_ns,
            residency_since_ns=completion_time_ns,
            mandatory_cpu_offload=self.pd_type == "decode",
        )
        self.session_kv_states[req.session_id] = state
        self._record_session_occupancy()
        self._record_kv_occupancy()
        return state

    def _claim_parked_session_kv(self, req):
        """Claim a compatible colocated NPU state for suffix-only prefill."""
        state = self.session_kv_states.get(req.session_id)
        if state is None:
            return 0
        if state.invalidated:
            return 0
        if (state.expires_at_ns is not None and
                req.arrival >= state.expires_at_ns):
            self._release_session_kv(
                req.session_id, req.arrival, reason="ttl")
            return 0
        expected_index = state.sub_request_index + 1
        if req.sub_request_index != expected_index:
            raise ValueError(
                f"Session {req.session_id} expected turn {expected_index}, "
                f"got {req.sub_request_index}.")
        pd_cpu_claim = (
            self.pd_type == "prefill" and
            state.residency is KVResidency.CPU and
            state.owner_node_id == self.node_id
        )
        layout_compatible = (
            (state.num_npus == self.num_npus and
             state.tp_size == self.tp_size) or
            pd_cpu_claim
        )
        compatible = (
            state.model_name == req.model and
            state.owner_node_id == self.node_id and
            state.owner_instance_id == self.instance_id and
            layout_compatible and
            state.block_size == self.memory.block_size and
            state.kv_fp == self.memory.kv_fp and
            state.residency in {
                KVResidency.NPU,
                KVResidency.CPU,
                KVResidency.NPU_TO_CPU,
            }
        )
        if not compatible:
            self._release_session_kv(req.session_id, req.arrival)
            return 0

        if req.reused_prefix_toks is not None:
            reused_tokens = req.reused_prefix_toks
        elif req.reuse_previous_kv:
            reused_tokens = min(state.cached_tokens, req.original_input)
        else:
            reused_tokens = 0
        if reused_tokens > state.cached_tokens:
            raise ValueError(
                f"Session {req.session_id} requests {reused_tokens} reused "
                f"tokens but only {state.cached_tokens} are retained.")
        if reused_tokens == 0:
            self._release_session_kv(req.session_id, req.arrival)
            return 0

        num_blocks = (
            reused_tokens + self.memory.block_size - 1
        ) // self.memory.block_size
        claimed_bytes = self.memory.get_kv(
            num_blocks * self.memory.block_size)
        claim_full_cluster = claimed_bytes * self.num_npus
        if ((not pd_cpu_claim and claimed_bytes > state.bytes_per_rank) or
                (pd_cpu_claim and
                 claim_full_cluster > state.bytes_full_cluster)):
            raise RuntimeError(
                f"Session {req.session_id} claim exceeds retained allocation: "
                f"{claim_full_cluster} > {state.bytes_full_cluster} "
                "full-cluster bytes.")
        req.num_computed_tokens = reused_tokens
        req.prefix_cache_hit = reused_tokens
        req.session_kv_hit_tokens = reused_tokens
        if state.residency in {
                KVResidency.CPU, KVResidency.NPU_TO_CPU}:
            req.pending_session_kv = True
            req.pending_session_kv_bytes_per_rank = claimed_bytes
            req.session_kv_hit_tier = "CPU_PENDING"
            return reused_tokens

        excess_bytes = state.bytes_per_rank - claimed_bytes
        if excess_bytes:
            self.memory.free(excess_bytes, Device.NPU)
        self.session_kv_states.pop(req.session_id)
        self._account_session_residency(state, req.arrival)
        req.session_kv_hit_tier = "NPU"
        self._record_session_occupancy()
        self._record_kv_occupancy()
        return reused_tokens

    def _account_session_residency(self, state, current_time_ns):
        duration_ns = max(0, current_time_ns - state.residency_since_ns)
        stats = self._get_kv_offload_stats()
        if state.residency is KVResidency.NPU:
            stats.session_parked_npu_byte_ns += (
                state.bytes_per_rank * duration_ns)
        elif state.residency is KVResidency.CPU:
            stats.session_parked_cpu_byte_ns += (
                state.bytes_full_cluster * duration_ns)
        state.residency_since_ns = current_time_ns

    def _record_session_occupancy(self):
        stats = self._get_kv_offload_stats()
        if not hasattr(self, "session_kv_states"):
            stats.session_current_parked = 0
            return
        npu_bytes = sum(
            state.bytes_per_rank for state in self.session_kv_states.values()
            if state.residency is KVResidency.NPU and not state.invalidated)
        cpu_bytes = sum(
            state.bytes_full_cluster for state in self.session_kv_states.values()
            if state.residency is KVResidency.CPU and not state.invalidated)
        stats.session_peak_parked_npu_bytes_per_rank = max(
            stats.session_peak_parked_npu_bytes_per_rank, npu_bytes)
        stats.session_peak_parked_cpu_bytes = max(
            stats.session_peak_parked_cpu_bytes, cpu_bytes)
        stats.session_current_parked = sum(
            not state.invalidated
            for state in self.session_kv_states.values())

    def _release_session_kv(self, session_id, current_time_ns=None, reason=None):
        """Drop one parked state and release its currently owned allocation."""
        state = self.session_kv_states.get(session_id)
        if state is None:
            return None
        if state.residency not in {KVResidency.NPU, KVResidency.CPU}:
            raise RuntimeError(
                f"Cannot release migrating session {session_id} before its "
                "transfer completes.")
        self.session_kv_states.pop(session_id)
        if current_time_ns is None:
            current_time_ns = state.last_access_ns
        self._account_session_residency(state, current_time_ns)
        if state.residency is KVResidency.NPU:
            self.memory.free(state.bytes_per_rank, Device.NPU)
        elif state.residency is KVResidency.CPU:
            if self.memory.cpu_kv_pool is not None:
                self.memory.cpu_kv_pool.unregister_parked_session(
                    self, session_id)
            self.memory.free(state.bytes_full_cluster, Device.CPU)
        stats = self._get_kv_offload_stats()
        if reason == "ttl":
            stats.session_ttl_expiration_count += 1
            if state.residency is KVResidency.NPU:
                stats.session_ttl_npu_bytes_freed += state.bytes_per_rank
            else:
                stats.session_ttl_cpu_bytes_freed += state.bytes_full_cluster
        elif reason == "capacity":
            stats.session_capacity_drop_count += 1
            stats.session_capacity_drop_bytes += state.bytes_full_cluster
        self._record_session_occupancy()
        self._record_kv_occupancy()
        return state

    def _drop_cpu_session_for_capacity(self, session_id, current_time_ns):
        state = self.session_kv_states.get(session_id)
        if state is None or state.residency is not KVResidency.CPU:
            return 0
        bytes_freed = state.bytes_full_cluster
        self._reset_pending_session_claim(session_id)
        self._release_session_kv(
            session_id, current_time_ns, reason="capacity")
        return bytes_freed

    def _adopt_pd_cpu_session(self, session_id):
        """Move a node-shared CPU session record from decode to prefill."""
        if self.pd_type != "prefill" or self.memory.cpu_kv_pool is None:
            return None
        record = self.memory.cpu_kv_pool.find_parked_session(session_id)
        if record is None:
            return None
        source, state = record
        if source is self:
            return state
        if (source.node_id != self.node_id or source.pd_type != "decode" or
                state.residency is not KVResidency.CPU):
            return None
        if session_id in self.session_kv_states:
            raise RuntimeError(
                f"Prefill instance {self.instance_id} already owns session "
                f"{session_id}.")
        source.session_kv_states.pop(session_id)
        self.memory.cpu_kv_pool.unregister_parked_session(
            source, session_id)
        state.owner_instance_id = self.instance_id
        self.session_kv_states[session_id] = state
        self.memory.cpu_kv_pool.register_parked_session(self, state)
        source._record_session_occupancy()
        self._record_session_occupancy()
        return state

    def get_next_session_kv_expiry(self):
        """Return the earliest live parked-state expiry, if any."""
        expiries = [
            state.expires_at_ns
            for state in self.session_kv_states.values()
            if not state.invalidated and state.expires_at_ns is not None
        ]
        return min(expiries) if expiries else None

    def _reset_pending_session_claim(self, session_id):
        """Convert a queued or reloading continuation into a full miss."""
        candidates = list(self.request)
        for batch in self.inflight:
            candidates.extend(batch.requests)
        seen = set()
        for req in candidates:
            if req.id in seen:
                continue
            seen.add(req.id)
            if req.session_id != session_id or not req.pending_session_kv:
                continue
            self._record_session_miss(req)
            req.pending_session_kv = False
            req.pending_session_kv_bytes_per_rank = 0
            req.num_computed_tokens = 0
            req.prefix_cache_hit = 0
            req.session_kv_hit_tokens = 0
            req.session_kv_hit_tier = None

    @staticmethod
    def _is_session_reuse_attempt(req):
        return (
            req.session_id is not None and
            (req.sub_request_index or 0) > 0 and
            (req.reuse_previous_kv or req.reused_prefix_toks is not None)
        )

    def _record_session_hit(self, req, tier):
        """Commit one usable session-cache hit to the aggregate metrics."""
        if req.session_kv_outcome_recorded:
            return
        if not self._is_session_reuse_attempt(req):
            return
        if req.session_kv_hit_tokens <= 0:
            raise RuntimeError(
                f"Session hit for request #{req.id} has no reused tokens.")
        stats = self._get_kv_offload_stats()
        if tier == "NPU":
            stats.session_npu_hit_count += 1
            stats.session_npu_hit_tokens += req.session_kv_hit_tokens
        elif tier == "CPU":
            stats.session_cpu_hit_count += 1
            stats.session_cpu_hit_tokens += req.session_kv_hit_tokens
        else:
            raise ValueError(f"Unknown session KV hit tier '{tier}'.")
        req.session_kv_outcome_recorded = True

    def _record_session_miss(self, req):
        """Commit one full-prefill fallback to the aggregate metrics."""
        if req.session_kv_outcome_recorded:
            return
        if not self._is_session_reuse_attempt(req):
            return
        stats = self._get_kv_offload_stats()
        stats.session_miss_count += 1
        stats.session_recomputed_prompt_tokens += req.original_input
        req.session_kv_outcome_recorded = True

    def expire_session_kv(self, current_time_ns):
        """Hard-expire inactive session KV before processing equal-time arrivals."""
        expired = []
        for session_id, state in list(self.session_kv_states.items()):
            if (state.invalidated or state.expires_at_ns is None or
                    current_time_ns < state.expires_at_ns):
                continue
            expired.append(session_id)
            self._reset_pending_session_claim(session_id)
            if state.residency in {KVResidency.NPU, KVResidency.CPU}:
                self._release_session_kv(
                    session_id, current_time_ns, reason="ttl")
            else:
                # The backend migration is synchronous and cannot be
                # cancelled. Hide the logical cache now; completion releases
                # both the source and destination without exposing a hit.
                state.invalidated = True
                state.expires_at_ns = None
                stats = self._get_kv_offload_stats()
                stats.session_ttl_expiration_count += 1
                if state.residency is KVResidency.NPU_TO_CPU:
                    stats.session_ttl_npu_bytes_freed += state.bytes_per_rank
                else:
                    stats.session_ttl_cpu_bytes_freed += state.bytes_full_cluster
        self._record_session_occupancy()
        return expired

    def _get_reload_size(self, batch_req, batch_len):
        load_size = 0
        for req in batch_req[:batch_len]:
            if req.pending_session_kv:
                load_size += req.pending_session_kv_bytes_per_rank
            elif req.is_kv_on_cpu():
                load_size += self.memory.get_evict_kv(req)
        return load_size

    def _select_offload_victim(self, requests):
        """Return one inactive decode request according to the configured policy."""
        if not requests:
            return None
        if self.kv_offload_victim_policy == 'lru':
            return min(requests, key=lambda req: (req.last_scheduled_ns, req.id))
        if self.kv_offload_victim_policy == 'largest-kv':
            return min(
                requests,
                key=lambda req: (
                    -self.memory.get_evict_kv(req),
                    req.last_scheduled_ns,
                    req.id,
                ),
            )
        raise RuntimeError(
            f"Unsupported KV offload victim policy '{self.kv_offload_victim_policy}'. "
            "Supported policies: 'lru', 'largest-kv'.")

    def _select_parked_session_victims(self, projected, target):
        """Select parked NPU sessions before considering active requests."""
        candidates = sorted(
            (state for state in self.session_kv_states.values()
             if state.residency is KVResidency.NPU),
            key=lambda state: (state.last_access_ns, state.session_id),
        )
        cpu_available = self._cpu_kv_capacity_with_session_drops()
        selected = []
        selected_cpu_bytes = 0
        while projected > target and candidates:
            state = candidates.pop(0)
            if (selected_cpu_bytes + state.bytes_full_cluster >
                    cpu_available):
                continue
            selected.append(state)
            selected_cpu_bytes += state.bytes_full_cluster
            projected -= state.bytes_per_rank
        return selected

    def _get_kv_offload_stats(self):
        if not hasattr(self, "kv_offload_stats"):
            self.kv_offload_stats = KVOffloadStats()
        return self.kv_offload_stats

    def _record_kv_occupancy(self):
        stats = self._get_kv_offload_stats()
        stats.npu_peak_used_bytes_per_rank = max(
            stats.npu_peak_used_bytes_per_rank, self.memory.npu_used)
        stats.npu_peak_reserved_bytes_per_rank = max(
            stats.npu_peak_reserved_bytes_per_rank, self.memory.npu_reserved)
        if self.memory.cpu_kv_pool is not None:
            cpu_used = self.memory.cpu_kv_pool.used
            cpu_reserved = self.memory.cpu_kv_pool.reserved
        else:
            cpu_used = self.memory.cpu_used
            cpu_reserved = self.memory.cpu_reserved
        stats.cpu_peak_used_bytes = max(stats.cpu_peak_used_bytes, cpu_used)
        stats.cpu_peak_reserved_bytes = max(
            stats.cpu_peak_reserved_bytes, cpu_reserved)

    def get_kv_offload_stats(self):
        """Return a stable copy suitable for reporting and validation."""
        self._record_kv_occupancy()
        return asdict(self._get_kv_offload_stats())

    def _cpu_kv_available_bytes(self):
        if self.memory.cpu_kv_pool is not None:
            pool = self.memory.cpu_kv_pool
            return max(0, pool.capacity - pool.used - pool.reserved)
        return max(
            0,
            self.memory.cpu_mem - self.memory.cpu_used - self.memory.cpu_reserved,
        )

    def _ensure_cpu_kv_capacity(self, size, current_time_ns):
        if self.memory.cpu_kv_pool is not None:
            self.memory.cpu_kv_pool.drop_parked_sessions_for(
                size, current_time_ns)
        return self.memory.is_avail(size, Device.CPU)

    def _cpu_kv_capacity_with_session_drops(self):
        available = self._cpu_kv_available_bytes()
        if self.memory.cpu_kv_pool is not None:
            available += self.memory.cpu_kv_pool.reclaimable_parked_bytes()
        return available

    def _get_migration_id(self):
        self.migration_ids += 1
        return self.migration_ids

    def _start_kv_eviction(self, requests, current, sys):
        """Reserve CPU capacity and create one migration-only eviction batch."""
        if not requests or self.inflight:
            return None
        if len({req.id for req in requests}) != len(requests):
            raise ValueError("A KV eviction plan cannot contain duplicate requests.")
        invalid = [req.id for req in requests if not req.is_kv_on_npu()]
        if invalid:
            raise RuntimeError(
                f"KV eviction requires NPU-resident requests; invalid ids: {invalid}")

        sizes = []
        for req in requests:
            size = self.memory.get_evict_kv(req)
            if size > 0:
                sizes.append((req, size))
        if not sizes:
            return None
        total_per_rank = sum(size for _, size in sizes)
        total_cpu = total_per_rank * self.num_npus
        if not self._ensure_cpu_kv_capacity(total_cpu, current):
            self.logger.warning(
                "CPU KV capacity prevents eviction; required=%.2fMB",
                total_cpu / MB_TO_BYTE)
            return None

        migration_id = self._get_migration_id()
        self.memory.reserve_live_kv(total_cpu, Device.CPU)
        self._record_kv_occupancy()
        migrations = []
        try:
            for req, size_per_rank in sizes:
                req.begin_kv_offload(migration_id)
                migrations.append(KVMigration(
                    migration_id=migration_id,
                    request_id=req.id,
                    direction=KVMigrationDirection.NPU_TO_CPU,
                    bytes_per_rank=size_per_rank,
                    bytes_full_cluster=size_per_rank * self.num_npus,
                    submit_time_ns=current,
                ))
            batch = Batch(
                self.get_batch_id(), self.model, 0, 0, [], [], 0, 0,
                [], [], [], current, 0, evict=total_per_rank,
                kind=BatchKind.KV_EVICT, migrations=migrations)
            batch.requests.extend(req for req, _ in sizes)
            batch.fired.append(sys)
        except Exception:
            for req, _ in sizes:
                if req.kv_migration_id == migration_id:
                    req.cancel_kv_migration(migration_id)
            self.memory.cancel_live_kv_reservation(total_cpu, Device.CPU)
            self._record_kv_occupancy()
            raise

        self.inflight.append(batch)
        self.logger.info(
            "Scheduling KV eviction batch #%d for %d request(s), %.2fMB per rank",
            batch.batch_id, len(sizes), total_per_rank / MB_TO_BYTE)
        return batch

    def _start_session_kv_eviction(self, states, current, sys):
        """Move parked NPU session state to the shared CPU KV pool."""
        if not states or self.inflight:
            return None
        if len({state.session_id for state in states}) != len(states):
            raise ValueError(
                "A session KV eviction plan cannot contain duplicates.")
        invalid = [
            state.session_id for state in states
            if state.residency is not KVResidency.NPU
        ]
        if invalid:
            raise RuntimeError(
                f"Session KV eviction requires NPU residency: {invalid}")
        total_per_rank = sum(state.bytes_per_rank for state in states)
        total_cpu = sum(state.bytes_full_cluster for state in states)
        if total_per_rank <= 0:
            return None
        if not self._ensure_cpu_kv_capacity(total_cpu, current):
            return None

        migration_id = self._get_migration_id()
        self.memory.reserve_live_kv(total_cpu, Device.CPU)
        self._record_kv_occupancy()
        migrations = []
        try:
            for state in states:
                self._account_session_residency(state, current)
                state.begin_offload(migration_id)
                migrations.append(KVMigration(
                    migration_id=migration_id,
                    request_id=None,
                    session_id=state.session_id,
                    direction=KVMigrationDirection.NPU_TO_CPU,
                    bytes_per_rank=state.bytes_per_rank,
                    bytes_full_cluster=state.bytes_full_cluster,
                    submit_time_ns=current,
                ))
            batch = Batch(
                self.get_batch_id(), self.model, 0, 0, [], [], 0, 0,
                [], [], [], current, 0, evict=total_per_rank,
                kind=BatchKind.KV_EVICT, migrations=migrations)
            batch.fired.append(sys)
        except Exception:
            for state in states:
                if state.migration_id == migration_id:
                    state.cancel_migration(migration_id)
            self.memory.cancel_live_kv_reservation(total_cpu, Device.CPU)
            self._record_kv_occupancy()
            raise

        self.inflight.append(batch)
        self.logger.info(
            "Scheduling parked-session eviction batch #%d for %d "
            "session(s), %.2fMB per rank",
            batch.batch_id, len(states), total_per_rank / MB_TO_BYTE)
        return batch

    def _start_kv_reload(self, requests, current, sys):
        """Reserve NPU capacity and create one migration-only reload batch."""
        if not requests or self.inflight:
            return None
        if len({req.id for req in requests}) != len(requests):
            raise ValueError("A KV reload plan cannot contain duplicate requests.")
        invalid = [req.id for req in requests if not req.is_kv_on_cpu()]
        if invalid:
            raise RuntimeError(
                f"KV reload requires CPU-resident requests; invalid ids: {invalid}")

        sizes = []
        for req in requests:
            size = self.memory.get_evict_kv(req)
            if size > 0:
                sizes.append((req, size))
        if not sizes:
            return None
        total_per_rank = sum(size for _, size in sizes)
        if not self.memory.is_avail(total_per_rank, Device.NPU):
            return None

        migration_id = self._get_migration_id()
        self.memory.reserve_live_kv(total_per_rank, Device.NPU)
        self._record_kv_occupancy()
        migrations = []
        try:
            for req, size_per_rank in sizes:
                req.begin_kv_reload(migration_id)
                migrations.append(KVMigration(
                    migration_id=migration_id,
                    request_id=req.id,
                    direction=KVMigrationDirection.CPU_TO_NPU,
                    bytes_per_rank=size_per_rank,
                    bytes_full_cluster=size_per_rank * self.num_npus,
                    submit_time_ns=current,
                ))
            batch = Batch(
                self.get_batch_id(), self.model, 0, 0, [], [], 0, 0,
                [], [], [], current, 0, load=total_per_rank,
                kind=BatchKind.KV_RELOAD, migrations=migrations)
            batch.requests.extend(req for req, _ in sizes)
            batch.fired.append(sys)
        except Exception:
            for req, _ in sizes:
                if req.kv_migration_id == migration_id:
                    req.cancel_kv_migration(migration_id)
            self.memory.cancel_live_kv_reservation(total_per_rank, Device.NPU)
            self._record_kv_occupancy()
            raise

        self.inflight.append(batch)
        self.logger.info(
            "Scheduling KV reload batch #%d for %d request(s), %.2fMB per rank",
            batch.batch_id, len(sizes), total_per_rank / MB_TO_BYTE)
        return batch

    def _start_session_kv_reload(self, req, current, sys):
        """Reload one parked CPU session before its continuation computes."""
        if self.inflight or not req.pending_session_kv:
            return None
        state = self.session_kv_states.get(req.session_id)
        if state is None or state.residency is not KVResidency.CPU:
            raise RuntimeError(
                f"Request #{req.id} has no CPU-resident session state.")
        size_per_rank = req.pending_session_kv_bytes_per_rank
        if size_per_rank <= 0:
            raise RuntimeError(
                f"Request #{req.id} has an invalid session reload size.")
        if not self.memory.is_avail(size_per_rank, Device.NPU):
            return None

        migration_id = self._get_migration_id()
        self.memory.reserve_live_kv(size_per_rank, Device.NPU)
        self._record_kv_occupancy()
        try:
            self._account_session_residency(state, current)
            state.begin_reload(migration_id)
            migration = KVMigration(
                migration_id=migration_id,
                request_id=req.id,
                session_id=state.session_id,
                direction=KVMigrationDirection.CPU_TO_NPU,
                bytes_per_rank=size_per_rank,
                bytes_full_cluster=size_per_rank * self.num_npus,
                submit_time_ns=current,
            )
            batch = Batch(
                self.get_batch_id(), self.model, 0, 0, [], [], 0, 0,
                [], [], [], current, 0, load=size_per_rank,
                kind=BatchKind.KV_RELOAD, migrations=[migration])
            batch.requests.append(req)
            batch.fired.append(sys)
        except Exception:
            if state.migration_id == migration_id:
                state.cancel_migration(migration_id)
            self.memory.cancel_live_kv_reservation(
                size_per_rank, Device.NPU)
            self._record_kv_occupancy()
            raise

        self.inflight.append(batch)
        self.logger.info(
            "Scheduling session reload batch #%d for session %s, "
            "%.2fMB per rank",
            batch.batch_id, state.session_id,
            size_per_rank / MB_TO_BYTE)
        return batch

    # batch the request scheduling method
    def schedule_base(self, current, sys, batch_id=-1):
        # first NPU to process new batch
        if sys == self.start_npu:
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                # wait it to be done
                return None
            # The correctness-first baseline does not overlap a migration with
            # compute or with another migration.
            if any(batch.kind is not BatchKind.COMPUTE for batch in self.inflight):
                return None

            mandatory_session_states = [
                state for state in self.session_kv_states.values()
                if (state.mandatory_cpu_offload and
                    state.residency is KVResidency.NPU and
                    not state.invalidated)
            ]
            if mandatory_session_states:
                migration_batch = self._start_session_kv_eviction(
                    mandatory_session_states, current, sys)
                if migration_batch is not None:
                    return migration_batch
                # Session KV is optional. If correctness-owned active KV has
                # consumed the CPU pool, discard the parked decode state and
                # let the continuation perform full prefill instead of
                # deadlocking the PD pipeline.
                for state in mandatory_session_states:
                    self._release_session_kv(
                        state.session_id, current, reason="capacity")
                return None

            handoff_batch = self._admit_pending_pd_handoff(current, sys)
            if handoff_batch is not None:
                return handoff_batch

            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None

            # scheduling start
            eligible_requests = [
                req for req in self.request
                if req.arrival <= current and not req.is_kv_migrating()
            ]
            if self.enable_kv_offloading:
                # A CPU-resident request is preempted waiting work, not a
                # running decode. Drain the resident queue before considering
                # swapped work. Merely ordering both groups is insufficient:
                # when physical capacity exceeds the high watermark, the same
                # candidate can reload a victim and evict it again next turn.
                resident_requests = [
                    req for req in eligible_requests if req.is_kv_on_npu()
                ]
                swapped_requests = [
                    req for req in eligible_requests if req.is_kv_on_cpu()
                ]
                eligible_requests = (
                    resident_requests if resident_requests else swapped_requests
                )

            # max_num_seqs limits total running requests (vLLM behavior)
            running_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - running_reqs)
            batch_len = min(len(eligible_requests), available_slots)

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = eligible_requests[:batch_len]

            kv_size = 0
            evict_size = 0

            # Get decode requests for preemption decisions
            # Victims can come from any runnable decode request on this
            # instance, not only the requests admitted into this candidate
            # batch. This lets a prefill-heavy candidate reclaim NPU space
            # from an older decode request without scheduling that victim.
            gen_req = [
                req for req in self.request
                if (req.arrival <= current and not req.is_prefill() and
                    req.is_kv_on_npu())
            ]
            
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Chunked prefill: process decode requests first, then prefill requests
            if self.enable_chunked_prefill:
                prefills = [req for req in batch_req if req.is_prefill()]
                if self.enable_kv_offloading:
                    npu_decodes = [
                        req for req in batch_req
                        if not req.is_prefill() and req.is_kv_on_npu()
                    ]
                    cpu_decodes = [
                        req for req in batch_req
                        if not req.is_prefill() and req.is_kv_on_cpu()
                    ]
                    batch_req = npu_decodes + prefills + cpu_decodes
                else:
                    decodes = [req for req in batch_req if not req.is_prefill()]
                    batch_req = decodes + prefills
                batch_len = len(batch_req)
            
            # ============ STEP 1: Token budget allocation (FIRST) ============
            # Build scheduled_tokens dict: req.id -> tokens to process this step
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # vLLM-style chunked prefill: schedule running (decode + ongoing prefill)
                # first, then waiting (new prefill) requests. Token budget is the main
                # constraint; long_prefill_token_threshold caps per-request tokens per step.
                token_budget = self.max_num_batched_tokens
                new_batch_req = []
                threshold = self.long_prefill_token_threshold
                # Decode requests first (each decode request = 1 token)
                for req in batch_req:
                    if not req.is_prefill():
                        if self.enable_kv_offloading and req.is_kv_on_cpu():
                            continue
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                # Then prefill requests (chunked)
                for req in batch_req:
                    if req.is_prefill():
                        if token_budget <= 0:
                            break
                        remaining = req.original_input - req.num_computed_tokens
                        # Per-request cap: long_prefill_token_threshold
                        if 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk
                # Swapped decodes are waiting work. Consider reload only after
                # NPU-resident decode and prefill work has consumed its budget.
                if self.enable_kv_offloading:
                    for req in batch_req:
                        if req.is_prefill() or not req.is_kv_on_cpu():
                            continue
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                batch_req = new_batch_req
                batch_len = len(batch_req)

            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        scheduled_tokens[req.id] = req.input
                        total_len += req.input
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    # print(f"[NON_CHUNKED] total_len({total_len} = sum([req 0 ~ {batch_len - 1}])) exceed 'max_num_batched_tokens'")
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1
                
                # DEBUG: Check if total_len reached max
                # if total_len >= self.max_num_batched_tokens * 0.9:
                #     print(f"[NON-CHUNKED] Near max tokens! total_len: {total_len}/{self.max_num_batched_tokens}")
                #     print(f"              Batch: {batch_len} reqs, scheduled_tokens: {scheduled_tokens}")
            
                # Early return due to max_num_batched_tokens limitation (It occurs only when No chunked-prefill)
                if batch_len == 0:
                    print("     [WARNNING] Cannot load the request to batch due to max_num_batched_tokens limitation")
                    return None
            # ============ STEP 2: KV size calculation and migration planning ============
            full_kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            full_load_size = self._get_reload_size(batch_req, batch_len)
            admission_limit = (
                self.memory.npu_mem * self.kv_offload_high_watermark
                if self.enable_kv_offloading else self.memory.npu_mem
            )
            low_watermark_limit = (
                self.memory.npu_mem * self.kv_offload_low_watermark
                if self.enable_kv_offloading else self.memory.npu_mem
            )
            projected = (
                self.memory.npu_used + self.memory.npu_reserved +
                full_kv_size + full_load_size
            )

            if self.enable_kv_offloading and projected > admission_limit:
                parked_victims = self._select_parked_session_victims(
                    projected, low_watermark_limit)
                if parked_victims:
                    migration_batch = self._start_session_kv_eviction(
                        parked_victims, current, sys)
                    if migration_batch is not None:
                        return migration_batch
                    return None

            # Plan victims without mutating memory or request state. Crossing
            # the high watermark triggers eviction toward the low watermark.
            # If a candidate request becomes a victim, remove it from this
            # compute plan first so no request is migrated and executed in the
            # same admission decision.
            selected_victims = []
            if self.enable_kv_offloading and projected > admission_limit:
                background = [req for req in gen_req if req not in batch_req]
                remaining = [req for req in gen_req if req in batch_req]
                candidates = background + remaining
                cpu_available = self._cpu_kv_capacity_with_session_drops()
                selected_cpu_bytes = 0
                while projected > low_watermark_limit and candidates:
                    victim = self._select_offload_victim(candidates)
                    candidates.remove(victim)
                    victim_size = self.memory.get_evict_kv(victim)
                    victim_cpu_size = victim_size * self.num_npus
                    if victim_size <= 0:
                        continue
                    if victim in batch_req and not any(
                            req is not victim and req.is_kv_on_npu()
                            for req in batch_req):
                        # Do not evict the last NPU-runnable request merely to
                        # leave swapped work behind; that produces D2H/H2D
                        # ping-pong without advancing model tokens.
                        continue
                    if selected_cpu_bytes + victim_cpu_size > cpu_available:
                        continue
                    selected_victims.append(victim)
                    selected_cpu_bytes += victim_cpu_size
                    if victim in batch_req:
                        batch_req.remove(victim)
                        scheduled_tokens.pop(victim.id, None)
                        batch_len = len(batch_req)
                    full_kv_size = self.memory.get_block_kv(
                        batch_req, batch_len, scheduled_tokens)
                    full_load_size = self._get_reload_size(batch_req, batch_len)
                    projected = (
                        self.memory.npu_used + self.memory.npu_reserved +
                        full_kv_size + full_load_size -
                        sum(self.memory.get_evict_kv(req)
                            for req in selected_victims)
                    )

            if selected_victims:
                migration_batch = self._start_kv_eviction(selected_victims, current, sys)
                if migration_batch is not None:
                    return migration_batch
                return None

            # If there is no eligible/capacity-feasible victim, the watermark
            # cannot be a hard stop: allow progress up to physical capacity
            # and shrink the candidate as necessary.
            effective_limit = admission_limit
            if self.enable_kv_offloading and projected > admission_limit:
                effective_limit = self.memory.npu_mem

            temp_len = 0
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                load_size = self._get_reload_size(batch_req, i)
                usage_after = (
                    self.memory.npu_used + self.memory.npu_reserved +
                    kv_size + load_size
                )
                if usage_after <= effective_limit:
                    temp_len = i
                    break

            batch_len = temp_len
            batch_req = batch_req[:batch_len]

            if batch_len == 0:
                return None

            scheduled_tokens = {
                req.id: scheduled_tokens[req.id]
                for req in batch_req
                if req.id in scheduled_tokens
            }

            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            load_size = self._get_reload_size(batch_req, batch_len)

            # Reload is a separate workload. The request stays queued and no
            # model tokens advance until the reload completes.
            session_reload_requests = [
                req for req in batch_req if req.pending_session_kv]
            if session_reload_requests:
                migration_batch = self._start_session_kv_reload(
                    session_reload_requests[0], current, sys)
                if migration_batch is not None:
                    return migration_batch
                return None

            reload_requests = [req for req in batch_req if req.is_kv_on_cpu()]
            if reload_requests:
                migration_batch = self._start_kv_reload(reload_requests, current, sys)
                if migration_batch is not None:
                    return migration_batch
                return None

            # ============ STEP 4: Build the compute batch without mutation ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            for req in batch_req:
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size
                    chunk_size = scheduled_tokens.get(req.id, req.original_input - req.num_computed_tokens)

                    total_len += chunk_size
                    q_list.append(chunk_size)
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                    # k_list: total kv cache after this step (computed + new)
                    # k_list.append(req.num_computed_tokens + chunk_size)
                    num_prefill += 1

                else:
                    # Decode
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens
                    decode_k_list.append(req.num_computed_tokens)
                    # k_list.append(req.num_computed_tokens)

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size)

            # ============ STEP 5: Commit admission ============
            # Capacity allocation is the only fallible operation below. Keep
            # request fields and the scheduler queue unchanged until it passes.
            if kv_size > 0:
                self.memory.allocate(kv_size, Device.NPU)
                self._record_kv_occupancy()

            admitted_ids = {req.id for req in batch_req}
            self.request = [
                req for req in self.request if req.id not in admitted_ids
            ]
            for req in batch_req:
                if req.is_prefill():
                    req.chunk_len = scheduled_tokens[req.id]
                    if req.is_init:
                        req.set_que_delay(current)

            # add already fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            # batch.log()
            # add scheduled_tokens to batch for debugging
            batch.scheduled_tokens = scheduled_tokens
            return batch
        
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch == None:
                    return None
                # check if this has been runned in the system
                if sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
    
    def schedule_with_prefix(self, current, sys, batch_id=-1):
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]

            # max_num_seqs limits total running requests (vLLM behavior)
            running_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - running_reqs)
            batch_len = min(len(batch_req), available_slots)

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            # Prioritize prefill (without chunked prefill) or reorder for chunked prefill
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]
                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Chunked prefill: process decode requests first, then prefill requests
            if self.enable_chunked_prefill:
                prefills = [req for req in batch_req if req.is_prefill()]
                decodes = [req for req in batch_req if not req.is_prefill()]
                batch_req = decodes + prefills
                batch_len = len(batch_req)

            # Get decode requests for preemption decisions
            gen_req = [req for req in batch_req if not req.is_prefill()]
            # gen_req = [req for req in batch_req if not (req.num_computed_tokens >= req.original_input)]
            
            # ============ STEP 0: Prefix Matching ============
            # Only match prefix for NEW prefill requests (first chunk)
            # Ongoing chunked prefills already have their prefix cache info
            # for req in batch_req:
            #     if req.is_prefill():
            #         self.memory.prefix_match(req)
            
            # ============ STEP 1: Token budget allocation ============
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # Chunked prefill: assign token budget to requests
                token_budget = self.max_num_batched_tokens
                new_batch_req = []
                
                # Decode requests first (each decode request = 1 token)
                for req in batch_req:
                    if not req.is_prefill():
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                
                # Then prefill requests (chunked)
                threshold = self.long_prefill_token_threshold
                for req in batch_req:
                    if req.is_prefill():
                        if token_budget <= 0:
                            break
                        # Calculate remaining tokens without considering prefix cache
                        # because it is already considered in "self.memory.prefix_match(req)" -> req.num_computed_tokens
                        if req.num_computed_tokens == 0:
                            self.memory.prefix_match(req)
                        remaining = req.original_input - req.num_computed_tokens
                        # Per-request cap: long_prefill_token_threshold
                        if 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break

                        req.chunk_len = chunk
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk

                batch_req = new_batch_req
                batch_len = len(batch_req)
            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        if req.num_computed_tokens == 0:
                            self.memory.prefix_match(req)
                        # Consider prefix cache hit for non-chunked prefill
                        prefix_hit = req.prefix_cache_hit
                        tokens_to_compute = max(req.original_input - prefix_hit, 1)
                        scheduled_tokens[req.id] = tokens_to_compute
                        req.chunk_len = tokens_to_compute  # Set chunk_len for add_done()
                        total_len += tokens_to_compute
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1
            
            # ============ STEP 1.5: Lock prefix for scheduled requests ============
            newly_locked = set()
            for req in batch_req:
                # if req.is_prefill() and req.num_computed_tokens == 0:
                if req.is_prefill() and req.npu_last_node is not None and not req._prefix_locked:
                    self.memory.lock_prefix(req, Device.NPU)
                    req._prefix_locked = True
                    newly_locked.add(req.id)
            
            # ============ STEP 2: KV size calculation ============
            kv_size = 0
            evict_size = 0
            temp_len = batch_len
            total_useable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
            
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                if total_useable_size >= kv_size:
                    temp_len = i
                    break
            
            # ============ STEP 3: Eviction if needed ============
            evicted_req = []
            while temp_len == 0:
                # print("eviction occurs!!")
                if len(gen_req) == 0:
                    # print("gen_req length == 0 (No decode) => return None (No Batch)")
                    # No request to evict but no memory - rollback prefix cache lock
                    for req in batch_req:
                        if req.is_prefill() and req._prefix_locked:
                            
                            self.memory.unlock_prefix(req, Device.NPU)
                            self.memory.erase_prefix_info(req)
                            req._prefix_locked = False
                    return None
                
                # Check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue
                
                # Evict the last decode request
                # (DEPRECATED) self.memory.unlock_prefix(gen_req[-1], Device.NPU)
                # (DEPRECATED) self.memory.erase_prefix_info(gen_req[-1])
                if gen_req[-1].is_prefill() and getattr(gen_req[-1], '_prefix_locked', False):
                    self.memory.unlock_prefix(gen_req[-1], Device.NPU)
                    # self.memory.erase_prefix_info(gen_req[-1])
                    gen_req[-1]._prefix_locked = False
                
                current_usable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
                
                gen_req[-1].evict = True
                evicted_req.append(gen_req[-1])
                self.logger.info("Eviction of the request #%d", gen_req[-1].id)
                gen_req = gen_req[:-1]
                
                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)
                
                # Check if can batch now
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    if current_usable_size >= kv_size:
                        temp_len = i
                        break

            # Unlock prefix for requests that didn't make it into the batch
            for req in batch_req[temp_len:]:
                if req.is_prefill() and req._prefix_locked:
                    self.memory.unlock_prefix(req, Device.NPU)
                    self.memory.erase_prefix_info(req)
                    req._prefix_locked = False

            batch_len = temp_len
            batch_req = batch_req[:batch_len]
            
            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            evict_size = (kv_size - self.memory.avail_size(Device.NPU)) if kv_size > self.memory.avail_size(Device.NPU) else 0
            
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)

            # ============ STEP 4: Allocate memory & handle evicted requests ============
            evict_load_size = 0
            prefix_load_size = 0
            
            for req in batch_req:
                # Remove from request queue
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                # Load prefix cache from storage if needed
                if req.is_prefill() and req.storage_cache_hit > req.npu_cache_hit:
                    prefix_load_size += (req.storage_cache_hit - req.npu_cache_hit) * self.memory.get_kv(1)

                # Handle evicted requests
                if req.evict:
                    self.memory.prefix_match(req)
                    self.memory.lock_prefix(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.unlock_prefix(req, Device.CPU)
                    evict_load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # ============ STEP 5: Build batch with lists ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            
            # Evict storage prefix cache if needed
            total_size = 0
            for req in batch_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            for req in evicted_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            
            if self.prefix_storage is not None:
                storage_evict_size = (total_size - self.memory.avail_size(self.prefix_storage)) if total_size > self.memory.avail_size(self.prefix_storage) else 0
                if storage_evict_size > 0:
                    self.memory.evict_prefix_cache(storage_evict_size, self.prefix_storage)

            for req in batch_req:
                # Update the prefix cache for incoming batch
                # NOTE: Moved to add_done() to ensure prefix cache is updated after chunk computation
                # self.memory.cache_unfinished_req(req, Device.NPU)
                # if self.prefix_storage is not None:
                #     self.memory.cache_unfinished_req(req, self.prefix_storage)
                
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size. num_computed_tokens
                    # already includes any prefix-cache hit (memory_model.py
                    # bumps it on first prefix_match), so chunk_size is already
                    # the count of tokens actually computed this iteration —
                    # no further prefix-hit subtraction is needed downstream.
                    chunk_size = scheduled_tokens.get(req.id, req.original_input - req.num_computed_tokens)
                    if chunk_size > self.max_num_batched_tokens:
                        raise Exception("Chunk length exceeds max num batched tokens")

                    total_len += chunk_size
                    if req.is_init:  # Only set queuing delay on first chunk
                        req.set_que_delay(current)

                    q_list.append(chunk_size)
                    num_prefill += 1
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                else:
                    # Decode: use num_computed_tokens (inevitable modification)
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens  # inevitable modification: was req.input
                    decode_k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
                
                k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
            
            # Storage needs to hold evicted cache
            if self.prefix_storage is not None:
                for req in evicted_req:
                    self.memory.storage_cache_evicted_req(req)

            
            # For debugging
            # self.memory.npu_prefix_cache.pretty_print()
            # self.memory.npu_prefix_cache.print_prefix_info()
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, evict_load_size + prefix_load_size)
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            batch.scheduled_tokens = scheduled_tokens
            # batch.log()
            return batch
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch is None or sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
        
    # pop inflight, add to done
    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs
        batch = None
        # find batch
        id -= 1
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            return prompt_t, gen_t, end_reqs
        # already done
        if sys in batch.end:
            return prompt_t, gen_t, end_reqs
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            if batch.kind is not BatchKind.COMPUTE:
                end_npu = self.start_npu + self.num_npus - 1
                if self.pd_type == "prefill":
                    end_npu = self.start_npu + self.num_npus * 2 - 1
                if self.start_npu not in batch.end or end_npu not in batch.end:
                    return prompt_t, gen_t, end_reqs
            elif self.pd_type != "prefill":
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
            else:
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus * 2 - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
        self.logger.info(
            "Batch #%d is done",
            batch.batch_id,
        )

        if batch.kind in {BatchKind.KV_EVICT, BatchKind.KV_RELOAD}:
            requests_by_id = {req.id: req for req in batch.requests}
            request_migrations = [
                migration for migration in batch.migrations
                if migration.session_id is None
            ]
            session_migrations = [
                migration for migration in batch.migrations
                if migration.session_id is not None
            ]
            stats = self._get_kv_offload_stats()
            duration_ns = max(0, finish - batch.batch_time)
            stats.migration_time_ns += duration_ns
            if batch.kind is BatchKind.KV_EVICT:
                total_cpu = sum(m.bytes_full_cluster for m in batch.migrations)
                total_per_rank = sum(m.bytes_per_rank for m in batch.migrations)
                self.memory.commit_live_kv_reservation(total_cpu, Device.CPU)
                self.memory.free(total_per_rank, Device.NPU)
                for migration in request_migrations:
                    requests_by_id[migration.request_id].complete_kv_offload(
                        migration.migration_id)
                for migration in session_migrations:
                    state = self.session_kv_states[migration.session_id]
                    state.complete_offload(migration.migration_id)
                    if state.invalidated:
                        self.memory.free(
                            state.bytes_full_cluster, Device.CPU)
                        self.session_kv_states.pop(migration.session_id)
                    else:
                        state.residency_since_ns = finish
                        if self.memory.cpu_kv_pool is not None:
                            self.memory.cpu_kv_pool.register_parked_session(
                                self, state)
                stats.preemption_count += len(request_migrations)
                stats.evict_bytes += total_cpu
                stats.eviction_batches += 1
                stats.eviction_time_ns += duration_ns
            else:
                total_per_rank = sum(m.bytes_per_rank for m in batch.migrations)
                request_cpu = sum(
                    migration.bytes_full_cluster
                    for migration in request_migrations)
                session_cpu = sum(
                    self.session_kv_states[
                        migration.session_id].bytes_full_cluster
                    for migration in session_migrations)
                total_cpu = request_cpu + session_cpu
                self.memory.commit_live_kv_reservation(total_per_rank, Device.NPU)
                self.memory.free(total_cpu, Device.CPU)
                for migration in request_migrations:
                    requests_by_id[migration.request_id].complete_kv_reload(
                        migration.migration_id)
                for migration in session_migrations:
                    state = self.session_kv_states[migration.session_id]
                    if self.memory.cpu_kv_pool is not None:
                        self.memory.cpu_kv_pool.unregister_parked_session(
                            self, migration.session_id)
                    state.complete_reload(
                        migration.migration_id, migration.bytes_per_rank)
                    req = requests_by_id[migration.request_id]
                    self.session_kv_states.pop(migration.session_id)
                    req.pending_session_kv = False
                    req.pending_session_kv_bytes_per_rank = 0
                    req.kv_residency = KVResidency.NPU
                    if state.invalidated:
                        self._record_session_miss(req)
                        self.memory.free(
                            migration.bytes_per_rank, Device.NPU)
                        req.num_computed_tokens = 0
                        req.prefix_cache_hit = 0
                        req.session_kv_hit_tokens = 0
                        req.session_kv_hit_tier = None
                    else:
                        state.cached_tokens = req.num_computed_tokens
                        req.session_kv_hit_tier = "CPU"
                        self._record_session_hit(req, "CPU")
                        stats.session_cpu_reload_bytes += (
                            migration.bytes_full_cluster)
                        stats.session_cpu_reload_time_ns += duration_ns
                        stats.session_reload_wait_ns += max(
                            0, finish - req.arrival)
                stats.reload_bytes += sum(
                    migration.bytes_full_cluster
                    for migration in batch.migrations)
                stats.reload_batches += 1
                stats.reload_stall_count += len(batch.migrations)
                stats.reload_stall_ns += duration_ns * len(batch.migrations)
                stats.reload_time_ns += duration_ns
            self._record_kv_occupancy()
            self._record_session_occupancy()
            del self.inflight[idx]
            return prompt_t, gen_t, end_reqs

        pool = []
        for req in batch.requests:
            req.last_scheduled_ns = finish
            # For chunked prefill, use computed tokens to determine prefill vs decode
            # Use is_prefill() method which checks num_computed_tokens < original_input
            is_prefill_req = req.is_prefill()
            
            # change phase
            if is_prefill_req:
                # Get chunk_len from scheduling step
                chunk_len = req.chunk_len if req.chunk_len > 0 else (req.original_input - req.num_computed_tokens)
                if chunk_len > self.max_num_batched_tokens:
                    raise Exception("Chunk length exceeds max num batched tokens")

                # Update num_computed_tokens
                req.num_computed_tokens += chunk_len
                req.chunk_len = 0  # Reset for next step
                
                # Check if prefill is complete
                if req.num_computed_tokens >= req.original_input:
                    # Update prefix cache before clearing is_init (for stats tracking)
                    if self.enable_prefix_caching:
                        self.memory.cache_unfinished_req(req, Device.NPU)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    req.is_init = False
                    # Include prefix cache hit tokens in prompt throughput
                    prompt_t += chunk_len + req.prefix_cache_hit
                    req.set_ttft(finish)
                    
                    if self.pd_type == "prefill":
                        # Prefill instance: send to decode instance
                        self.logger.info("Request #%d is prefill done", req.id)
                        self.logger.info("Request #%d is sent to decode instance", req.id)
                        # The final prefill token passes through lm_head and
                        # produces the first output token, just as in the
                        # colocated path below. The token is implicit in
                        # num_computed_tokens but must be counted in throughput.
                        gen_t += 1
                        
                        # In the offload-aware same-node PD path, source KV
                        # remains owned by this instance until the decode
                        # scheduler reserves and commits destination capacity.
                        if self.enable_prefix_caching:
                            self.memory.unlock_prefix(req, Device.NPU)
                        elif not self.enable_kv_offloading:
                            kv_size = self.memory.get_evict_kv(req)
                            self.memory.free(kv_size, Device.NPU)

                        end_reqs.append(req)
                        continue
                    else:
                        # Non-PD: prefill complete, first output token generated
                        # The last prefill token passing through lm_head generates the first output
                        gen_t += 1
                        # req.num_computed_tokens += 1  # Count the first generated token
                        # req.set_ttft(finish)
                        # pool.append(req)
                        # continue
                else:
                    # Prefill not complete, return to pool for next chunk
                    prompt_t += chunk_len
                    # pool.append(req)
                    # continue
            else:
                # Decode phase
                if req.is_init:
                    # Full prefix cache hit: all input tokens were cached, so the
                    # request never entered the prefill-complete path where is_init
                    # is cleared. Lock the prefix node (was skipped because
                    # is_prefill() returned False during scheduling), count prefix
                    # stats once, then clear is_init.
                    if self.enable_prefix_caching:
                        if req.npu_last_node is not None and not req._prefix_locked:
                            self.memory.lock_prefix(req, Device.NPU)
                            req._prefix_locked = True
                        self.memory.cache_unfinished_req(req, Device.NPU)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    req.is_init = False
                    req.set_ttft(finish)
                    # Full prefix hit: count all cached tokens as prompt throughput
                    prompt_t += req.prefix_cache_hit
                gen_t += 1
                req.add_itl(finish)
                req.num_computed_tokens += 1

            # Update computed tokens for decode
            # req.num_computed_tokens += 1

            # check done
            if req.output <= req.num_computed_tokens + 1:
                # print("Request #{} is done".format(req.id))
                self.logger.info("Request #%d is done", req.id)
                # remove kv cache here
                if self.enable_prefix_caching:
                    self.memory.cache_finished_req(req, Device.NPU) # insert happens here
                    if self.prefix_storage is not None:
                        self.memory.cache_finished_req(req, Device.CPU)
                elif (self.enable_session_kv_retention and
                        req.session_id is not None and
                        req.retain_session_kv):
                    self._park_completed_session_kv(req, finish)
                else:
                    terminal_cleanup_bytes = 0
                    if (self.enable_session_kv_retention and
                            req.session_id is not None):
                        stale = self._release_session_kv(
                            req.session_id, finish, reason="terminal")
                        if stale is not None:
                            terminal_cleanup_bytes += stale.bytes_full_cluster
                    kv_size = self.memory.get_evict_kv(req)
                    self.memory.free(kv_size, Device.NPU)
                    if (self.enable_session_kv_retention and
                            req.session_id is not None and
                            not req.session_has_next):
                        stats = self._get_kv_offload_stats()
                        stats.session_terminal_cleanup_count += 1
                        stats.session_terminal_cleanup_bytes += (
                            terminal_cleanup_bytes +
                            kv_size * self.num_npus)
                req.add_latency(finish)
                self.done.append(req)
                end_reqs.append(req)

            # return to pool
            else:
                # print("Request #{} is not finished => go to pool".format(req.id))
                # Update prefix cache after chunk completion (moved from schedule_with_prefix())
                if self.enable_prefix_caching:
                    self.memory.cache_unfinished_req(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.cache_unfinished_req(req, self.prefix_storage)
                pool.append(req)
        # return to request pool, both are already sorted with arrival_time
        if self.prioritize_prefill:
            self.request = self._merge_by_arrival_id(pool, self.request)
        else:
            self.request = pool + self.request
        del self.inflight[idx]
        del batch
        self._record_kv_occupancy()

        return prompt_t, gen_t, end_reqs
    

    ##### Helper Functions ######
    # get new batch id
    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    # add a request
    def add_request(self, req, is_init=True, session_metadata=None):
        session_metadata = session_metadata or {}
        new_req = Request(*(req), is_init=is_init, **session_metadata)
        session_hit_tokens = 0
        if (self.enable_session_kv_retention and
                new_req.session_id is not None and
                new_req.session_id not in self.session_kv_states):
            self._adopt_pd_cpu_session(new_req.session_id)
        if (self.enable_session_kv_retention and
                new_req.session_id in self.session_kv_states):
            session_hit_tokens = self._claim_parked_session_kv(new_req)
        if (self.enable_session_kv_retention and
                self._is_session_reuse_attempt(new_req)):
            if session_hit_tokens:
                if new_req.session_kv_hit_tier == "NPU":
                    self._record_session_hit(new_req, "NPU")
                # CPU claims remain provisional until H2D completion.
            else:
                self._record_session_miss(new_req)
        # Maintain arrival-time sort order (required by schedule_base/schedule_with_prefix)
        bisect.insort(self.request, new_req, key=lambda r: (r.arrival, r.id))
        return new_req
    
    # add decode request to decode instance from prefill instnace
    def add_decode(self, req):
        source_instance_id = req.kv_owner_instance_id
        if self.enable_prefix_caching:
            req.transfer_kv_owner(source_instance_id, self.instance_id)
            self.request.append(req)
            self.memory.prefix_match(req)
            kv_size = self.memory.get_evict_kv(req)
            evict_size = max(0, kv_size - self.memory.avail_size(Device.NPU))
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)
            self.memory.cache_unfinished_req(req, Device.NPU)
        else:
            kv_size = self.memory.get_total_kv(req)
            self.memory.allocate(kv_size, Device.NPU)
            req.transfer_kv_owner(source_instance_id, self.instance_id)
            bisect.insort(self.request, req, key=lambda item: (item.arrival, item.id))
    
    # get first request's arrival time
    def get_first_arrival_time(self):
        return self.first_arrival_time if self.first_arrival_time != 0 else 1 # need to add event handler at first
    
    # merge requests in the request pool, ensuring they are sorted by arrival time
    def _merge_by_arrival_id(self, left, right):
        if not left:  
            return right
        if not right: 
            return left

        # Fast path: if ranges don't overlap, just concatenate
        if (left[-1].arrival, left[-1].id) <= (right[0].arrival, right[0].id):
            return left + right
        if (right[-1].arrival, right[-1].id) <= (left[0].arrival, left[0].id):
            return right + left

        # General merge
        i = j = 0
        out = []
        while i < len(left) and j < len(right):
            li, rj = left[i], right[j]
            if (li.arrival, li.id) <= (rj.arrival, rj.id):
                out.append(li); i += 1
            else:
                out.append(rj); j += 1
        if i < len(left):  
            out.extend(left[i:])
        if j < len(right): 
            out.extend(right[j:])
        return out
    
    # print total system request metrics (TTFT, TPOT, ITL)
    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        def _render(title: str, values, num_space=0):
            print_rule(f"[sim.tagline]{title}[/]")
            if not values:
                print_markup(f"No {title.split()[0]} data available")
                return
            mean = np.mean(values) / 1_000_000
            p50 = np.percentile(values, 50) / 1_000_000
            p95 = np.percentile(values, 95) / 1_000_000
            p99 = np.percentile(values, 99) / 1_000_000
            label = title.split()[-1] if title != "Time to First Token" else "TTFT"
            # Map to the metric short-name used in the detail rows.
            short = {
                "Time to First Token": "TTFT",
                "Time per Output Token (excl. 1st token)": "TPOT",
                "Inter-token Latency": "ITL",
            }[title]
            spacing = " " * num_space
            print_markup(f"Mean {short} (ms){spacing}:                                                     {mean:.2f}")
            print_markup(f"P50 {short} (ms){spacing}:                                                      {p50:.2f}")
            print_markup(f"P95 {short} (ms){spacing}:                                                      {p95:.2f}")
            print_markup(f"P99 {short} (ms){spacing}:                                                      {p99:.2f}")

        _render("Time to First Token", ttft_values)
        _render("Time per Output Token (excl. 1st token)", tpot_values)
        _render("Inter-token Latency", itl_values, num_space=1)

        if self.enable_kv_offloading:
            stats = self.get_kv_offload_stats()
            print_rule("[sim.tagline]CPU KV Offloading[/]")
            print_markup(
                f"Preemptions: {stats['preemption_count']}, "
                f"evicted/reloaded: "
                f"{stats['evict_bytes'] / MB_TO_BYTE:.2f}/"
                f"{stats['reload_bytes'] / MB_TO_BYTE:.2f} MB")
            print_markup(
                f"Migration time: {stats['migration_time_ns'] / 1_000_000:.3f} ms, "
                f"reload request stalls: {stats['reload_stall_count']} "
                f"({stats['reload_stall_ns'] / 1_000_000:.3f} ms total)")
            print_markup(
                "Peak NPU used/reserved per rank: "
                f"{stats['npu_peak_used_bytes_per_rank'] / MB_TO_BYTE:.2f}/"
                f"{stats['npu_peak_reserved_bytes_per_rank'] / MB_TO_BYTE:.2f} MB, "
                "peak CPU used/reserved: "
                f"{stats['cpu_peak_used_bytes'] / MB_TO_BYTE:.2f}/"
                f"{stats['cpu_peak_reserved_bytes'] / MB_TO_BYTE:.2f} MB")
            if stats['pd_handoff_count']:
                print_markup(
                    f"PD handoffs: {stats['pd_handoff_count']}, "
                    f"bytes: {stats['pd_handoff_bytes'] / MB_TO_BYTE:.2f} MB, "
                    f"admission wait: "
                    f"{stats['pd_handoff_wait_ns'] / 1_000_000:.3f} ms")
        if self.enable_session_kv_retention:
            stats = self.get_kv_offload_stats()
            print_rule("[sim.tagline]Session KV Retention[/]")
            print_markup(
                "NPU/CPU hits: "
                f"{stats['session_npu_hit_count']}/"
                f"{stats['session_cpu_hit_count']}, misses: "
                f"{stats['session_miss_count']}")
            print_markup(
                "TTL expirations/capacity drops: "
                f"{stats['session_ttl_expiration_count']}/"
                f"{stats['session_capacity_drop_count']}, currently parked: "
                f"{stats['session_current_parked']}")

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def is_request_empty(self):
        if (len(self.request) == 0 and len(self.inflight) == 0 and
                len(self.pending_pd_handoffs) == 0):
            return True
        else:
            return False
        
    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        if not os.path.isabs(output_file):
            output_file = f'../{output_file}'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow([
                    'instance id', 'request id', 'session id',
                    'sub request index', 'session kv hit tier',
                    'session kv hit tokens', 'model', 'input', 'output',
                    'arrival', 'end_time', 'latency', 'queuing_delay',
                    'TTFT', 'TPOT', 'ITL'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.session_id,
                    req.sub_request_index,
                    req.session_kv_hit_tier,
                    req.session_kv_hit_tokens,
                    req.model,
                    req.input,
                    req.output - req.input,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot,
                    req.itl
                ])

    def save_kv_offload_output(self, output_file, is_append=False):
        """Write one instance-level row of CPU KV offload metrics."""
        if not os.path.isabs(output_file):
            output_file = f'../{output_file}'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        stats = self.get_kv_offload_stats()
        columns = [
            'instance id', 'node id', 'preemption count',
            'eviction batches', 'reload batches', 'evict bytes',
            'reload bytes', 'migration time ns', 'eviction time ns',
            'reload time ns', 'reload stall count', 'reload stall ns',
            'npu peak used bytes per rank',
            'npu peak reserved bytes per rank', 'cpu peak used bytes',
            'cpu peak reserved bytes', 'pd handoff count',
            'pd handoff bytes', 'pd handoff wait ns',
            'session npu hit count', 'session npu hit tokens',
            'session cpu hit count', 'session cpu hit tokens',
            'session cpu reload bytes', 'session cpu reload time ns',
            'session miss count', 'session recomputed prompt tokens',
            'session ttl expiration count',
            'session ttl npu bytes freed', 'session ttl cpu bytes freed',
            'session capacity drop count', 'session capacity drop bytes',
            'session parked npu byte ns', 'session parked cpu byte ns',
            'session peak parked npu bytes per rank',
            'session peak parked cpu bytes', 'session reload wait ns',
            'session current parked', 'session terminal cleanup count',
            'session terminal cleanup bytes',
        ]
        row = [
            self.instance_id, self.node_id, stats['preemption_count'],
            stats['eviction_batches'], stats['reload_batches'],
            stats['evict_bytes'], stats['reload_bytes'],
            stats['migration_time_ns'], stats['eviction_time_ns'],
            stats['reload_time_ns'], stats['reload_stall_count'],
            stats['reload_stall_ns'],
            stats['npu_peak_used_bytes_per_rank'],
            stats['npu_peak_reserved_bytes_per_rank'],
            stats['cpu_peak_used_bytes'], stats['cpu_peak_reserved_bytes'],
            stats['pd_handoff_count'], stats['pd_handoff_bytes'],
            stats['pd_handoff_wait_ns'],
            stats['session_npu_hit_count'],
            stats['session_npu_hit_tokens'],
            stats['session_cpu_hit_count'],
            stats['session_cpu_hit_tokens'],
            stats['session_cpu_reload_bytes'],
            stats['session_cpu_reload_time_ns'],
            stats['session_miss_count'],
            stats['session_recomputed_prompt_tokens'],
            stats['session_ttl_expiration_count'],
            stats['session_ttl_npu_bytes_freed'],
            stats['session_ttl_cpu_bytes_freed'],
            stats['session_capacity_drop_count'],
            stats['session_capacity_drop_bytes'],
            stats['session_parked_npu_byte_ns'],
            stats['session_parked_cpu_byte_ns'],
            stats['session_peak_parked_npu_bytes_per_rank'],
            stats['session_peak_parked_cpu_bytes'],
            stats['session_reload_wait_ns'],
            stats['session_current_parked'],
            stats['session_terminal_cleanup_count'],
            stats['session_terminal_cleanup_bytes'],
        ]
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            writer = csv.writer(file)
            if not is_append:
                writer.writerow(columns)
            writer.writerow(row)


def main():
    pass

if __name__ == "__main__":
    main()
