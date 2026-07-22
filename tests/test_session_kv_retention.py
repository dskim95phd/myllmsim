import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.__main__ import (
    _build_instance_runtime_configs,
    _next_idle_event,
    _validate_kv_offload_node_scope,
)
from serving.core.memory_model import MemoryModel, NodeCPUKVPool
from serving.core.request import Batch, KVResidency, Request
from serving.core.router import Router
from serving.core.scheduler import KVOffloadStats, Scheduler


class SessionKVOutputTest(unittest.TestCase):
    def test_per_request_output_contains_session_metadata(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.done = [SimpleNamespace(
            instance_id=0,
            id=7,
            session_id="session-7",
            sub_request_index=2,
            session_kv_hit_tier="CPU",
            session_kv_hit_tokens=96,
            model="test/model",
            input=128,
            output=160,
            arrival=10,
            end_time=30,
            latency=20,
            queuing_delay=4,
            ttft=6,
            tpot=1,
            itl=[1, 1],
        )]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "requests.csv"
            scheduler.save_output(str(output))
            with output.open(newline="", encoding="utf-8") as output_file:
                row = next(csv.DictReader(output_file))

        self.assertEqual(row["session id"], "session-7")
        self.assertEqual(row["sub request index"], "2")
        self.assertEqual(row["session kv hit tier"], "CPU")
        self.assertEqual(row["session kv hit tokens"], "96")


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class SessionKVRetentionFoundationTest(unittest.TestCase):
    def _scheduler(
            self, npu_used=60, cpu_pool=None, instance_id=0,
            pd_type=None):
        memory = MemoryModel.__new__(MemoryModel)
        memory.node_id = 0
        memory.instance_id = instance_id
        memory.num_npus = 1
        memory.npu_mem = 100
        memory.npu_used = npu_used
        memory.npu_reserved = 0
        memory.cpu_mem = 200
        memory.cpu_used = 0
        memory.cpu_reserved = 0
        memory.cpu_kv_pool = cpu_pool
        memory.weight = 0
        memory.block_size = 16
        memory.kv_fp = 2
        memory.prefix_storage = None
        memory.enable_prefix_sharing = False
        memory.enable_prefix_caching = False
        memory.logger = _Logger()
        memory.get_kv = lambda tokens: (tokens // 16) * 20
        memory.get_evict_kv = lambda req: memory.get_kv(
            ((req.num_computed_tokens + 15) // 16) * 16)
        memory.get_total_kv = memory.get_evict_kv

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = "test/model"
        scheduler.node_id = 0
        scheduler.instance_id = instance_id
        scheduler.num_npus = 1
        scheduler.tp_size = 1
        scheduler.start_npu = 0
        scheduler.pd_type = pd_type
        scheduler.max_num_batched_tokens = 32
        scheduler.max_num_seqs = 8
        scheduler.pp_size = 1
        scheduler.prioritize_prefill = False
        scheduler.enable_chunked_prefill = True
        scheduler.long_prefill_token_threshold = 0
        scheduler.enable_prefix_caching = False
        scheduler.enable_session_kv_retention = True
        scheduler.session_kv_ttl_ns = 0
        scheduler.enable_kv_offloading = cpu_pool is not None
        scheduler.kv_offload_high_watermark = 0.9
        scheduler.kv_offload_low_watermark = 0.8
        scheduler.kv_offload_victim_policy = "lru"
        scheduler.prefix_storage = None
        scheduler.request = []
        scheduler.inflight = []
        scheduler.done = []
        scheduler.pending_pd_handoffs = []
        scheduler.session_kv_states = {}
        scheduler.batch_ids = -1
        scheduler.migration_ids = -1
        scheduler.memory = memory
        scheduler.kv_offload_stats = KVOffloadStats()
        scheduler.logger = _Logger()
        return scheduler

    def _completed_decode_batch(self, request):
        batch = Batch(
            0, "test/model", 1, 0, [], [], 0, 1, [], [], [10],
            100, 0)
        batch.requests.append(request)
        batch.fired.append(0)
        return batch

    def test_non_terminal_turn_parks_existing_allocation(self):
        scheduler = self._scheduler()
        request = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            reuse_previous_kv=True,
            session_kv_ttl_ns=1_000)
        request.num_computed_tokens = 10
        request.is_init = False
        scheduler.inflight.append(self._completed_decode_batch(request))

        _, generated, finished = scheduler.add_done(1, 0, 120)

        self.assertEqual(generated, 1)
        self.assertEqual([req.id for req in finished], [0])
        self.assertEqual(scheduler.memory.npu_used, 60)
        self.assertEqual(request.kv_owner_session_id, "session-0")
        state = scheduler.session_kv_states["session-0"]
        self.assertEqual(state.source_request_id, request.id)
        self.assertEqual(state.cached_tokens, 11)
        self.assertEqual(state.bytes_per_rank, 20)
        self.assertEqual(state.bytes_full_cluster, 20)
        self.assertEqual(state.residency, KVResidency.NPU)
        self.assertEqual(state.parked_at_ns, 120)
        self.assertEqual(state.expires_at_ns, 1_120)

    def test_terminal_turn_frees_current_and_predecessor_state(self):
        scheduler = self._scheduler(npu_used=80)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        scheduler._park_completed_session_kv(predecessor, 100)

        request = Request(
            1, "test/model", 12, 16, 110, 0,
            session_id="session-0", sub_request_index=1,
            session_has_next=False, reuse_previous_kv=True)
        request.num_computed_tokens = 14
        request.is_init = False
        scheduler.inflight.append(self._completed_decode_batch(request))

        _, generated, finished = scheduler.add_done(1, 0, 140)

        self.assertEqual(generated, 1)
        self.assertEqual([req.id for req in finished], [1])
        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertIsNone(request.kv_owner_session_id)

    def test_non_reusing_session_does_not_park_kv(self):
        scheduler = self._scheduler()
        request = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=False,
            reuse_previous_kv=False)
        request.num_computed_tokens = 10
        request.is_init = False
        scheduler.inflight.append(self._completed_decode_batch(request))

        scheduler.add_done(1, 0, 120)

        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertIsNone(request.kv_owner_session_id)

    def test_append_only_continuation_claims_npu_state(self):
        scheduler = self._scheduler()
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        scheduler._park_completed_session_kv(predecessor, 100)

        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 110, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
                "reused_prefix_toks": None,
                "session_kv_ttl_ns": None,
            },
        )

        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(scheduler.memory.npu_used, 60)
        self.assertEqual(continuation.session_id, "session-0")
        self.assertEqual(continuation.sub_request_index, 1)
        self.assertTrue(continuation.reuse_previous_kv)
        self.assertEqual(continuation.num_computed_tokens, 11)
        self.assertEqual(continuation.prefix_cache_hit, 11)
        self.assertEqual(continuation.session_kv_hit_tokens, 11)
        self.assertEqual(continuation.session_kv_hit_tier, "NPU")
        self.assertEqual(
            continuation.original_input - continuation.num_computed_tokens,
            1,
        )

    def test_partial_override_frees_excess_blocks(self):
        scheduler = self._scheduler(npu_used=100)
        predecessor = Request(
            0, "test/model", 24, 42, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 40
        scheduler._park_completed_session_kv(predecessor, 100)

        continuation = scheduler.add_request(
            [1, "test/model", 20, 24, 110, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
                "reused_prefix_toks": 10,
                "session_kv_ttl_ns": None,
            },
        )

        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(continuation.num_computed_tokens, 10)
        self.assertEqual(continuation.session_kv_hit_tokens, 10)
        self.assertEqual(scheduler.memory.npu_used, 60)

    def test_expired_state_is_released_before_equal_time_arrival(self):
        scheduler = self._scheduler()
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=10)
        predecessor.num_computed_tokens = 11
        scheduler._park_completed_session_kv(predecessor, 100)

        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 110, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )

        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertEqual(continuation.session_kv_hit_tokens, 0)
        self.assertEqual(scheduler.memory.npu_used, 40)

    def test_continuation_before_ttl_claims_state_and_disarms_expiry(self):
        scheduler = self._scheduler()
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=20)
        predecessor.num_computed_tokens = 11
        scheduler._park_completed_session_kv(predecessor, 100)

        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 119, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )

        self.assertEqual(continuation.session_kv_hit_tokens, 11)
        self.assertEqual(continuation.session_kv_hit_tier, "NPU")
        self.assertEqual(scheduler.session_kv_states, {})
        self.assertIsNone(scheduler.get_next_session_kv_expiry())
        self.assertEqual(scheduler.expire_session_kv(120), [])

    def test_continuation_after_ttl_is_full_prefill_miss(self):
        scheduler = self._scheduler()
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=20)
        predecessor.num_computed_tokens = 11
        scheduler._park_completed_session_kv(predecessor, 100)
        scheduler.expire_session_kv(120)

        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 121, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )

        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertEqual(continuation.session_kv_hit_tokens, 0)
        self.assertIsNone(continuation.session_kv_hit_tier)
        self.assertEqual(
            scheduler.get_kv_offload_stats()["session_miss_count"], 1)

    def test_incompatible_kv_layout_invalidates_state_and_recomputes(self):
        scheduler = self._scheduler()
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)
        state.block_size = 32

        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 110, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )

        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertEqual(continuation.session_kv_hit_tokens, 0)
        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertEqual(
            scheduler.get_kv_offload_stats()["session_miss_count"], 1)

    def test_default_ttl_hard_expires_idle_npu_state(self):
        scheduler = self._scheduler()
        scheduler.session_kv_ttl_ns = 25
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)

        self.assertEqual(state.expires_at_ns, 125)
        self.assertEqual(scheduler.get_next_session_kv_expiry(), 125)
        self.assertEqual(scheduler.expire_session_kv(124), [])
        self.assertEqual(scheduler.memory.npu_used, 60)

        self.assertEqual(
            scheduler.expire_session_kv(125), ["session-0"])
        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertIsNone(scheduler.get_next_session_kv_expiry())

    def test_session_zero_ttl_disables_instance_default(self):
        scheduler = self._scheduler()
        scheduler.session_kv_ttl_ns = 25
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=0)
        predecessor.num_computed_tokens = 11

        state = scheduler._park_completed_session_kv(predecessor, 100)

        self.assertIsNone(state.expires_at_ns)
        self.assertEqual(scheduler.expire_session_kv(1_000), [])
        self.assertIn("session-0", scheduler.session_kv_states)

    def test_hard_expiry_releases_cpu_state(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        scheduler = self._scheduler(cpu_pool=pool)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=50)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)
        eviction = scheduler._start_session_kv_eviction([state], 110, 0)
        scheduler.add_done(eviction.batch_id + 1, 0, 130)

        scheduler.expire_session_kv(150)

        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(pool.used, 0)
        self.assertEqual(scheduler.memory.npu_used, 40)

    def test_expiry_during_d2h_cleans_destination_after_completion(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        scheduler = self._scheduler(cpu_pool=pool)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=20)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)
        eviction = scheduler._start_session_kv_eviction([state], 110, 0)

        scheduler.expire_session_kv(120)

        self.assertTrue(state.invalidated)
        self.assertEqual(pool.reserved, 20)
        scheduler.add_done(eviction.batch_id + 1, 0, 130)
        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 0)

    def test_expiry_during_h2d_turns_continuation_into_full_miss(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        scheduler = self._scheduler(cpu_pool=pool)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=100)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)
        eviction = scheduler._start_session_kv_eviction([state], 110, 0)
        scheduler.add_done(eviction.batch_id + 1, 0, 130)
        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 190, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )
        reload_batch = scheduler._start_session_kv_reload(
            continuation, 190, 0)

        scheduler.expire_session_kv(200)
        scheduler.add_done(reload_batch.batch_id + 1, 0, 210)

        self.assertEqual(scheduler.session_kv_states, {})
        self.assertFalse(continuation.pending_session_kv)
        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertEqual(continuation.prefix_cache_hit, 0)
        self.assertEqual(continuation.session_kv_hit_tokens, 0)
        self.assertIsNone(continuation.session_kv_hit_tier)
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertEqual(scheduler.memory.npu_reserved, 0)
        self.assertEqual(pool.used, 0)
        stats = scheduler.get_kv_offload_stats()
        self.assertEqual(stats["session_cpu_hit_count"], 0)
        self.assertEqual(stats["session_cpu_hit_tokens"], 0)
        self.assertEqual(stats["session_miss_count"], 1)
        self.assertEqual(stats["session_recomputed_prompt_tokens"], 12)

    def test_cpu_state_expiry_resets_queued_continuation(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        scheduler = self._scheduler(cpu_pool=pool)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            session_kv_ttl_ns=50)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)
        eviction = scheduler._start_session_kv_eviction([state], 110, 0)
        scheduler.add_done(eviction.batch_id + 1, 0, 130)
        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 140, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )
        self.assertTrue(continuation.pending_session_kv)

        scheduler.expire_session_kv(150)

        self.assertFalse(continuation.pending_session_kv)
        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertEqual(continuation.prefix_cache_hit, 0)
        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(pool.used, 0)
        stats = scheduler.get_kv_offload_stats()
        self.assertEqual(stats["session_cpu_hit_count"], 0)
        self.assertEqual(stats["session_miss_count"], 1)
        self.assertEqual(stats["session_recomputed_prompt_tokens"], 12)

    def test_different_session_cannot_claim_parked_state(self):
        scheduler = self._scheduler()
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-a", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        scheduler._park_completed_session_kv(predecessor, 100)

        unrelated = scheduler.add_request(
            [1, "test/model", 12, 16, 110, 0],
            session_metadata={
                "session_id": "session-b",
                "sub_request_index": 0,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )

        self.assertEqual(unrelated.num_computed_tokens, 0)
        self.assertIn("session-a", scheduler.session_kv_states)
        self.assertEqual(scheduler.memory.npu_used, 60)

    def test_parked_session_uses_migration_batches_for_cpu_round_trip(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        scheduler = self._scheduler(cpu_pool=pool)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)

        eviction = scheduler._start_session_kv_eviction([state], 110, 0)

        self.assertEqual(state.residency, KVResidency.NPU_TO_CPU)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 20)
        self.assertEqual(scheduler.memory.npu_used, 60)

        scheduler.add_done(eviction.batch_id + 1, 0, 130)

        self.assertEqual(state.residency, KVResidency.CPU)
        self.assertEqual(pool.used, 20)
        self.assertEqual(pool.reserved, 0)
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertEqual(
            scheduler.get_kv_offload_stats()["preemption_count"], 0)

        continuation = scheduler.add_request(
            [1, "test/model", 12, 16, 140, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )
        self.assertTrue(continuation.pending_session_kv)
        self.assertEqual(continuation.num_computed_tokens, 11)
        self.assertEqual(continuation.session_kv_hit_tier, "CPU_PENDING")
        self.assertEqual(
            scheduler.get_kv_offload_stats()["session_cpu_hit_count"], 0)

        reload_batch = scheduler._start_session_kv_reload(
            continuation, 150, 0)
        self.assertEqual(state.residency, KVResidency.CPU_TO_NPU)
        self.assertEqual(scheduler.memory.npu_reserved, 20)
        self.assertEqual(pool.used, 20)

        scheduler.add_done(reload_batch.batch_id + 1, 0, 170)

        self.assertFalse(continuation.pending_session_kv)
        self.assertEqual(continuation.session_kv_hit_tier, "CPU")
        self.assertEqual(scheduler.session_kv_states, {})
        self.assertEqual(scheduler.memory.npu_used, 60)
        self.assertEqual(scheduler.memory.npu_reserved, 0)
        self.assertEqual(pool.used, 0)
        stats = scheduler.get_kv_offload_stats()
        self.assertEqual(stats["evict_bytes"], 20)
        self.assertEqual(stats["reload_bytes"], 20)
        self.assertEqual(stats["session_cpu_hit_count"], 1)
        self.assertEqual(stats["session_cpu_hit_tokens"], 11)
        self.assertEqual(stats["session_reload_wait_ns"], 30)

    def test_cpu_capacity_failure_leaves_parked_state_on_npu(self):
        pool = NodeCPUKVPool(node_id=0, capacity=0)
        scheduler = self._scheduler(cpu_pool=pool)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        state = scheduler._park_completed_session_kv(predecessor, 100)

        batch = scheduler._start_session_kv_eviction([state], 110, 0)

        self.assertIsNone(batch)
        self.assertEqual(state.residency, KVResidency.NPU)
        self.assertIsNone(state.migration_id)
        self.assertEqual(scheduler.memory.npu_used, 60)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 0)

    def test_partial_cpu_hit_reloads_prefix_and_frees_full_source(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        scheduler = self._scheduler(npu_used=100, cpu_pool=pool)
        predecessor = Request(
            0, "test/model", 24, 42, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 40
        state = scheduler._park_completed_session_kv(predecessor, 100)

        eviction = scheduler._start_session_kv_eviction([state], 110, 0)
        scheduler.add_done(eviction.batch_id + 1, 0, 130)
        self.assertEqual(state.bytes_per_rank, 60)
        self.assertEqual(pool.used, 60)

        continuation = scheduler.add_request(
            [1, "test/model", 20, 24, 140, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
                "reused_prefix_toks": 10,
            },
        )
        self.assertEqual(continuation.pending_session_kv_bytes_per_rank, 20)

        reload_batch = scheduler._start_session_kv_reload(
            continuation, 150, 0)
        scheduler.add_done(reload_batch.batch_id + 1, 0, 170)

        self.assertEqual(continuation.num_computed_tokens, 10)
        self.assertEqual(scheduler.memory.npu_used, 60)
        self.assertEqual(pool.used, 0)
        self.assertEqual(
            scheduler.get_kv_offload_stats()["reload_bytes"], 20)

    def test_active_eviction_drops_node_wide_parked_cpu_state_first(self):
        pool = NodeCPUKVPool(node_id=0, capacity=20)
        session_owner = self._scheduler(
            cpu_pool=pool, instance_id=0)
        predecessor = Request(
            0, "test/model", 8, 12, 0, 0,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True)
        predecessor.num_computed_tokens = 11
        state = session_owner._park_completed_session_kv(predecessor, 100)
        eviction = session_owner._start_session_kv_eviction(
            [state], 110, 0)
        session_owner.add_done(eviction.batch_id + 1, 0, 130)
        self.assertEqual(pool.used, 20)

        active_owner = self._scheduler(
            cpu_pool=pool, instance_id=1)
        active = Request(9, "test/model", 8, 20, 0, 1)
        active.num_computed_tokens = 11
        active.is_init = False

        active_eviction = active_owner._start_kv_eviction(
            [active], 150, 0)

        self.assertIsNotNone(active_eviction)
        self.assertEqual(active.kv_residency, KVResidency.NPU_TO_CPU)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 20)
        self.assertEqual(session_owner.session_kv_states, {})
        stats = session_owner.get_kv_offload_stats()
        self.assertEqual(stats["session_capacity_drop_count"], 1)
        self.assertEqual(stats["session_capacity_drop_bytes"], 20)
        self.assertEqual(stats["session_parked_cpu_byte_ns"], 400)

        continuation = session_owner.add_request(
            [1, "test/model", 12, 16, 160, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )
        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertEqual(
            session_owner.get_kv_offload_stats()["session_miss_count"], 1)

    def test_pd_continuation_uses_cpu_between_decode_and_prefill(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        decode = self._scheduler(
            cpu_pool=pool, instance_id=1, pd_type="decode")
        predecessor = Request(
            0, "test/model", 8, 12, 0, 1,
            session_id="session-0", sub_request_index=0,
            session_has_next=True, retain_session_kv=True,
            reuse_previous_kv=True)
        predecessor.num_computed_tokens = 11
        state = decode._park_completed_session_kv(predecessor, 100)
        self.assertTrue(state.mandatory_cpu_offload)
        d2h = decode._start_session_kv_eviction([state], 110, 0)
        decode.add_done(d2h.batch_id + 1, 0, 130)
        self.assertEqual(state.residency, KVResidency.CPU)
        self.assertEqual(pool.used, 20)

        prefill = self._scheduler(
            npu_used=40, cpu_pool=pool, instance_id=0,
            pd_type="prefill")
        continuation = prefill.add_request(
            [1, "test/model", 12, 16, 140, 0],
            session_metadata={
                "session_id": "session-0",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
        )

        self.assertEqual(decode.session_kv_states, {})
        self.assertTrue(continuation.pending_session_kv)
        self.assertEqual(continuation.num_computed_tokens, 11)
        h2d = prefill._start_session_kv_reload(
            continuation, 150, 0)
        prefill.add_done(h2d.batch_id + 1, 0, 170)
        prefill.add_done(h2d.batch_id + 1, 1, 170)

        self.assertEqual(prefill.session_kv_states, {})
        self.assertEqual(pool.used, 0)
        self.assertEqual(prefill.memory.npu_used, 60)
        self.assertEqual(continuation.session_kv_hit_tier, "CPU")
        self.assertEqual(
            continuation.original_input - continuation.num_computed_tokens,
            1,
        )
        stats = prefill.get_kv_offload_stats()
        self.assertEqual(stats["session_cpu_hit_count"], 1)
        self.assertEqual(stats["session_cpu_reload_bytes"], 20)


class SessionKVRuntimeConfigTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "enable_attn_offloading": False,
            "enable_sub_batch_interleaving": False,
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048,
            "long_prefill_token_threshold": 0,
            "block_size": 16,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "enable_session_kv_retention": True,
            "session_kv_ttl_ns": 0,
            "enable_kv_offloading": False,
            "kv_offload_high_watermark": 0.9,
            "kv_offload_low_watermark": 0.8,
            "kv_offload_victim_policy": "lru",
            "prioritize_prefill": False,
            "enable_local_offloading": False,
            "enable_block_copy": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _instance(self, pd_type=None):
        return {
            "instance_id": 0,
            "model_name": "meta-llama/Llama-3.1-8B",
            "pd_type": pd_type,
        }

    def test_enables_colocated_npu_session_hits(self):
        configs = _build_instance_runtime_configs(
            [self._instance()], self._args(), {"bfloat16": 16})

        self.assertTrue(configs[0]["enable_session_kv_retention"])

    def test_accepts_cpu_offload_integration(self):
        configs = _build_instance_runtime_configs(
            [self._instance()],
            self._args(enable_kv_offloading=True),
            {"bfloat16": 16},
        )

        self.assertTrue(configs[0]["enable_session_kv_retention"])
        self.assertTrue(configs[0]["enable_kv_offloading"])

    def test_rejects_negative_instance_default_ttl(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            _build_instance_runtime_configs(
                [self._instance()],
                self._args(session_kv_ttl_ns=-1),
                {"bfloat16": 16},
            )

    def test_rejects_generic_prefix_cache_and_pd(self):
        cases = [
            (self._args(enable_prefix_caching=True), self._instance()),
            (self._args(), self._instance(pd_type="prefill")),
        ]
        for args, instance in cases:
            with self.subTest(args=args, instance=instance):
                with self.assertRaises(ValueError):
                    _build_instance_runtime_configs(
                        [instance], args, {"bfloat16": 16})

    def test_accepts_fully_enabled_same_node_pd_policy(self):
        instances = [
            {**self._instance(pd_type="prefill"), "node_id": 0},
            {**self._instance(pd_type="decode"), "instance_id": 1,
             "node_id": 0},
        ]
        configs = _build_instance_runtime_configs(
            instances,
            self._args(enable_kv_offloading=True),
            {"bfloat16": 16},
        )

        offload_nodes = _validate_kv_offload_node_scope(
            instances, configs, "None")

        self.assertEqual(offload_nodes, {0})
        self.assertTrue(all(
            config["enable_session_kv_retention"]
            for config in configs))

    def test_rejects_partially_enabled_pd_session_policy(self):
        instances = [
            {**self._instance(pd_type="prefill"), "node_id": 0},
            {**self._instance(pd_type="decode"), "instance_id": 1,
             "node_id": 0, "enable_session_kv_retention": False},
        ]
        configs = _build_instance_runtime_configs(
            instances,
            self._args(enable_kv_offloading=True),
            {"bfloat16": 16},
        )

        with self.assertRaisesRegex(ValueError, "every prefill/decode"):
            _validate_kv_offload_node_scope(instances, configs, "None")


class RouterSessionMetadataTest(unittest.TestCase):
    class FakeScheduler:
        def __init__(self, instance_id):
            self.instance_id = instance_id
            self.node_id = 0
            self.pd_type = None
            self.model = "test/model"
            self.max_num_seqs = 8
            self.request = []
            self.inflight = []
            self.enable_prefix_caching = False
            self.enable_session_kv_retention = True
            self.session_kv_states = {}
            self.received = []

        def add_request(self, req, is_init=True, session_metadata=None):
            self.received.append((req, session_metadata))

    def test_router_propagates_metadata_and_preserves_affinity(self):
        schedulers = [self.FakeScheduler(0), self.FakeScheduler(1)]
        router = Router(2, schedulers, 0, routing_policy="RR")
        router._load_agentic_session({
            "session_id": "session-0",
            "arrival_time_ns": 0,
            "reuse_previous_kv": True,
            "session_kv_ttl_ns": 1_000,
            "sub_requests": [
                {
                    "input_toks": 8,
                    "output_toks": 4,
                    "tool_duration_ns": 10,
                },
                {
                    "input_toks": 14,
                    "output_toks": 4,
                    "reused_prefix_toks": 11,
                    "tool_duration_ns": 0,
                },
            ],
        }, enable_prefix_caching=False)

        router.route_arrived_requests(0)
        first_meta = schedulers[0].received[0][1]
        self.assertEqual(first_meta["session_id"], "session-0")
        self.assertEqual(first_meta["sub_request_index"], 0)
        self.assertTrue(first_meta["session_has_next"])
        self.assertTrue(first_meta["reuse_previous_kv"])
        self.assertEqual(first_meta["session_kv_ttl_ns"], 1_000)

        router.notify_request_completed(0, 100)
        router.route_arrived_requests(110)
        second_meta = schedulers[0].received[1][1]
        self.assertEqual(schedulers[1].received, [])
        self.assertEqual(second_meta["sub_request_index"], 1)
        self.assertFalse(second_meta["session_has_next"])
        self.assertEqual(second_meta["reused_prefix_toks"], 11)

    def test_router_rejects_negative_session_scalars(self):
        router = Router(
            1, [self.FakeScheduler(0)], 0, routing_policy="RR")
        row = {
            "session_id": "session-0",
            "arrival_time_ns": 0,
            "session_kv_ttl_ns": -1,
            "sub_requests": [{
                "input_toks": 8,
                "output_toks": 4,
                "tool_duration_ns": 0,
            }],
        }

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            router._load_agentic_session(row, enable_prefix_caching=False)

    def test_router_rejects_duplicate_session_ids(self):
        router = Router(
            1, [self.FakeScheduler(0)], 0, routing_policy="RR")
        row = {
            "session_id": "session-0",
            "arrival_time_ns": 0,
            "sub_requests": [{
                "input_toks": 8,
                "output_toks": 4,
                "tool_duration_ns": 0,
            }],
        }
        router._load_agentic_session(row, enable_prefix_caching=False)

        with self.assertRaisesRegex(ValueError, "Duplicate agentic session id"):
            router._load_agentic_session(row, enable_prefix_caching=False)

    def test_pd_bridge_wait_does_not_block_unrelated_arrival(self):
        prefill = self.FakeScheduler(0)
        prefill.pd_type = "prefill"
        decode = self.FakeScheduler(1)
        decode.pd_type = "decode"
        router = Router(
            2, [prefill, decode], 0, routing_policy="RR",
            same_node_pd_only=True)
        router._session_schedulers["blocked-session"] = prefill
        state = SimpleNamespace(
            mandatory_cpu_offload=True,
            residency=KVResidency.NPU_TO_CPU,
            invalidated=False,
        )
        decode.session_kv_states["blocked-session"] = state
        router._pending_requests = [
            {
                "index": 0,
                "input_toks": 12,
                "output_toks": 16,
                "arrival_time_ns": 100,
                "session_id": "blocked-session",
                "sub_request_index": 1,
                "session_has_next": False,
                "reuse_previous_kv": True,
            },
            {
                "index": 1,
                "input_toks": 8,
                "output_toks": 12,
                "arrival_time_ns": 100,
            },
        ]

        self.assertEqual(router.route_arrived_requests(100), 1)
        self.assertEqual([item[0][0] for item in prefill.received], [1])
        self.assertEqual(router.get_next_pending_arrival(), 100)

        state.residency = KVResidency.CPU
        self.assertEqual(router.route_arrived_requests(110), 1)
        self.assertEqual([item[0][0] for item in prefill.received], [1, 0])


class SessionKVIdleEventTest(unittest.TestCase):
    def test_expiry_precedes_later_request_arrival(self):
        router = SimpleNamespace(get_next_pending_arrival=lambda: 200)
        schedulers = [
            SimpleNamespace(get_next_session_kv_expiry=lambda: 150),
            SimpleNamespace(get_next_session_kv_expiry=lambda: None),
        ]

        self.assertEqual(_next_idle_event(router, schedulers), 150)


if __name__ == "__main__":
    unittest.main()
