import csv
import unittest
from collections import deque
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

from serving.core.memory_model import Device, MemoryModel, NodeCPUKVPool
from serving.core.request import (
    Batch,
    BatchKind,
    KVResidency,
    Request,
    SessionKVState,
)
from serving.core.router import Router
from serving.core.scheduler import Scheduler
from serving.core.trace_generator import generate_trace
from serving.core.config_builder import _host_transfer_config
from serving.__main__ import (
    _build_instance_runtime_configs,
    _dp_execution_idle,
    _kv_offload_output_file,
    _monotonic_event_time,
    _validate_kv_offload_node_scope,
)


class SimulationTimeSafetyTest(unittest.TestCase):
    def test_event_time_never_moves_backward(self):
        self.assertEqual(_monotonic_event_time(100, 250), 250)
        self.assertEqual(_monotonic_event_time(250, 100), 250)

    def test_dp_execution_is_not_idle_with_pending_or_active_waves(self):
        pending = {"A": {0: deque(), 1: deque()}}
        active = {"A": {}}
        self.assertTrue(_dp_execution_idle(pending, active))

        pending["A"][0].append(object())
        self.assertFalse(_dp_execution_idle(pending, active))
        pending["A"][0].clear()

        active["A"][3] = 1
        self.assertFalse(_dp_execution_idle(pending, active))


class KVResidencyTest(unittest.TestCase):
    def setUp(self):
        self.request = Request(7, "test/model", 16, 32, 0, 0)

    def test_offload_and_reload_state_machine(self):
        self.request.begin_kv_offload(10)
        self.assertEqual(self.request.kv_residency, KVResidency.NPU_TO_CPU)
        self.assertTrue(self.request.is_kv_migrating())

        self.request.complete_kv_offload(10)
        self.assertEqual(self.request.kv_residency, KVResidency.CPU)

        self.request.begin_kv_reload(11)
        self.assertEqual(self.request.kv_residency, KVResidency.CPU_TO_NPU)

        self.request.complete_kv_reload(11)
        self.assertEqual(self.request.kv_residency, KVResidency.NPU)

    def test_cancel_restores_source_residency(self):
        self.request.begin_kv_offload(10)
        self.request.cancel_kv_migration(10)
        self.assertEqual(self.request.kv_residency, KVResidency.NPU)

        self.request.mark_kv_on_cpu()
        self.request.begin_kv_reload(11)
        self.request.cancel_kv_migration(11)
        self.assertEqual(self.request.kv_residency, KVResidency.CPU)

    def test_migration_id_must_match(self):
        self.request.begin_kv_offload(10)
        with self.assertRaises(RuntimeError):
            self.request.complete_kv_offload(11)


class CapacityReservationTest(unittest.TestCase):
    def test_node_cpu_pool_reserve_commit_and_release(self):
        pool = NodeCPUKVPool(node_id=0, capacity=100)
        pool.allocate(40)
        pool.reserve(50)

        self.assertEqual(pool.used, 40)
        self.assertEqual(pool.reserved, 50)
        self.assertFalse(pool.is_avail(11))

        pool.commit_reservation(50)
        self.assertEqual(pool.used, 90)
        self.assertEqual(pool.reserved, 0)

        pool.free(90)
        self.assertEqual(pool.used, 0)

    def test_node_cpu_pool_cancel_keeps_used_unchanged(self):
        pool = NodeCPUKVPool(node_id=0, capacity=100)
        pool.allocate(40)
        pool.reserve(50)
        pool.cancel_reservation(50)

        self.assertEqual(pool.used, 40)
        self.assertEqual(pool.reserved, 0)

    def test_npu_reservation_counts_against_capacity(self):
        memory = MemoryModel.__new__(MemoryModel)
        memory.node_id = 0
        memory.instance_id = 0
        memory.npu_mem = 100
        memory.npu_used = 60
        memory.npu_reserved = 0
        memory.cpu_mem = 200
        memory.cpu_used = 0
        memory.cpu_reserved = 0
        memory.cpu_kv_pool = None

        memory.reserve_live_kv(30, Device.NPU)
        self.assertEqual(memory.npu_used, 60)
        self.assertEqual(memory.npu_reserved, 30)
        self.assertFalse(memory.is_avail(11, Device.NPU))

        memory.commit_live_kv_reservation(30, Device.NPU)
        self.assertEqual(memory.npu_used, 90)
        self.assertEqual(memory.npu_reserved, 0)

    def test_cpu_reservation_is_full_cluster_capacity(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        memory = MemoryModel.__new__(MemoryModel)
        memory.node_id = 0
        memory.instance_id = 0
        memory.npu_mem = 100
        memory.npu_used = 60
        memory.npu_reserved = 0
        memory.cpu_mem = 200
        memory.cpu_used = 0
        memory.cpu_reserved = 0
        memory.cpu_kv_pool = pool

        memory.reserve_live_kv(120, Device.CPU)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 120)

        memory.commit_live_kv_reservation(120, Device.CPU)
        self.assertEqual(pool.used, 120)
        self.assertEqual(pool.reserved, 0)


class HostTransferConfigTest(unittest.TestCase):
    def test_translates_pipelined_host_link(self):
        result = _host_transfer_config({
            "host_link_bw": 64,
            "host_link_latency": 800,
        }, required=True)
        self.assertEqual(result, {
            "host-link-bw": 64,
            "host-link-latency": 800,
            "host-transfer-model": "pipelined",
        })

    def test_accepts_serial_model(self):
        result = _host_transfer_config({
            "host_link_bw": 64,
            "host_link_latency": 800,
            "host_transfer_model": "serial",
        }, required=True)
        self.assertEqual(result["host-transfer-model"], "serial")

    def test_requires_link_when_cpu_offload_is_enabled(self):
        with self.assertRaises(KeyError):
            _host_transfer_config({}, required=True)

    def test_rejects_partial_or_invalid_link(self):
        with self.assertRaises(KeyError):
            _host_transfer_config({"host_link_bw": 64})
        with self.assertRaises(KeyError):
            _host_transfer_config({"host_transfer_model": "serial"})
        with self.assertRaises(ValueError):
            _host_transfer_config({
                "host_link_bw": 0,
                "host_link_latency": 800,
            })
        with self.assertRaises(ValueError):
            _host_transfer_config({
                "host_link_bw": 64,
                "host_link_latency": -1,
            })
        with self.assertRaises(ValueError):
            _host_transfer_config({
                "host_link_bw": 64,
                "host_link_latency": 800,
                "host_transfer_model": "unknown",
            })


class OffloadNodeScopeTest(unittest.TestCase):
    def test_rejects_prefix_caching_on_offload_instance(self):
        args = SimpleNamespace(
            dtype="bfloat16",
            kv_cache_dtype="auto",
            enable_attn_offloading=False,
            enable_sub_batch_interleaving=False,
            max_num_seqs=128,
            max_num_batched_tokens=2048,
            long_prefill_token_threshold=0,
            block_size=16,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            enable_session_kv_retention=False,
            session_kv_ttl_ns=0,
            enable_kv_offloading=True,
            kv_offload_high_watermark=0.9,
            kv_offload_low_watermark=0.8,
            kv_offload_victim_policy="lru",
            prioritize_prefill=False,
            enable_local_offloading=False,
            enable_block_copy=True,
        )
        instances = [{
            "instance_id": 0,
            "model_name": "meta-llama/Llama-3.1-8B",
        }]

        with self.assertRaisesRegex(ValueError, "does not support prefix caching"):
            _build_instance_runtime_configs(
                instances,
                args,
                {"bfloat16": 2},
            )

    def test_kv_offload_output_path_preserves_extension(self):
        self.assertEqual(
            _kv_offload_output_file("outputs/run.csv"),
            "outputs/run_kv_offload.csv",
        )
        self.assertEqual(
            _kv_offload_output_file("outputs/run"),
            "outputs/run_kv_offload.csv",
        )

    def test_rejects_partially_enabled_pd_node(self):
        instances = [
            {"node_id": 0, "pd_type": "prefill"},
            {"node_id": 0, "pd_type": "decode"},
        ]
        runtime = [
            {"enable_kv_offloading": True, "enable_prefix_caching": False},
            {"enable_kv_offloading": False, "enable_prefix_caching": False},
        ]

        with self.assertRaises(ValueError):
            _validate_kv_offload_node_scope(instances, runtime, "None")

    def test_rejects_cpu_prefix_cache_on_offload_node(self):
        instances = [
            {"node_id": 0, "pd_type": None},
            {"node_id": 0, "pd_type": None},
        ]
        runtime = [
            {"enable_kv_offloading": True, "enable_prefix_caching": False},
            {"enable_kv_offloading": False, "enable_prefix_caching": True},
        ]

        with self.assertRaises(ValueError):
            _validate_kv_offload_node_scope(instances, runtime, "CPU")

    def test_returns_only_nodes_with_offloading(self):
        instances = [
            {"node_id": 0, "pd_type": None},
            {"node_id": 1, "pd_type": None},
        ]
        runtime = [
            {"enable_kv_offloading": True, "enable_prefix_caching": False},
            {"enable_kv_offloading": False, "enable_prefix_caching": True},
        ]

        self.assertEqual(
            _validate_kv_offload_node_scope(instances, runtime, "CPU"),
            {0},
        )


class MigrationBatchTest(unittest.TestCase):
    def test_batch_defaults_to_compute(self):
        batch = Batch(0, "test/model", 0, 0, [], [], 0, 0, [], [], [], 0, 0)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertEqual(batch.migrations, [])

    def _scheduler(self):
        class Logger:
            def info(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

        pool = NodeCPUKVPool(node_id=0, capacity=200)
        memory = MemoryModel.__new__(MemoryModel)
        memory.node_id = 0
        memory.instance_id = 0
        memory.num_npus = 1
        memory.npu_mem = 100
        memory.npu_used = 60
        memory.npu_reserved = 0
        memory.cpu_mem = 200
        memory.cpu_used = 0
        memory.cpu_reserved = 0
        memory.cpu_kv_pool = pool
        memory.weight = 0
        memory.logger = Logger()
        memory.get_evict_kv = lambda req: 20

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = "test/model"
        scheduler.num_npus = 1
        scheduler.start_npu = 0
        scheduler.pd_type = None
        scheduler.max_num_batched_tokens = 32
        scheduler.prioritize_prefill = False
        scheduler.inflight = []
        scheduler.request = []
        scheduler.pending_pd_handoffs = []
        scheduler.session_kv_states = {}
        scheduler.batch_ids = -1
        scheduler.migration_ids = -1
        scheduler.memory = memory
        scheduler.logger = Logger()
        return scheduler, pool

    def test_eviction_and_reload_commit_only_on_batch_completion(self):
        scheduler, pool = self._scheduler()
        request = Request(7, "test/model", 16, 32, 0, 0)
        request.num_computed_tokens = 20
        original_tokens = request.num_computed_tokens

        evict_batch = scheduler._start_kv_eviction([request], 100, 0)
        self.assertEqual(evict_batch.kind, BatchKind.KV_EVICT)
        self.assertEqual(request.kv_residency, KVResidency.NPU_TO_CPU)
        self.assertEqual(scheduler.memory.npu_used, 60)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 20)

        scheduler.add_done(evict_batch.batch_id + 1, 0, 150)
        self.assertEqual(request.kv_residency, KVResidency.CPU)
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertEqual(pool.used, 20)
        self.assertEqual(pool.reserved, 0)
        self.assertEqual(request.num_computed_tokens, original_tokens)

        reload_batch = scheduler._start_kv_reload([request], 200, 0)
        self.assertEqual(reload_batch.kind, BatchKind.KV_RELOAD)
        self.assertEqual(request.kv_residency, KVResidency.CPU_TO_NPU)
        self.assertEqual(scheduler.memory.npu_used, 40)
        self.assertEqual(scheduler.memory.npu_reserved, 20)
        self.assertEqual(pool.used, 20)

        scheduler.add_done(reload_batch.batch_id + 1, 0, 260)
        self.assertEqual(request.kv_residency, KVResidency.NPU)
        self.assertEqual(scheduler.memory.npu_used, 60)
        self.assertEqual(scheduler.memory.npu_reserved, 0)
        self.assertEqual(pool.used, 0)
        self.assertEqual(request.num_computed_tokens, original_tokens)

        stats = scheduler.get_kv_offload_stats()
        self.assertEqual(stats["preemption_count"], 1)
        self.assertEqual(stats["evict_bytes"], 20)
        self.assertEqual(stats["reload_bytes"], 20)
        self.assertEqual(stats["migration_time_ns"], 110)
        self.assertEqual(stats["reload_stall_count"], 1)
        self.assertEqual(stats["reload_stall_ns"], 60)
        self.assertEqual(stats["npu_peak_reserved_bytes_per_rank"], 20)
        self.assertEqual(stats["cpu_peak_reserved_bytes"], 20)

    def test_migration_trace_contains_only_cpu_operation(self):
        batch = Batch(
            3, "test/model", 0, 0, [], [], 0, 0, [], [], [], 0, 0,
            evict=4096, kind=BatchKind.KV_EVICT)
        with TemporaryDirectory() as temp_dir:
            generate_trace(
                batch, "test-hardware", 1, 1, 1, 1,
                node_id=2, instance_id=0, inputs_root=temp_dir,
                kv_offload_cpu=True)
            trace_path = Path(temp_dir) / "trace" / "test-hardware" / "test" / "model" / "instance0_batch3.txt"
            trace = trace_path.read_text()

        self.assertIn("KV_EVICT_CPU_0", trace)
        self.assertIn("REMOTE:2", trace)
        self.assertIn("4096", trace)
        self.assertNotIn("embedding", trace)

    def test_reload_trace_contains_only_cpu_operation(self):
        batch = Batch(
            4, "test/model", 0, 0, [], [], 0, 0, [], [], [], 0, 0,
            load=8192, kind=BatchKind.KV_RELOAD)
        with TemporaryDirectory() as temp_dir:
            generate_trace(
                batch, "test-hardware", 1, 1, 1, 1,
                pd_type="prefill",
                node_id=3, instance_id=0, inputs_root=temp_dir,
                kv_offload_cpu=True)
            trace_path = Path(temp_dir) / "trace" / "test-hardware" / "test" / "model" / "instance0_batch4.txt"
            trace = trace_path.read_text()

        self.assertIn("KV_RELOAD_CPU_0", trace)
        self.assertIn("REMOTE:3", trace)
        self.assertIn("8192", trace)
        self.assertNotIn("embedding", trace)
        self.assertTrue(trace.startswith("PREFILL"))

    def _pressure_scheduler(self, npu_used, new_kv_size):
        scheduler, pool = self._scheduler()
        scheduler.pp_size = 1
        scheduler.max_num_seqs = 8
        scheduler.prioritize_prefill = False
        scheduler.enable_chunked_prefill = True
        scheduler.enable_prefix_caching = False
        scheduler.max_num_batched_tokens = 8
        scheduler.long_prefill_token_threshold = 0
        scheduler.enable_kv_offloading = True
        scheduler.kv_offload_high_watermark = 0.90
        scheduler.kv_offload_low_watermark = 0.80
        scheduler.kv_offload_victim_policy = "lru"
        scheduler.memory.npu_used = npu_used
        scheduler.memory.get_block_kv = (
            lambda requests, count, scheduled_tokens=None: new_kv_size if count else 0)
        scheduler.memory.get_evict_kv = lambda req: 8
        return scheduler, pool

    def test_recompute_preemption_discards_kv_and_preserves_progress(self):
        scheduler, _ = self._pressure_scheduler(100, 1)
        scheduler.enable_kv_offloading = False
        older = Request(1, "test/model", 8, 32, 0, 0)
        newer = Request(2, "test/model", 8, 32, 1, 0)
        for request in (older, newer):
            request.num_computed_tokens = 16
            request.is_init = False
        scheduler.request = [older, newer]

        batch = scheduler.schedule_base(10, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertIn(older, batch.requests)
        self.assertEqual(older.num_computed_tokens, 16)
        self.assertEqual(newer.num_computed_tokens, 0)
        self.assertEqual(newer.recompute_kv_target_tokens, 16)
        self.assertTrue(newer.is_prefill())
        self.assertEqual(scheduler.memory.npu_used, 93)
        stats = scheduler.get_kv_offload_stats()
        self.assertEqual(stats["preemption_count"], 1)
        self.assertEqual(stats["recompute_preemption_count"], 1)
        self.assertEqual(stats["recompute_preemption_bytes"], 8)
        self.assertEqual(stats["recompute_preemption_tokens"], 16)

    def test_full_offload_tiers_fall_back_to_recompute_preemption(self):
        scheduler, pool = self._pressure_scheduler(100, 1)
        pool.capacity = 0
        request = Request(1, "test/model", 8, 32, 0, 0)
        request.num_computed_tokens = 16
        request.is_init = False
        scheduler.request = [request]

        batch = scheduler.schedule_base(10, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertEqual([req.id for req in batch.requests], [request.id])
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertEqual(request.recompute_kv_target_tokens, 16)
        self.assertEqual(scheduler.memory.npu_used, 93)
        stats = scheduler.get_kv_offload_stats()
        self.assertEqual(stats["recompute_preemption_count"], 1)
        self.assertEqual(stats["recompute_preemption_bytes"], 8)
        self.assertEqual(stats["recompute_preemption_tokens"], 16)

    def test_recompute_prefill_does_not_duplicate_token_throughput(self):
        scheduler, _ = self._pressure_scheduler(60, 1)
        scheduler.enable_kv_offloading = False
        request = Request(1, "test/model", 8, 16, 0, 0)
        request.num_computed_tokens = 8
        request.is_init = False
        request.ttft = 5
        request.begin_recompute_preemption()
        request.chunk_len = 8
        batch = Batch(
            0, "test/model", 8, 0, [8], [], 1, 0,
            [8], [0], [], 10, 0)
        batch.requests.append(request)
        batch.fired.append(0)
        scheduler.inflight.append(batch)

        prompt_tokens, generated_tokens, finished = scheduler.add_done(
            1, 0, 20)

        self.assertEqual(prompt_tokens, 0)
        self.assertEqual(generated_tokens, 0)
        self.assertEqual(finished, [])
        self.assertEqual(request.num_computed_tokens, 8)
        self.assertIsNone(request.recompute_kv_target_tokens)
        self.assertEqual(request.ttft, 5)

    def test_partial_prefill_can_be_recompute_preempted(self):
        scheduler, _ = self._pressure_scheduler(100, 1)
        scheduler.enable_kv_offloading = False
        older = Request(1, "test/model", 32, 8, 0, 0)
        newer = Request(2, "test/model", 32, 8, 1, 0)
        for request in (older, newer):
            request.num_computed_tokens = 8
        scheduler.request = [older, newer]

        batch = scheduler.schedule_base(10, 0)

        self.assertIsNotNone(batch)
        self.assertIn(older, batch.requests)
        self.assertEqual(newer.num_computed_tokens, 0)
        self.assertEqual(newer.recompute_kv_target_tokens, 8)
        self.assertTrue(newer.is_init)

        newer.num_computed_tokens = 8
        newer.finish_recompute_prefill()
        self.assertTrue(newer.is_prefill())
        self.assertEqual(newer.prefill_target_tokens(), 32)

    def test_pressure_emits_migration_without_releasing_source(self):
        scheduler, pool = self._pressure_scheduler(95, 10)
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        decode = Request(2, "test/model", 8, 32, 0, 0)
        decode.num_computed_tokens = 10
        decode.is_init = False
        scheduler.request = [prefill, decode]

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.kind, BatchKind.KV_EVICT)
        self.assertEqual(decode.kv_residency, KVResidency.NPU_TO_CPU)
        self.assertEqual(scheduler.memory.npu_used, 95)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 8)

    def test_high_watermark_does_not_block_physically_valid_prefill(self):
        scheduler, _ = self._pressure_scheduler(85, 6)
        scheduler.request = [Request(1, "test/model", 8, 16, 0, 0)]

        batch = scheduler.schedule_base(0, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertEqual(scheduler.memory.npu_used, 91)

    def test_chunked_prefill_shrinks_to_remaining_physical_capacity(self):
        scheduler, pool = self._pressure_scheduler(95, 8)
        pool.capacity = 0
        scheduler.memory.get_block_kv = (
            lambda requests, count, scheduled_tokens=None:
                sum(scheduled_tokens[req.id] for req in requests[:count]))
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        scheduler.request = [prefill]

        batch = scheduler.schedule_base(0, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertEqual(batch.total_len, 5)
        self.assertEqual(prefill.chunk_len, 5)
        self.assertEqual(scheduler.memory.npu_used, 100)

    def test_full_tiers_drop_oldest_parked_npu_session_for_progress(self):
        scheduler, pool = self._pressure_scheduler(100, 1)
        pool.capacity = 0
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        scheduler.request = [prefill]
        scheduler.session_kv_states["old-session"] = SessionKVState(
            session_id="old-session",
            model_name="test/model",
            sub_request_index=0,
            source_request_id=9,
            cached_tokens=8,
            bytes_per_rank=8,
            bytes_full_cluster=8,
            residency=KVResidency.NPU,
            owner_node_id=0,
            owner_instance_id=0,
            num_npus=1,
            tp_size=1,
            block_size=16,
            kv_fp=2,
            parked_at_ns=0,
            expires_at_ns=None,
            last_access_ns=0,
        )

        batch = scheduler.schedule_base(10, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertNotIn("old-session", scheduler.session_kv_states)
        self.assertEqual(scheduler.memory.npu_used, 93)
        stats = scheduler.get_kv_offload_stats()
        self.assertEqual(stats["session_capacity_drop_count"], 1)
        self.assertEqual(stats["session_capacity_drop_bytes"], 8)

    def test_failed_mandatory_offload_reschedules_after_session_drop(self):
        scheduler, pool = self._pressure_scheduler(100, 1)
        scheduler.pd_type = "decode"
        pool.capacity = 0
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        scheduler.request = [prefill]
        scheduler.session_kv_states["mandatory"] = SessionKVState(
            session_id="mandatory",
            model_name="test/model",
            sub_request_index=0,
            source_request_id=9,
            cached_tokens=8,
            bytes_per_rank=8,
            bytes_full_cluster=8,
            residency=KVResidency.NPU,
            owner_node_id=0,
            owner_instance_id=0,
            num_npus=1,
            tp_size=1,
            block_size=16,
            kv_fp=2,
            parked_at_ns=0,
            expires_at_ns=None,
            last_access_ns=0,
            mandatory_cpu_offload=True,
        )

        batch = scheduler.schedule_base(10, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertNotIn("mandatory", scheduler.session_kv_states)
        self.assertEqual(scheduler.memory.npu_used, 93)

    def test_only_runnable_decode_is_not_evicted(self):
        scheduler, _ = self._pressure_scheduler(85, 1)
        decode = Request(1, "test/model", 8, 32, 0, 0)
        decode.num_computed_tokens = 16
        decode.is_init = False
        scheduler.request = [decode]

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertEqual([req.id for req in batch.requests], [decode.id])
        self.assertEqual(decode.kv_residency, KVResidency.NPU)

    def test_npu_decode_runs_before_swapped_decode_without_ping_pong(self):
        # Physical capacity can fit resident KV + reload, but the configured
        # high watermark cannot. The swapped request must remain queued until
        # resident work drains instead of entering an evict/reload cycle.
        scheduler, pool = self._pressure_scheduler(85, 1)
        npu_decode = Request(1, "test/model", 8, 32, 0, 0)
        cpu_decode = Request(2, "test/model", 8, 32, 0, 0)
        for request in (npu_decode, cpu_decode):
            request.num_computed_tokens = 16
            request.is_init = False
        cpu_decode.mark_kv_on_cpu()
        pool.used = 8
        scheduler.request = [npu_decode, cpu_decode]

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertEqual([req.id for req in batch.requests], [npu_decode.id])
        self.assertEqual(cpu_decode.kv_residency, KVResidency.CPU)

    def test_lru_and_largest_kv_victim_policies(self):
        scheduler, _ = self._pressure_scheduler(80, 1)
        older = Request(1, "test/model", 8, 32, 0, 0)
        newer = Request(2, "test/model", 8, 32, 0, 0)
        older.num_computed_tokens = 16
        newer.num_computed_tokens = 32
        older.last_scheduled_ns = 10
        newer.last_scheduled_ns = 20
        scheduler.memory.get_evict_kv = (
            lambda req: 8 if req.id == older.id else 16)

        self.assertIs(scheduler._select_offload_victim([newer, older]), older)

        scheduler.kv_offload_victim_policy = "largest-kv"
        self.assertIs(scheduler._select_offload_victim([older, newer]), newer)

    def test_high_watermark_evicts_toward_low_watermark(self):
        scheduler, pool = self._pressure_scheduler(95, 1)
        scheduler.max_num_seqs = 1
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        older = Request(2, "test/model", 8, 32, 0, 0)
        newer = Request(3, "test/model", 8, 32, 0, 0)
        for request in (older, newer):
            request.num_computed_tokens = 16
            request.is_init = False
        older.last_scheduled_ns = 10
        newer.last_scheduled_ns = 20
        scheduler.request = [prefill, older, newer]

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.kind, BatchKind.KV_EVICT)
        self.assertEqual([m.request_id for m in batch.migrations], [2, 3])
        self.assertEqual(batch.evict, 16)
        self.assertEqual(pool.reserved, 16)
        self.assertEqual(scheduler.memory.npu_used, 95)

    def test_cpu_capacity_uses_feasible_victim_subset(self):
        scheduler, pool = self._pressure_scheduler(95, 1)
        scheduler.max_num_seqs = 1
        pool.capacity = 8
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        older = Request(2, "test/model", 8, 32, 0, 0)
        newer = Request(3, "test/model", 8, 32, 0, 0)
        for request in (older, newer):
            request.num_computed_tokens = 16
            request.is_init = False
        scheduler.request = [prefill, older, newer]

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.kind, BatchKind.KV_EVICT)
        self.assertEqual(len(batch.migrations), 1)
        self.assertEqual(batch.migrations[0].request_id, older.id)
        self.assertEqual(pool.reserved, 8)

    def test_full_cpu_tier_recompute_preempts_instead_of_stalling(self):
        scheduler, pool = self._pressure_scheduler(95, 10)
        pool.capacity = 0
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        decode = Request(2, "test/model", 8, 32, 0, 0)
        decode.num_computed_tokens = 16
        decode.is_init = False
        scheduler.request = [prefill, decode]

        batch = scheduler.schedule_base(0, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, BatchKind.COMPUTE)
        self.assertEqual([req.id for req in batch.requests], [1])
        self.assertEqual(prefill.chunk_len, 8)
        self.assertEqual(prefill.queuing_delay, 0)
        self.assertEqual(decode.kv_residency, KVResidency.NPU)
        self.assertEqual(decode.num_computed_tokens, 0)
        self.assertEqual(decode.recompute_kv_target_tokens, 16)
        self.assertEqual([req.id for req in scheduler.request], [2])
        self.assertEqual(scheduler.memory.npu_used, 97)
        self.assertEqual(scheduler.memory.npu_reserved, 0)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 0)
        self.assertEqual(scheduler.inflight, [batch])

    def test_evicted_request_is_not_immediately_reloaded_under_pressure(self):
        scheduler, _ = self._pressure_scheduler(95, 10)
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        decode = Request(2, "test/model", 8, 32, 0, 0)
        decode.num_computed_tokens = 16
        decode.is_init = False
        scheduler.request = [prefill, decode]

        eviction = scheduler.schedule_base(100, 0)
        self.assertEqual(eviction.kind, BatchKind.KV_EVICT)
        scheduler.add_done(eviction.batch_id + 1, 0, 150)

        next_batch = scheduler.schedule_base(200, 0)

        self.assertEqual(next_batch.kind, BatchKind.COMPUTE)
        self.assertEqual([req.id for req in next_batch.requests], [prefill.id])
        self.assertEqual(decode.kv_residency, KVResidency.CPU)


class PDHandoffTest(unittest.TestCase):
    class Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    def _scheduler(self, instance_id, pd_type, npu_used, pool=None):
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
        memory.cpu_kv_pool = pool
        memory.weight = 0
        memory.logger = self.Logger()
        memory.get_total_kv = lambda req: 20
        memory.get_evict_kv = lambda req: 20

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model = "test/model"
        scheduler.node_id = 0
        scheduler.instance_id = instance_id
        scheduler.num_npus = 1
        scheduler.start_npu = 0
        scheduler.pd_type = pd_type
        scheduler.enable_prefix_caching = False
        scheduler.enable_kv_offloading = True
        scheduler.kv_offload_high_watermark = 0.90
        scheduler.kv_offload_low_watermark = 0.80
        scheduler.kv_offload_victim_policy = "lru"
        scheduler.max_num_batched_tokens = 32
        scheduler.max_num_seqs = 8
        scheduler.prioritize_prefill = False
        scheduler.enable_chunked_prefill = True
        scheduler.long_prefill_token_threshold = 0
        scheduler.pp_size = 1
        scheduler.inflight = []
        scheduler.request = []
        scheduler.done = []
        scheduler.pending_pd_handoffs = []
        scheduler.batch_ids = -1
        scheduler.migration_ids = -1
        scheduler.memory = memory
        scheduler.logger = self.Logger()
        return scheduler

    def test_direct_handoff_transfers_capacity_and_ownership_without_cpu(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        source = self._scheduler(0, "prefill", 60, pool)
        destination = self._scheduler(1, "decode", 60, pool)
        request = Request(7, "test/model", 16, 32, 0, source.instance_id)
        request.num_computed_tokens = 16

        destination.enqueue_pd_handoff(request, source, 100)
        batch = destination._admit_pending_pd_handoff(150, 0)

        self.assertIsNone(batch)
        self.assertEqual(source.memory.npu_used, 40)
        self.assertEqual(destination.memory.npu_used, 80)
        self.assertEqual(destination.memory.npu_reserved, 0)
        self.assertEqual(request.instance_id, destination.instance_id)
        self.assertEqual(request.kv_owner_instance_id, destination.instance_id)
        self.assertEqual([req.id for req in destination.request], [request.id])
        self.assertEqual(destination.pending_pd_handoffs, [])
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 0)
        stats = destination.get_kv_offload_stats()
        self.assertEqual(stats["pd_handoff_count"], 1)
        self.assertEqual(stats["pd_handoff_bytes"], 20)
        self.assertEqual(stats["pd_handoff_wait_ns"], 50)

    def test_cross_node_handoff_is_rejected_before_state_changes(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        source = self._scheduler(0, "prefill", 60, pool)
        destination = self._scheduler(1, "decode", 60, pool)
        destination.node_id = 1
        request = Request(7, "test/model", 16, 32, 0, source.instance_id)
        request.num_computed_tokens = 16

        with self.assertRaisesRegex(RuntimeError, "same-node PD handoff only"):
            destination.enqueue_pd_handoff(request, source, 100)

        self.assertEqual(source.memory.npu_used, 60)
        self.assertEqual(destination.memory.npu_used, 60)
        self.assertEqual(request.kv_owner_instance_id, source.instance_id)
        self.assertEqual(destination.pending_pd_handoffs, [])

    def test_decode_pressure_evicts_before_handoff_commit(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        source = self._scheduler(0, "prefill", 60, pool)
        destination = self._scheduler(1, "decode", 95, pool)
        source.memory.get_total_kv = lambda req: 10
        destination.memory.get_total_kv = lambda req: 10
        victim = Request(1, "test/model", 8, 32, 0, destination.instance_id)
        victim.num_computed_tokens = 16
        victim.is_init = False
        incoming = Request(2, "test/model", 8, 32, 0, source.instance_id)
        incoming.num_computed_tokens = 16
        destination.request = [victim]
        destination.enqueue_pd_handoff(incoming, source, 100)

        eviction = destination._admit_pending_pd_handoff(120, 0)

        self.assertEqual(eviction.kind, BatchKind.KV_EVICT)
        self.assertEqual(incoming.kv_owner_instance_id, source.instance_id)
        self.assertEqual(source.memory.npu_used, 60)
        self.assertEqual(destination.memory.npu_used, 95)
        self.assertEqual(pool.reserved, 20)

        destination.add_done(eviction.batch_id + 1, 0, 140)
        destination._admit_pending_pd_handoff(150, 0)

        self.assertEqual(victim.kv_residency, KVResidency.CPU)
        self.assertEqual(incoming.kv_owner_instance_id, destination.instance_id)
        self.assertEqual(source.memory.npu_used, 50)
        self.assertEqual(destination.memory.npu_used, 85)
        self.assertEqual(pool.used, 20)

    def test_blocked_handoff_keeps_source_and_request_state(self):
        pool = NodeCPUKVPool(node_id=0, capacity=0)
        source = self._scheduler(0, "prefill", 60, pool)
        destination = self._scheduler(1, "decode", 100, pool)
        source.memory.get_total_kv = lambda req: 10
        destination.memory.get_total_kv = lambda req: 10
        victim = Request(1, "test/model", 8, 32, 0, destination.instance_id)
        victim.num_computed_tokens = 16
        incoming = Request(2, "test/model", 8, 32, 0, source.instance_id)
        incoming.num_computed_tokens = 16
        destination.request = [victim]
        destination.enqueue_pd_handoff(incoming, source, 100)

        batch = destination._admit_pending_pd_handoff(120, 0)

        self.assertIsNone(batch)
        self.assertEqual(source.memory.npu_used, 60)
        self.assertEqual(destination.memory.npu_used, 100)
        self.assertEqual(destination.memory.npu_reserved, 0)
        self.assertEqual(incoming.kv_owner_instance_id, source.instance_id)
        self.assertEqual([item.request.id for item in destination.pending_pd_handoffs], [2])
        self.assertEqual([req.id for req in destination.request], [1])

    def test_handoff_failure_rolls_back_destination_reservation(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        source = self._scheduler(0, "prefill", 5, pool)
        destination = self._scheduler(1, "decode", 60, pool)
        source.memory.get_total_kv = lambda req: 10
        destination.memory.get_total_kv = lambda req: 10
        incoming = Request(2, "test/model", 8, 32, 0, source.instance_id)
        incoming.num_computed_tokens = 16
        destination.enqueue_pd_handoff(incoming, source, 100)

        with self.assertRaises(RuntimeError):
            destination._admit_pending_pd_handoff(120, 0)

        self.assertEqual(source.memory.npu_used, 5)
        self.assertEqual(destination.memory.npu_used, 60)
        self.assertEqual(destination.memory.npu_reserved, 0)
        self.assertEqual(incoming.kv_owner_instance_id, source.instance_id)
        self.assertEqual(destination.request, [])
        self.assertEqual([item.request.id for item in destination.pending_pd_handoffs], [2])

    def test_prefill_completion_retains_source_until_handoff(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        source = self._scheduler(0, "prefill", 60, pool)
        request = Request(7, "test/model", 10, 20, 0, source.instance_id)
        request.num_computed_tokens = 9
        request.chunk_len = 1
        batch = Batch(
            0, "test/model", 1, 0, [1], [], 1, 0, [1], [9], [],
            100, 0)
        batch.requests.append(request)
        batch.fired.append(0)
        source.inflight.append(batch)

        source.add_done(1, 0, 120)
        generated, _, finished = source.add_done(1, 1, 120)

        self.assertEqual(generated, 1)
        self.assertEqual([req.id for req in finished], [request.id])
        self.assertEqual(source.memory.npu_used, 60)
        self.assertEqual(request.kv_owner_instance_id, source.instance_id)

    def test_writes_instance_level_kv_offload_metrics(self):
        pool = NodeCPUKVPool(node_id=0, capacity=200)
        scheduler = self._scheduler(1, "decode", 60, pool)
        scheduler._get_kv_offload_stats().preemption_count = 2
        scheduler._get_kv_offload_stats().evict_bytes = 40
        scheduler._get_kv_offload_stats().reload_bytes = 20
        scheduler._get_kv_offload_stats().pd_handoff_count = 3
        scheduler._get_kv_offload_stats().pd_handoff_bytes = 60
        scheduler._get_kv_offload_stats().session_capacity_drop_count = 4
        scheduler._get_kv_offload_stats().session_capacity_drop_bytes = 80
        scheduler._get_kv_offload_stats().session_cpu_hit_count = 5
        scheduler._record_kv_occupancy()

        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "offload.csv"
            scheduler.save_kv_offload_output(str(output))
            with output.open(newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["instance id"], "1")
        self.assertEqual(rows[0]["preemption count"], "2")
        self.assertEqual(rows[0]["recompute preemption count"], "0")
        self.assertEqual(rows[0]["evict bytes"], "40")
        self.assertEqual(rows[0]["reload bytes"], "20")
        self.assertEqual(rows[0]["pd handoff count"], "3")
        self.assertEqual(rows[0]["pd handoff bytes"], "60")
        self.assertEqual(rows[0]["session capacity drop count"], "4")
        self.assertEqual(rows[0]["session capacity drop bytes"], "80")
        self.assertEqual(rows[0]["session cpu hit count"], "5")
        self.assertEqual(rows[0]["npu peak used bytes per rank"], "60")

    def test_router_uses_selected_local_decode_object(self):
        class FakeScheduler:
            def __init__(self, instance_id, node_id, pd_type):
                self.instance_id = instance_id
                self.node_id = node_id
                self.pd_type = pd_type
                self.enable_kv_offloading = True
                self.max_num_seqs = 8
                self.request = []
                self.inflight = []
                self.received = []

            def enqueue_pd_handoff(self, req, source, current):
                self.received.append((req.id, source.instance_id, current))

        prefill0 = FakeScheduler(0, 0, "prefill")
        decode0 = FakeScheduler(1, 0, "decode")
        prefill1 = FakeScheduler(2, 1, "prefill")
        decode1 = FakeScheduler(3, 1, "decode")
        router = Router(
            4, [prefill0, decode0, prefill1, decode1], 0,
            routing_policy="LOAD", same_node_pd_only=True)
        request = Request(9, "test/model", 8, 16, 0, prefill1.instance_id)

        router.transfer_prefill_request([request], 123)

        self.assertEqual(decode0.received, [])
        self.assertEqual(decode1.received, [(9, prefill1.instance_id, 123)])


if __name__ == "__main__":
    unittest.main()
