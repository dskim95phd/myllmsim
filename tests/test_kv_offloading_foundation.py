import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from serving.core.memory_model import Device, MemoryModel, NodeCPUKVPool
from serving.core.request import Batch, BatchKind, KVResidency, Request
from serving.core.scheduler import Scheduler
from serving.core.trace_generator import generate_trace
from serving.core.config_builder import _host_transfer_config


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
                node_id=3, instance_id=0, inputs_root=temp_dir,
                kv_offload_cpu=True)
            trace_path = Path(temp_dir) / "trace" / "test-hardware" / "test" / "model" / "instance0_batch4.txt"
            trace = trace_path.read_text()

        self.assertIn("KV_RELOAD_CPU_0", trace)
        self.assertIn("REMOTE:3", trace)
        self.assertIn("8192", trace)
        self.assertNotIn("embedding", trace)

    def _pressure_scheduler(self, npu_used, new_kv_size):
        scheduler, pool = self._scheduler()
        scheduler.pp_size = 1
        scheduler.max_num_seqs = 8
        scheduler.prioritize_prefill = False
        scheduler.enable_chunked_prefill = True
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

    def test_failed_admission_does_not_mutate_request_or_memory(self):
        scheduler, pool = self._pressure_scheduler(95, 10)
        pool.capacity = 0
        prefill = Request(1, "test/model", 8, 16, 0, 0)
        decode = Request(2, "test/model", 8, 32, 0, 0)
        decode.num_computed_tokens = 16
        decode.is_init = False
        scheduler.request = [prefill, decode]

        batch = scheduler.schedule_base(0, 0)

        self.assertIsNone(batch)
        self.assertEqual(prefill.chunk_len, 0)
        self.assertEqual(prefill.queuing_delay, -1)
        self.assertEqual(decode.kv_residency, KVResidency.NPU)
        self.assertEqual([req.id for req in scheduler.request], [1, 2])
        self.assertEqual(scheduler.memory.npu_used, 95)
        self.assertEqual(scheduler.memory.npu_reserved, 0)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.reserved, 0)
        self.assertEqual(scheduler.inflight, [])

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


if __name__ == "__main__":
    unittest.main()
