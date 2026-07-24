import unittest
import socket
import struct
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_chakra_pipeline import aggregate
from serving.proto import llmservingsim_workload_pb2 as workload_pb2
from serving.core.chakra_template import (
    BoundedTemplateCache,
    CompiledSlotExpression,
    CompiledTracePlan,
    RuntimeTraceMarker,
    RuntimeTraceRow,
    RuntimeTraceSnapshot,
    RuntimeValueSource,
    _runtime_source_for_node,
    _runtime_value_from_trace,
)
from serving.core.config_builder import (
    _compute_network_dims,
    _resolve_dp_groups,
)
from serving.core.graph_generator import generate_graph, _load_llm_converter
from serving.core.host_timing import HostTimingRecorder, timed_stage
from serving.core.router import Router
from serving.core.trace_generator import (
    _load_compiled_profile_cache,
    _write_compiled_profile_cache,
)
from serving.core.utils import get_config
from serving.core.workload_protocol import (
    HEADER_SIZE,
    MessageType,
    ProtocolError,
    decode_header,
    encode_frame,
    receive_frame,
    send_frame,
)
from serving.core.workload_transport import (
    FileWorkloadTransport,
    IpcFileWorkloadTransport,
)


class ModelConfigCacheTest(unittest.TestCase):
    def tearDown(self):
        get_config.cache_clear()

    def test_model_config_is_loaded_once(self):
        get_config.cache_clear()

        first = get_config("meta-llama/Llama-3.1-8B")
        second = get_config("meta-llama/Llama-3.1-8B")

        self.assertIs(first, second)
        self.assertEqual(get_config.cache_info().misses, 1)
        self.assertEqual(get_config.cache_info().hits, 1)


class DpPipelineTopologyTest(unittest.TestCase):
    @staticmethod
    def _instance(pp_size=2):
        return {
            "dp_group": "A",
            "tp_size": 1,
            "pp_size": pp_size,
            "ep_size": 2,
        }

    def test_pp_dimension_is_preserved_and_excluded_from_collectives(self):
        instances = [self._instance(), self._instance()]

        _resolve_dp_groups(instances)

        self.assertEqual(_compute_network_dims(instances), [1, 2, 2])
        for instance in instances:
            self.assertEqual(instance["tp_dim"], [True, False, False])
            self.assertEqual(instance["ep_dim"], [False, False, True])

    def test_dp_group_rejects_mixed_pp_layouts(self):
        instances = [self._instance(2), self._instance(1)]

        with self.assertRaisesRegex(ValueError, "pp_size mismatch"):
            _resolve_dp_groups(instances)


class CompiledProfileCacheTest(unittest.TestCase):
    @staticmethod
    def _perf_db():
        return {
            "meta": {"engine_effective": {"max_num_seqs": 8}},
            "architecture": {"catalog": {}, "sequence": {}},
            "available_tps": [1],
            "tables": {
                1: {
                    "dense": {
                        "embedding": {"keys": [1], "values": [10]}},
                    "attention": {
                        "pc_vals": [0],
                        "nd_vals": [1],
                        "pc_nd_pairs": [(0, 1)],
                        "slices": {
                            (0, 1): {
                                "kv_prefill_vals": [0],
                                "rows": [{"keys": [1], "values": [20]}],
                            },
                        },
                    },
                },
            },
        }

    def test_safe_cache_round_trip_and_fingerprint_invalidation(self):
        identity = {
            "hardware": "test-hw",
            "model": "test-model",
            "variant": "bf16",
            "model_type": "test",
            "tp_degrees": [1],
        }
        fingerprints = [{
            "path": "tp1/dense.csv",
            "size": 12,
            "sha256": "abc",
        }]
        with TemporaryDirectory() as root:
            path = str(Path(root, "cache.json.gz"))
            self.assertTrue(_write_compiled_profile_cache(
                path, identity, fingerprints, self._perf_db()))

            loaded = _load_compiled_profile_cache(
                path, identity, fingerprints)
            invalidated = _load_compiled_profile_cache(
                path, identity, [{**fingerprints[0], "sha256": "changed"}])

        self.assertEqual(
            loaded["tables"][1]["attention"]["pc_nd_pairs"], [(0, 1)])
        self.assertEqual(
            loaded["tables"][1]["attention"]["slices"][(0, 1)]
            ["rows"][0]["values"],
            [20],
        )
        self.assertIsNone(invalidated)


class BoundedTemplateCacheTest(unittest.TestCase):
    def test_lru_hit_and_eviction(self):
        cache = BoundedTemplateCache(2)
        self.assertIsNone(cache.put("dense-a", 1))
        self.assertIsNone(cache.put("dense-b", 2))
        self.assertEqual(cache.get("dense-a"), 1)

        evicted = cache.put("pim-a", 3)

        self.assertEqual(evicted, ("dense-b", 2))
        self.assertIsNone(cache.get("dense-b"))
        self.assertEqual(cache.get("dense-a"), 1)
        self.assertEqual(cache.get("pim-a"), 3)
        self.assertEqual(len(cache), 2)

    def test_rejects_non_positive_capacity(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            BoundedTemplateCache(0)


class InProcessGraphConversionTest(unittest.TestCase):
    def test_pipeline_partitions_only_at_tensor_compatible_boundaries(self):
        converter_class = _load_llm_converter(str(
            Path("astra-sim/extern/graph_frontend/chakra").resolve()))
        converter = object.__new__(converter_class)
        layers = [
            SimpleNamespace(
                input_memory_size=10,
                output_memory_size=10,
                is_expert=False,
                is_pim=False,
            )
            for _ in range(12)
        ]
        layers[2].output_memory_size = 30
        layers[3].input_memory_size = 20

        ends = converter.get_pipeline_partition_ends(layers, 4)

        self.assertEqual(ends[0], 0)
        self.assertEqual(ends[-1], len(layers))
        self.assertEqual(len(ends), 5)
        self.assertNotIn(3, ends)
        for boundary in ends[1:-1]:
            self.assertEqual(
                layers[boundary - 1].output_memory_size,
                layers[boundary].input_memory_size,
            )

    def test_pipeline_partitions_skip_expert_markers_without_tensor_fields(self):
        converter_class = _load_llm_converter(str(
            Path("astra-sim/extern/graph_frontend/chakra").resolve()))
        converter = object.__new__(converter_class)
        normal = lambda: SimpleNamespace(
            input_memory_size=10,
            output_memory_size=10,
            is_expert=False,
            is_pim=False,
        )
        marker = lambda: SimpleNamespace(is_expert=True, is_pim=False)
        layers = [normal(), normal(), marker(), normal(), marker(), normal(),
                  normal(), normal()]

        ends = converter.get_pipeline_partition_ends(layers, 2)

        self.assertEqual(ends[0], 0)
        self.assertEqual(ends[-1], len(layers))
        for boundary in ends[1:-1]:
            self.assertFalse(layers[boundary - 1].is_expert)
            self.assertFalse(layers[boundary].is_expert)

    def test_in_process_converter_receives_existing_trace_and_cleans_it(self):
        calls = []

        class FakeConverter:
            def __init__(self, input_path, output_path, num_npus, npu_offset,
                         local_offloading):
                calls.append(
                    (input_path, output_path, num_npus, npu_offset,
                     local_offloading)
                )
                self.output_path = output_path

            def convert(self):
                Path(f"{self.output_path}.3.et").write_bytes(b"graph")

        with TemporaryDirectory() as root:
            trace_path = Path(root, "trace", "test-hw", "test-model",
                              "instance2_batch7.txt")
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text("trace", encoding="utf-8")
            batch = SimpleNamespace(model="test-model", batch_id=7)

            with patch(
                "serving.core.graph_generator._load_llm_converter",
                return_value=FakeConverter,
            ):
                generate_graph(
                    batch,
                    "test-hw",
                    4,
                    instance_id=2,
                    npu_offset=3,
                    enable_local_offloading=True,
                    inputs_root=root,
                    cleanup_trace=True,
                    converter_mode="in-process",
                )

            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(calls[0][0]).resolve(), trace_path.resolve())
            self.assertEqual(calls[0][2:], (4, 3, True))
            self.assertFalse(trace_path.exists())
            self.assertTrue(Path(f"{calls[0][1]}.3.et").exists())

    def test_subprocess_mode_remains_available_as_reference(self):
        with TemporaryDirectory() as root:
            trace_path = Path(root, "trace", "event_handler.txt")
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text("trace", encoding="utf-8")

            with patch("serving.core.graph_generator.subprocess.run") as run:
                generate_graph(
                    None,
                    None,
                    1,
                    event=True,
                    inputs_root=root,
                    cleanup_trace=False,
                    converter_mode="subprocess",
                )

            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[1:5], [
                "-m", "chakra.src.converter.converter", "LLM", "--input",
            ])
            self.assertEqual(run.call_args.kwargs["check"], True)
            self.assertTrue(trace_path.exists())

    def test_unknown_converter_mode_is_rejected(self):
        with TemporaryDirectory() as root:
            trace_path = Path(root, "trace", "event_handler.txt")
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text("trace", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported Chakra converter mode"):
                generate_graph(
                    None,
                    None,
                    1,
                    event=True,
                    inputs_root=root,
                    converter_mode="unknown",
                )


class HostTimingRecorderTest(unittest.TestCase):
    def test_summary_reports_stage_distribution_and_counters(self):
        recorder = HostTimingRecorder()
        recorder.record("stage", 1.0)
        recorder.record("stage", 3.0)
        recorder.increment("messages", 2)
        recorder.observe("batch_size", 1)
        recorder.observe("batch_size", 3)
        checkpoint = recorder.record_rss_checkpoint("test")

        summary = recorder.summary({"run_id": "test"})

        self.assertEqual(summary["metadata"]["run_id"], "test")
        self.assertEqual(summary["stages"]["stage"]["count"], 2)
        self.assertEqual(summary["stages"]["stage"]["p50_seconds"], 2.0)
        self.assertEqual(summary["stages"]["stage"]["p90_seconds"], 2.8)
        self.assertEqual(
            summary["stages"]["stage"]["cold_first_seconds"], 1.0)
        self.assertEqual(summary["stages"]["stage"]["warm_count"], 1)
        self.assertEqual(
            summary["stages"]["stage"]["warm_p50_seconds"], 3.0)
        self.assertEqual(summary["counters"]["messages"], 2)
        self.assertEqual(summary["metrics"]["batch_size"]["p50"], 2.0)
        self.assertEqual(summary["metrics"]["batch_size"]["max"], 3.0)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["rss_checkpoints"][0]["name"], "test")
        self.assertEqual(
            summary["rss_checkpoints"][0]["frontend"],
            checkpoint["frontend"],
        )

    def test_timed_stage_is_optional_for_existing_callers(self):
        recorder = HostTimingRecorder()

        @timed_stage("function")
        def add(left, right):
            return left + right

        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(2, 3, host_timing=recorder), 5)
        self.assertEqual(recorder.sample_count("function"), 1)


class ConcurrencyAccountingTest(unittest.TestCase):
    def test_time_weighted_idle_and_runnable_metrics(self):
        scheduler = SimpleNamespace(
            pd_type=None,
            request=[],
            inflight=[],
            pending_pd_handoffs=[],
        )
        router = Router(1, [scheduler], 0)
        router._agentic_session_count = 1
        router._agentic_turn_count = 3
        router._live_sessions.add("session")

        router.observe_concurrency(0)
        router.observe_concurrency(50)
        router._pending_requests.append({"arrival_time_ns": 0})
        router.observe_concurrency(50)
        summary = router.concurrency_summary(100)

        self.assertEqual(summary["peak_live_sessions"], 1)
        self.assertEqual(summary["peak_runnable_requests"], 1)
        self.assertEqual(summary["all_idle_interval_count"], 1)
        self.assertEqual(summary["all_idle_duration_ns"], 50)
        self.assertEqual(summary["mean_live_sessions"], 1.0)
        self.assertEqual(summary["mean_runnable_requests"], 0.5)


class RuntimeTracePatchTest(unittest.TestCase):
    @staticmethod
    def _trace(embedding_time=100, input_size=40, comm_size=80,
               sampler_input=32):
        return (
            "COLOCATED\t\tmodel_parallel_NPU_group: 1\n"
            "2\n"
            "Layername comp_time input_loc input_size weight_loc weight_size "
            "output_loc output_size comm_type comm_size misc\n"
            f"embedding_0 {embedding_time} REMOTE:0 {input_size} LOCAL 1000 "
            f"LOCAL 80 ALLREDUCE {comm_size} NONE\n"
            f"sampler_1 25 LOCAL {sampler_input} LOCAL 0 REMOTE:0 8 "
            "NONE 0 NONE\n"
        )

    def test_runtime_trace_values_map_to_dense_chakra_nodes(self):
        trace = RuntimeTraceSnapshot.parse(self._trace())

        self.assertEqual(
            _runtime_value_from_trace(
                SimpleNamespace(name="COMP_NODE_embedding_0"),
                "duration", trace),
            100,
        )
        self.assertEqual(
            _runtime_value_from_trace(
                SimpleNamespace(name="MEM_LOAD_NODE_embedding_0_INPUT"),
                "tensor_size", trace),
            40,
        )
        self.assertEqual(
            _runtime_value_from_trace(
                SimpleNamespace(name="COMM_COLL_NODE_embedding_0_ALLREDUCE"),
                "comm_size", trace),
            80,
        )
        self.assertEqual(
            _runtime_value_from_trace(
                SimpleNamespace(name="MEM_STORE_NODE_sampler_1_OUTPUT"),
                "tensor_size", trace),
            32,
        )

    def test_runtime_trace_structure_ignores_dynamic_values(self):
        template = RuntimeTraceSnapshot.parse(self._trace())
        current = RuntimeTraceSnapshot.parse(
            self._trace(embedding_time=200, input_size=64,
                        comm_size=128, sampler_input=48))

        template.assert_same_structure(current)
        with self.assertRaisesRegex(ValueError, "structure changed"):
            template.assert_same_structure(RuntimeTraceSnapshot.parse(
                self._trace().replace("LOCAL 80", "CXL:0 80", 1)))

    def test_marker_trace_preserves_structure_and_dynamic_comm_size(self):
        marker_trace = (
            "COLOCATED model_parallel_NPU_group: 1\n"
            "1\n"
            "header\n"
            "EXPERT 0 ALLTOALL 128\n"
        )
        trace = RuntimeTraceSnapshot.parse(marker_trace)

        self.assertEqual(trace.signature, (
            ("MARKER", "EXPERT", "0", "ALLTOALL"),
        ))
        self.assertEqual(trace.expert_markers("START")[0].comm_size, 128)

    def test_kv_migration_rows_map_to_dynamic_memory_bytes(self):
        evict = RuntimeTraceSnapshot.parse(
            "COLOCATED model_parallel_NPU_group: 1\n"
            "1\nheader\n"
            "KV_EVICT_CPU_0 0 LOCAL 0 REMOTE:0 4096 LOCAL 0 NONE 0 NONE\n"
        )
        reload = RuntimeTraceSnapshot.parse(
            "COLOCATED model_parallel_NPU_group: 1\n"
            "1\nheader\n"
            "KV_RELOAD_CPU_0 0 LOCAL 0 REMOTE:0 8192 LOCAL 0 NONE 0 NONE\n"
        )

        self.assertNotEqual(evict.structure_key, reload.structure_key)
        self.assertEqual(
            _runtime_value_from_trace(
                SimpleNamespace(
                    name="MEM_STORE_NODE_KV_EVICT_CPU_0_WEIGHT"),
                "tensor_size", evict),
            4096,
        )
        self.assertEqual(
            _runtime_value_from_trace(
                SimpleNamespace(
                    name="MEM_LOAD_NODE_KV_RELOAD_CPU_0_WEIGHT"),
                "tensor_size", reload),
            8192,
        )

    def test_pd_kv_communication_uses_qkv_rows_in_layer_order(self):
        trace = RuntimeTraceSnapshot.parse(
            "PREFILL model_parallel_NPU_group: 1\n"
            "2\nheader\n"
            "qkv_proj_1 10 LOCAL 16 LOCAL 32 LOCAL 64 NONE 0 NONE\n"
            "qkv_proj_2 20 LOCAL 32 LOCAL 64 LOCAL 128 NONE 0 NONE\n"
        )
        indices = {}

        first = _runtime_source_for_node(
            SimpleNamespace(name="COMM_SEND_NODE_kv_proj_NONE_0_1"),
            "comm_size", trace, indices, 64)
        second = _runtime_source_for_node(
            SimpleNamespace(name="COMM_SEND_NODE_kv_proj_NONE_0_1"),
            "comm_size", trace, indices, 128)
        sync = _runtime_source_for_node(
            SimpleNamespace(name="COMP_NODE_kv_migration_sync"),
            "duration", trace, indices, 1)

        self.assertEqual(first.resolve(trace), 64)
        self.assertEqual(second.resolve(trace), 128)
        self.assertEqual(sync.resolve(trace), 1)

    def test_structured_trace_round_trips_without_value_drift(self):
        structured = RuntimeTraceSnapshot.from_records(
            "COLOCATED\t\tmodel_parallel_NPU_group: 1",
            (
                RuntimeTraceRow(
                    "embedding_0", 100, "REMOTE:0", 40, "LOCAL", 1000,
                    "LOCAL", 80, "ALLREDUCE:1,0,0", 380928, "NONE"),
                RuntimeTraceMarker("EXPERT", "0", "ALLTOALL", 128),
                RuntimeTraceMarker("EXPERT", "END", "ALLTOALL", 128),
            ),
        )

        parsed = RuntimeTraceSnapshot.parse(structured.render())

        self.assertEqual(parsed.signature, structured.signature)
        self.assertEqual(parsed.rows, structured.rows)
        self.assertEqual(parsed.markers, structured.markers)

    def test_compiled_trace_plan_evaluates_structured_records(self):
        template = RuntimeTraceSnapshot.parse(self._trace())
        current = RuntimeTraceSnapshot.parse(
            self._trace(embedding_time=250, input_size=64, comm_size=160))
        plan = CompiledTracePlan(
            template_signature=template.signature,
            expressions=(
                CompiledSlotExpression(
                    system_id=0,
                    slot_id=3,
                    source=RuntimeValueSource(
                        "comp_time", row_name="embedding_0"),
                ),
                CompiledSlotExpression(
                    system_id=0,
                    slot_id=4,
                    source=RuntimeValueSource(
                        "comm_size", row_name="embedding_0"),
                ),
            ),
        )

        patch = plan.evaluate(7, 11, current)

        self.assertEqual(patch.template_id, 7)
        self.assertEqual(patch.batch_id, 11)
        self.assertEqual(
            list(patch.systems[0].packed_values),
            [3, 250, 4, 160],
        )


class WorkloadProtocolTest(unittest.TestCase):
    def test_frame_round_trip_over_socket(self):
        left, right = socket.socketpair()
        try:
            sent = send_frame(left, MessageType.RUN_BATCH, b"patch")
            message_type, payload = receive_frame(right)
        finally:
            left.close()
            right.close()

        self.assertEqual(sent, HEADER_SIZE + len(b"patch"))
        self.assertEqual(message_type, MessageType.RUN_BATCH)
        self.assertEqual(payload, b"patch")

    def test_header_rejects_bad_magic_version_type_and_size(self):
        valid = encode_frame(MessageType.HELLO)
        with self.assertRaisesRegex(ProtocolError, "magic"):
            decode_header(b"BAD!" + valid[4:HEADER_SIZE])

        bad_version = struct.pack("!4sHHQ", b"LSIM", 3, 1, 0)
        with self.assertRaisesRegex(ProtocolError, "version"):
            decode_header(bad_version)

        bad_type = struct.pack("!4sHHQ", b"LSIM", 2, 99, 0)
        with self.assertRaisesRegex(ProtocolError, "message type"):
            decode_header(bad_type)

        too_large = struct.pack("!4sHHQ", b"LSIM", 2, 1, 11)
        with self.assertRaisesRegex(ProtocolError, "exceeds limit"):
            decode_header(too_large, max_payload_bytes=10)

    def test_payload_must_be_bytes(self):
        with self.assertRaisesRegex(TypeError, "bytes-like"):
            encode_frame(MessageType.HELLO, "not bytes")


class FileWorkloadTransportTest(unittest.TestCase):
    def test_commands_include_the_target_system(self):
        controller = unittest.mock.Mock()
        process = object()
        recorder = HostTimingRecorder()
        transport = FileWorkloadTransport(
            controller, process, host_timing=recorder)

        with self.assertRaisesRegex(RuntimeError, "target system"):
            transport.pass_system()

        transport.set_system(3)
        transport.run_batch("batch.et")
        transport.run_wave("wave.et", [3, 4])
        transport.advance_time(123456)
        transport.pass_system()
        transport.sleep_system()
        transport.close()

        self.assertEqual(
            controller.write_flush.call_args_list,
            [
                unittest.mock.call(process, "@3\tbatch.et"),
                unittest.mock.call(process, "@3\twave:3,4\twave.et"),
                unittest.mock.call(process, "@3\tadvance:123456"),
                unittest.mock.call(process, "@3\tpass"),
                unittest.mock.call(process, "@3\tdone"),
                unittest.mock.call(process, "@3\texit"),
            ],
        )
        counters = recorder.summary()["counters"]
        self.assertEqual(counters["run_batch_submissions"], 1)
        self.assertEqual(counters["run_wave_submissions"], 1)

    def test_ipc_file_bridge_handshake_and_commands(self):
        client, server = socket.socketpair()
        received = []

        def serve():
            message_type, payload = receive_frame(server)
            received.append((message_type, payload))
            send_frame(server, MessageType.HELLO)
            message_type, payload = receive_frame(server)
            received.append((message_type, payload))
            registration = workload_pb2.RegisterTemplate.FromString(payload)
            response = workload_pb2.TemplateReady(
                request_id=registration.request_id,
                template_id=7,
                template_key=registration.template_key,
            )
            for graph in registration.graphs:
                response.graphs.add(
                    system_id=graph.system_id,
                    node_count=2,
                    root_count=1,
                    iteration_state_validated=True,
                )
            send_frame(
                server,
                MessageType.TEMPLATE_READY,
                response.SerializeToString(),
            )
            message_type, payload = receive_frame(server)
            received.append((message_type, payload))
            run_batch = workload_pb2.RunBatch.FromString(payload)
            accepted = workload_pb2.BatchAccepted(
                request_id=run_batch.request_id,
                batch_id=run_batch.patch.batch_id,
                template_id=run_batch.patch.template_id,
                system_count=len(run_batch.patch.systems),
                patched_value_count=sum(
                    len(system.values)
                    for system in run_batch.patch.systems),
            )
            send_frame(
                server,
                MessageType.BATCH_ACCEPTED,
                accepted.SerializeToString(),
            )
            message_type, payload = receive_frame(server)
            received.append((message_type, payload))
            run_wave = workload_pb2.RunWave.FromString(payload)
            wave_accepted = workload_pb2.BatchAccepted(
                request_id=run_wave.request_id,
                batch_id=run_wave.wave_id,
                template_id=0,
                system_count=sum(
                    len(run.patch.systems) for run in run_wave.runs),
                patched_value_count=sum(
                    len(system.values)
                    for run in run_wave.runs
                    for system in run.patch.systems),
            )
            send_frame(
                server,
                MessageType.BATCH_ACCEPTED,
                wave_accepted.SerializeToString(),
            )
            for _ in range(6):
                received.append(receive_frame(server))

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            recorder = HostTimingRecorder()
            transport = IpcFileWorkloadTransport(
                "unused", host_timing=recorder, connection=client)
            with self.assertRaisesRegex(RuntimeError, "target system"):
                transport.pass_system()
            template_id = transport.register_template(
                "dense-test",
                {1: b"graph-one", 0: b"graph-zero"},
                bindings=[{
                    "slot_id": 3,
                    "system_id": 0,
                    "node_id": 9,
                    "attribute": "duration",
                    "required": True,
                }],
            )
            patch = workload_pb2.BatchPatch(batch_id=11, template_id=template_id)
            patch.systems.add(system_id=0).values.add(slot_id=3, value=91)
            patch.systems.add(system_id=1)
            transport.set_system(3)
            transport.prepare_batch(4, patch)
            transport.prepare_wave(23, [(4, patch)])
            transport.run_batch("batch.et")
            transport.run_wave("wave.et")
            transport.advance_time(123456)
            transport.pass_system()
            transport.sleep_system()
            transport.close()
            thread.join(timeout=2)
        finally:
            server.close()

        self.assertFalse(thread.is_alive())
        self.assertEqual(template_id, 7)
        self.assertEqual(received[0], (MessageType.HELLO, b""))
        self.assertEqual(received[1][0], MessageType.REGISTER_TEMPLATE)
        registration = workload_pb2.RegisterTemplate.FromString(received[1][1])
        self.assertEqual(registration.template_key, "dense-test")
        self.assertEqual(
            [(graph.system_id, graph.chakra_graph)
             for graph in registration.graphs],
            [(0, b"graph-zero"), (1, b"graph-one")],
        )
        self.assertEqual(received[2][0], MessageType.RUN_BATCH)
        run_batch = workload_pb2.RunBatch.FromString(received[2][1])
        self.assertEqual(run_batch.instance_id, 4)
        self.assertEqual(run_batch.controller_system_id, 3)
        self.assertFalse(run_batch.execute)
        self.assertEqual(run_batch.patch.batch_id, 11)
        self.assertEqual(received[3][0], MessageType.RUN_WAVE)
        run_wave = workload_pb2.RunWave.FromString(received[3][1])
        self.assertEqual(run_wave.wave_id, 23)
        self.assertEqual(run_wave.controller_system_id, 3)
        self.assertEqual(len(run_wave.runs), 1)
        batch_file = workload_pb2.FileWorkload.FromString(received[4][1])
        wave_file = workload_pb2.FileWorkload.FromString(received[5][1])
        advance = workload_pb2.AdvanceTime.FromString(received[6][1])
        controls = [
            workload_pb2.SystemCommand.FromString(payload)
            for _, payload in received[7:10]
        ]
        self.assertEqual(received[4][0], MessageType.FILE_WORKLOAD)
        self.assertEqual((batch_file.system_id, batch_file.path), (3, "batch.et"))
        self.assertEqual(received[5][0], MessageType.FILE_WORKLOAD)
        self.assertEqual((wave_file.system_id, wave_file.path), (3, "wave.et"))
        self.assertEqual(received[6][0], MessageType.ADVANCE_TIME)
        self.assertEqual(
            (advance.system_id, advance.current_time_ns), (3, 123456))
        self.assertEqual(
            [message_type for message_type, _ in received[7:10]],
            [MessageType.PASS, MessageType.SLEEP, MessageType.EXIT],
        )
        self.assertEqual([control.system_id for control in controls], [3, 3, 3])
        counters = recorder.summary()["counters"]
        self.assertEqual(counters["ipc_messages_sent"], 10)
        self.assertEqual(counters["ipc_messages_received"], 4)
        self.assertEqual(counters["template_registrations"], 1)
        self.assertEqual(counters["compiled_template_graphs"], 2)
        self.assertEqual(counters["compiled_template_nodes"], 4)
        self.assertEqual(counters["validated_iteration_states"], 2)
        self.assertEqual(counters["run_batch_patch_submissions"], 1)
        self.assertEqual(counters["batch_patch_values_sent"], 1)
        self.assertEqual(counters["run_wave_patch_submissions"], 1)
        self.assertEqual(counters["wave_participants"], 1)
        self.assertEqual(counters["time_advance_submissions"], 1)


class BenchmarkDeterminismTest(unittest.TestCase):
    @staticmethod
    def _record(mode, digest):
        return {
            "scenario": "pd-stress",
            "mode": mode,
            "warmup": False,
            "wall_seconds": 1.0,
            "correctness": {"combined_sha256": "result", "rows": 1},
            "total_clocks_ns": 10,
            "reported_simulation_loop_seconds": None,
            "resource_usage": None,
            "transport_counters": {},
            "stage_timing": {},
            "rss_checkpoints": [],
            "sequence_digests": {
                "batch_done_sequence": {"sha256": digest},
            },
            "workload_metrics": {},
            "stage_reconciliation": {},
        }

    def test_completion_sequence_is_part_of_determinism_gate(self):
        records = [
            self._record("oracle", "stable"),
            self._record("oracle", "stable"),
            self._record("direct", "stable"),
            self._record("direct", "racy"),
        ]

        summary = aggregate(
            records, ["pd-stress"], ["oracle", "direct"])["scenarios"][
                "pd-stress"]

        self.assertFalse(summary["modes"]["direct"]["internally_deterministic"])
        self.assertFalse(
            summary["comparisons"]["oracle_vs_direct"]["correctness_exact"])

    def test_common_completion_sequences_must_match(self):
        records = [
            self._record("oracle", "oracle-order"),
            self._record("direct", "direct-order"),
        ]

        comparison = aggregate(
            records, ["pd-stress"], ["oracle", "direct"])["scenarios"][
                "pd-stress"]["comparisons"]["oracle_vs_direct"]

        self.assertFalse(comparison["correctness_exact"])


if __name__ == "__main__":
    unittest.main()
