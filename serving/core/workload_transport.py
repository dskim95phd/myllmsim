from abc import ABC, abstractmethod
from collections import deque
from itertools import count
import socket
import struct
from time import monotonic, sleep

from serving.proto import llmservingsim_workload_pb2 as workload_pb2

from .workload_protocol import HEADER_SIZE, MessageType, receive_frame, send_frame


def _system_patch_value_count(system_patch):
    if len(system_patch.packed_values) % 2:
        raise ValueError("Packed system patch must contain slot/value pairs.")
    return len(system_patch.values) + len(system_patch.packed_values) // 2


class WorkloadTransport(ABC):
    """Submission interface between the serving loop and ASTRA-Sim."""

    def set_system(self, system_id):
        """Select the system that will consume the next legacy command."""

        return None

    @abstractmethod
    def run_batch(self, workload):
        pass

    def prepare_batch(self, instance_id, patch, execute=False):
        return None

    def prepare_wave(self, wave_id, runs):
        return None

    def advance_time(self, current_time_ns):
        """Advance the backend clock across an external idle interval."""
        return None

    @abstractmethod
    def run_wave(self, workload, system_ids=None):
        pass

    @abstractmethod
    def pass_system(self):
        pass

    @abstractmethod
    def sleep_system(self):
        pass

    @abstractmethod
    def close(self):
        pass


class FileWorkloadTransport(WorkloadTransport):
    """Preserve the stdin workload-path protocol used by file mode."""

    def __init__(self, controller, process, host_timing=None):
        self.controller = controller
        self.process = process
        self.host_timing = host_timing
        self.system_id = None

    def set_system(self, system_id):
        self.system_id = int(system_id)

    def _send(self, value, counter):
        if self.system_id is None:
            raise RuntimeError(
                "File workload transport has no selected target system.")
        command = f"@{self.system_id}\t{value}"
        self.controller.write_flush(self.process, command)
        if self.host_timing is not None:
            self.host_timing.increment(counter)

    def run_batch(self, workload):
        self._send(workload, "run_batch_submissions")

    def run_wave(self, workload, system_ids=None):
        if not system_ids:
            raise ValueError("File RUN_WAVE requires participant system ids.")
        participants = ",".join(str(system_id) for system_id in system_ids)
        self._send(
            f"wave:{participants}\t{workload}", "run_wave_submissions")

    def pass_system(self):
        self._send("pass", "pass_submissions")

    def advance_time(self, current_time_ns):
        self._send(
            f"advance:{int(current_time_ns)}", "time_advance_submissions")

    def sleep_system(self):
        self._send("done", "sleep_submissions")

    def close(self):
        self._send("exit", "exit_submissions")


class IpcFileWorkloadTransport(WorkloadTransport):
    """Use framed IPC for control while ASTRA-Sim still loads `.et` files."""

    def __init__(self, socket_path, process=None, host_timing=None,
                 connect_timeout=10.0, connection=None):
        self.socket_path = socket_path
        self.process = process
        self.host_timing = host_timing
        self.connection = connection or self._connect(connect_timeout)
        self.closed = False
        self._request_ids = count(1)
        self._completion_queue = deque()
        self._hello()

    def _connect(self, timeout):
        deadline = monotonic() + timeout
        last_error = None
        while monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                stderr = ""
                if self.process.stderr is not None:
                    stderr = self.process.stderr.read().strip()
                detail = f" ASTRA-Sim stderr: {stderr}" if stderr else ""
                raise RuntimeError(
                    "ASTRA-Sim exited before the workload IPC socket became "
                    f"ready.{detail}")
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.connect(self.socket_path)
                return connection
            except (FileNotFoundError, ConnectionRefusedError, OSError) as error:
                last_error = error
                connection.close()
                sleep(0.01)
        raise TimeoutError(
            f"Timed out connecting to ASTRA-Sim workload IPC socket "
            f"{self.socket_path}: {last_error}")

    def _record_send(self, message_type, byte_count):
        if self.host_timing is None:
            return
        self.host_timing.increment("ipc_messages_sent")
        self.host_timing.increment("ipc_bytes_sent", byte_count)
        self.host_timing.increment(
            f"ipc_{message_type.name.lower()}_messages_sent")

    def _send(self, message_type, payload=b""):
        byte_count = send_frame(self.connection, message_type, payload)
        self._record_send(message_type, byte_count)

    def _receive(self):
        message_type, payload = receive_frame(self.connection)
        if self.host_timing is not None:
            self.host_timing.increment("ipc_messages_received")
            self.host_timing.increment(
                "ipc_bytes_received", HEADER_SIZE + len(payload))
            self.host_timing.increment(
                f"ipc_{message_type.name.lower()}_messages_received")
        return message_type, payload

    def _hello(self):
        measurement = (
            self.host_timing.measure("ipc_hello")
            if self.host_timing is not None else _NullMeasurement()
        )
        with measurement:
            self._send(MessageType.HELLO)
            message_type, payload = self._receive()
        if message_type is MessageType.ERROR:
            raise RuntimeError(
                f"ASTRA-Sim rejected workload IPC HELLO: "
                f"{payload.decode('utf-8', errors='replace')}")
        if message_type is not MessageType.HELLO or payload:
            raise RuntimeError(
                f"Invalid workload IPC HELLO response: "
                f"type={message_type.name}, payload_bytes={len(payload)}")

    @staticmethod
    def _parse_batch_done(payload):
        completion = workload_pb2.BatchDone()
        if not completion.ParseFromString(payload):
            raise RuntimeError("BATCH_DONE payload is not valid protobuf.")
        if completion.request_id == 0:
            raise RuntimeError("BATCH_DONE has no request id.")
        return completion

    def _receive_response(self, expected_type):
        while True:
            message_type, payload = self._receive()
            if message_type is MessageType.BATCH_DONE:
                self._completion_queue.append(
                    self._parse_batch_done(payload))
                continue
            if message_type is MessageType.ERROR:
                raise RuntimeError(
                    f"ASTRA-Sim rejected workload IPC request: "
                    f"{payload.decode('utf-8', errors='replace')}")
            if message_type is not expected_type:
                raise RuntimeError(
                    f"Invalid workload IPC response: expected "
                    f"{expected_type.name}, got {message_type.name}")
            return payload

    def wait_batch_done(self):
        measurement = (
            self.host_timing.measure("astra_execution_wait")
            if self.host_timing is not None else _NullMeasurement()
        )
        with measurement:
            if self._completion_queue:
                completion = self._completion_queue.popleft()
            else:
                while True:
                    message_type, payload = self._receive()
                    if message_type is MessageType.BATCH_DONE:
                        completion = self._parse_batch_done(payload)
                        break
                    if message_type is MessageType.ERROR:
                        raise RuntimeError(
                            "ASTRA-Sim reported an error while waiting for "
                            f"completion: {payload.decode('utf-8', errors='replace')}")
                    raise RuntimeError(
                        "Expected BATCH_DONE while waiting for execution, got "
                        f"{message_type.name}.")
        if self.host_timing is not None:
            self.host_timing.increment("batch_done_events")
        return completion

    def register_template(self, template_key, graphs, bindings=()):
        request = workload_pb2.RegisterTemplate(
            request_id=next(self._request_ids),
            template_key=template_key,
        )
        for system_id, chakra_graph in sorted(graphs.items()):
            request.graphs.add(
                system_id=system_id,
                chakra_graph=chakra_graph,
            )
        for binding in bindings:
            request.bindings.add(**binding)

        measurement = (
            self.host_timing.measure("template_registration")
            if self.host_timing is not None else _NullMeasurement()
        )
        with measurement:
            self._send(MessageType.REGISTER_TEMPLATE,
                       request.SerializeToString())
            payload = self._receive_response(MessageType.TEMPLATE_READY)

        response = workload_pb2.TemplateReady()
        if not response.ParseFromString(payload):
            raise RuntimeError("TEMPLATE_READY payload is not valid protobuf.")
        if response.request_id != request.request_id:
            raise RuntimeError(
                "TEMPLATE_READY request id does not match REGISTER_TEMPLATE.")
        if response.template_key != template_key or response.template_id == 0:
            raise RuntimeError("TEMPLATE_READY contains invalid template identity.")
        summaries = {summary.system_id: summary for summary in response.graphs}
        if set(summaries) != set(graphs):
            raise RuntimeError(
                "TEMPLATE_READY graph summaries do not match registration.")
        if any(summary.node_count == 0 or summary.root_count == 0
               for summary in summaries.values()):
            raise RuntimeError("TEMPLATE_READY contains an empty compiled graph.")
        if any(not summary.iteration_state_validated
               for summary in summaries.values()):
            raise RuntimeError(
                "TEMPLATE_READY contains an unvalidated iteration state.")
        if self.host_timing is not None:
            self.host_timing.increment("template_registrations")
            self.host_timing.increment(
                "compiled_template_graphs", len(summaries))
            self.host_timing.increment(
                "compiled_template_nodes",
                sum(summary.node_count for summary in summaries.values()),
            )
            self.host_timing.increment(
                "validated_iteration_states", len(summaries))
        return response.template_id

    @staticmethod
    def _path_payload(workload):
        return str(workload).encode("utf-8")

    def run_batch(self, workload):
        self._send(MessageType.FILE_WORKLOAD, self._path_payload(workload))
        if self.host_timing is not None:
            self.host_timing.increment("run_batch_submissions")

    def prepare_batch(self, instance_id, patch, execute=False):
        request = workload_pb2.RunBatch(
            request_id=next(self._request_ids),
            instance_id=instance_id,
            execute=execute,
        )
        request.patch.CopyFrom(patch)
        if self.host_timing is not None:
            normalized = workload_pb2.RunBatch()
            normalized.CopyFrom(request)
            normalized.request_id = 0
            self.host_timing.update_digest(
                "run_batch_sequence",
                normalized.SerializeToString(deterministic=True),
            )
        measurement = (
            self.host_timing.measure("batch_patch_submission")
            if self.host_timing is not None else _NullMeasurement()
        )
        with measurement:
            serialization = (
                self.host_timing.measure("batch_patch_serialization")
                if self.host_timing is not None else _NullMeasurement()
            )
            with serialization:
                payload = request.SerializeToString()
            self._send(MessageType.RUN_BATCH, payload)
            payload = self._receive_response(MessageType.BATCH_ACCEPTED)

        response = workload_pb2.BatchAccepted()
        if not response.ParseFromString(payload):
            raise RuntimeError("BATCH_ACCEPTED payload is not valid protobuf.")
        expected_value_count = sum(
            _system_patch_value_count(system) for system in patch.systems)
        if (response.request_id != request.request_id or
                response.batch_id != patch.batch_id or
                response.template_id != patch.template_id or
                response.system_count != len(patch.systems) or
                response.patched_value_count != expected_value_count):
            raise RuntimeError("BATCH_ACCEPTED does not match RUN_BATCH.")
        if self.host_timing is not None:
            self.host_timing.increment("run_batch_patch_submissions")
            if execute:
                self.host_timing.increment("direct_batch_submissions")
            self.host_timing.increment(
                "batch_patch_values_sent", expected_value_count)

    def run_wave(self, workload, system_ids=None):
        self._send(MessageType.FILE_WORKLOAD, self._path_payload(workload))
        if self.host_timing is not None:
            self.host_timing.increment("run_wave_submissions")

    def prepare_wave(self, wave_id, runs):
        request = workload_pb2.RunWave(
            request_id=next(self._request_ids),
            wave_id=wave_id,
        )
        expected_system_count = 0
        expected_value_count = 0
        for instance_id, patch in runs:
            run = request.runs.add(instance_id=instance_id)
            run.patch.CopyFrom(patch)
            expected_system_count += len(patch.systems)
            expected_value_count += sum(
                _system_patch_value_count(system)
                for system in patch.systems)

        if self.host_timing is not None:
            normalized = workload_pb2.RunWave()
            normalized.CopyFrom(request)
            normalized.request_id = 0
            self.host_timing.update_digest(
                "run_wave_sequence",
                normalized.SerializeToString(deterministic=True),
            )

        measurement = (
            self.host_timing.measure("wave_patch_submission")
            if self.host_timing is not None else _NullMeasurement()
        )
        with measurement:
            serialization = (
                self.host_timing.measure("wave_patch_serialization")
                if self.host_timing is not None else _NullMeasurement()
            )
            with serialization:
                payload = request.SerializeToString()
            self._send(MessageType.RUN_WAVE, payload)
            payload = self._receive_response(MessageType.BATCH_ACCEPTED)

        response = workload_pb2.BatchAccepted()
        if not response.ParseFromString(payload):
            raise RuntimeError("RUN_WAVE acknowledgement is not valid protobuf.")
        if (response.request_id != request.request_id or
                response.batch_id != wave_id or
                response.template_id != 0 or
                response.system_count != expected_system_count or
                response.patched_value_count != expected_value_count):
            raise RuntimeError("RUN_WAVE acknowledgement does not match request.")
        if self.host_timing is not None:
            self.host_timing.increment("run_wave_patch_submissions")
            self.host_timing.increment("wave_participants", len(runs))
            self.host_timing.increment(
                "wave_patch_values_sent", expected_value_count)

    def pass_system(self):
        self._send(MessageType.PASS)
        if self.host_timing is not None:
            self.host_timing.increment("pass_submissions")

    def advance_time(self, current_time_ns):
        self._send(
            MessageType.ADVANCE_TIME,
            struct.pack("!Q", int(current_time_ns)),
        )
        if self.host_timing is not None:
            self.host_timing.increment("time_advance_submissions")

    def sleep_system(self):
        self._send(MessageType.SLEEP)
        if self.host_timing is not None:
            self.host_timing.increment("sleep_submissions")

    def close(self):
        if self.closed:
            return
        try:
            self._send(MessageType.EXIT)
            if self.host_timing is not None:
                self.host_timing.increment("exit_submissions")
        finally:
            self.closed = True
            self.connection.close()


class _NullMeasurement:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False
