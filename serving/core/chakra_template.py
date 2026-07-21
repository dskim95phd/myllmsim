import importlib
import hashlib
import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache

from serving.proto import llmservingsim_workload_pb2 as workload_pb2


_DYNAMIC_ATTRIBUTES = ("duration", "tensor_size", "comm_size")


class BoundedTemplateCache:
    """Small LRU cache for frontend-owned compiled template bundles."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Template cache capacity must be positive.")
        self.capacity = capacity
        self._entries = OrderedDict()

    def get(self, key):
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key, value):
        if key in self._entries:
            self._entries[key] = value
            self._entries.move_to_end(key)
            return None
        evicted = None
        if len(self._entries) >= self.capacity:
            evicted = self._entries.popitem(last=False)
        self._entries[key] = value
        return evicted

    def __len__(self):
        return len(self._entries)


@lru_cache(maxsize=1)
def _load_et_proto():
    try:
        return importlib.import_module("chakra.schema.protobuf.et_def_pb2")
    except ModuleNotFoundError:
        from .graph_generator import _load_llm_converter

        chakra_root = os.path.abspath("extern/graph_frontend/chakra")
        _load_llm_converter(chakra_root)
        return importlib.import_module("chakra.schema.protobuf.et_def_pb2")


def _read_varint32(data, offset):
    value = 0
    for byte_index in range(5):
        if offset >= len(data):
            raise ValueError("Truncated Chakra message length prefix.")
        byte = data[offset]
        offset += 1
        if byte_index == 4 and byte & 0xF0:
            raise ValueError("Chakra message length exceeds uint32 range.")
        value |= (byte & 0x7F) << (byte_index * 7)
        if not byte & 0x80:
            return value, offset
    raise ValueError("Invalid Chakra message length prefix.")


def _read_message(data, offset, message_type):
    size, offset = _read_varint32(data, offset)
    end = offset + size
    if end > len(data):
        raise ValueError("Truncated Chakra protobuf message.")
    message = message_type()
    message.ParseFromString(data[offset:end])
    return message, end


def _dynamic_attribute(node, attribute):
    if attribute == "duration":
        return node.duration_micros
    for item in node.attr:
        if item.name != attribute:
            continue
        value_field = item.WhichOneof("value")
        if value_field not in ("uint64_val", "int64_val"):
            raise ValueError(
                f"Node {node.id} has invalid {attribute} value type.")
        value = getattr(item, value_field)
        if value < 0:
            raise ValueError(f"Node {node.id} has negative {attribute}.")
        return value
    raise KeyError(attribute)


def _static_node_bytes(node, node_type):
    static_node = node_type()
    static_node.CopyFrom(node)
    static_node.duration_micros = 0
    for attribute in static_node.attr:
        if attribute.name in _DYNAMIC_ATTRIBUTES[1:]:
            value_field = attribute.WhichOneof("value")
            if value_field is None:
                raise ValueError(
                    f"Node {node.id} has an empty {attribute.name} value.")
            attribute.ClearField(value_field)
    return static_node.SerializeToString(deterministic=True)


def _static_metadata_bytes(metadata, metadata_type):
    static_metadata = metadata_type()
    static_metadata.CopyFrom(metadata)
    retained_attributes = [
        attribute for attribute in static_metadata.attr
        if attribute.name != "input_file"
    ]
    static_metadata.ClearField("attr")
    static_metadata.attr.extend(retained_attributes)
    return static_metadata.SerializeToString(deterministic=True)


@dataclass(frozen=True)
class ChakraGraphSnapshot:
    raw_bytes: bytes
    metadata_bytes: bytes
    nodes: dict
    static_node_bytes: dict
    dynamic_values: dict

    @classmethod
    def parse(cls, raw_bytes):
        et_proto = _load_et_proto()
        offset = 0
        metadata, offset = _read_message(
            raw_bytes, offset, et_proto.GlobalMetadata)
        nodes = {}
        static_nodes = {}
        dynamic_values = {}
        while offset < len(raw_bytes):
            node, offset = _read_message(raw_bytes, offset, et_proto.Node)
            if node.id in nodes:
                raise ValueError(f"Duplicate Chakra node id: {node.id}")
            nodes[node.id] = node
            static_nodes[node.id] = _static_node_bytes(node, et_proto.Node)
            dynamic_values[(node.id, "duration")] = _dynamic_attribute(
                node, "duration")
            for attribute in _DYNAMIC_ATTRIBUTES[1:]:
                try:
                    value = _dynamic_attribute(node, attribute)
                except KeyError:
                    continue
                dynamic_values[(node.id, attribute)] = value
        if not nodes:
            raise ValueError("Chakra graph contains no nodes.")
        return cls(
            raw_bytes=bytes(raw_bytes),
            metadata_bytes=_static_metadata_bytes(
                metadata, et_proto.GlobalMetadata),
            nodes=nodes,
            static_node_bytes=static_nodes,
            dynamic_values=dynamic_values,
        )

    def assert_same_structure(self, current):
        if self.metadata_bytes != current.metadata_bytes:
            raise ValueError("Chakra graph metadata changed within a template.")
        if self.static_node_bytes != current.static_node_bytes:
            raise ValueError("Chakra graph structure changed within a template.")
        if self.dynamic_values.keys() != current.dynamic_values.keys():
            raise ValueError("Chakra graph dynamic fields changed shape.")


@dataclass(frozen=True)
class RuntimeTraceRow:
    name: str
    comp_time: int
    input_loc: str
    input_size: int
    weight_loc: str
    weight_size: int
    output_loc: str
    output_size: int
    comm_type: str
    comm_size: int
    misc: str


@dataclass(frozen=True)
class RuntimeTraceMarker:
    kind: str
    label: str
    comm_type: str = "NONE"
    comm_size: int = 0


@dataclass(frozen=True)
class RuntimeTraceSnapshot:
    instance_header: str
    rows: dict
    markers: tuple
    signature: tuple
    ordered_records: tuple = ()

    @classmethod
    def from_records(cls, instance_header, records):
        rows = {}
        markers = []
        signature = []
        ordered_records = tuple(records)
        for record in ordered_records:
            if isinstance(record, RuntimeTraceMarker):
                markers.append(record)
                if record.kind == "EXPERT":
                    signature.append((
                        "MARKER", record.kind, record.label,
                        record.comm_type))
                else:
                    signature.append((
                        "MARKER", record.kind, record.label))
                continue
            if not isinstance(record, RuntimeTraceRow):
                raise TypeError(
                    f"Unsupported runtime trace record: {type(record)!r}")
            if record.name in rows:
                raise ValueError(
                    f"Runtime trace contains duplicate row {record.name}.")
            rows[record.name] = record
            signature.append((
                record.name,
                record.input_loc,
                record.weight_loc,
                record.output_loc,
                record.comm_type,
                record.misc,
            ))
        return cls(
            instance_header=instance_header,
            rows=rows,
            markers=tuple(markers),
            signature=tuple(signature),
            ordered_records=ordered_records,
        )

    @classmethod
    def parse(cls, trace_text):
        lines = trace_text.splitlines()
        if len(lines) < 3:
            raise ValueError("Runtime trace is missing its header.")
        try:
            expected_rows = int(lines[1].strip())
        except ValueError as error:
            raise ValueError("Runtime trace has an invalid row count.") from error
        data_lines = lines[3:]
        if len(data_lines) != expected_rows:
            raise ValueError(
                "Runtime trace row count does not match its header.")

        rows = {}
        markers = []
        signature = []
        ordered_records = []
        for line in data_lines:
            columns = line.split()
            if not columns:
                raise ValueError("Runtime trace contains an empty row.")
            if columns[0] == "EXPERT":
                if len(columns) < 2:
                    raise ValueError("EXPERT marker is missing its label.")
                comm_type = columns[2] if len(columns) > 2 else "NONE"
                try:
                    comm_size = int(columns[3]) if len(columns) > 3 else 0
                except ValueError as error:
                    raise ValueError(
                        "EXPERT marker contains an invalid communication size.") from error
                if comm_size < 0:
                    raise ValueError(
                        "EXPERT marker contains a negative communication size.")
                marker = RuntimeTraceMarker(
                    "EXPERT", columns[1], comm_type, comm_size)
                markers.append(marker)
                ordered_records.append(marker)
                signature.append((
                    "MARKER", marker.kind, marker.label, marker.comm_type))
                continue
            if columns[0] == "PIM":
                if len(columns) < 2:
                    raise ValueError("PIM marker is missing its label.")
                marker = RuntimeTraceMarker("PIM", columns[1])
                markers.append(marker)
                ordered_records.append(marker)
                signature.append(("MARKER", marker.kind, marker.label))
                continue
            if len(columns) != 11:
                raise ValueError(
                    f"Runtime trace row has {len(columns)} columns, expected 11.")
            name = columns[0]
            if name in rows:
                raise ValueError(f"Runtime trace contains duplicate row {name}.")
            try:
                numeric = [int(columns[index]) for index in (1, 3, 5, 7, 9)]
            except ValueError as error:
                raise ValueError(
                    f"Runtime trace row {name} contains a non-integer value.") from error
            if any(value < 0 for value in numeric):
                raise ValueError(
                    f"Runtime trace row {name} contains a negative value.")
            row = RuntimeTraceRow(
                name=name,
                comp_time=numeric[0],
                input_loc=columns[2],
                input_size=numeric[1],
                weight_loc=columns[4],
                weight_size=numeric[2],
                output_loc=columns[6],
                output_size=numeric[3],
                comm_type=columns[8],
                comm_size=numeric[4],
                misc=columns[10],
            )
            rows[name] = row
            ordered_records.append(row)
            signature.append((
                row.name,
                row.input_loc,
                row.weight_loc,
                row.output_loc,
                row.comm_type,
                row.misc,
            ))
        return cls(
            instance_header=lines[0].strip(),
            rows=rows,
            markers=tuple(markers),
            signature=tuple(signature),
            ordered_records=tuple(ordered_records),
        )

    def render(self):
        from .utils import formatter, header

        rendered = [
            f"{self.instance_header}\n",
            f"{len(self.ordered_records)}\n",
            header(),
        ]
        for record in self.ordered_records:
            if isinstance(record, RuntimeTraceRow):
                rendered.append(formatter(
                    record.name,
                    str(record.comp_time),
                    record.input_loc,
                    str(record.input_size),
                    record.weight_loc,
                    str(record.weight_size),
                    record.output_loc,
                    str(record.output_size),
                    record.comm_type,
                    str(record.comm_size),
                    record.misc,
                ))
                continue
            marker = f"{record.kind} {record.label}"
            if record.kind == "EXPERT":
                marker += f" {record.comm_type} {record.comm_size}"
            rendered.append(formatter(
                marker, '', '', '', '', '', '', '', '', '', ''))
        return "".join(rendered)

    def assert_same_structure(self, current):
        if (self.instance_header != current.instance_header or
                self.signature != current.signature):
            raise ValueError("Runtime trace structure changed within a template.")

    @property
    def structure_key(self):
        payload = repr((self.instance_header, self.signature)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    @property
    def template_class(self):
        row_names = tuple(self.rows)
        if any(name.startswith("KV_EVICT") for name in row_names):
            return "kv_evict"
        if any(name.startswith("KV_RELOAD") for name in row_names):
            return "kv_reload"
        marker_kinds = {marker.kind for marker in self.markers}
        if "PIM" in marker_kinds:
            return "pim"
        if "EXPERT" in marker_kinds:
            return "moe"
        if any(
                row.comm_type.startswith(("SEND", "RECV"))
                for row in self.rows.values()):
            return "pipeline"
        instance_type = self.instance_header.split("\t", 1)[0].lower()
        return instance_type or "compute"

    def expert_markers(self, label):
        if label == "START":
            return tuple(
                marker for marker in self.markers
                if marker.kind == "EXPERT" and marker.label != "END" and
                marker.comm_type != "NONE" and marker.comm_size > 0
            )
        return tuple(
            marker for marker in self.markers
            if marker.kind == "EXPERT" and marker.label == label and
            marker.comm_type != "NONE" and marker.comm_size > 0
        )


@dataclass(frozen=True)
class RuntimeValueSource:
    field: str
    row_name: str = None
    marker_label: str = None
    marker_index: int = None
    constant: int = None

    def resolve(self, trace):
        if self.constant is not None:
            return self.constant
        if self.marker_label is not None:
            markers = trace.expert_markers(self.marker_label)
            try:
                return markers[self.marker_index].comm_size
            except IndexError as error:
                raise ValueError(
                    "Runtime trace expert marker count changed.") from error
        try:
            row = trace.rows[self.row_name]
        except KeyError as error:
            raise ValueError(
                f"Runtime trace row {self.row_name} is missing.") from error
        if self.field == "pim_tensor_size":
            return row.input_size + row.output_size
        return getattr(row, self.field)


@dataclass(frozen=True)
class CompiledSlotExpression:
    system_id: int
    slot_id: int
    source: RuntimeValueSource


@dataclass(frozen=True)
class CompiledTracePlan:
    """Immutable mapping from structured runtime values to patch slots."""

    template_signature: tuple
    expressions: tuple

    def evaluate(self, template_id, batch_id, runtime_trace):
        if runtime_trace.signature != self.template_signature:
            raise ValueError("Runtime trace structure changed within a template.")
        patch = workload_pb2.BatchPatch(
            batch_id=batch_id,
            template_id=template_id,
        )
        system_patches = {}
        for expression in self.expressions:
            system_patch = system_patches.get(expression.system_id)
            if system_patch is None:
                system_patch = patch.systems.add(
                    system_id=expression.system_id)
                system_patches[expression.system_id] = system_patch
            system_patch.packed_values.append(expression.slot_id)
            system_patch.packed_values.append(
                expression.source.resolve(runtime_trace))
        return patch


def _row_for_node(trace, node_name, prefix, suffix=""):
    if not node_name.startswith(prefix) or (
            suffix and not node_name.endswith(suffix)):
        raise KeyError(node_name)
    end = -len(suffix) if suffix else None
    row_name = node_name[len(prefix):end]
    try:
        return trace.rows[row_name]
    except KeyError as error:
        raise KeyError(
            f"Chakra node {node_name} has no runtime trace row.") from error


def _row_name_from_prefixed_node(trace, node_name, prefix):
    remainder = node_name[len(prefix):]
    for row_name in sorted(trace.rows, key=len, reverse=True):
        if remainder == row_name or remainder.startswith(f"{row_name}_"):
            return row_name
    raise KeyError(f"Chakra node {node_name} has no runtime trace row.")


def _pd_kv_row_name(trace, marker_indices, direction):
    counter_key = f"PD_KV_{direction}"
    index = marker_indices.get(counter_key, 0)
    marker_indices[counter_key] = index + 1
    rows = tuple(
        row_name for row_name in trace.rows
        if row_name.startswith("qkv_proj_")
    )
    try:
        return rows[index]
    except IndexError as error:
        raise KeyError(
            "Prefill/decode KV communication count does not match the trace."
        ) from error


def _choose_size_source(trace, row_name, expected, preferred):
    row = trace.rows[row_name]
    candidates = [preferred, "input_size", "output_size"]
    for field in candidates:
        if getattr(row, field) == expected:
            return RuntimeValueSource(field, row_name=row_name)
    raise KeyError(
        f"Neither tensor size on trace row {row_name} matches {expected}.")


def _runtime_source_for_node(node, attribute, trace, marker_indices, expected):
    node_name = node.name
    if attribute == "duration":
        if node_name == "COMP_NODE_kv_migration_sync":
            return RuntimeValueSource("constant", constant=expected)
        if node_name.startswith("COMP_NODE_"):
            row = _row_for_node(trace, node_name, "COMP_NODE_")
            return RuntimeValueSource("comp_time", row_name=row.name)
        if (node_name.startswith("PIM_COMP_NODE_") and
                node_name.endswith("_PIM")):
            row = _row_for_node(
                trace, node_name, "PIM_COMP_NODE_", "_PIM")
            return RuntimeValueSource("comp_time", row_name=row.name)
        return RuntimeValueSource("constant", constant=0)

    if attribute == "tensor_size":
        if node_name.startswith("MEM_LOAD_NODE_"):
            if node_name.endswith("_INPUT"):
                row = _row_for_node(
                    trace, node_name, "MEM_LOAD_NODE_", "_INPUT")
                return RuntimeValueSource("input_size", row_name=row.name)
            if node_name.endswith("_WEIGHT"):
                row = _row_for_node(
                    trace, node_name, "MEM_LOAD_NODE_", "_WEIGHT")
                return RuntimeValueSource("weight_size", row_name=row.name)
        if (node_name.startswith("MEM_STORE_NODE_") and
                node_name.endswith("_OUTPUT")):
            # The current Chakra LLM converter sizes the final store from
            # the last layer's input tensor.
            row = _row_for_node(
                trace, node_name, "MEM_STORE_NODE_", "_OUTPUT")
            return RuntimeValueSource("input_size", row_name=row.name)
        if (node_name.startswith("MEM_STORE_NODE_") and
                node_name.endswith("_WEIGHT")):
            row = _row_for_node(
                trace, node_name, "MEM_STORE_NODE_", "_WEIGHT")
            return RuntimeValueSource("weight_size", row_name=row.name)
        if (node_name.startswith("PIM_COMP_NODE_") and
                node_name.endswith("_PIM")):
            row = _row_for_node(
                trace, node_name, "PIM_COMP_NODE_", "_PIM")
            if row.input_size + row.output_size != expected:
                raise KeyError(
                    f"PIM tensor size for {node_name} does not match the trace.")
            return RuntimeValueSource("pim_tensor_size", row_name=row.name)
        raise KeyError(
            f"Unsupported trace source for {node_name}.tensor_size")

    if attribute == "comm_size":
        if node_name.startswith("COMM_COLL_NODE_expert_start_"):
            index = marker_indices.get("START", 0)
            marker_indices["START"] = index + 1
            return RuntimeValueSource(
                "comm_size", marker_label="START", marker_index=index)
        if node_name.startswith("COMM_COLL_NODE_expert_end_"):
            index = marker_indices.get("END", 0)
            marker_indices["END"] = index + 1
            return RuntimeValueSource(
                "comm_size", marker_label="END", marker_index=index)
        if node_name.startswith("COMM_COLL_NODE_"):
            row_name = _row_name_from_prefixed_node(
                trace, node_name, "COMM_COLL_NODE_")
            return RuntimeValueSource("comm_size", row_name=row_name)
        if node_name.startswith("COMM_SEND_NODE_"):
            if node_name.startswith("COMM_SEND_NODE_kv_proj_"):
                row_name = _pd_kv_row_name(
                    trace, marker_indices, "SEND")
                return _choose_size_source(
                    trace, row_name, expected, "output_size")
            row_name = _row_name_from_prefixed_node(
                trace, node_name, "COMM_SEND_NODE_")
            return _choose_size_source(
                trace, row_name, expected, "output_size")
        if node_name.startswith("COMM_RECV_NODE_"):
            if node_name.startswith("COMM_RECV_NODE_kv_proj_"):
                row_name = _pd_kv_row_name(
                    trace, marker_indices, "RECV")
                return _choose_size_source(
                    trace, row_name, expected, "output_size")
            row_name = _row_name_from_prefixed_node(
                trace, node_name, "COMM_RECV_NODE_")
            return _choose_size_source(
                trace, row_name, expected, "input_size")
        raise KeyError(
            f"Unsupported trace source for {node_name}.comm_size")

    raise KeyError(f"Unsupported runtime trace attribute {attribute}.")


def _runtime_value_from_trace(node, attribute, trace):
    source = _runtime_source_for_node(
        node, attribute, trace, {}, expected=None)
    if source.field == "pim_tensor_size":
        row = trace.rows[source.row_name]
        return row.input_size + row.output_size
    return source.resolve(trace)


class ChakraTemplateBundle:
    def __init__(self, template_key, graphs, trace_text=None,
                 runtime_trace=None, host_timing=None):
        self.template_key = template_key
        self.graphs = {
            system_id: ChakraGraphSnapshot.parse(graph_bytes)
            for system_id, graph_bytes in sorted(graphs.items())
        }
        self.bindings = []
        self.slot_by_target = {}
        slot_id = 0
        for system_id, graph in self.graphs.items():
            for node_id, attribute in sorted(graph.dynamic_values):
                self.slot_by_target[(system_id, node_id, attribute)] = slot_id
                self.bindings.append({
                    "slot_id": slot_id,
                    "system_id": system_id,
                    "node_id": node_id,
                    "attribute": attribute,
                    "required": True,
                })
                slot_id += 1
        self.runtime_trace = None
        self.runtime_sources = {}
        self.compiled_plan = None
        self.trace_patch_reason = "No bootstrap runtime trace was provided."
        if runtime_trace is not None or trace_text is not None:
            try:
                if runtime_trace is None:
                    measurement = (
                        host_timing.measure("runtime_trace_parsing")
                        if host_timing is not None else nullcontext()
                    )
                    with measurement:
                        runtime_trace = RuntimeTraceSnapshot.parse(trace_text)
                for system_id, graph in self.graphs.items():
                    marker_indices = {}
                    for (node_id, attribute), expected in graph.dynamic_values.items():
                        source = _runtime_source_for_node(
                            graph.nodes[node_id], attribute, runtime_trace,
                            marker_indices, expected)
                        actual = source.resolve(runtime_trace)
                        if actual != expected:
                            raise ValueError(
                                f"Trace source mismatch for node {node_id} "
                                f"{attribute}: expected {expected}, got {actual}.")
                        self.runtime_sources[
                            (system_id, node_id, attribute)] = source
                self.runtime_trace = runtime_trace
                expressions = []
                for (system_id, node_id, attribute), source in sorted(
                        self.runtime_sources.items()):
                    expressions.append(CompiledSlotExpression(
                        system_id=system_id,
                        slot_id=self.slot_by_target[
                            (system_id, node_id, attribute)],
                        source=source,
                    ))
                self.compiled_plan = CompiledTracePlan(
                    template_signature=runtime_trace.signature,
                    expressions=tuple(expressions),
                )
                self.trace_patch_reason = None
            except (KeyError, ValueError) as error:
                self.trace_patch_reason = str(error)

    @property
    def supports_trace_patch(self):
        return self.runtime_trace is not None

    def build_patch(self, template_id, batch_id, graphs):
        current_graphs = {
            system_id: ChakraGraphSnapshot.parse(graph_bytes)
            for system_id, graph_bytes in sorted(graphs.items())
        }
        if current_graphs.keys() != self.graphs.keys():
            raise ValueError("Batch graph systems do not match the template.")

        patch = workload_pb2.BatchPatch(
            batch_id=batch_id,
            template_id=template_id,
        )
        for system_id, template_graph in self.graphs.items():
            current_graph = current_graphs[system_id]
            template_graph.assert_same_structure(current_graph)
            system_patch = patch.systems.add(system_id=system_id)
            for (node_id, attribute), value in sorted(
                    current_graph.dynamic_values.items()):
                system_patch.packed_values.append(self.slot_by_target[
                    (system_id, node_id, attribute)])
                system_patch.packed_values.append(value)
        return patch

    def build_patch_from_trace(self, template_id, batch_id, trace_text,
                               host_timing=None):
        if self.runtime_trace is None:
            raise ValueError(
                f"Template does not support trace patches: {self.trace_patch_reason}")
        measurement = (
            host_timing.measure("runtime_trace_parsing")
            if host_timing is not None else nullcontext()
        )
        with measurement:
            current = RuntimeTraceSnapshot.parse(trace_text)
        return self.build_patch_from_snapshot(
            template_id, batch_id, current, host_timing=host_timing)

    def build_patch_from_snapshot(self, template_id, batch_id, runtime_trace,
                                  host_timing=None):
        if self.compiled_plan is None:
            raise ValueError(
                f"Template does not support direct patches: "
                f"{self.trace_patch_reason}")
        self.runtime_trace.assert_same_structure(runtime_trace)

        measurement = (
            host_timing.measure("direct_patch_evaluation")
            if host_timing is not None else nullcontext()
        )
        with measurement:
            return self.compiled_plan.evaluate(
                template_id, batch_id, runtime_trace)
