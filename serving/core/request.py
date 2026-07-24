from dataclasses import dataclass
from enum import Enum, auto


class KVResidency(Enum):
    """Location of a request's live KV state in the CPU-offload baseline."""

    NPU = auto()
    CPU = auto()
    NPU_TO_CPU = auto()
    CPU_TO_NPU = auto()


class KVMigrationDirection(Enum):
    NPU_TO_CPU = auto()
    CPU_TO_NPU = auto()


class BatchKind(Enum):
    COMPUTE = auto()
    KV_EVICT = auto()
    KV_RELOAD = auto()
    PD_HANDOFF = auto()


@dataclass(frozen=True)
class KVMigration:
    migration_id: int
    request_id: int | None
    direction: KVMigrationDirection
    bytes_per_rank: int
    bytes_full_cluster: int
    submit_time_ns: int
    session_id: str | None = None


@dataclass
class SessionKVState:
    """KV allocation retained between turns of one linear session."""

    session_id: str
    model_name: str
    sub_request_index: int
    source_request_id: int
    cached_tokens: int
    bytes_per_rank: int
    bytes_full_cluster: int
    residency: KVResidency
    owner_node_id: int
    owner_instance_id: int
    num_npus: int
    tp_size: int
    block_size: int
    kv_fp: int
    parked_at_ns: int
    expires_at_ns: int | None
    last_access_ns: int
    migration_id: int | None = None
    invalidated: bool = False
    residency_since_ns: int = 0
    mandatory_cpu_offload: bool = False

    def begin_offload(self, migration_id):
        if self.residency is not KVResidency.NPU:
            raise RuntimeError(
                f"Session {self.session_id} cannot start NPU-to-CPU "
                f"migration from {self.residency.name}.")
        self.residency = KVResidency.NPU_TO_CPU
        self.migration_id = migration_id

    def complete_offload(self, migration_id):
        self._check_migration(KVResidency.NPU_TO_CPU, migration_id)
        self.residency = KVResidency.CPU
        self.migration_id = None

    def begin_reload(self, migration_id):
        if self.residency is not KVResidency.CPU:
            raise RuntimeError(
                f"Session {self.session_id} cannot start CPU-to-NPU "
                f"migration from {self.residency.name}.")
        self.residency = KVResidency.CPU_TO_NPU
        self.migration_id = migration_id

    def complete_reload(self, migration_id, bytes_per_rank):
        self._check_migration(KVResidency.CPU_TO_NPU, migration_id)
        self.residency = KVResidency.NPU
        self.migration_id = None
        self.bytes_per_rank = bytes_per_rank
        self.bytes_full_cluster = bytes_per_rank * self.num_npus

    def cancel_migration(self, migration_id):
        if self.migration_id != migration_id:
            raise RuntimeError(
                f"Session {self.session_id} migration id mismatch: expected "
                f"{self.migration_id}, got {migration_id}.")
        if self.residency is KVResidency.NPU_TO_CPU:
            self.residency = KVResidency.NPU
        elif self.residency is KVResidency.CPU_TO_NPU:
            self.residency = KVResidency.CPU
        else:
            raise RuntimeError(
                f"Session {self.session_id} is not migrating from "
                f"{self.residency.name}.")
        self.migration_id = None

    def _check_migration(self, expected_residency, migration_id):
        if self.residency is not expected_residency:
            raise RuntimeError(
                f"Session {self.session_id} expected "
                f"{expected_residency.name}, got {self.residency.name}.")
        if self.migration_id != migration_id:
            raise RuntimeError(
                f"Session {self.session_id} migration id mismatch: expected "
                f"{self.migration_id}, got {migration_id}.")


# class that manages request of astra-sim
class Request:
    def __init__(
            self, id, model, input, output, arrival, instance_id,
            input_hash_ids=None, output_hash_ids=None, is_init=True,
            session_id=None, sub_request_index=None, session_has_next=False,
            retain_session_kv=False,
            reuse_previous_kv=False, reused_prefix_toks=None,
            session_kv_ttl_ns=None):
        self.id = id
        self.model = model
        self.input = input  # Always keep original input length
        self.output = output
        self.arrival = arrival
        self.instance_id = instance_id
        self.kv_owner_instance_id = instance_id
        self.is_init = is_init
        self.original_input = input
        self.num_computed_tokens = 0  # Tracks actual computed tokens (vLLM style)
        # When live KV is discarded, rebuild it through this logical token
        # position before generating another output token.
        self.recompute_kv_target_tokens = None
        # ``evict`` is retained for compatibility with the existing scheduler
        # paths. New CPU-offload code uses ``kv_residency`` as the source of
        # truth and keeps the two values synchronized.
        self.evict = False
        self.kv_residency = KVResidency.NPU
        self.kv_migration_id = None
        self.last_scheduled_ns = 0
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        self.ttft = -1
        self.tpot = -1
        self.itl = []
        self.recent_end = 0

        # For chunked prefill
        self.chunk_len = 0  # tokens scheduled for this request in the current step

        # For prefix caching modeling
        self.input_hash_ids = input_hash_ids
        self.output_hash_ids = output_hash_ids
        self.prefix_cache_hit = 0
        self.npu_cache_hit = 0
        self.storage_cache_hit = 0
        self.npu_last_node = None
        self.cpu_last_node = None
        self.storage_last_node = None

        # For prefix cache lock tracking
        self._prefix_locked = False
        self._prefix_npu_stats_counted = False
        self._prefix_storage_stats_counted = False

        # Agentic session identity and declared continuation semantics.
        self.session_id = session_id
        self.sub_request_index = sub_request_index
        self.session_has_next = session_has_next
        self.retain_session_kv = retain_session_kv
        self.reuse_previous_kv = reuse_previous_kv
        self.reused_prefix_toks = reused_prefix_toks
        self.session_kv_ttl_ns = session_kv_ttl_ns
        self.session_kv_hit_tokens = 0
        self.session_kv_hit_tier = None
        # Session hit/miss metrics are committed exactly once. CPU claims stay
        # provisional until the H2D migration completes successfully.
        self.session_kv_outcome_recorded = False
        self.pending_session_kv = False
        self.pending_session_kv_bytes_per_rank = 0
        # None means the active Request owns its KV. A session id means the
        # completed Request transferred ownership to SessionKVState.
        self.kv_owner_session_id = None

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def add_latency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        self.input = self.original_input
        if self.output == self.input + 1:
            self.tpot = 0
        else:
            self.tpot = (self.latency - self.ttft) // (self.output - self.input - 1)
    
    def add_itl(self, current): # 
        self.itl.append(current - self.recent_end)
        self.recent_end = current

    def set_que_delay(self, current):
        self.queuing_delay = current - self.arrival
    
    def set_ttft(self, current):
        self.ttft = current - self.arrival
        self.recent_end = current
    
    def log(self):
        print("         scheduled request : {}".format(self.__dict__))
    
    def is_prefill(self):
        """Check if request is still in prefill phase (has tokens left to compute)"""
        return self.num_computed_tokens < self.prefill_target_tokens()

    def prefill_target_tokens(self):
        if self.recompute_kv_target_tokens is not None:
            return self.recompute_kv_target_tokens
        return self.original_input

    def begin_recompute_preemption(self):
        """Discard live KV while preserving logical generation progress."""
        if self.num_computed_tokens <= 0:
            raise RuntimeError(
                f"Request #{self.id} has no computed KV to discard.")
        if self.kv_residency is not KVResidency.NPU:
            raise RuntimeError(
                f"Request #{self.id} cannot be recompute-preempted from "
                f"{self.kv_residency.name} residency.")
        target_tokens = self.num_computed_tokens
        self.recompute_kv_target_tokens = target_tokens
        self.num_computed_tokens = 0
        self.chunk_len = 0
        self.prefix_cache_hit = 0
        self.npu_cache_hit = 0
        self.storage_cache_hit = 0
        return target_tokens

    def finish_recompute_prefill(self):
        if self.recompute_kv_target_tokens is None:
            raise RuntimeError(
                f"Request #{self.id} has no recompute prefill to finish.")
        if self.num_computed_tokens < self.recompute_kv_target_tokens:
            raise RuntimeError(
                f"Request #{self.id} has rebuilt only "
                f"{self.num_computed_tokens} of "
                f"{self.recompute_kv_target_tokens} tokens.")
        self.recompute_kv_target_tokens = None

    def is_kv_on_cpu(self):
        return self.kv_residency is KVResidency.CPU

    def is_kv_on_npu(self):
        return self.kv_residency is KVResidency.NPU

    def is_kv_migrating(self):
        return self.kv_residency in {
            KVResidency.NPU_TO_CPU,
            KVResidency.CPU_TO_NPU,
        }

    def begin_kv_offload(self, migration_id):
        if self.kv_residency is not KVResidency.NPU:
            raise RuntimeError(
                f"Request #{self.id} cannot start NPU-to-CPU migration from "
                f"{self.kv_residency.name}.")
        self.kv_residency = KVResidency.NPU_TO_CPU
        self.kv_migration_id = migration_id
        self.evict = True

    def complete_kv_offload(self, migration_id):
        self._check_migration(KVResidency.NPU_TO_CPU, migration_id)
        self.kv_residency = KVResidency.CPU
        self.kv_migration_id = None
        self.evict = True

    def begin_kv_reload(self, migration_id):
        if self.kv_residency is not KVResidency.CPU:
            raise RuntimeError(
                f"Request #{self.id} cannot start CPU-to-NPU migration from "
                f"{self.kv_residency.name}.")
        self.kv_residency = KVResidency.CPU_TO_NPU
        self.kv_migration_id = migration_id
        self.evict = True

    def complete_kv_reload(self, migration_id):
        self._check_migration(KVResidency.CPU_TO_NPU, migration_id)
        self.kv_residency = KVResidency.NPU
        self.kv_migration_id = None
        self.evict = False

    def cancel_kv_migration(self, migration_id):
        if self.kv_migration_id != migration_id:
            raise RuntimeError(
                f"Request #{self.id} migration id mismatch: expected "
                f"{self.kv_migration_id}, got {migration_id}.")
        if self.kv_residency is KVResidency.NPU_TO_CPU:
            self.kv_residency = KVResidency.NPU
            self.evict = False
        elif self.kv_residency is KVResidency.CPU_TO_NPU:
            self.kv_residency = KVResidency.CPU
            self.evict = True
        else:
            raise RuntimeError(
                f"Request #{self.id} is not migrating from "
                f"{self.kv_residency.name}.")
        self.kv_migration_id = None

    def _check_migration(self, expected_residency, migration_id):
        if self.kv_residency is not expected_residency:
            raise RuntimeError(
                f"Request #{self.id} expected {expected_residency.name}, got "
                f"{self.kv_residency.name}.")
        if self.kv_migration_id != migration_id:
            raise RuntimeError(
                f"Request #{self.id} migration id mismatch: expected "
                f"{self.kv_migration_id}, got {migration_id}.")

    def mark_kv_on_cpu(self):
        self.kv_residency = KVResidency.CPU
        self.evict = True

    def mark_kv_on_npu(self):
        self.kv_residency = KVResidency.NPU
        self.evict = False

    def transfer_kv_owner(self, source_instance_id, destination_instance_id):
        """Move NPU-resident live-KV ownership between PD instances."""
        if self.kv_residency is not KVResidency.NPU:
            raise RuntimeError(
                f"Request #{self.id} cannot transfer KV ownership from "
                f"{self.kv_residency.name} residency.")
        if self.kv_owner_instance_id != source_instance_id:
            raise RuntimeError(
                f"Request #{self.id} KV owner mismatch: expected "
                f"{source_instance_id}, got {self.kv_owner_instance_id}.")
        self.kv_owner_instance_id = destination_instance_id
        self.instance_id = destination_instance_id

    def transfer_kv_to_session(self):
        """Mark this completed turn's KV as owned by its session record."""
        if self.session_id is None:
            raise RuntimeError(
                f"Request #{self.id} cannot park KV without a session id.")
        if not self.session_has_next:
            raise RuntimeError(
                f"Terminal request #{self.id} cannot retain session KV.")
        if not self.retain_session_kv:
            raise RuntimeError(
                f"Request #{self.id} has no declared continuation reuse.")
        if self.kv_residency is not KVResidency.NPU:
            raise RuntimeError(
                f"Request #{self.id} cannot park KV from "
                f"{self.kv_residency.name} residency.")
        if self.kv_owner_session_id is not None:
            raise RuntimeError(
                f"Request #{self.id} KV is already owned by session "
                f"{self.kv_owner_session_id}.")
        self.kv_owner_session_id = self.session_id

# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, batch_time, kv_size, evict=0, load=0, kind=BatchKind.COMPUTE, migrations=None):
        self.batch_id = batch_id
        self.kind = kind
        self.migrations = list(migrations or [])
        self.model = model
        self.total_len = total_len
        self.kv_len = kv_len
        self.batch_time = batch_time
        self.fired = [] # systems that fired this batch
        self.requests = []
        self.end = []
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
        # for attn prediction
        self.q_list = q_list
        self.k_list = k_list
        self.num_prefill = num_prefill
        self.num_decode = num_decode
        self.prefill_q_list = prefill_q_list
        self.prefill_k_list = prefill_k_list
        self.decode_k_list = decode_k_list

        # for debugging
        self.scheduled_tokens = None
    def log(self):
        print("-------------------------Batch Log------------------------")
        for key in self.__dict__.keys():
            if key == 'requests':
                continue
            print("         {} : {}".format(key, self.__dict__[key]))
        for req in self.requests:
            req.log()
        print("----------------------------------------------------------")
