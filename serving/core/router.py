import bisect
import json
import os
import random
from .logger import get_logger
from .request import KVResidency


def _optional_nonnegative_int(value, field_name):
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative, got {result}.")
    return result


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            same_node_pd_only=False,
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self.same_node_pd_only = same_node_pd_only
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._is_init = True

        # Agentic session dependency tracking
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._session_schedulers = {}    # session_id -> affinity scheduler
        self._next_request_id = 0        # monotonic counter for unique request IDs
        self._agentic_session_count = 0
        self._agentic_turn_count = 0
        self._terminal_session_count = 0
        self._live_sessions = set()
        self._tool_gap_count = 0
        self._tool_gap_duration_ns = 0

        # Time-weighted realized-concurrency accounting. Counts are sampled
        # at simulation events and integrated over simulated time.
        self._concurrency_origin_ns = None
        self._concurrency_time_ns = None
        self._live_session_area_ns = 0
        self._runnable_request_area_ns = 0
        self._peak_live_sessions = 0
        self._peak_runnable_requests = 0
        self._last_live_sessions = 0
        self._last_runnable_requests = 0
        self._all_idle_interval_count = 0
        self._all_idle_duration_ns = 0

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._custom_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, CUSTOM")
        self.logger = get_logger(self.__class__)

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, role):
        return self.decode_rr_counter if role == "decode" else self.prefill_rr_counter

    def _set_counter(self, role, value):
        if role == "decode":
            self.decode_rr_counter = value
        else:
            self.prefill_rr_counter = value

    def _rr_select(self, schedulers, role):
        num_instances = len(schedulers)
        idx = self._get_counter(role) % num_instances
        self._set_counter(role, idx + 1)
        return idx

    def _rand_select(self, schedulers, role):
        return self._rnd.randrange(len(schedulers))

    def _least_load_select(self, schedulers, role):
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        best_idx = 0
        best_score = float('inf')
        num_instances = len(schedulers)
        start = self._get_counter(role) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.request) + len(
                getattr(sched, "pending_pd_handoffs", ()))
            running = sum(len(b.requests) for b in sched.inflight)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(role, (best_idx + 1) % num_instances)
        return best_idx

    def _custom_select(self, schedulers, role):
        raise NotImplementedError("Implement custom routing policy.")

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        if not os.path.isabs(path):
            path = os.path.join('..', path)
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        loaded_lines = 0

        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                row = json.loads(line)
                if 'sub_requests' in row:
                    self._load_agentic_session(row, enable_prefix_caching)
                else:
                    self._load_flat_request(row, enable_prefix_caching)
                loaded_lines += 1

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=lambda r: r['arrival_time_ns'])

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions))

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = row.get('input_tok_ids', [])
            req_data['output_hash_ids'] = row.get('output_tok_ids', [])
        self._pending_requests.append(req_data)

    def _load_agentic_session(self, row, enable_prefix_caching):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        session_id = row.get('session_id', f'session_{self._next_request_id}')
        if session_id in self._deferred_sessions:
            raise ValueError(f"Duplicate agentic session id: {session_id}")
        session_kv_ttl_ns = _optional_nonnegative_int(
            row.get('session_kv_ttl_ns'), 'session_kv_ttl_ns')
        reused_prefix_toks = []
        for index, sub_req in enumerate(sub_reqs):
            reused = _optional_nonnegative_int(
                sub_req.get('reused_prefix_toks'),
                f'sub_requests[{index}].reused_prefix_toks')
            input_toks = int(sub_req['input_toks'])
            if reused is not None and reused > input_toks:
                raise ValueError(
                    f"sub_requests[{index}].reused_prefix_toks cannot exceed "
                    f"input_toks ({reused} > {input_toks}).")
            reused_prefix_toks.append(reused)
        reuse_previous_kv = bool(row.get('reuse_previous_kv', False))
        retain_for_next = [
            index + 1 < len(sub_reqs) and (
                reuse_previous_kv or
                (reused_prefix_toks[index + 1] or 0) > 0)
            for index in range(len(sub_reqs))
        ]
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        arrival_ns = int(row['arrival_time_ns'])

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
            'reuse_previous_kv': reuse_previous_kv,
            'session_kv_ttl_ns': session_kv_ttl_ns,
            'reused_prefix_toks': reused_prefix_toks,
            'retain_for_next': retain_for_next,
        }
        self._agentic_session_count += 1
        self._agentic_turn_count += len(sub_reqs)

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = {
            'index': base_id,
            'input_toks': int(first['input_toks']),
            'output_toks': int(first['input_toks'] + first['output_toks']),
            'arrival_time_ns': arrival_ns,
            'session_id': session_id,
            'sub_request_index': 0,
            'session_has_next': len(sub_reqs) > 1,
            'retain_session_kv': retain_for_next[0],
            'reuse_previous_kv': reuse_previous_kv,
            'reused_prefix_toks': reused_prefix_toks[0],
            'session_kv_ttl_ns': session_kv_ttl_ns,
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = first.get('input_tok_ids', [])
            req_data['output_hash_ids'] = first.get('output_tok_ids', [])
        self._pending_requests.append(req_data)
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def route_arrived_requests(self, current_time_ns):
        """Route requests that have arrived by current_time_ns to instances.

        Called at the start of each iteration in the main simulation loop.
        Returns the number of newly routed requests.
        """
        routed = 0
        blocked = []
        while self._pending_idx < len(self._pending_requests):
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break

            session_id = req_data.get('session_id')
            if (session_id is not None and
                    req_data.get('sub_request_index') == 0):
                self._live_sessions.add(session_id)
            sched = self._session_schedulers.get(session_id)
            if (sched is not None and sched.pd_type == "prefill" and
                    self._pd_session_cpu_bridge_pending(session_id)):
                # Preserve the request in the pending queue until the decode
                # scheduler commits its mandatory D2H park. This is required
                # when tool_duration_ns=0 and release shares a timestamp with
                # decode completion.
                blocked.append(req_data)
                self._pending_requests.pop(self._pending_idx)
                continue
            if sched is None:
                instance_id = self._select_instance(
                    self.prefill_schedulers, "prefill")
                sched = self.prefill_schedulers[instance_id]
                if (session_id is not None and
                        sched.enable_session_kv_retention):
                    self._session_schedulers[session_id] = sched

            session_metadata = {
                'session_id': session_id,
                'sub_request_index': req_data.get('sub_request_index'),
                'session_has_next': req_data.get('session_has_next', False),
                'retain_session_kv': req_data.get(
                    'retain_session_kv', False),
                'reuse_previous_kv': req_data.get('reuse_previous_kv', False),
                'reused_prefix_toks': req_data.get('reused_prefix_toks'),
                'session_kv_ttl_ns': req_data.get('session_kv_ttl_ns'),
            }

            if sched.enable_prefix_caching:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                    req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                ], is_init=self._is_init, session_metadata=session_metadata)
            else:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                ], is_init=self._is_init, session_metadata=session_metadata)

            self._pending_idx += 1
            routed += 1

        # Keep bridge-blocked requests at the front of the unconsumed region,
        # without letting one session delay unrelated requests that have also
        # arrived. Their relative order remains unchanged.
        if blocked:
            self._pending_requests[self._pending_idx:self._pending_idx] = blocked

        return routed

    def _pd_session_cpu_bridge_pending(self, session_id):
        for scheduler in self.decode_schedulers:
            state = getattr(scheduler, "session_kv_states", {}).get(session_id)
            if (state is not None and state.mandatory_cpu_offload and
                    state.residency is not KVResidency.CPU and
                    not state.invalidated):
                return True
        return False

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return self._pending_idx < len(self._pending_requests)

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request_id, completion_time_ns):
        """Called when a request finishes. Releases the next sub-request in
        the session chain after the tool_call duration elapses.

        For flat requests (not in a session), this is a no-op.
        """
        session_info = self._request_to_session.pop(request_id, None)
        if session_info is None:
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(sub_reqs[completed_idx].get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            if tool_duration_ns > 0:
                self._tool_gap_count += 1
                self._tool_gap_duration_ns += tool_duration_ns
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            req_data = {
                'index': next_id,
                'input_toks': int(next_sub['input_toks']),
                'output_toks': int(next_sub['input_toks'] + next_sub['output_toks']),
                'arrival_time_ns': release_time_ns,
                'session_id': session_id,
                'sub_request_index': next_idx,
                'session_has_next': next_idx + 1 < len(sub_reqs),
                'retain_session_kv': session['retain_for_next'][next_idx],
                'reuse_previous_kv': session['reuse_previous_kv'],
                'reused_prefix_toks': session['reused_prefix_toks'][next_idx],
                'session_kv_ttl_ns': session['session_kv_ttl_ns'],
            }
            if self._enable_prefix_caching:
                req_data['input_hash_ids'] = next_sub.get('input_tok_ids', [])
                req_data['output_hash_ids'] = next_sub.get('output_tok_ids', [])
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(req_data)
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete — all sub-requests have been released
            del self._deferred_sessions[session_id]
            self._session_schedulers.pop(session_id, None)
            self._live_sessions.discard(session_id)
            self._terminal_session_count += 1

    def _insert_pending_sorted(self, req_data):
        """Insert a request into _pending_requests maintaining arrival-time
        sort order for the not-yet-consumed portion (from _pending_idx onward)."""
        arrival = req_data['arrival_time_ns']
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_requests[mid]['arrival_time_ns'] <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there are agentic sessions with unreleased sub-requests."""
        return bool(self._deferred_sessions)

    @property
    def terminal_session_count(self):
        return self._terminal_session_count

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        if self._pending_idx < len(self._pending_requests):
            return self._pending_requests[self._pending_idx]['arrival_time_ns']
        return None

    def _count_runnable_requests(self, current_time_ns):
        pending = sum(
            request['arrival_time_ns'] <= current_time_ns
            for request in self._pending_requests[self._pending_idx:]
        )
        waiting = 0
        running = 0
        handoffs = 0
        for scheduler in self.schedulers:
            waiting += sum(
                request.arrival <= current_time_ns
                for request in scheduler.request
            )
            running += sum(
                len(batch.requests) for batch in scheduler.inflight)
            handoffs += len(getattr(scheduler, 'pending_pd_handoffs', ()))
        return pending + waiting + running + handoffs

    def observe_concurrency(self, current_time_ns):
        """Integrate realized session/request concurrency at an event time."""
        now = int(current_time_ns)
        if self._concurrency_time_ns is None:
            self._concurrency_origin_ns = now
            self._concurrency_time_ns = now
        if now < self._concurrency_time_ns:
            now = self._concurrency_time_ns
        elapsed = now - self._concurrency_time_ns
        self._live_session_area_ns += self._last_live_sessions * elapsed
        self._runnable_request_area_ns += self._last_runnable_requests * elapsed
        if self._last_live_sessions > 0 and self._last_runnable_requests == 0:
            self._all_idle_duration_ns += elapsed

        live = len(self._live_sessions)
        runnable = self._count_runnable_requests(now)
        was_all_idle = (
            self._last_live_sessions > 0 and
            self._last_runnable_requests == 0
        )
        if live > 0 and runnable == 0 and not was_all_idle:
            self._all_idle_interval_count += 1
        self._peak_live_sessions = max(self._peak_live_sessions, live)
        self._peak_runnable_requests = max(
            self._peak_runnable_requests, runnable)
        self._last_live_sessions = live
        self._last_runnable_requests = runnable
        self._concurrency_time_ns = now

    def concurrency_summary(self, current_time_ns):
        self.observe_concurrency(current_time_ns)
        observed_duration = max(
            0, self._concurrency_time_ns - self._concurrency_origin_ns)
        denominator = observed_duration or 1
        return {
            'agentic_sessions': self._agentic_session_count,
            'agentic_turns': self._agentic_turn_count,
            'terminal_sessions': self._terminal_session_count,
            'peak_live_sessions': self._peak_live_sessions,
            'mean_live_sessions': self._live_session_area_ns / denominator,
            'peak_runnable_requests': self._peak_runnable_requests,
            'mean_runnable_requests': (
                self._runnable_request_area_ns / denominator),
            'all_idle_interval_count': self._all_idle_interval_count,
            'all_idle_duration_ns': self._all_idle_duration_ns,
            'tool_gap_count': self._tool_gap_count,
            'tool_gap_duration_ns': self._tool_gap_duration_ns,
            'observed_duration_ns': observed_duration,
        }

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def transfer_prefill_request(self, requests, current_time_ns=0):
        for req in requests:
            source_scheduler = self.schedulers[req.instance_id]
            if not self.same_node_pd_only:
                instance_id = self._select_instance(self.decode_schedulers, "decode")
                self.decode_schedulers[instance_id].add_decode(req)
                continue
            prefill_node_id = source_scheduler.node_id
            local_decodes = [
                scheduler for scheduler in self.decode_schedulers
                if scheduler.node_id == prefill_node_id
            ]
            if not local_decodes:
                raise RuntimeError(
                    f"No decode instance on node {prefill_node_id} for prefill "
                    f"request #{req.id}. CPU KV offloading Phase 1 supports "
                    "same-node prefill/decode disaggregation only.")
            decode_index = self._select_instance(local_decodes, "decode")
            target_scheduler = local_decodes[decode_index]
            if (source_scheduler.enable_kv_offloading and
                    target_scheduler.enable_kv_offloading):
                target_scheduler.enqueue_pd_handoff(
                    req, source_scheduler, current_time_ns)
            else:
                target_scheduler.add_decode(req)
